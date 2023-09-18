import logging
import math

import torch
import torch_scatter

from transformers import (
    TrainerCallback,
)

from .layer import Linear
from .optimizer import SM3

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ReselectionCallback(TrainerCallback):

    def __init__(self, trainer, initial_reallocation=True):
        self.trainer = trainer
        self.initial_reallocation = initial_reallocation

    def on_step_begin(self, args, state, control, **kwargs):
        if (
            state.global_step % self.trainer.sft_args.reselection_steps == 0 and
            (self.initial_reallocation or state.global_step > 0)
        ):
            self.trainer.select()


def SftTrainer(_Trainer):

    class _SftTrainer(_Trainer):

        def __init__(
            self,
            *args,
            sft_args=None,
            **kwargs
        ):
            super().__init__(*args, **kwargs)
            #logger.setLevel(self.args.get_process_log_level())

            if sft_args is None:
                self.sft_args = SftArguments()
            else:
                self.sft_args = sft_args

            if self.sft_args.selection_algorithm != "none":
                self.add_callback(
                    ReselectionCallback(
                        self,
                        initial_reallocation=self.sft_args.selection_algorithm != "importance"
                    )
                )

            if self.args.max_steps > 0:
                max_steps = self.args.max_steps
            else:
                train_dataloader = self.get_train_dataloader()
                len_dataloader = len(train_dataloader)
                num_update_steps_per_epoch = (
                    len_dataloader //
                    self.args.gradient_accumulation_steps
                )
                num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
                max_steps = math.ceil(self.args.num_train_epochs * num_update_steps_per_epoch)
            num_reselections = max_steps // self.sft_args.reselection_steps
            if num_reselections == 0:
                logger.warning(
                    f'--reselection_steps = {self.sft_args.reselection_steps}, '
                    f'but training is only expected to last {max_steps} steps.'
                )
            else:
                self._reselection_rate_exponent = (
                    math.log(0.001 / self.sft_args.initial_reselection_rate) /
                    (max_steps // self.sft_args.reselection_steps)
                )

            self._reg_loss = 0.0
            self.reallocation_scores = {}

            for n, p in self.model.named_parameters():
                logger.info(f'{n}: {p.size()}, {p.requires_grad}, {p.dtype}')

        def replacement_rate(self):
            if self.state.global_step == 0 and self.sft_args.selection_algorithm == 'rigl':
                return 1.0

            return (
                self.sft_args.initial_reselection_rate *
                math.exp(
                    self._reselection_rate_exponent * (
                        self.state.global_step // self.sft_args.reselection_steps
                    )
                )
            )

        def reallocation_hook(self, module_name):

            @torch.no_grad()
            def _reallocation_hook(grad):
                m = self.model.get_submodule(module_name)
                grad = grad.view(-1)
                if module_name in self.reallocation_scores:
                    candidate_indices, candidate_grads = self.reallocation_scores[module_name]
                    candidate_grads += grad[candidate_indices]
                else:
                    #num_candidates = int(
                    #    self.replacement_rate() *
                    #    len(m.sft_delta[m.active_adapter].values) *
                    #    self.sft_args.candidates_per_replacement_slot
                    #)
                    num_candidates = len(m.sft_delta[m.active_adapter].values)

                    _, candidate_indices = torch.topk(
                        torch.abs(grad),
                        num_candidates,
                        largest=True,
                        sorted=False,
                    )
                    candidate_grads = grad[candidate_indices]
                    self.reallocation_scores[module_name] = (candidate_indices, candidate_grads)

            return _reallocation_hook

        def select(self):
            if self.sft_args.selection_algorithm == "importance":
                self.select_by_importance()
            if self.sft_args.selection_algorithm == "rigl":
                self.select_rigl()
            else:
                raise ValueError(
                    f'Invalid selection method {self.sft_args.selection_algorithm}'
                )

        def active_sft_deltas(self):
            for n, m in self.model.named_modules():
                if (
                    isinstance(m, Linear) and
                    m.active_adapter is not None and
                    m.active_adapter in m.sft_delta
                ):
                    yield (
                        f'{n}.sft_delta.{m.active_adapter}',
                        m.sft_delta[m.active_adapter]
                    )

        def select_rigl(self):
            self.reallocation_scores = {}
            for n, m in self.model.named_modules():
                if (
                    isinstance(m, Linear) and
                    m.active_adapter is not None and
                    m.active_adapter in m.sft_delta
                ):
                    m.apply_hook(self.reallocation_hook(n))

            dataloader = self.get_train_dataloader()
            for i, batch in enumerate(dataloader):
                if i == 0:
                    logger.info(f'First batch: {batch}')
                if i >= self.sft_args.selection_accumulation_steps:
                    break
                self.training_step(self.model, batch)

            for n, m in self.model.named_modules():
                if not (
                    isinstance(m, Linear) and
                    m.active_adapter is not None and
                    m.active_adapter in m.sft_delta
                ):
                    continue
                m.apply_hook(None)

            with torch.no_grad():
                n_replacements = 0
                total_params = 0

                for module_name, (candidate_indices, candidate_grads) in self.reallocation_scores.items():
                    m = self.model.get_submodule(module_name)
                    delta = m.sft_delta[m.active_adapter]
                    delta.values.grad = None

                    optimizer_state = self.optimizer.state[delta.values]

                    num_to_reallocate = int(len(delta.values) * self.replacement_rate())
                    _, changing_indices = torch.topk(
                        torch.abs(delta.values),
                        num_to_reallocate,
                        largest=False,
                        sorted=True,
                    )
                    outgoing_params = delta.indices[changing_indices]
                    is_outgoing = torch_scatter.scatter(
                        torch.ones_like(outgoing_params, dtype=torch.bool),
                        outgoing_params,
                        dim_size=delta.dense_numel,
                    )
                    assert torch.sum(is_outgoing) == num_to_reallocate
                    is_current = torch_scatter.scatter(
                        torch.ones_like(delta.indices, dtype=torch.bool),
                        delta.indices,
                        dim_size=delta.dense_numel,
                    )
                    is_remaining = is_current & ~is_outgoing

                    is_valid_candidate = ~is_remaining[candidate_indices]
                    candidate_indices = candidate_indices[is_valid_candidate]
                    candidate_grads = candidate_grads[is_valid_candidate]
                    _, best_candidate_indices = torch.topk(
                        torch.abs(candidate_grads),
                        min(num_to_reallocate, len(candidate_grads)),
                        largest=True,
                        sorted=False,
                    )
                    incoming_params = candidate_indices[best_candidate_indices]
                    is_incoming = torch_scatter.scatter(
                        torch.ones_like(incoming_params, dtype=torch.bool),
                        incoming_params,
                        dim_size=delta.dense_numel,
                    )
                    assert torch.sum(is_incoming) == len(best_candidate_indices)
                    outgoing_is_incoming = is_incoming[outgoing_params]
                    changing_indices = changing_indices[~outgoing_is_incoming]
                    incoming_is_outgoing = is_outgoing[incoming_params]
                    assert torch.sum(outgoing_is_incoming) == torch.sum(incoming_is_outgoing)
                    #logger.info(f'{module_name}: {torch.sum(outgoing_is_incoming)}/{len(best_candidate_indices)} overlapping params')
                    incoming_params = incoming_params[~incoming_is_outgoing]
                    changing_indices = changing_indices[:len(incoming_params)]

                    n_replacements += len(changing_indices)
                    total_params += len(delta.indices)

                    delta.indices[changing_indices] = incoming_params
                    delta.values[changing_indices] = 0.0

                    delta.indices.data, sort_order = torch.sort(delta.indices)
                    delta.values.data = delta.values[sort_order]

                    for optim_aux in ['exp_avg', 'exp_avg_sq']:
                        optimizer_params = optimizer_state.get(optim_aux, None)
                        if optimizer_params is not None:
                            optimizer_params[changing_indices] = 0.0
                            optimizer_state[optim_aux] = optimizer_params[sort_order]
                        else:
                            assert self.state.global_step == 0

                logger.info(
                    f'Replacing {n_replacements} ({100*n_replacements/total_params:.4f}%)'
                )

                self.reallocation_scores = {}

        def sm3_importances(self, state):
            row_grads_sq = state['accumulator_0']
            col_grads_sq = state['accumulator_1']
            return torch.outer(row_grads_sq, col_grads_sq)

        def adam_importances(self, state, delta):
            row_indices = delta.indices // delta.shape[1]
            col_indices = delta.indices - (row_indices * delta.shape[1])
            #abs_momentum = torch.abs(state['exp_avg'])
            #abs_momentum = state['exp_avg']
            abs_momentum = torch.abs(delta.values)
            row_means = torch_scatter.scatter(
                abs_momentum,
                row_indices.long(),
                dim_size=delta.shape[0],
                reduce="mean",
            )
            col_means = torch_scatter.scatter(
                abs_momentum,
                col_indices.long(),
                dim_size=delta.shape[1],
                reduce="mean",
            )
            #return torch.outer(row_means.abs(), col_means.abs())
            return row_means.view(-1, 1) + col_means.view(1, -1)

        def weight_importances(self, delta):
            optimizer_state = self.optimizer.state[delta.values]
            if all(
                f'accumulator_{i}' in optimizer_state
                for i in range(len(delta.shape))
            ):
                return self.sm3_importances(optimizer_state)
            elif 'exp_avg' in optimizer_state and 'exp_avg_sq' in optimizer_state:
                return self.adam_importances(optimizer_state, delta)
            else:
                raise ValueError(f'Unsupported optimizer type {type(self.optimizer)}')

        @torch.no_grad()
        def select_by_importance(self):
            #for n, p in sorted(list(self.model.named_parameters())):
            #    logger.info(f'{n}: {p.size()}, {p.requires_grad}, {p.device}, {p.dtype}')
            #logger.info(self.optimizer.state)

            n_replacements = 0
            total_params = 0

            for _, delta in self.active_sft_deltas():
                delta.values.grad = None

                num_to_reallocate = int(len(delta.values) * self.replacement_rate())
                _, changing_indices = torch.topk(
                    torch.abs(delta.values),
                    num_to_reallocate,
                    largest=False,
                    sorted=True,
                )
                outgoing_params = delta.indices[changing_indices]
                is_outgoing = torch_scatter.scatter(
                    torch.ones_like(outgoing_params, dtype=torch.bool),
                    outgoing_params.long(),
                    dim_size=delta.dense_numel,
                )
                assert torch.sum(is_outgoing) == num_to_reallocate
                is_current = torch_scatter.scatter(
                    torch.ones_like(delta.indices, dtype=torch.bool),
                    delta.indices.long(),
                    dim_size=delta.dense_numel,
                )
                is_valid_candidate = ~(is_current & ~is_outgoing)

                importances = self.weight_importances(delta)
                importances = importances.view(-1)[is_valid_candidate]
                importance_indices = torch.arange(
                    0,
                    delta.dense_numel,
                    device=is_valid_candidate.device,
                )
                importance_indices = importance_indices[is_valid_candidate]
                _, best_candidate_indices = torch.topk(
                    importances,
                    num_to_reallocate,
                    largest=True,
                    sorted=False,
                )
                incoming_params = importance_indices[best_candidate_indices]
                is_incoming = torch_scatter.scatter(
                    torch.ones_like(incoming_params, dtype=torch.bool),
                    incoming_params,
                    dim_size=delta.dense_numel,
                )
                assert torch.sum(is_incoming) == len(best_candidate_indices)
                outgoing_is_incoming = is_incoming[outgoing_params]
                changing_indices = changing_indices[~outgoing_is_incoming]
                incoming_is_outgoing = is_outgoing[incoming_params]
                assert torch.sum(outgoing_is_incoming) == torch.sum(incoming_is_outgoing)
                #logger.info(f'{delta_name}: {torch.sum(outgoing_is_incoming)}/{len(best_candidate_indices)} overlapping params')
                incoming_params = incoming_params[~incoming_is_outgoing]
                changing_indices = changing_indices[:len(incoming_params)]

                n_replacements += len(changing_indices)
                total_params += len(delta.indices)

                delta.indices[changing_indices] = incoming_params.to(dtype=torch.int32)
                delta.values[changing_indices] = 0.0

                delta.indices.data, sort_order = torch.sort(delta.indices)
                delta.values.data = delta.values[sort_order]

                optimizer_state = self.optimizer.state[delta.values]
                for optim_aux in ['exp_avg', 'exp_avg_sq']:
                    optimizer_params = optimizer_state.get(optim_aux, None)
                    if optimizer_params is not None:
                        optimizer_params[changing_indices] = 0.0
                        optimizer_state[optim_aux] = optimizer_params[sort_order]

            logger.info(
                f'Replacing {n_replacements} ({100*n_replacements/total_params:.4f}%)'
            )

        def training_step(self, *args, **kwargs):
            loss = super().training_step(*args, **kwargs)

            l1_reg = self.sft_args.l1_reg
            if l1_reg != 0.0:
                l1_dists = []
                n_params = 0
                for n, p in self.model.named_parameters():
                    if p.requires_grad and 'sft_delta' in n:
                        l1_dists.append(torch.sum(torch.abs(p)))
                        n_params += p.numel()
                if l1_dists:
                    reg_loss = l1_reg * torch.sum(torch.stack(l1_dists)) / n_params
                    if self.do_grad_scaling:
                        self.scaler.scale(reg_loss).backward()
                    #elif self.use_apex:
                    #    with amp.scale_loss(reg_loss, self.optimizer) as scaled_loss:
                    #        scaled_loss.backward()
                    else:
                        self.accelerator.backward(reg_loss)
                    self._reg_loss += float(reg_loss)

            return loss

        #def create_optimizer(self):
        #    if self.sft_args.selection_algorithm != "SM3":
        #        return super().create_optimizer()

        #    if self.optimizer is None:
        #        optimizer_grouped_parameters = [
        #            {
        #                "params": [
        #                    p for n, p in self.model.named_parameters() if p.requires_grad
        #                ],
        #            },
        #        ]

        #        optimizer_kwargs = {
        #            'lr': self.args.learning_rate,
        #            #'momentum': 0.9,
        #        }

        #        shapes = {
        #            delta.values: delta.shape
        #            for _, delta in self.active_sft_deltas()
        #        }
        #        indices = {
        #            delta.values: delta.indices
        #            for _, delta in self.active_sft_deltas()
        #        }

        #        self.optimizer = SM3(
        #            optimizer_grouped_parameters,
        #            indices,
        #            shapes,
        #            **optimizer_kwargs
        #        )

        #    return self.optimizer

        def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
            if self.control.should_log:
                logs: Dict[str, float] = {}
                tr_loss_scalar = tr_loss.item()
                # reset tr_loss to zero
                tr_loss -= tr_loss

                steps_delta = self.state.global_step - self._globalstep_last_logged
                logs["loss"] = round(tr_loss_scalar / steps_delta, 4)
                logs["learning_rate"] = self._get_learning_rate()

                if self._reg_loss != 0.0:
                    logs['l1_reg_loss'] = round(self._reg_loss / steps_delta, 4)
                    self._reg_loss = 0.0
                if hasattr(self.model, 'losses'):
                    acc_steps_delta = steps_delta * self.args.gradient_accumulation_steps
                    for loss_name, loss_value in self.model.losses.items():
                        logs[loss_name] = round(loss_value / acc_steps_delta, 4)
                    self.model.losses.clear()

                self._total_loss_scalar += tr_loss_scalar
                self._globalstep_last_logged = self.state.global_step
                self.store_flos()

                self.log(logs)

            metrics = None
            if self.control.should_evaluate:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
                self._report_to_hp_search(trial, epoch, metrics)

            if self.control.should_save:
                self._save_checkpoint(model, trial, metrics=metrics)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    return _SftTrainer