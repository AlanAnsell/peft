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
            logger.setLevel(self.args.get_process_log_level())

            if sft_args is None:
                self.sft_args = SftArguments()
            else:
                self.sft_args = sft_args

            if self.sft_args.selection_algorithm != "none":
                self.add_callback(
                    ReselectionCallback(
                        self,
                        initial_reallocation=self.sft_args.selection_algorithm != "SM3"
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

        def replacement_rate(self):
            return (
                self.sft_args.initial_reselection_rate *
                math.exp(
                    self._reselection_rate_exponent * (
                        self.state.global_step // self.sft_args.reselection_steps
                    )
                )
            )

        def select(self):
            if self.sft_args.selection_algorithm == "SM3":
                self.select_sm3()
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

        @torch.no_grad()
        def select_sm3(self):
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

                optimizer_state = self.optimizer.state[delta.values]
                #logger.info(optimizer_state)
                row_grads_sq = optimizer_state['accumulator_0']
                col_grads_sq = optimizer_state['accumulator_1']
                importances = torch.outer(row_grads_sq, col_grads_sq).view(-1)
                importance_indices = torch.arange(
                    0,
                    delta.dense_numel,
                    device=is_valid_candidate.device,
                )
                importances = importances[is_valid_candidate]
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

            logger.info(
                f'Replacing {n_replacements} ({100*n_replacements/total_params:.4f}%)'
            )

        def create_optimizer(self):
            if self.sft_args.selection_algorithm != "SM3":
                return super().create_optimizer()

            if self.optimizer is None:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in self.model.named_parameters() if p.requires_grad
                        ],
                    },
                ]

                optimizer_kwargs = {
                    'lr': self.args.learning_rate,
                    #'momentum': 0.9,
                }

                shapes = {
                    delta.values: delta.shape
                    for _, delta in self.active_sft_deltas()
                }
                indices = {
                    delta.values: delta.indices
                    for _, delta in self.active_sft_deltas()
                }

                self.optimizer = SM3(
                    optimizer_grouped_parameters,
                    indices,
                    shapes,
                    **optimizer_kwargs
                )

            return self.optimizer

    return _SftTrainer
