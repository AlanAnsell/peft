import logging
import math

import numpy as np
import torch
import torch_scatter

from transformers import (
    TrainerCallback,
)
from accelerate.optimizer import AcceleratedOptimizer
from deepspeed.runtime import ZeROOptimizer

from .layer import Linear
from .optimizer import SftAdamW, SftSM3

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def zero_and_reorder(optimizer, param, param_name, changing_indices, reorder=None, init_momenta={}):
    if isinstance(optimizer, AcceleratedOptimizer):
        optimizer = optimizer.optimizer

    if isinstance(optimizer, ZeROOptimizer):
        param_name = '.'.join(param_name.split('.')[1:])
        mapping = optimizer._param_slice_mappings[0][param_name]
        start = mapping.start
        end = start + mapping.numel
        state = optimizer.optimizer.state
        assert len(state) == 1
        momenta = list(optimizer.optimizer.state.values())[0]
        #logger.info(f'momentum_flat_tensor {momentum_flat_tensor.size()}: {momentum_flat_tensor}')
        for optim_aux in ['age', 'exp_avg', 'exp_avg_sq']:
            momentum_flat_tensor = momenta[optim_aux]
            init = init_momenta.get(optim_aux, None)
            if init is not None:
                if isinstance(init, torch.Tensor):
                    init = init.to(dtype=momentum_flat_tensor.dtype)
                momentum_flat_tensor[changing_indices + start] = init
            else:
                momentum_flat_tensor[changing_indices + start] = 0.0
            if reorder is not None:
                momentum_flat_tensor[start:end] = momentum_flat_tensor[start:end][reorder]
    else:
        optimizer_state = optimizer.state[param]
        for optim_aux in ['age', 'exp_avg', 'exp_avg_sq']:
            optimizer_params = optimizer_state[optim_aux]
            init = init_momenta.get(optim_aux, None)
            if init is not None:
                if isinstance(init, torch.Tensor):
                    init = init.to(dtype=optimizer_params.dtype)
                optimizer_params[changing_indices] = init
            else:
                optimizer_params[changing_indices] = 0.0
            if reorder is not None:
                optimizer_state[optim_aux] = optimizer_params[reorder]


class SftSelector:

    def __init__(self, model, optimizer, sft_args, total_update_steps, grad_accumulation_steps):
        self.model = model
        self.optimizer = optimizer
        self.sft_args = sft_args
        self.total_update_steps = total_update_steps
        self.grad_accumulation_steps = grad_accumulation_steps
        self.completed_steps = 0
        self.begin_selection_phase()

    def step(self):
        self.completed_steps += 1
        
        if (self.completed_steps + 1) % self.sft_args.reselection_steps == 0:
            self.begin_selection_phase()
        
        if self.completed_steps % self.sft_args.reselection_steps == self.sft_args.selection_accumulation_steps:
            self.end_selection_phase()

    def begin_selection_phase(self):
        if self.sft_args.selection_algorithm == "sm3":
            return

        logger.info('Beginning selection phase')
        self.reallocation_scores = {}
        for n, m in self.model.named_modules():
            if (
                isinstance(m, Linear) and
                m.active_adapter is not None and
                m.active_adapter in m.sft_delta
            ):
                m.apply_hook(self.reallocation_hook(n))

    def end_selection_phase(self):
        logger.info('Ending selection phase')
        if self.completed_steps > self.total_update_steps:
            return

        if self.sft_args.selection_algorithm != "sm3":
            for n, m in self.model.named_modules():
                if (
                    isinstance(m, Linear) and
                    m.active_adapter is not None and
                    m.active_adapter in m.sft_delta
                ):
                    m.apply_hook(None)

        if self.completed_steps == self.sft_args.selection_accumulation_steps:
            p = 1
        else:
            p = self.sft_args.initial_reselection_rate * (1 - self.completed_steps / self.total_update_steps)
        self.select(p)
        self.reallocation_scores = {}

    def grad_scores(self, grads, grads_sq, samples):
        variances = (grads_sq - grads**2 / samples) / (samples - 1)
        return torch.abs(grads / samples) - 1.65 * torch.sqrt(variances / samples)
        #return torch.abs(grads) / samples

    def reallocation_hook(self, module_name):

        accumulation_steps = 0
        #previous_change_indices = None
        #original_candidates = None

        @torch.no_grad()
        def _reallocation_hook(grad):
            nonlocal accumulation_steps
            #nonlocal original_candidates
            #nonlocal previous_change_indices

            m = self.model.get_submodule(module_name)
            grad = grad.reshape(-1)
            if module_name in self.reallocation_scores:
                candidate_indices, candidate_grads, candidate_grads_sq, samples = self.reallocation_scores[module_name]
                candidate_grads += grad[candidate_indices]
                candidate_grads_sq.addcmul_(grad[candidate_indices], grad[candidate_indices])
                samples += 1

                if (
                    self.sft_args.candidate_reselection_steps is not None and
                    accumulation_steps % self.sft_args.candidate_reselection_steps == 0
                ):
                    is_invalid = torch_scatter.scatter(
                        torch.ones_like(candidate_indices, dtype=torch.bool),
                        candidate_indices.long(),
                        dim_size=grad.numel(),
                    )

                    num_changes = int(self.sft_args.candidate_reselection_proportion * len(candidate_grads))
                    _, incoming_candidates = torch.topk(
                        torch.abs(grad),
                        num_changes,
                        largest=True,
                        sorted=False,
                    )
                    is_valid_candidate = ~(is_invalid[incoming_candidates])
                    is_invalid = torch_scatter.scatter(
                        torch.ones_like(incoming_candidates, dtype=torch.bool),
                        incoming_candidates,
                        dim_size=grad.numel(),
                    )
                    incoming_candidates = incoming_candidates[is_valid_candidate]

                    candidate_scores = self.grad_scores(candidate_grads, candidate_grads_sq, samples)
                    _, change_indices = torch.topk(
                        candidate_scores, 
                        num_changes,
                        largest=False,
                        sorted=True,
                    )
                    outgoing_candidates = candidate_indices[change_indices]
                    change_indices = change_indices[~(is_invalid[outgoing_candidates])]
                    change_indices = change_indices[:incoming_candidates.numel()]
                    assert change_indices.numel() == incoming_candidates.numel()

                    incoming_grads = grad[incoming_candidates]
                    candidate_indices[change_indices] = incoming_candidates.to(candidate_indices.dtype)
                    candidate_grads[change_indices] = incoming_grads
                    candidate_grads_sq[change_indices] = incoming_grads * incoming_grads
                    samples[change_indices] = 1

                    #is_original_candidate = torch_scatter.scatter(
                    #    torch.ones_like(original_candidates, dtype=torch.bool),
                    #    original_candidates.long(),
                    #    dim_size=grad.numel(),
                    #)
                    #is_current_candidate = torch_scatter.scatter(
                    #    torch.ones_like(candidate_indices, dtype=torch.bool),
                    #    candidate_indices.long(),
                    #    dim_size=grad.numel(),
                    #)
                    #num_overlaps = torch.sum(is_original_candidate & is_current_candidate).item()
                    #overlap_prop = num_overlaps / candidate_indices.numel()
                    #logger.info(
                    #    f'{100 * overlap_prop:.4f}% ({num_overlaps}/{candidate_indices.numel()}) of original '
                    #    f'candidates remain.'
                    #)
                #previous_change_indices = change_indices
            else:
                num_candidates = len(m.sft_delta[m.active_adapter].values)
                _, candidate_indices = torch.topk(
                    torch.abs(grad),
                    num_candidates,
                    largest=True,
                    sorted=False,
                )
                candidate_grads = grad[candidate_indices]
                self.reallocation_scores[module_name] = (
                    candidate_indices.to(m.sft_delta[m.active_adapter].indices.dtype),
                    candidate_grads,
                    candidate_grads * candidate_grads,
                    torch.ones_like(candidate_grads)
                )
                #original_candidates = candidate_indices
            accumulation_steps += 1

        return _reallocation_hook

    def select(self, p):
        if self.sft_args.selection_algorithm == "sm3":
            self.select_sm3(p)
        elif self.sft_args.selection_algorithm == "rigl":
            self.select_rigl(p)
        elif self.sft_args.selection_algorithm == "acc":
            self.select_accumulative()
        #elif self.sft_args.selection_algorithm == "proj":
        #    self.select_proj(p)
        #elif self.sft_args.selection_algorithm == "lt-sft":
        #    self.merge_and_reselect()
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
    def select_rigl(self, p):
        n_replacements = 0
        total_params = 0

        for module_name, (
            candidate_indices,
            candidate_grads,
            candidate_grads_sq,
            candidate_samples
        ) in self.reallocation_scores.items():
            m = self.model.get_submodule(module_name)
            delta = m.sft_delta[m.active_adapter]
            delta.values.grad = None

            num_to_reallocate = int(len(delta.values) * p)
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
            is_remaining = is_current & ~is_outgoing

            is_valid_candidate = ~is_remaining[candidate_indices]
            candidate_indices = candidate_indices[is_valid_candidate]
            candidate_grads = candidate_grads[is_valid_candidate]
            candidate_grads_sq = candidate_grads_sq[is_valid_candidate]
            candidate_samples = candidate_samples[is_valid_candidate]
            if self.sft_args.candidate_reselection_steps is None:
                candidate_scores = torch.abs(candidate_grads)
            else:
                candidate_scores = self.grad_scores(
                    candidate_grads,
                    candidate_grads_sq,
                    candidate_samples
                )
            best_scores, best_candidate_indices = torch.topk(
                candidate_scores,
                min(num_to_reallocate, len(candidate_grads)),
                largest=True,
                sorted=True,
            )
            #logger.info(f'Regrowth range = ({best_scores[-1]:.8f}, {best_scores[0]:.8f})')
            incoming_params = candidate_indices[best_candidate_indices]
            incoming_grads = candidate_grads[best_candidate_indices]
            incoming_grads_sq = candidate_grads_sq[best_candidate_indices]
            incoming_samples = candidate_samples[best_candidate_indices]
            is_incoming = torch_scatter.scatter(
                torch.ones_like(incoming_params, dtype=torch.bool),
                incoming_params.long(),
                dim_size=delta.dense_numel,
            )
            assert torch.sum(is_incoming) == len(best_candidate_indices)
            outgoing_is_incoming = is_incoming[outgoing_params]
            changing_indices = changing_indices[~outgoing_is_incoming]
            incoming_is_outgoing = is_outgoing[incoming_params]
            assert torch.sum(outgoing_is_incoming) == torch.sum(incoming_is_outgoing)
            incoming_params = incoming_params[~incoming_is_outgoing]
            incoming_grads = incoming_grads[~incoming_is_outgoing]
            incoming_grads_sq = incoming_grads_sq[~incoming_is_outgoing]
            incoming_samples = incoming_samples[~incoming_is_outgoing]
            changing_indices = changing_indices[:len(incoming_params)]
            #logger.info(f'{module_name}: {len(changing_indices)}/{len(delta.indices)}')

            n_replacements += len(changing_indices)
            #logger.info(f'{module_name} has {len(delta.indices)} indices')
            total_params += len(delta.indices)

            delta.indices[changing_indices] = incoming_params.to(delta.indices.dtype)
            delta.values[changing_indices] = 0.0

            #delta.indices.data, sort_order = torch.sort(delta.indices)
            #delta.values.data = delta.values[sort_order]

            zero_and_reorder(
                self.optimizer,
                delta.values,
                f'{module_name}.sft_delta.{m.active_adapter}.values',
                changing_indices,
                #sort_order,
                init_momenta={
                    'age': incoming_samples / self.grad_accumulation_steps,
                    'exp_avg': incoming_grads / incoming_samples,
                    'exp_avg_sq': incoming_grads_sq / incoming_samples,
                }
            )
                #else:
                #    assert self.state.global_step == 0

        logger.info(
            f'Replacing {n_replacements} ({100*n_replacements/total_params:.4f}%)'
        )

    @torch.no_grad()
    def select_sm3(self, p):
        n_replacements = 0
        total_params = 0

        for _, delta in self.active_sft_deltas():
            num_to_reallocate = int(len(delta.values) * p)
            _, changing_indices = torch.topk(
                torch.abs(delta.values),
                num_to_reallocate,
                largest=False,
                sorted=True,
            )

            is_current = torch_scatter.scatter(
                torch.ones_like(delta.indices, dtype=torch.bool),
                delta.indices.long(),
                dim_size=delta.dense_numel,
            )
            is_valid_candidate = ~is_current

            optimizer_state = self.optimizer.state[delta.values]
            row_grads_sq = optimizer_state['accumulator_0']
            col_grads_sq = optimizer_state['accumulator_1']
            estimated_momenta = torch.outer(row_grads_sq, col_grads_sq)
            estimated_momenta = estimated_momenta.view(-1)[is_valid_candidate]
            candidate_indices = torch.arange(
                0,
                delta.dense_numel,
                device=is_valid_candidate.device,
            )
            candidate_indices = candidate_indices[is_valid_candidate]
            _, best_candidate_indices = torch.topk(
                estimated_momenta,
                num_to_reallocate,
                largest=True,
                sorted=False,
            )
            incoming_params = candidate_indices[best_candidate_indices]

            n_replacements += len(changing_indices)
            total_params += len(delta.indices)

            delta.indices[changing_indices] = incoming_params.to(delta.indices.dtype) #.to(dtype=torch.int32)
            delta.values[changing_indices] = 0.0

            delta.indices.data, sort_order = torch.sort(delta.indices)
            delta.values.data = delta.values[sort_order]

        logger.info(
            f'Replacing {n_replacements} ({100*n_replacements/total_params:.4f}%)'
        )

    @torch.no_grad()
    def select_accumulative(self):
        num_overlaps = 0
        total_indices = 0
        for module_name, (
            candidate_indices,
            candidate_grads,
            candidate_grads_sq,
            candidate_samples
        ) in self.reallocation_scores.items():
            m = self.model.get_submodule(module_name)
            delta = m.sft_delta[m.active_adapter]
            delta.merge(m.weight)
            delta.values.grad = None

            _, new_candidate_indices = torch.topk(
                torch.abs(candidate_grads),
                len(delta.values),
                largest=True,
                sorted=False,
            )
            new_params = candidate_indices[new_candidate_indices]
            new_samples = candidate_samples[new_candidate_indices]
            new_grads = candidate_grads[new_candidate_indices]
            new_grads_sq = candidate_grads_sq[new_candidate_indices]

            is_old_param = torch_scatter.scatter(
                torch.ones_like(delta.indices, dtype=torch.bool),
                delta.indices.long(),
                dim_size=delta.dense_numel,
            )
            is_new_param = torch_scatter.scatter(
                torch.ones_like(new_params, dtype=torch.bool),
                new_params.long(),
                dim_size=delta.dense_numel,
            )
            is_incoming = (~is_old_param)[new_params]
            is_leaving = (~is_new_param)[delta.indices]
            
            delta.indices[is_leaving] = new_params[is_incoming]
            delta.values.zero_()

            changing_indices = torch.nonzero(is_leaving).squeeze(1)
            new_samples = new_samples[is_incoming]
            zero_and_reorder(
                self.optimizer,
                delta.values,
                f'{module_name}.sft_delta.{m.active_adapter}.values',
                changing_indices,
                init_momenta={
                    'age': new_samples,
                    'exp_avg': new_grads[is_incoming] / new_samples,
                    'exp_avg_sq': new_grads_sq[is_incoming] / new_samples,
                }
            )

            is_remaining_param = is_old_param & is_new_param
            num_overlaps += torch.sum(is_remaining_param).item()
            total_indices += delta.indices.numel()
        
        logger.info(f'Replacement overlap: {100*num_overlaps/total_indices:.4f}%')

        self.reallocation_scores = {}


    #    def project_candidates(self, module_name):
    #        candidate_indices, candidate_grads = self.reallocation_scores[module_name]
    #        m = self.model.get_submodule(module_name)
    #        delta = m.sft_delta[m.active_adapter]

    #        is_current = torch_scatter.scatter(
    #            torch.ones_like(delta.indices, dtype=torch.bool),
    #            delta.indices,
    #            dim_size=delta.dense_numel,
    #        )
    #        is_valid_candidate = ~is_current[candidate_indices]
    #        candidate_indices = candidate_indices[is_valid_candidate]
    #        candidate_grads = candidate_grads[is_valid_candidate]
    #        candidate_grads /= self.sft_args.selection_accumulation_steps
    #        candidate_projections = -self.sft_args.projection_alpha * candidate_grads 
    #        return candidate_projections, candidate_indices
    #    
    #    def project_incumbents(self, module_name):
    #        m = self.model.get_submodule(module_name)
    #        delta = m.sft_delta[m.active_adapter]
    #        optimizer_state = self.optimizer.state[delta.values]
    #        incumbent_grads = optimizer_state.get('exp_avg', None)
    #        if incumbent_grads is None:
    #            assert self.state.global_step == 0
    #            incumbent_projections = torch.zeros_like(delta.values)
    #        else:
    #            incumbent_projections = delta.values - self.sft_args.projection_alpha * incumbent_grads

    #        return incumbent_projections

    #    def reallocate(self, module_name, changing_indices, incoming_params):
    #        m = self.model.get_submodule(module_name)
    #        delta = m.sft_delta[m.active_adapter]

    #        delta.indices[changing_indices] = incoming_params
    #        delta.values[changing_indices] = 0.0

    #        delta.indices.data, sort_order = torch.sort(delta.indices)
    #        delta.values.data = delta.values[sort_order]

    #        optimizer_buffers = ['exp_avg', 'exp_avg_sq']
    #        optimizer_state = self.optimizer.state[delta.values]
    #        for optim_aux in optimizer_buffers:
    #            optimizer_params = optimizer_state.get(optim_aux, None)
    #            if optimizer_params is not None:
    #                optimizer_params[changing_indices] = 0.0
    #                optimizer_state[optim_aux] = optimizer_params[sort_order]

    #    def select_proj(self):
    #        self.model.eval()
    #        self.reallocation_scores = {}
    #        for n, m in self.model.named_modules():
    #            if (
    #                isinstance(m, Linear) and
    #                m.active_adapter is not None and
    #                m.active_adapter in m.sft_delta
    #            ):
    #                m.apply_hook(self.reallocation_hook(n))

    #        dataloader = self.get_train_dataloader()
    #        for i, batch in enumerate(dataloader):
    #            if i >= self.sft_args.selection_accumulation_steps:
    #                break
    #            self.training_step(self.model, batch, selecting=True)

    #        for n, m in self.model.named_modules():
    #            if not (
    #                isinstance(m, Linear) and
    #                m.active_adapter is not None and
    #                m.active_adapter in m.sft_delta
    #            ):
    #                continue
    #            m.apply_hook(None)
    #        self.model.train()

    #        with torch.no_grad():
    #            n_replacements = 0
    #            total_params = 0

    #            for module_name in self.reallocation_scores.keys():
    #                m = self.model.get_submodule(module_name)
    #                delta = m.sft_delta[m.active_adapter]
    #                delta.values.grad = None

    #                candidate_projections, candidate_indices = self.project_candidates(module_name)
    #                candidate_projections, candidate_order = torch.sort(
    #                    torch.abs(candidate_projections),
    #                    descending=True,
    #                )
    #                candidate_indices = candidate_indices[candidate_order]
    #                incumbent_projections = self.project_incumbents(module_name)
    #                incumbent_projections, incumbent_indices = torch.sort(
    #                    torch.abs(incumbent_projections),
    #                    descending=False,
    #                )

    #                max_replacements = min(len(candidate_indices), len(incumbent_indices))
    #                replace = candidate_projections[:max_replacements] > incumbent_projections[:max_replacements]
    #                local_replacements = torch.sum(replace)
    #                #logger.info(
    #                #    f'Replacing {local_replacements} ({100*local_replacements/len(incumbent_indices):.4f}%) '
    #                #    f'of {module_name} sparse params.'
    #                #)
    #                n_replacements += local_replacements
    #                total_params += len(incumbent_indices)

    #                incoming_params = candidate_indices[:local_replacements]
    #                changing_indices = incumbent_indices[:local_replacements]

    #                self.reallocate(module_name, changing_indices, incoming_params)

    #            logger.info(
    #                f'Replacing {n_replacements} ({100*n_replacements/total_params:.4f}%)'
    #            )

    #            self.reallocation_scores = {}

    #    def sm3_importances(self, state):
    #        row_grads_sq = state['accumulator_0']
    #        col_grads_sq = state['accumulator_1']
    #        return torch.outer(row_grads_sq, col_grads_sq)

    #    def adam_importances(self, state, delta):
    #        row_indices = delta.indices // delta.shape[1]
    #        col_indices = delta.indices - (row_indices * delta.shape[1])
    #        abs_momentum = torch.abs(state['exp_avg'])
    #        abs_momentum = state['exp_avg']
    #        #abs_momentum = torch.abs(delta.values)
    #        row_means = torch_scatter.scatter(
    #            abs_momentum,
    #            row_indices.long(),
    #            dim_size=delta.shape[0],
    #            reduce="mean",
    #        )
    #        col_means = torch_scatter.scatter(
    #            abs_momentum,
    #            col_indices.long(),
    #            dim_size=delta.shape[1],
    #            reduce="mean",
    #        )
    #        #return torch.outer(row_means.abs(), col_means.abs())
    #        return row_means.view(-1, 1) + col_means.view(1, -1)

    #    def weight_importances(self, delta):
    #        optimizer_state = self.optimizer.state[delta.values]
    #        if all(
    #            f'accumulator_{i}' in optimizer_state
    #            for i in range(len(delta.shape))
    #        ):
    #            return self.sm3_importances(optimizer_state)
    #        elif 'exp_avg' in optimizer_state and 'exp_avg_sq' in optimizer_state:
    #            return self.adam_importances(optimizer_state, delta)
    #        else:
    #            raise ValueError(f'Unsupported optimizer type {type(self.optimizer)}')



class SelectorStepCallback(TrainerCallback):

    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        self.trainer._selector.step()


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

            self._selector = SftSelector(
                self.model,
                self.create_optimizer(),
                self.sft_args,
                max_steps,
                self.args.gradient_accumulation_steps,
            )
            self.add_callback(SelectorStepCallback(self))


        #def num_trainable_with_threshold(self, threshold):
        #    num_trainable = 0
        #    for n, m in self.model.named_modules():
        #        if (
        #            isinstance(m, Linear) and
        #            m.active_adapter is not None and
        #            m.active_adapter in m.sft_delta
        #        ):
        #            num_trainable += torch.sum(m.weight >= threshold)
        #    return num_trainable

        #def train(self, *args, **kwargs):
        #    if self.sft_args.selection_algorithm == 'lt-sft':
        #        self._pretrained_weights = {}
        #        self._ltsft_phase = 'dense'
        #        for n, m in self.model.named_modules():
        #            if (
        #                isinstance(m, Linear) and
        #                m.active_adapter is not None and
        #                m.active_adapter in m.sft_delta
        #            ):
        #                self._pretrained_weights[n] = m.weight.detach().cpu()

        #    outputs = super().train(*args, **kwargs)
        #    
        #    if self.sft_args.selection_algorithm == 'lt-sft':
        #        self._ltsft_phase = 'sparse'
        #        with torch.no_grad():
        #            thresh_lb = 0.0
        #            thresh_ub = None
        #            total_trainable_params = 0
        #            for n, m in self.model.named_modules():
        #                if (
        #                    isinstance(m, Linear) and
        #                    m.active_adapter is not None and
        #                    m.active_adapter in m.sft_delta
        #                ):
        #                    # temporarily replace weights with their absolute differences
        #                    m.weight.sub_(self._pretrained_weights[n].to(m.weight.device)).abs_()
        #                    if thresh_ub is None:
        #                        thresh_ub = torch.max(m.weight)
        #                    else:
        #                        thresh_ub = max(thresh_ub, torch.max(m.weight))
        #                    total_trainable_params += m.sft_delta[m.active_adapter].values.numel()

        #            num_lb = total_trainable_params
        #            num_ub = 0
        #            while num_ub < num_lb:
        #                logger.info(f'({num_lb} @ {thresh_lb:.6f}, {num_ub} @ {thresh_ub:.6f})')
        #                mid = (thresh_lb + thresh_ub) / 2
        #                res = self.num_trainable_with_threshold(mid)
        #                if res >= total_trainable_params:
        #                    if mid == thresh_lb:
        #                        break
        #                    thresh_lb = mid
        #                    num_lb = res
        #                else:
        #                    if mid == thresh_ub:
        #                        break
        #                    thresh_ub = mid
        #                    num_ub = res
        #            threshold = thresh_lb

        #            num_trainable = 0
        #            for n, m in self.model.named_modules():
        #                if (
        #                    isinstance(m, Linear) and
        #                    m.active_adapter is not None and
        #                    m.active_adapter in m.sft_delta
        #                ):
        #                    delta = m.sft_delta[m.active_adapter]
        #                    tunable_indices = torch.nonzero(m.weight.view(-1) >= threshold, as_tuple=False).view(-1)
        #                    num_trainable += tunable_indices.numel()
        #                    delta.indices.data = tunable_indices
        #                    delta.values.data = torch.zeros_like(
        #                        delta.indices.data,
        #                        dtype=delta.values.dtype,
        #                    )
        #                    m.weight.copy_(self._pretrained_weights[n])
        #                    del self._pretrained_weights[n]
        #            logger.info(f'Selected {num_trainable} params with threshold {threshold:.6f}')

        #        for n, p in self.model.named_parameters():
        #            logger.info(f'{n}: {p.size()}, {p.requires_grad}, {p.dtype}')

        #        self.remove_callback(ReselectionCallback)
        #        self.optimizer = None
        #        self.lr_scheduler = None
        #        outputs = super().train(*args, **kwargs)

        #    return outputs


        #def training_step(self, *args, selecting=False, **kwargs):
        #    #if not selecting:
        #    #    for n, p in self.model.named_parameters():
        #    #        assert p.grad is None or torch.all(p.grad == 0.0)

        #    loss = super().training_step(*args, **kwargs)

        #    l1_reg = self.sft_args.l1_reg
        #    if l1_reg != 0.0:
        #        l1_dists = []
        #        n_params = 0
        #        for n, p in self.model.named_parameters():
        #            if p.requires_grad and 'sft_delta' in n:
        #                l1_dists.append(torch.sum(torch.abs(p)))
        #                n_params += p.numel()
        #        if l1_dists:
        #            reg_loss = l1_reg * torch.sum(torch.stack(l1_dists)) / n_params
        #            if self.do_grad_scaling:
        #                self.scaler.scale(reg_loss).backward()
        #            #elif self.use_apex:
        #            #    with amp.scale_loss(reg_loss, self.optimizer) as scaled_loss:
        #            #        scaled_loss.backward()
        #            else:
        #                self.accelerator.backward(reg_loss)
        #            self._reg_loss += float(reg_loss)

        #    return loss

        def create_optimizer(self):
            if self.optimizer is None:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in self.model.named_parameters() if p.requires_grad
                        ],
                    },
                ]

                _, optimizer_kwargs = _Trainer.get_optimizer_cls_and_kwargs(self.args)

                if self.sft_args.selection_algorithm == "sm3":
                    shapes = {
                        delta.values: delta.shape
                        for _, delta in self.active_sft_deltas()
                    }
                    indices = {
                        delta.values: delta.indices
                        for _, delta in self.active_sft_deltas()
                    }

                    self.optimizer = SftSM3(
                        optimizer_grouped_parameters,
                        indices,
                        shapes,
                        **optimizer_kwargs
                    )
                else:
                    self.optimizer = SftAdamW(optimizer_grouped_parameters, **optimizer_kwargs)

            return self.optimizer

        def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
            if self.control.should_log:
                logs: Dict[str, float] = {}
                tr_loss_scalar = tr_loss.item()
                # reset tr_loss to zero
                tr_loss -= tr_loss

                steps_delta = self.state.global_step - self._globalstep_last_logged
                logs["loss"] = round(tr_loss_scalar / steps_delta, 4)
                logs["learning_rate"] = self._get_learning_rate()

                if hasattr(self.model, '_losses'):
                    acc_steps_delta = steps_delta * self.args.gradient_accumulation_steps
                    for loss_name, loss_value in self.model._losses.items():
                        logs[loss_name] = round(loss_value / acc_steps_delta, 8)
                    self.model._losses.clear()

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
