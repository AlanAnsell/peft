import logging
import math

import numpy as np
import torch

from transformers import (
    TrainerCallback,
)
from accelerate.optimizer import AcceleratedOptimizer

from .layer import Linear
from .optimizer import SftAdamW, SftSM3

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def update_optimizer(
    optimizer,
    param,
    changing_indices,
    init_momenta={},
):
    """
    Updates optimizer state for a PEFT parameter tensor after dropping and regrowth.

    Args:
      - optimizer: the optimizer to update.
      - param: the parameter tensor.
      - changing_indices: the indices in the optimizer state that need to be updated.
      - init_momenta: dict mapping state keys to seed values. If not supplied, values
          are seeded to zero.
    """
    if isinstance(optimizer, AcceleratedOptimizer):
        optimizer = optimizer.optimizer

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


class SftSelector:
    """
    Implements SFT tunable parameter reselection. Simply construct the SftSelector and call
    .step() after each training update step.
    """

    def __init__(
        self,
        model,
        optimizer,
        sft_config,
        total_update_steps,
        grad_accumulation_steps,
        completed_steps=0, # number of already completed steps if resuming from saved ckpt.
    ):
        self.model = model
        self.optimizer = optimizer
        self.sft_config = sft_config
        self.total_update_steps = total_update_steps
        self.grad_accumulation_steps = grad_accumulation_steps
        self.completed_steps = completed_steps
        self.begin_selection_phase()

    def step(self):
        self.completed_steps += 1
        
        if (self.completed_steps + 1) % self.sft_config.reselection_steps == 0:
            self.begin_selection_phase()
        
        if (
            self.completed_steps % self.sft_config.reselection_steps ==
            self.sft_config.selection_accumulation_steps
        ):
            self.end_selection_phase()

    def begin_selection_phase(self):
        if self.sft_config.selection_algorithm == "sm3":
            return

        logger.info('Beginning selection phase')
        self.reselection_scores = {}
        # Apply hooks to gather gradients for growth selection
        for n, m in self.model.named_modules():
            if (
                isinstance(m, Linear) and
                m.active_adapter is not None and
                m.active_adapter in m.sft_delta
            ):
                m.apply_hook(self.gradient_accumulation_hook(n))

    def end_selection_phase(self):
        logger.info('Ending selection phase')
        if self.completed_steps > self.total_update_steps:
            return

        if self.sft_config.selection_algorithm != "sm3":
            # Remove hooks
            for n, m in self.model.named_modules():
                if (
                    isinstance(m, Linear) and
                    m.active_adapter is not None and
                    m.active_adapter in m.sft_delta
                ):
                    m.apply_hook(None)

        # Replace all parameters if it's the first reselection step, linear
        # decay from initial rate otherwise.
        if self.sft_config.reselection_rate_policy == "linear":
            if self.completed_steps == self.sft_config.selection_accumulation_steps:
                p = 1
            else:
                p = self.sft_config.initial_reselection_rate * (
                    1 - self.completed_steps / self.total_update_steps
                )
        elif self.sft_config.reselection_rate_policy == "cosine":
            p = self.sft_config.initial_reselection_rate * (
                1 + math.cos(math.pi * self.completed_steps / self.total_update_steps)
            ) / 2
        else:
            raise ValueError(f'Unsupported reselection rate policy {self.sft_config.reselection_rate_policy}')
        self.select(p)
        self.reselection_scores = {}

    def select(self, p):
        if self.sft_config.selection_algorithm == "sm3":
            self.select_sm3(p)
        elif self.sft_config.selection_algorithm == "rigl":
            self.select_rigl(p)
        else:
            raise ValueError(
                f'Invalid selection method {self.sft_config.selection_algorithm}'
            )

    def gradient_accumulation_hook(self, module_name):

        @torch.no_grad()
        def _gradient_accumulation_hook(grad):
            m = self.model.get_submodule(module_name)
            grad = grad.reshape(-1)
            if module_name in self.reselection_scores:
                candidate_indices, candidate_grads, candidate_grads_sq, samples = self.reselection_scores[module_name]
                new_grads = grad[candidate_indices]
                candidate_grads += new_grads
                candidate_grads_sq.addcmul_(new_grads, new_grads)
                samples += 1
            else:
                num_candidates = len(m.sft_delta[m.active_adapter].values)
                _, candidate_indices = torch.topk(
                    torch.abs(grad),
                    num_candidates,
                    largest=True,
                    sorted=False,
                )
                candidate_grads = grad[candidate_indices]
                self.reselection_scores[module_name] = (
                    candidate_indices.to(m.sft_delta[m.active_adapter].indices.dtype),
                    candidate_grads,
                    candidate_grads * candidate_grads,
                    torch.ones_like(candidate_grads)
                )

        return _gradient_accumulation_hook

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
    def select_rigl(self, change_proportion):
        n_replacements = 0
        total_params = 0

        betas = {}
        for group in self.optimizer.param_groups:
            for p in group['params']:
                betas[p] = group['betas']

        for module_name, (
            candidate_indices,
            candidate_grads,
            candidate_grads_sq,
            candidate_samples
        ) in self.reselection_scores.items():
            m = self.model.get_submodule(module_name)
            delta = m.sft_delta[m.active_adapter]
            delta.values.grad = None

            num_to_reallocate = int(len(delta.values) * change_proportion)
            # Find the k deltas with smallest absolute values
            _, changing_indices = torch.topk(
                torch.abs(delta.values),
                num_to_reallocate,
                largest=False,
                sorted=True,
            )
            outgoing_params = delta.indices[changing_indices]
            # binary mask of weights to drop
            is_outgoing = torch.zeros(
                [delta.dense_numel],
                dtype=torch.bool,
                device=outgoing_params.device,
            )
            is_outgoing[outgoing_params] = True
            assert torch.sum(is_outgoing) == num_to_reallocate
            # binary mask of currently active weights
            is_current = torch.zeros(
                [delta.dense_numel],
                dtype=torch.bool,
                device=delta.indices.device,
            )
            is_current[delta.indices] = True
            # weights that will stil be active after dropping
            is_remaining = is_current & ~is_outgoing

            # don't consider growing any already active candidate
            is_valid_candidate = ~is_remaining[candidate_indices]
            candidate_indices = candidate_indices[is_valid_candidate]
            candidate_grads = candidate_grads[is_valid_candidate]
            candidate_grads_sq = candidate_grads_sq[is_valid_candidate]
            candidate_samples = candidate_samples[is_valid_candidate]
            candidate_scores = torch.abs(candidate_grads)

            if self.sft_config.sample_for_growth:
                best_candidate_indices = torch.multinomial(
                    candidate_scores,
                    min(num_to_reallocate, len(candidate_grads)),
                )
            else:
                # take the top k growth candidates with highest gradient magnitudes
                best_scores, best_candidate_indices = torch.topk(
                    candidate_scores,
                    min(num_to_reallocate, len(candidate_grads)),
                    largest=True,
                    sorted=True,
                )
            incoming_params = candidate_indices[best_candidate_indices]
            incoming_grads = candidate_grads[best_candidate_indices]
            incoming_grads_sq = candidate_grads_sq[best_candidate_indices]
            incoming_samples = candidate_samples[best_candidate_indices]
            # binary mask of weights to grow
            is_incoming = torch.zeros(
                [delta.dense_numel],
                dtype=torch.bool,
                device=incoming_params.device,
            )
            is_incoming[incoming_params] = True

            # filter out weights which have been selected to be dropped and
            # grown simultaneously
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

            n_replacements += len(changing_indices)
            total_params += len(delta.indices)

            # update delta indices and values
            delta.indices[changing_indices] = incoming_params.to(delta.indices.dtype)
            delta.values[changing_indices] = 0.0

            # seed the optimizer momenta appropriately
            incoming_grads /= incoming_samples
            incoming_grads_sq /= incoming_samples
            incoming_ages = incoming_samples / self.grad_accumulation_steps
            beta1, beta2 = betas[delta.values]
            # bias counter-correction: these are unbiased estimates of the momenta,
            # so bias them in order that they will be unbiased after Adam's bias
            # correction
            incoming_grads *= (1.0 - beta1 ** incoming_ages)
            incoming_grads_sq *= (1.0 - beta2 ** incoming_ages)
            update_optimizer(
                self.optimizer,
                delta.values,
                changing_indices,
                init_momenta={
                    'age': incoming_ages,
                    'exp_avg': incoming_grads,
                    'exp_avg_sq': incoming_grads_sq,
                }
            )

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

            is_current = torch.zeros(
                [delta.dense_numel],
                dtype=torch.bool,
                device=delta.indices.device,
            )
            is_current[delta.indices] = True
            is_valid_candidate = ~is_current

            optimizer_state = self.optimizer.state[delta.values]
            row_grads_sq = optimizer_state['accumulator_0']
            col_grads_sq = optimizer_state['accumulator_1']
            # take outer product of row and column wise SM3 buffers
            # (here we assume 2D parameter tensors).
            estimated_momenta = torch.outer(row_grads_sq, col_grads_sq)
            estimated_momenta = estimated_momenta.view(-1)[is_valid_candidate]
            candidate_indices = torch.arange(
                0,
                delta.dense_numel,
                device=is_valid_candidate.device,
            )
            candidate_indices = candidate_indices[is_valid_candidate]

            if self.sft_config.sample_for_growth:
                best_candidate_indices = torch.multinomial(
                    estimated_momenta,
                    num_to_reallocate,
                )
            else:
                _, best_candidate_indices = torch.topk(
                    estimated_momenta,
                    num_to_reallocate,
                    largest=True,
                    sorted=False,
                )
            incoming_params = candidate_indices[best_candidate_indices]

            n_replacements += len(changing_indices)
            total_params += len(delta.indices)

            delta.indices[changing_indices] = incoming_params.to(delta.indices.dtype)
            delta.values[changing_indices] = 0.0

        logger.info(
            f'Replacing {n_replacements} ({100*n_replacements/total_params:.4f}%)'
        )


class SelectorStepCallback(TrainerCallback):

    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        self.trainer._selector.step()


def SftTrainer(_Trainer):
    """
    Wraps a Trainer or subclass thereof for SFT training. The resulting class
    should be constructed with a SftModel as the model and passing sft_config as
    a SftConfig instance.
    """

    class _SftTrainer(_Trainer):

        def __init__(
            self,
            *args,
            sft_config=None,
            **kwargs
        ):
            super().__init__(*args, **kwargs)
            logger.setLevel(self.args.get_process_log_level())

            if sft_config is None:
                raise ValueError('Missing sft_config')
            self.sft_config = sft_config

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
                self.sft_config,
                max_steps,
                self.args.gradient_accumulation_steps,
            )
            self.add_callback(SelectorStepCallback(self))

        def create_optimizer(self):
            if self.optimizer is None:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in self.model.named_parameters() if p.requires_grad
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                ]

                _, optimizer_kwargs = _Trainer.get_optimizer_cls_and_kwargs(self.args)
                logger.info(f'optimizer_kwargs: {optimizer_kwargs}')

                if self.sft_config.selection_algorithm == "sm3":
                    deltas = {
                        delta.values: delta
                        for _1, _2, delta in self.model.active_deltas()
                    }

                    self.optimizer = SftSM3(
                        optimizer_grouped_parameters,
                        deltas,
                        **optimizer_kwargs
                    )
                else:
                    self.optimizer = SftAdamW(optimizer_grouped_parameters, **optimizer_kwargs)

            return self.optimizer

    return _SftTrainer
