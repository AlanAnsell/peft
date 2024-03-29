import logging
import math
from typing import Callable, Iterable, Optional, Tuple, Union

import torch


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SftSM3(torch.optim.Optimizer):
    """Implements SM3 algorithm.

    Adapted from https://github.com/Enealor/PyTorch-SM3

    .. _Memory-Efficient Adaptive Optimization:
        https://arxiv.org/abs/1901.11150
    """
    def __init__(
        self,
        params,
        deltas,
        lr=0.1,
        momentum=0.0,
        weight_decay=0.0,
        eps=1e-8,
        row_cover_only=False,
        **kwargs
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {0}".format(lr))
        if not 0.0 <= momentum < 1.0:
            raise ValueError("Invalid momentum: {0}".format(momentum))
        if not 0.0 <= eps:
            raise ValueError("Invalid eps: {0}".format(eps))

        defaults = {
            'lr': lr,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'eps': eps,
            'row_cover_only': row_cover_only,
        }
        super(SftSM3, self).__init__(params, defaults)
        self.deltas = deltas

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            eps = group['eps']
            row_cover_only = group['row_cover_only']
            for p in group['params']:
                grad = p.grad
                if grad is None:
                    continue
                grad = grad.to(torch.float32)

                sparse = p in self.deltas
                if sparse:
                    indices = self.deltas[p].indices
                    shape = self.deltas[p].shape
                else:
                    shape = grad.size()
                rank = len(shape)

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = 0.
                    _add_initial_accumulators(state, grad, shape, sparse)

                if sparse and row_cover_only:
                    expanded_indices = indices // shape[1]
                    acc = state['accumulator_0']
                    update = acc[expanded_indices]
                    update.addcmul_(grad, grad)
                    acc.scatter_reduce_(
                        0,
                        expanded_indices.long(),
                        update,
                        "amax",
                        include_self=True,
                    )
                elif sparse and len(shape) == 2:
                    row_indices = indices // shape[1]
                    col_indices = indices - (row_indices * shape[1])
                    row_acc = state['accumulator_0']
                    col_acc = state['accumulator_1']
                    update = torch.minimum(
                        row_acc[row_indices],
                        col_acc[col_indices]
                    )
                    update.addcmul_(grad, grad)
                    row_acc.scatter_reduce_(
                        0,
                        row_indices.long(),
                        update,
                        "amax",
                        include_self=True,
                    )
                    col_acc.scatter_reduce_(
                        0,
                        col_indices.long(),
                        update,
                        "amax",
                        include_self=True,
                    )
                else:
                    if sparse:
                        expanded_indices = expand_indices(indices.long(), shape)
                    else:
                        expanded_indices = None

                    acc_list = [state[_key(i)] for i in range(len(shape))]

                    # Get update from accumulators and gradients
                    update = _compute_update(acc_list, grad, expanded_indices)

                    # Update accumulators.
                    self._update_accumulator(acc_list, update, shape, expanded_indices)

                # Add small amount for numerical stability
                update.add_(eps).rsqrt_().mul_(grad)

                if momentum > 0.:
                    m = state['momentum_buffer']
                    update.mul_(1. - momentum).add_(m, alpha=momentum)
                    state['momentum_buffer'] = update.detach()

                p.sub_(update, alpha=group['lr'])

                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

                state['step'] += 1
        return loss

    def _update_accumulator(self, acc_list, update, shape, expanded_indices):
        for i, acc in enumerate(acc_list):
            if expanded_indices is None:
                acc.copy_(_max_reduce_except_dim(update, i))
            else:
                acc.scatter_reduce_(
                    0,
                    expanded_indices[i, :].long(),
                    update,
                    "amax",
                    include_self=True,
                )

def _compute_update(acc_list, grad, expanded_indices):
    rank = len(acc_list)
    if expanded_indices is None:
        update = acc_list[0].clone()
    else:
        update = acc_list[0][expanded_indices[0, :]]
    for i in range(1, rank):
        if expanded_indices is None:
            update = torch.min(update, acc_list[i])
        else:
            update = torch.min(update, acc_list[i][expanded_indices[i, :]])
    update.addcmul_(grad, grad)

    return update

def _key(i):
    # Returns key used for accessing accumulators
    return 'accumulator_' + str(i)

def _add_initial_accumulators(state, grad, shape, sparse):
    # Creates initial accumulators. For a dense tensor of shape (n1, n2, n3),
    # then our initial accumulators are of shape (n1, 1, 1), (1, n2, 1) and
    # (1, 1, n3). For a sparse tensor of shape (n, *), we use a single
    # accumulator of shape (n,).
    rank = len(shape)

    for i in range(rank):
        if sparse:
            state[_key(i)] = torch.zeros([shape[i]], dtype=torch.float32, device=grad.device)
        else:
            acc_shape = [1] * i + [shape[i]] + [1] * (rank - 1 - i)
            state[_key(i)] = torch.zeros(acc_shape, dtype=torch.float32, device=grad.device)

def _max_reduce_except_dim(tensor, dim):
    # Computes max along all dimensions except the given dim.
    # If tensor is a scalar, it returns tensor.
    rank = len(tensor.shape)
    result = tensor
    if rank > 0:
        assert dim < rank
        for d in range(rank):
            if d != dim:
                result = result.max(dim=d, keepdim=True).values
    return result


class SftAdamW(torch.optim.Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Adapted from Huggingface AdamW optimizer.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        momentum_dtype: torch.dtype = None,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}
        super().__init__(params, defaults)
        self.momentum_dtype = momentum_dtype

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["age"] = torch.ones_like(p, dtype=self.momentum_dtype)
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p, dtype=self.momentum_dtype)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=self.momentum_dtype)

                age, exp_avg, exp_avg_sq = state["age"], state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                grad = grad.to(dtype=self.momentum_dtype)
                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                # 1.0 - beta might become 0 in low-precision dtypes like bfloat16
                age = age.to(dtype=torch.float32)
                # Per-parameter bias correction
                bias1_correction = 1.0 - beta1 ** age
                bias2_correction = 1.0 - beta2 ** age
                denom.mul_(bias1_correction.to(denom.dtype))
                denom.div_(torch.sqrt(bias2_correction).to(denom.dtype))

                step_size = group["lr"]
                p.addcdiv_(exp_avg, denom, value=-step_size)

                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

                age += 1

        return loss
