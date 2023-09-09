import torch
import torch_scatter

from typing import Callable, Iterable, Optional, Tuple, Union


class SM3(torch.optim.Optimizer):
    """Implements SM3 algorithm.

    It has been proposed in `Memory-Efficient Adaptive Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): coefficient that scale delta before it is applied
            to the parameters (default: 0.1)
        momentum (float, optional): coefficient used to scale prior updates
            before adding. This drastically increases memory usage if
            `momentum > 0.0`. This is ignored if the parameter's gradient
            is sparse. (default: 0.0)
        beta (float, optional): coefficient used for exponential moving
            averages (default: 0.0)
        eps (float, optional): Term added to square-root in denominator to
            improve numerical stability (default: 1e-30)

    .. _Memory-Efficient Adaptive Optimization:
        https://arxiv.org/abs/1901.11150
    """
    def __init__(self, params, indices, shapes, lr=0.1, momentum=0.0, beta=0.0, eps=1e-8, row_cover_only=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {0}".format(lr))
        if not 0.0 <= momentum < 1.0:
            raise ValueError("Invalid momentum: {0}".format(momentum))
        if not 0.0 <= beta < 1.0:
            raise ValueError("Invalid beta: {0}".format(beta))
        if not 0.0 <= eps:
            raise ValueError("Invalid eps: {0}".format(eps))

        defaults = {'lr': lr, 'momentum': momentum, 'beta': beta, 'eps': eps, 'row_cover_only': row_cover_only}
        super(SM3, self).__init__(params, defaults)
        self.indices = indices
        self.shapes = shapes

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
            beta = group['beta']
            eps = group['eps']
            row_cover_only = group['row_cover_only']
            for p in group['params']:
                grad = p.grad
                if grad is None:
                    continue

                indices = self.indices[p]
                shape = self.shapes[p]
                rank = len(shape)
                #assert torch.all(indices[1:] >= indices[:-1])
                #expanded_indices = expand_indices(indices, shape)
                #first_indices = expanded_indices[0, :]
                #assert torch.all(first_indices[1:] >= first_indices[:-1])

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = 0.
                    _add_initial_accumulators(state, grad, shape)

                if row_cover_only:
                    expanded_indices = indices // shape[1]
                    acc = state['accumulator_0']
                    update = acc[expanded_indices]
                    update.addcmul_(grad, grad)
                    torch_scatter.scatter(
                        update,
                        expanded_indices,
                        dim_size=shape[0],
                        reduce='max',
                        out=acc,
                    )
                elif len(shape) == 2:
                    row_indices = indices // shape[1]
                    col_indices = indices - (row_indices * shape[1])
                    row_acc = state['accumulator_0']
                    col_acc = state['accumulator_1']
                    update = torch.minimum(
                        row_acc[row_indices],
                        col_acc[col_indices]
                    )
                    update.addcmul_(grad, grad)
                    torch_scatter.scatter(
                        update,
                        row_indices,
                        dim_size=shape[0],
                        reduce='max',
                        out=row_acc
                    )
                    torch_scatter.scatter(
                        update,
                        col_indices,
                        dim_size=shape[1],
                        reduce='max',
                        out=col_acc
                    )
                else:
                    expanded_indices = expand_indices(indices, shape)

                    acc_list = [state[_key(i)] for i in range(len(shape))]

                    # Get update from accumulators and gradients
                    update = _compute_update(beta, acc_list, grad, expanded_indices)

                    # Update accumulators.
                    self._update_accumulator(beta, acc_list, update, shape, expanded_indices)

                # Add small amount for numerical stability
                update.add_(eps).rsqrt_().mul_(grad)

                if momentum > 0.:
                    m = state['momentum_buffer']
                    update.mul_(1. - momentum).add_(m, alpha=momentum)
                    state['momentum_buffer'] = update.detach()

                p.sub_(update, alpha=group['lr'])
                state['step'] += 1
        return loss

    def _update_accumulator(self, beta, acc_list, update, shape, expanded_indices):
        for i, acc in enumerate(acc_list):
            torch_scatter.scatter(
                update,
                expanded_indices[i, :],
                dim_size=shape[i],
                reduce='max',
                out=acc
            )

def _compute_update(beta, acc_list, grad, expanded_indices):
    rank = len(acc_list)
    update = acc_list[0][expanded_indices[0, :]]
    for i in range(1, rank):
        update = torch.min(update, acc_list[i][expanded_indices[i, :]])
    if beta > 0.:
        update.mul_(beta)
    update.addcmul_(grad, grad, value=1. - beta)

    return update

def _key(i):
    # Returns key used for accessing accumulators
    return 'accumulator_' + str(i)

def _add_initial_accumulators(state, grad, shape):
    # Creates initial accumulators. For a dense tensor of shape (n1, n2, n3),
    # then our initial accumulators are of shape (n1, 1, 1), (1, n2, 1) and
    # (1, 1, n3). For a sparse tensor of shape (n, *), we use a single
    # accumulator of shape (n,).
    rank = len(shape)

    for i in range(rank):
        state[_key(i)] = torch.zeros([shape[i]], dtype=grad.dtype, device=grad.device)


