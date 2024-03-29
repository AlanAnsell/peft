import logging
import warnings
from typing import Optional, Tuple, Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.import_utils import is_bnb_available
from peft.tuners.tuners_utils import BaseTunerLayer
import linear_sd

BNB_AVAILABLE = is_bnb_available()
if BNB_AVAILABLE:
    import bitsandbytes as bnb


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LinearWithSparseDelta(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, dv, di, bias, weight_grad_hook, compute_dtype):
        ctx.save_for_backward(input, weight, dv, di, bias)
        ctx.weight_grad_hook = weight_grad_hook
        ctx.compute_dtype = compute_dtype
        if BNB_AVAILABLE and isinstance(weight, bnb.nn.Params4bit):
            weight = bnb.functional.dequantize_4bit(
                weight,
                quant_state=weight.quant_state,
            ).to(compute_dtype)

        return linear_sd.forward(input, weight, dv, di, bias)

    @staticmethod
    def backward(ctx, output_grad):
        input, weight, dv, di, bias = ctx.saved_tensors
        if BNB_AVAILABLE and isinstance(weight, bnb.nn.Params4bit):
            weight = bnb.functional.dequantize_4bit(
                weight,
                quant_state=weight.quant_state,
            ).to(ctx.compute_dtype)

        grads = linear_sd.backward(
            output_grad, input, weight, dv, di, 
            ctx.needs_input_grad[0],
            ctx.weight_grad_hook is not None or ctx.needs_input_grad[1],
            ctx.needs_input_grad[2],
            bias is not None and ctx.needs_input_grad[4],
            bias,
        )
        if ctx.weight_grad_hook is not None:
            ctx.weight_grad_hook(grads[1])

        # need to return extra values corresponding to weight_grad_hook and compute_dtype
        grads.extend([None, None]) 
        if ctx.needs_input_grad[1]:
            return tuple(grads)
        else:
            return (grads[0], None) + tuple(grads[2:])

def linear_sd_op(input, weight, dv, di, bias=None, weight_grad_hook=None, compute_dtype=None):
    return LinearWithSparseDelta.apply(input, weight, dv, di, bias, weight_grad_hook, compute_dtype)


def flatten_indices(indices, shape):
    dim_values = torch.tensor(shape[1:] + (1,), device=indices.device)
    dim_multipliers = torch.flip(torch.cumprod(torch.flip(dim_values, [0]), 0), [0])
    dim_multipliers = dim_multipliers.unsqueeze(1)
    return torch.sum(dim_multipliers * indices, 0)


def expand_indices(indices, shape):
    expanded_indices = []
    indices_1d = indices.clone()
    for i in reversed(range(1, len(shape))):
        expanded_indices.append(indices_1d % shape[i])
        indices_1d //= shape[i]
    expanded_indices.append(indices_1d)
    return torch.stack(list(reversed(expanded_indices)), 0)


def random_subset(shape, k, device=None, dtype=None):
    scores = torch.rand(shape, dtype=torch.float32, device=device).view(-1)
    _, indices = torch.topk(scores, k, sorted=False)
    if dtype is not None:
        indices = indices.to(dtype=dtype)
    return indices


class SparseDelta(nn.Module):

    def __init__(self, k, shape, dtype=None, device=None):
        super().__init__()
        self.shape = shape
        self.dense_numel = np.prod(shape)
        self.values = nn.Parameter(torch.zeros([k], dtype=dtype, device=device))
        initial_indices = random_subset(
            self.shape,
            k,
            dtype=torch.int32 if self.dense_numel < 2**31 else torch.int64,
            device=device,
        )
        self.register_buffer('indices', torch.sort(initial_indices).values)

    def merge(self, tensor, negate=False):
        # can be used with quantization, but this is not recommended
        if BNB_AVAILABLE and isinstance(tensor, bnb.nn.Params4bit):
            target = bnb.functional.dequantize_4bit(
                tensor.data,
                quant_state=tensor.quant_state,
            )
        else:
            target = tensor

        if target.size() != self.shape:
            raise ValueError(
                f'SparseDelta has shape {self.shape}, but is being applied to '
                f'tensor of shape {target.size()}.'
            )
        values = self.values.to(target.dtype)
        target.view(-1).scatter_reduce_(
            0,
            self.indices.long(),
            -values if negate else values,
            "sum",
            include_self=True,
        )

        if BNB_AVAILABLE and isinstance(tensor, bnb.nn.Params4bit):
            _, tensor.quant_state = bnb.functional.quantize_4bit(
                target,
                out=tensor.data,
                blocksize=tensor.blocksize,
                compress_statistics=tensor.compress_statistics,
                quant_type=tensor.quant_type,
            )

    def unmerge(self, tensor):
        self.merge(tensor, negate=True)


class Linear(BaseTunerLayer):
    pass


def AddSparseDelta(_LinearType):

    if not isinstance(_LinearType, type):
        raise ValueError(
            'AddSparseDelta can only be called on a type, which must be a '
            'subclass of torch.nn.Linear'
        )

    if not issubclass(_LinearType, nn.Linear):
        raise ValueError(
            f'Can only add sparse delta to a subclass of torch.nn.Linear, '
            f'but received {_LinearType}.'
        )

    class _LinearWithSparseDelta(_LinearType, Linear):

        # Lora implemented in a dense layer
        def __init__(
            self,
            adapter_name: str,
            in_features: int,
            out_features: int,
            k: int,
            dtype: torch.dtype = None,
            device=None,
            **kwargs
        ) -> None:
            _LinearType.__init__(
                self,
                in_features,
                out_features,
                device=device,
                **kwargs
            )
            self.sft_delta = nn.ModuleDict({})

            self.merged = False
            self.disable_adapters = False

            self.update_layer(
                adapter_name,
                k,
                dtype=dtype,
                device=device,
            )
            self.active_adapter = adapter_name
            self.hook = None

            if not hasattr(self, "compute_dtype"):
                self.compute_dtype = self.weight.dtype

        def apply_hook(self, hook):
            self.hook = hook

        def update_layer(self, adapter_name, k, dtype=None, device=None):
            self.sft_delta[adapter_name] = SparseDelta(
                k,
                self.weight.size(),
                dtype=dtype,
                device=device,
            )

        def merge(self, module=None) -> None:
            if self.active_adapter not in self.sft_delta.keys():
                return
            if module is None:
                module = self
            if module is self and self.merged:
                warnings.warn("Already merged. Nothing to do.")
                return
            self.sft_delta[self.active_adapter].merge(module.weight)
            if module is self:
                self.merged = True

        def unmerge(self) -> None:
            if self.active_adapter not in self.sft_delta.keys():
                return
            if not self.merged:
                warnings.warn("Already unmerged. Nothing to do.")
                return
            self.sft_delta[self.active_adapter].unmerge(self.weight)
            self.merged = False

        def _linear(self, input: torch.Tensor) -> torch.Tensor:
            return super().forward(input)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.active_adapter not in self.sft_delta.keys():
                return self._linear(x)

            if self.disable_adapters:
                if self.merged:
                    self.unmerge()
                result = self._linear(x)
            elif self.merged:
                result = self._linear(x)
            else:
                sft = self.sft_delta[self.active_adapter]
                result = linear_sd_op(
                    x,
                    self.weight,
                    sft.values,
                    sft.indices,
                    bias=self.bias,
                    weight_grad_hook=self.hook,
                    compute_dtype=self.compute_dtype,
                )

            return result

    return _LinearWithSparseDelta
