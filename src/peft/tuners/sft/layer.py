import logging
import warnings
from typing import Optional, Tuple, Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTunerLayer

import bitsandbytes as bnb
import torch_scatter

import linear_sd_cpp

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class LinearWithSparseDelta(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, dv, di, bias, weight_grad_hook):
        ctx.save_for_backward(input, weight, dv, di, bias)
        ctx.weight_grad_hook = weight_grad_hook
        if isinstance(weight, bnb.nn.Params4bit):
            weight = bnb.functional.dequantize_4bit(
                weight,
                quant_state=weight.quant_state,
            )

        return linear_sd_cpp.forward(input, weight, dv, di, bias)

    @staticmethod
    def backward(ctx, output_grad):
        input, weight, dv, di, bias = ctx.saved_tensors
        if isinstance(weight, bnb.nn.Params4bit):
            weight = bnb.functional.dequantize_4bit(
                weight,
                quant_state=weight.quant_state,
            )

        grads = linear_sd_cpp.backward(
            output_grad, input, weight, dv, di, 
            ctx.needs_input_grad[0],
            ctx.weight_grad_hook is not None or ctx.needs_input_grad[1],
            ctx.needs_input_grad[2],
            bias is not None and ctx.needs_input_grad[4],
            bias,
        )
        if ctx.weight_grad_hook is not None:
            ctx.weight_grad_hook(grads[1])

        grads.append(None) # need to return an extra value corresponding to weight_grad_hook
        if ctx.needs_input_grad[1]:
            return tuple(grads)
        else:
            return (grads[0], None) + tuple(grads[2:])

def linear_sd(input, weight, dv, di, bias=None, weight_grad_hook=None):
    return LinearWithSparseDelta.apply(input, weight, dv, di, bias, weight_grad_hook)


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


def random_subset(shape, k, device=None):
    scores = torch.rand(shape, dtype=torch.float32, device=device).view(-1)
    _, indices = torch.topk(scores, k, sorted=False)
    return indices


class SparseDelta(nn.Module):

    def __init__(self, k, shape, dtype=None, dropout=0.0, device=None):
        super().__init__()
        self.shape = shape
        self.dense_numel = np.prod(shape)
        self.values = nn.Parameter(torch.zeros([k], dtype=dtype, device=device))
        initial_indices = random_subset(self.shape, k, device=device)
        self.register_buffer('indices', torch.sort(initial_indices).values)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tensor):
        if tensor.size() != self.shape:
            raise ValueError(
                f'SparseDelta has shape {self.shape}, but is being applied to '
                f'tensor of shape {tensor.size()}.'
            )
        #tensor = tensor.to(dtype=self.values.dtype)
        #if output is tensor:
        #    output = output.clone()
        #output = output.reshape(-1)
        #output = torch.flatten(output)
        deltas = self.dropout(self.values)
        output = tensor.reshape(-1) + torch_scatter.scatter(
            deltas.to(tensor.dtype),
            self.indices,
            dim_size=tensor.numel(),
            reduce="sum",
        )
        #logger.info(f'{output}')
        #assert self.values.requires_grad
        #assert output.requires_grad
        #output = output + tensor.view(-1)
        #assert output.requires_grad
        return output.view_as(tensor)

    def merge(self, tensor, negate=False):
        if isinstance(tensor, bnb.nn.Params4bit):
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
        torch_scatter.scatter(
            -values if negate else values,
            self.indices,
            out=target.view(-1),
            reduce="sum",
        )

        if isinstance(tensor, bnb.nn.Params4bit):
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
            dropout: float = 0.0,
            dtype: torch.dtype = None,
            device=None,
            **kwargs
        ) -> None:
            _LinearType.__init__(
                self,
                in_features,
                out_features,
                **kwargs
            )
            self.sft_delta = nn.ModuleDict({})

            self.merged = False
            self.disable_adapters = False

            self.update_layer(
                adapter_name,
                k,
                dtype=dtype,
                dropout=dropout,
                device=device,
            )
            self.active_adapter = adapter_name
            self.hook = None

        def apply_hook(self, hook):
            self.hook = hook

        def update_layer(self, adapter_name, k, dtype=None, dropout=0.0, device=None):
            self.sft_delta[adapter_name] = SparseDelta(
                k,
                self.weight.size(),
                dtype=dtype,
                dropout=dropout,
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

            #previous_dtype = x.dtype

            if self.disable_adapters:
                if self.merged:
                    self.unmerge()
                result = self._linear(x)
            elif self.merged:
                result = self._linear(x)
            else:
                sft = self.sft_delta[self.active_adapter]
                #deltas = sft.dropout(sft.values)
                result = linear_sd(x, self.weight, sft.values, sft.indices, bias=self.bias, weight_grad_hook=self.hook)
                #W = sft(self.weight)
                #if self.hook is not None and W.requires_grad:
                #    W.register_hook(self.hook)
                #result = F.linear(x, W, bias=self.bias)

            #result = result.to(previous_dtype)
            return result

    return _LinearWithSparseDelta
