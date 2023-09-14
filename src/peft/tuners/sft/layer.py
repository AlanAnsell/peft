import logging
import warnings
from typing import Optional, Tuple, Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTunerLayer

import torch_scatter

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#class LinearWithSparseDelta(torch.autograd.Function):
#
#    @staticmethod
#    def forward(ctx, input, weight, dv, di, bias):
#        ctx.save_for_backward(input, weight, dv, di, bias)
#
#        W = weight.to(dtype=input.dtype)
#        W = W.view(-1) + torch_scatter.segment_coo(
#            dv, di.long(), dim_size=W.numel(), reduce="sum"
#        )
#        W = W.view_as(weight)
#
#        return F.linear(input, W, bias=bias)
#
#    @staticmethod
#    def backward(ctx, output_grad):
#        input, weight, dv, di, bias = ctx.saved_tensors
#
#        W = weight.to(dtype=input.dtype)
#        W = W.view(-1) + torch_scatter.segment_coo(
#            dv, di.long(), dim_size=W.numel(), reduce="sum"
#        )
#        W = W.view_as(weight)
#
#        input_grad = weight_grad = dv_grad = bias_grad = None
#        if ctx.needs_input_grad[0]:
#            input_grad = torch.matmul(output_grad, W)
#        #logger.info(output_grad)
#        #input = input.view(-1, input.size(-1))
#        if ctx.needs_input_grad[1]:
#            assert False
#            weight_grad = torch.einsum('bij,bik->bkj', input, output_grad)
#            #output_grad = output_grad.contiguous().view(-1, output_grad.size(-1))
#            #weight_grad = torch.matmul(output_grad.T, input)
#            if ctx.needs_input_grad[2]:
#                dv_grad = weight_grad.reshape(-1)[di]
#        elif ctx.needs_input_grad[2]:
#            rows = di // weight.size(1)
#            cols = di - rows * weight.size(1)
#            output_grad = output_grad.transpose().view(output_grad.size(-1), -1)
#            output_grad_vectors = output_grad[rows, :]
#            input_vectors = input[:, cols]
#            dv_grad = torch.sum(output_grad_vectors * input_vectors, 0)
#
#        if ctx.needs_input_grad[4]:
#            bias_grad = torch.sum(output_grad, 0)
#
#        return input_grad, weight_grad, dv_grad, None, bias_grad
#
#def linear_sd(input, weight, dv, di, bias=None):
#    return LinearWithSparseDelta.apply(input, weight, dv, di, bias)


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


def random_subset(tensor, k):
    scores = torch.rand_like(tensor, dtype=torch.float32).view(-1)
    _, indices = torch.topk(scores, k, sorted=False)
    return indices


class SparseDelta(nn.Module):

    def __init__(self, k, shape, dtype=None):
        super().__init__()
        self.shape = shape
        self.dense_numel = np.prod(shape)
        self.values = nn.Parameter(torch.zeros([k], dtype=dtype))
        initial_indices = random_subset(self.values, k)
        self.register_buffer('indices', torch.sort(initial_indices).values)

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
        output = torch_scatter.segment_coo(
            self.values,
            self.indices,
            dim_size=tensor.numel(),
            reduce="sum",
        )
        #logger.info(f'{output}')
        #assert self.values.requires_grad
        #assert output.requires_grad
        output = output + tensor.view(-1)
        #assert output.requires_grad
        return output.view_as(tensor)

    #def apply(self, tensor, negate=False):
    #    if tensor.size() != self.shape:
    #        raise ValueError(
    #            f'SparseDelta has shape {self.shape}, but is being applied to '
    #            f'tensor of shape {tensor.size()}.'
    #        )
    #    torch_scatter.segment_coo(
    #        -self.values if negate else self.values,
    #        self.indices,
    #        out=tensor.view(-1),
    #        reduce="sum",
    #    )


class Linear(nn.Linear, BaseTunerLayer):

    # Lora implemented in a dense layer
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        k: int,
        **kwargs
    ) -> None:
        nn.Linear.__init__(
            self,
            in_features=in_features,
            out_features=out_features,
            **kwargs
        )
        self.sft_delta = nn.ModuleDict({})

        self.merged = False
        self.disable_adapters = False

        self.update_layer(adapter_name, k)
        self.active_adapter = adapter_name
        self.hook = None

    def apply_hook(self, hook):
        self.hook = hook

    def update_layer(self, adapter_name, k):
        self.sft_delta[adapter_name] = SparseDelta(
            k,
            self.weight.size(),
            dtype=self.weight.dtype,
        )

    def merge(self) -> None:
        if self.active_adapter not in self.sft_delta.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        self.sft_delta[self.active_adapter].apply(self.weight)
        self.merged = True

    def unmerge(self) -> None:
        if self.active_adapter not in self.sft_delta.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        self.sft_delta[self.active_adapter].apply(self.weight, negate=True)
        self.merged = False

    def _linear(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, bias=self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.active_adapter not in self.sft_delta.keys():
            return self._linear(x)

        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self._linear(x)
        elif self.merged:
            result = self._linear(x)
        else:
            sft = self.sft_delta[self.active_adapter]
            #result = linear_sd(x, self.weight, sft.values, sft.indices, bias=self.bias)
            merged_weight = sft(self.weight)
            if self.hook is not None and merged_weight.requires_grad:
                # check that merged_weight requires grad because this might not
                # be the case during the first pass of gradient checkpointing
                # if it is enabled
                merged_weight.register_hook(self.hook)
            result = F.linear(x, merged_weight, bias=self.bias)

        result = result.to(previous_dtype)
        return result
