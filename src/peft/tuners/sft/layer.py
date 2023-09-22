import logging
import warnings
from typing import Optional, Tuple, Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTunerLayer

import torch_scatter

import linear_sd_cpp

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


def random_subset(shape, k):
    scores = torch.rand(shape, dtype=torch.float32).view(-1)
    _, indices = torch.topk(scores, k, sorted=False)
    return indices



class SparseDelta(nn.Module):

    def __init__(self, k, shape, dtype=None):
        super().__init__()
        self.shape = shape
        self.dense_numel = np.prod(shape)
        self.values = nn.Parameter(torch.zeros([k], dtype=dtype))
        initial_indices = random_subset(self.shape, k)
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
        output = tensor.view(-1) + torch_scatter.segment_coo(
            self.values.to(tensor.dtype),
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
        if tensor.size() != self.shape:
            raise ValueError(
                f'SparseDelta has shape {self.shape}, but is being applied to '
                f'tensor of shape {tensor.size()}.'
            )
        values = self.values.to(tensor.dtype)
        torch_scatter.segment_coo(
            -values if negate else values,
            self.indices,
            out=tensor.view(-1),
            reduce="sum",
        )

    def unmerge(self, tensor):
        self.merge(tensor, negate=True)


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
            #dtype=self.weight.dtype,
        )

    def merge(self) -> None:
        if self.active_adapter not in self.sft_delta.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        self.sft_delta[self.active_adapter].merge(self.weight)
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
        return F.linear(input, self.weight, bias=self.bias)

    def cu_hook(self):
        def _hook(grad):
            assert grad.dtype == torch.float32
            if self.py_grad is None:
                self.cu_grad = grad.detach()
            else:
                assert torch.all(torch.abs(grad - self.py_grad) < 1e-5)
                #logger.info("OK")
                self.py_grad = None
        return _hook

    def py_hook(self):
        def _hook(grad):
            if self.cu_grad is None:
                self.py_grad = grad.detach()
            else:
                assert torch.all(torch.abs(self.cu_grad - grad) < 1e-5)
                #logger.info("OK")
                self.cu_grad = None
        return _hook

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
            #self.cu_grad = None
            #self.py_grad = None
            #if self.hook is None:
            #    values_cu = sft.values.clone()
            #    values_cu.register_hook(self.cu_hook())
            #    result1 = linear_sd_cpp.apply(x.to(self.weight.dtype), self.weight, values_cu, sft.indices, bias=self.bias)
            #    #x_leading = x.size()[:-1]
            #    #result = linear_sd_cpp.apply(x.view(-1, x.size(-1)), self.weight, sft.values, sft.indices, bias=self.bias)
            #    #result = result.view(*x_leading, result.size(-1))
            #
            #values_py = sft.values.clone()
            #if self.hook is None:
            #    values_py.register_hook(self.py_hook())
            #merged_weight = self.weight.view(-1) + torch_scatter.segment_coo(
            #    values_py.to(self.weight.dtype),
            #    sft.indices,
            #    dim_size=self.weight.numel(),
            #    reduce="sum",
            #)
            #merged_weight = merged_weight.view_as(self.weight)

            #if self.hook is not None and merged_weight.requires_grad:
            #    # check that merged_weight requires grad because this might not
            #    # be the case during the first pass of gradient checkpointing
            #    # if it is enabled
            #    merged_weight.register_hook(self.hook)

            #result2 = F.linear(x.to(dtype=merged_weight.dtype), merged_weight, bias=self.bias)
            #if self.hook is None:
            #    result = (result1 + result2) / 2
            #else:
            #    result = result2
            if self.hook is None:
                #values = F.dropout(sft.values, p=0.05, training=self.training)
                result = linear_sd_cpp.apply(x.to(self.weight.dtype), self.weight, sft.values, sft.indices, bias=self.bias)
            else:
                merged_weight = sft(self.weight)
                if merged_weight.requires_grad:
                    # check that merged_weight requires grad because this might not
                    # be the case during the first pass of gradient checkpointing
                    # if it is enabled
                    merged_weight.register_hook(self.hook)
                result = F.linear(x.to(merged_weight.dtype), merged_weight, bias=self.bias)

        result = result.to(previous_dtype)
        return result
