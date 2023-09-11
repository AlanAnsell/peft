import warnings
from typing import Optional, Tuple, Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTunerLayer

import torch_scatter


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


class SparseDelta(nn.Module):

    def __init__(self, k, shape, dtype=None):
        super().__init__()
        self.shape = shape
        self.dense_numel = np.prod(shape)
        self.values = nn.Parameter(torch.zeros([k], dtype=dtype))
        initial_indices = torch.multinomial(
            torch.ones(shape).view(-1),
            k,
            replacement=False,
        ).to(dtype=torch.int32)
        self.register_buffer('indices', torch.sort(initial_indices).values)

    def forward(self, tensor):
        if tensor.size() != self.shape:
            raise ValueError(
                f'SparseDelta has shape {self.shape}, but is being applied to '
                f'tensor of shape {tensor.size()}.'
            )
        output = tensor.to(dtype=self.values.dtype)
        if output is tensor:
            output = output.clone()
        #output = output.reshape(-1)
        output = torch.flatten(output)
        output = output + torch_scatter.segment_coo(
            self.values,
            self.indices.long(),
            dim_size=output.numel(),
            reduce="sum",
        )
        return output.view_as(tensor)

    def apply(self, tensor, negate=False):
        if tensor.size() != self.shape:
            raise ValueError(
                f'SparseDelta has shape {self.shape}, but is being applied to '
                f'tensor of shape {tensor.size()}.'
            )
        torch_scatter.segment_coo(
            -self.values if negate else self.values,
            self.indices,
            out=tensor.view(-1),
            reduce="sum",
        )


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
            merged_weight = self.sft_delta[self.active_adapter](self.weight)
            result = F.linear(x, merged_weight, bias=self.bias)

        result = result.to(previous_dtype)
        return result
