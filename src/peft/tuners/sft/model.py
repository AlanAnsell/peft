import collections
import logging
import re
import warnings
from dataclasses import asdict, replace
from enum import Enum

import torch
from torch import nn
from tqdm import tqdm

import bitsandbytes as bnb
import numpy as np

#from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import BaseTuner
from peft.utils import (
    COMMON_LAYERS_PATTERN,
    #TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _freeze_adapter,
    _get_submodules,
    #get_auto_gptq_quant_linear,
    #get_quantization_config,
)

from .config import SftConfig
#from .gptq import QuantLinear
from .layer import AddSparseDelta, Linear, SparseDelta


#if is_bnb_available():
#    import bitsandbytes as bnb
#
#    from .bnb import Linear8bitLt
#
#if is_bnb_4bit_available():
#    from .bnb import Linear4bit

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def original_numel(p):
    if isinstance(p, bnb.nn.Params4bit):
        return np.prod(p.quant_state[1])
    else:
        return p.numel()


class SftModel(BaseTuner):

    def __init__(self, model, config, adapter_name) -> None:
        self.total_params = 0
        for p in model.parameters():
            self.total_params += original_numel(p)
        self._losses = collections.defaultdict(float)
        self._replaced_modules = {}
        super().__init__(model, config, adapter_name)

    def _check_new_adapter_config(self, config: SftConfig) -> None:
        pass

    def active_deltas(self):
        for n, m in self.named_modules():
            if isinstance(m, Linear) and m.active_adapter in m.sft_delta:
                yield f'{n}.sft_delta.{m.active_adapter}', m.sft_delta[m.active_adapter]

    def _get_ancestry(self, module_name):
        current_module = self.model.get_submodule(module_name)
        original_module = self._replaced_modules.get(module_name, None)
        if original_module is None:
            original_module = current_module
        parts = module_name.split('.')
        child_name = parts[-1]
        parent_name = '.'.join(parts[:-1])
        parent = self.model.get_submodule(parent_name)
        return current_module, original_module, parent, child_name

    @staticmethod
    def _check_target_module_exists(sft_config, key):
        if isinstance(sft_config.target_modules, str):
            target_module_found = re.fullmatch(sft_config.target_modules, key)
        else:
            target_module_found = (
                any(
                    re.match(f".*\.{target_key}$", key)
                    for target_key in sft_config.target_modules
                ) or
                any(
                    target_key == key
                    for target_key in sft_config.target_modules
                )
            )
            is_using_layer_indexes = getattr(sft_config, "layers_to_transform", None) is not None
            layer_indexing_pattern = getattr(sft_config, "layers_pattern", None)

            if is_using_layer_indexes and target_module_found:
                layers_pattern = COMMON_LAYERS_PATTERN if layer_indexing_pattern is None else layer_indexing_pattern
                layers_pattern = [layers_pattern] if isinstance(layers_pattern, str) else layers_pattern

                for pattern in layers_pattern:
                    layer_index = re.match(f".*.{pattern}\.(\d+)\.*", key)
                    if layer_index is not None:
                        layer_index = int(layer_index.group(1))
                        if isinstance(sft_config.layers_to_transform, int):
                            target_module_found = layer_index == sft_config.layers_to_transform
                        else:
                            target_module_found = layer_index in sft_config.layers_to_transform

                        break
                    else:
                        target_module_found = False
        return target_module_found

    def _create_and_replace(
        self,
        peft_config,
        adapter_name,
        module_name,
        k=None,
        **optional_kwargs,
    ):
        if k is None:
            raise ValueError("k must be specified.")

        current_module, original_module, parent, child_name = self._get_ancestry(module_name)

        if not isinstance(original_module, nn.Linear):
            raise ValueError(
                f"Can only apply SFT to modules which are nn.Linear or subclasses thereof, "
                f"but {module_name} is a {type(original_module)}."
            )

        if peft_config.dtype is None or isinstance(peft_config.dtype, torch.dtype):
            dtype = peft_config.dtype
        elif peft_config.dtype == "auto":
            dtype = target.weight.dtype
        elif peft_config.dtype == "float32":
            dtype = torch.float32
        elif peft_config.dtype == "float16":
            dtype = torch.float16
        elif peft_config.dtype == "bfloat16":
            dtype = torch.bfloat16
        else:
            raise ValueError(
                f"Unsupported dtype requested for SFT delta: {peft_config.dtype}"
            )

        if current_module is original_module:
            linear_type = type(original_module)
            linear_with_sd_type = AddSparseDelta(linear_type)
            if linear_type == bnb.nn.Linear4bit:
                linear_kwargs = {
                    'compute_dtype': dtype,
                    'compress_statistics': original_module.weight.compress_statistics,
                    'quant_type': original_module.weight.quant_type,
                }
            else:
                linear_kwargs = {'dtype': dtype}
            linear_kwargs['dropout'] = peft_config.dropout
            new_module = linear_with_sd_type(
                adapter_name,
                original_module.in_features,
                original_module.out_features,
                k,
                bias=original_module.bias is not None,
                device=original_module.weight.device,
                **linear_kwargs
            )
            new_module.weight = original_module.weight
            new_module.bias = original_module.bias

            setattr(parent, child_name, new_module)
            self._replaced_modules[module_name] = original_module
        else:
            current_module.update_layer(
                adapter_name,
                k,
                dtype=dtype,
                dropout=peft_config.dropout,
                device=current_module.weight.device,
            )

    def _mark_only_adapters_as_trainable(self) -> None:
        active_adapter = self._get_active_adapter()

        for n, p in self.model.named_parameters():
            p.requires_grad = False
            parts = n.split('.')
            if len(parts) >= 2:
                module_name = '.'.join(parts[:-1])
                parent_module = self.model.get_submodule(module_name)
                adapter_name = parts[-2]
                if isinstance(parent_module, SparseDelta) and adapter_name == active_adapter:
                    p.requires_grad = True

    def inject_adapter(self, model: nn.Module, adapter_name: str):
        peft_config = self.peft_config[adapter_name]
        self._check_new_adapter_config(peft_config)

        is_target_modules_in_base_model = False

        model_config = getattr(model, "config", {"model_type": "custom"})
        if hasattr(model_config, "to_dict"):
            model_config = model_config.to_dict()

        peft_config = self._prepare_adapter_config(peft_config, model_config)

        module_list = [
            (n, m) for n, m in model.named_modules()
            if self._check_target_module_exists(peft_config, n)
        ]
        if not module_list:
            raise ValueError(
                f"Target modules {peft_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )
        weights_in_trainable_modules = sum(
            original_numel(m.weight) for _, m in module_list
        )

        if peft_config.num_deltas is None:
            if peft_config.num_tunable_weights is None:
                if peft_config.density <= 0.0 or peft_config.density > 1.0:
                    raise ValueError(
                        f"SFT density must be in the range (0, 1] (and should usually be << 1), "
                        f"but is {config.density}."
                    )
                num_tunable_weights = int(self.total_params * peft_config.density)
            else:
                num_tunable_weights = peft_config.num_tunable_weights
            if num_tunable_weights <= 0:
                raise ValueError(
                    f"Number of tunable weights must be positive, "
                    f"but is {num_tunable_weights}."
                )
            if num_tunable_weights > weights_in_trainable_modules:
                raise ValueError(
                    f"Number of tunable weights {num_tunable_weights} exceeds total "
                    f"number of weights in trainable tensors ({weights_in_trainable_modules})."
                )

            peft_config.num_deltas = {
                n: int(num_tunable_weights * original_numel(m.weight) / weights_in_trainable_modules)
                for n, m in module_list
            }

        for module_name, k in peft_config.num_deltas.items():
            #module_name = '.'.join(n.split('.')[:-1])
            self._create_and_replace(
                peft_config,
                adapter_name,
                module_name,
                k=k,
            )

        self._mark_only_adapters_as_trainable()

        if self.peft_config[adapter_name].inference_mode:
            for n, p in self.model.named_parameters():
                if adapter_name in n:
                    p.requires_grad = False

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, Linear):
                module.disable_adapters = False if enabled else True
            elif isinstance(module, ModulesToSaveWrapper):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def _get_active_adapter(self) -> str:
        active_adapter = None
        for module in self.model.modules():
            if isinstance(module, Linear):
                active_adapter = module.active_adapter

        if active_adapter is None:
            raise ValueError(
                "Something went wrong, no active adapter could be found, please report the issue on GitHub"
            )
        return active_adapter

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, Linear):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.active_adapter = adapter_name

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        return peft_config

    def _unload_and_optionally_merge(self, merge=True, module_regex=None, progressbar: bool = False):
        peft_config = self.peft_config[self._get_active_adapter()]
        #logger.info(peft_config)
        if peft_config.dtype is None or isinstance(peft_config.dtype, torch.dtype):
            dtype = peft_config.dtype
        elif peft_config.dtype == "auto":
            dtype = target.weight.dtype
        elif peft_config.dtype == "float32":
            dtype = torch.float32
        elif peft_config.dtype == "float16":
            dtype = torch.float16
        elif peft_config.dtype == "bfloat16":
            dtype = torch.bfloat16
        else:
            raise ValueError(
                f"Unsupported dtype requested for SFT delta: {peft_config.dtype}"
            )

        key_list = [key for key, _ in self.model.named_modules() if "sft_delta" not in key]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                current_module, original_module, parent, child_name = self._get_ancestry(key)
            except AttributeError:
                continue

            if isinstance(current_module, Linear):
                if merge:
                    if module_regex is None or re.fullmatch(module_regex, key) is not None:
                        logger.info(f'Applying SFT to module {key}')
                        current_module.merge(original_module)
                    else:
                        logger.info(f'Not applying SFT to module {key} due to filter regex')
                setattr(parent, child_name, original_module)
                del self._replaced_modules[key]

        return self.model

    def forward(self, *args, **kwargs):
        results = self.model.forward(*args, **kwargs)
        loss = results[0]
        config = self.peft_config[self._get_active_adapter()]
        if loss is not None and config.l2_reg != 0.0:
            assert loss.numel() == 1
            reg_losses = []
            total_params = sum(
                p.numel() for p in self.model.parameters()
                if p.requires_grad
            )
            assert total_params != 0
            reg_losses = [
                (config.l2_reg / total_params) * torch.sum(torch.square(p))
                for p in self.model.parameters()
                if p.requires_grad
            ]
            reg_loss = torch.sum(torch.stack(reg_losses))
            self._losses['reg_loss'] += reg_loss.item()
            loss += reg_loss
        return results

    def merge_and_unload(self, module_regex=None, progressbar: bool = False):
        return self._unload_and_optionally_merge(module_regex=module_regex, progressbar=progressbar)

    def unload(self):
        return self._unload_and_optionally_merge(merge=False)
