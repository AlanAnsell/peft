import re
import warnings
from dataclasses import asdict, replace
from enum import Enum

import torch
from torch import nn
from tqdm import tqdm

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
from .layer import Linear, SparseDelta


#if is_bnb_available():
#    import bitsandbytes as bnb
#
#    from .bnb import Linear8bitLt
#
#if is_bnb_4bit_available():
#    from .bnb import Linear4bit


class SftModel(BaseTuner):

    def __init__(self, model, config, adapter_name) -> None:
        self.total_params = 0
        for p in model.parameters():
            self.total_params += p.numel()
        super().__init__(model, config, adapter_name)

    def _check_new_adapter_config(self, config: SftConfig) -> None:
        pass

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
        sft_config,
        adapter_name,
        target,
        target_name,
        parent,
        k=None,
        **optional_kwargs,
    ):
        if k is None:
            raise ValueError("k must be specified.")
        new_module = self._create_new_module(
            sft_config,
            adapter_name,
            target,
            k,
            **optional_kwargs
        )
        self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _replace_module(parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable
        new_module.weight = child.weight
        if hasattr(child, "bias"):
            new_module.bias = child.bias

        new_module.to(child.weight.device)
        #if getattr(child, "state", None) is not None:
        #    new_module.state = child.state
        #    new_module.to(child.weight.device)

        ## dispatch to correct device
        #for name, module in new_module.named_modules():
        #    if "sft_delta" in name:
        #        module.to(child.weight.device)

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
            m.weight.numel() for _, m in module_list
        )
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
                f"number of weights in trainable tensors ({self.total_params})."
            )

        for key, _ in module_list:
            parent, target, target_name = _get_submodules(model, key)
            
            proportion = target.weight.numel() / weights_in_trainable_modules
            k = int(proportion * num_tunable_weights)
            self._create_and_replace(
                peft_config,
                adapter_name,
                target,
                target_name,
                parent,
                k=k,
            )

        self._mark_only_adapters_as_trainable()

        if self.peft_config[adapter_name].inference_mode:
            for n, p in self.model.named_parameters():
                if adapter_name in n:
                    p.requires_grad = False

    @staticmethod
    def _create_new_module(peft_config, adapter_name, target, k, **kwargs):
        if not isinstance(target, torch.nn.Linear):
            raise ValueError(
                f"Target module {type(target)} is not supported. Currently, "
                f"only the following modules are supported: `torch.nn.Linear`."
            )

        if isinstance(peft_config.dtype, torch.dtype):
            dtype = peft_config.dtype
        elif peft_config.dtype == "auto":
            dtype = target.weight.dtype
        elif peft_config == "float32":
            dtype = torch.float32
        elif peft_config.dtype == "float16":
            dtype = torch.float16
        elif peft_config.dtype == "bfloat16":
            dtype = torch.bfloat16
        else:
            raise ValueError(
                f"Unsupported dtype requested for SFT delta: {peft_config.dtype}"
            )
        new_module = Linear(
            adapter_name,
            target.in_features,
            target.out_features,
            k,
            bias=target.bias is not None,
            dtype=dtype,
        )

        return new_module

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

    def _unload_and_optionally_merge(self, merge=True, progressbar: bool = False):
        key_list = [key for key, _ in self.model.named_modules() if "sft_delta" not in key]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, Linear):
                new_module = torch.nn.Linear(
                    target.in_features,
                    target.out_features,
                    bias=target.bias is not None,
                )
                if merge:
                    target.merge()
                self._replace_module(parent, target_name, new_module, target)

            # save any additional trainable modules part of `modules_to_save`
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model

    #def delete_adapter(self, adapter_name):
    #    """
    #    Deletes an existing adapter.

    #    Args:
    #        adapter_name (str): Name of the adapter to be deleted.
    #    """
    #    if adapter_name not in list(self.peft_config.keys()):
    #        raise ValueError(f"Adapter {adapter_name} does not exist")
    #    del self.peft_config[adapter_name]
    #    key_list = [key for key, _ in self.model.named_modules() if "sft_args" not in key]
    #    for key in key_list:
    #        _, target, _ = _get_submodules(self.model, key)
    #        if isinstance(target, LoraLayer):
    #            for attr in [
    #                "r",
    #                "lora_alpha",
    #                "scaling",
    #                "lora_A",
    #                "lora_B",
    #                "lora_embedding_A",
    #                "lora_embedding_B",
    #                "lora_dropout",
    #            ]:
    #                if adapter_name in getattr(target, attr):
    #                    getattr(target, attr).pop(adapter_name)
    #            if target.active_adapter == adapter_name:
    #                resetting_active_adapter = list(self.peft_config.keys())[0]
    #                warnings.warn(
    #                    f"Adapter {adapter_name} was active which is now deleted. Setting active adapter to {resetting_active_adapter}. "
    #                )
    #                target.active_adapter = resetting_active_adapter

    def merge_and_unload(self, progressbar: bool = False):
        return self._unload_and_optionally_merge(progressbar=progressbar)

    def unload(self):
        return self._unload_and_optionally_merge(merge=False)
