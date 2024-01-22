from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class SftConfig(PeftConfig):

    density: float = field(default=0.01, metadata={"help": "Density of SFT, i.e. proportion of model weights which are tunable."})
    num_tunable_weights: int = field(
        default=None,
        metadata={
            "help": "Total number of tunable weights across all parameter tensors. Overrides --density if provided."
        }
    )
    num_deltas: Optional[Dict[str, int]] = field(
        default=None,
        metadata={
            "help": "Dict mapping linear module names to the number of deltas in their SFT. "
                    "Overrides --density, --num_tunable_weights and --target_modules if provided."
        },
    )

    selection_algorithm: Optional[str] = field(
        default="rigl",
        metadata={
            "help": (
                "SFT selection algorithm."
            ),
        },
    )

    reselection_steps: Optional[int] = field(
        default=20,
        metadata={"help": "Number of steps between reselections of tunable weights."},
    )
    selection_accumulation_steps: Optional[int] = field(
        default=5,
        metadata={
            "help": "Number of steps to accumulate gradients when selecting "
                    "params to regrow during RigL selection."
        },
    )
    initial_reselection_rate: Optional[float] = field(
        default=0.2,
        metadata={"help": "Proportion of weights to change in first reselection step."},
    )

    candidate_reselection_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Number of steps between reselections of tunable weights."},
    )
    candidate_reselection_proportion: Optional[float] = field(
        default=0.2,
        metadata={"help": "Number of steps between reselections of tunable weights."},
    )

    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": (
                "Torch dtype for SFT deltas."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    layers_to_transform: Optional[Union[List[int], int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index."
        },
    )
    layers_pattern: Optional[str] = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern."
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.SFT

