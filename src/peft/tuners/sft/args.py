from dataclasses import dataclass, field
from typing import Optional

@dataclass
class SftArguments:
    "Arguments pertaining to sparse fine-tuning configuration."""
    
    density: Optional[float] = field(
        default=0.01,
        metadata={
            "help": "Density of SFT, i.e. proportion of total weights to be fine-tuned.",
        },
    )
    selection_algorithm: Optional[str] = field(
        default="SM3",
        metadata={"help": "Method used to select subset of weights to fine-tune."},
    )

    reselection_steps: Optional[int] = field(
        default=100,
        metadata={"help": "Number of steps between reselections of tunable weights."},
    )
    selection_accumulation_steps: Optional[int] = field(
        default=5,
        metadata={
            "help": "Number of steps to accumulate gradients when selecting "
                    "params to regrow during RigL selection."
        },
    )
    candidates_per_replacement_slot: Optional[int] = field(
        default=5,
        metadata={
            "help": "Ratio of number of params to accumulate gradients for "
            ": params to be regrown during RigL selection."
        },
    )
    initial_reselection_rate: Optional[float] = field(
        default=0.2,
        metadata={"help": "Proportion of weights to change in first reselection step."},
    )
    l1_reg: Optional[float] = field(
        default=0.0,
        metadata={"help": "L1 regularization coefficient for SFT deltas."},
    )
