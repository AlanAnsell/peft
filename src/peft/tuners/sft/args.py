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
    dropout: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "Probability of dropping out each SFT delta.",
        },
    )
    selection_algorithm: Optional[str] = field(
        default="rigl",
        metadata={"help": "Method used to select subset of weights to fine-tune."},
    )
    selection_duration: Optional[float] = field(
        default=1.0,
        metadata={"help": "Proportion of training during which selection will take place."},
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
    projection_alpha: Optional[float] = field(
        default=0.1,
        metadata={"help": "Projection coefficien."},
    )
    l1_reg: Optional[float] = field(
        default=0.0,
        metadata={"help": "L1 regularization coefficient for SFT deltas."},
    )
