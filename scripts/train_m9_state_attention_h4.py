"""
M9-0b State-Dependent Field Attention H4 entry point.

Experiment type:
    Formal attention baseline candidate.
    This is not the complete M9 model.
"""

from train_m9_attention_h4_common import (
    main_for_variant,
)


if __name__ == "__main__":
    main_for_variant("state")
