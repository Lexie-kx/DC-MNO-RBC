import os
import sys
import argparse

import torch

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from models.blocks.param_conditioned_field_coupling import (
    ParamConditionedFieldCouplingBlock,
)


def parse_mask_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--mask_target",
        type=str,
        required=True,
        choices=[
            "buoyancy",
            "u_x",
            "u_y",
            "pressure",
            "all",
        ],
    )
    args, remaining = parser.parse_known_args()
    return args, remaining


MASK_INDEX = {
    "buoyancy": 0,
    "u_x": 1,
    "u_y": 2,
    "pressure": 3,
}


def main():
    mask_args, remaining = parse_mask_args()

    original_method = (
        ParamConditionedFieldCouplingBlock
        .conditioned_delta_matrix
    )

    def masked_conditioned_delta_matrix(self, param):
        delta = original_method(self, param)

        target_mask = torch.ones(
            self.num_fields,
            device=delta.device,
            dtype=delta.dtype,
        )

        if mask_args.mask_target == "all":
            target_mask.zero_()
        else:
            target_index = MASK_INDEX[
                mask_args.mask_target
            ]
            target_mask[target_index] = 0.0

        return delta * target_mask.view(
            1,
            self.num_fields,
            1,
        )

    (
        ParamConditionedFieldCouplingBlock
        .conditioned_delta_matrix
    ) = masked_conditioned_delta_matrix

    print(
        "🧪 M8-B dynamic target-row masking enabled"
    )
    print(
        f"📌 Mask target: {mask_args.mask_target}"
    )
    print(
        "📌 Static base coupling remains unchanged"
    )

    sys.argv = [sys.argv[0], *remaining]

    from evaluation.evaluate_cross_param_rollout_m8 import (
        main as original_main,
    )

    original_main()


if __name__ == "__main__":
    main()
