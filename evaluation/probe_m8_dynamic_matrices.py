import os
import sys
import csv
import math
import argparse
import itertools

import torch

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from evaluation.evaluate_cross_param_rollout_m8 import (
    build_m8_model_from_checkpoint,
)


FIELDS = [
    "buoyancy",
    "u_x",
    "u_y",
    "pressure",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint = os.path.abspath(args.checkpoint)
    output = os.path.abspath(args.output)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = build_m8_model_from_checkpoint(
        checkpoint_path=checkpoint,
        device=device,
        expected_coupling_mode="parameter_conditioned",
    )

    labels = []
    params = []

    for ra in [1e6, 1e7, 1e8]:
        for pr in [0.5, 1.0, 2.0]:
            labels.append(
                f"Ra={ra:.0e}, Pr={pr:g}"
            )
            params.append(
                [
                    math.log10(ra),
                    math.log10(pr),
                ]
            )

    param_tensor = torch.tensor(
        params,
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        delta = (
            model.field_coupling
            .conditioned_delta_matrix(param_tensor)
            .detach()
            .cpu()
        )

    mask = (
        model.field_coupling
        .offdiag_mask
        .detach()
        .cpu()
        .bool()
    )

    torch.set_printoptions(
        precision=6,
        sci_mode=False,
    )

    print("\n================ Dynamic Matrices ================")

    for index, label in enumerate(labels):
        print(f"\n[{label}]")
        print(delta[index])

    offdiag_values = delta[:, mask]

    across_condition_std = offdiag_values.std(
        dim=0,
        unbiased=False,
    )

    print("\n================ Condition Diversity ================")
    print(
        "Mean off-diagonal std across conditions: "
        f"{across_condition_std.mean().item():.8f}"
    )
    print(
        "Max off-diagonal std across conditions:  "
        f"{across_condition_std.max().item():.8f}"
    )

    pairwise_max_diffs = []

    for i, j in itertools.combinations(
        range(len(labels)),
        2,
    ):
        pairwise_max_diffs.append(
            (delta[i] - delta[j])
            .abs()
            .max()
            .item()
        )

    print(
        "Mean pairwise max-abs difference: "
        f"{sum(pairwise_max_diffs) / len(pairwise_max_diffs):.8f}"
    )
    print(
        "Max pairwise max-abs difference:  "
        f"{max(pairwise_max_diffs):.8f}"
    )

    rows = []

    for condition_index, label in enumerate(labels):
        for target_index, target in enumerate(FIELDS):
            for source_index, source in enumerate(FIELDS):
                rows.append(
                    {
                        "condition": label,
                        "target_field": target,
                        "source_field": source,
                        "dynamic_delta": (
                            delta[
                                condition_index,
                                target_index,
                                source_index,
                            ].item()
                        ),
                    }
                )

    os.makedirs(
        os.path.dirname(output),
        exist_ok=True,
    )

    with open(
        output,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "condition",
                "target_field",
                "source_field",
                "dynamic_delta",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Saved to: {output}")


if __name__ == "__main__":
    main()
