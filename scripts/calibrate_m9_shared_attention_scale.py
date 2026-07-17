"""
Calibrate one shared M9-0 attention output scale.

Rules:
- Training data only.
- Pool unseen-Pr and unseen-Ra training features.
- Use the same paired Norm / pre_proj / post_proj / gate initialization.
- Use multiple fixed random seeds.
- M9-0a and M9-0b must share the same lambda.
- Lambda is anchored by matching M9-0a Static Softmax branch RMS to
  the M7 FieldCoupling branch RMS.
- The same lambda is then applied to M9-0b without separate calibration.
- No model training and no validation/test metrics.

Frozen before this script:
    score_scale_mode = sqrt_token_dim
    score_temperature = 0.03
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from datasets.rbc_dataset import RBCDataset
from models.blocks.field_attention import (
    StaticSoftmaxFieldAttentionBlock,
    StateDependentFieldAttentionBlock,
)
from models.blocks.field_coupling import FieldCouplingBlock
from models.operators.fno2d_fieldwise import FieldWiseFNO2d


EPS = 1e-12
TEMPERATURE = 0.03
SCORE_SCALE_MODE = "sqrt_token_dim"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate one shared lambda for "
            "M9-0a and M9-0b."
        )
    )

    parser.add_argument("--pr_split", required=True)
    parser.add_argument("--pr_stats", required=True)
    parser.add_argument("--pr_m6_ckpt", required=True)

    parser.add_argument("--ra_split", required=True)
    parser.add_argument("--ra_stats", required=True)
    parser.add_argument("--ra_m6_ckpt", required=True)

    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_json", required=True)

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--num_batches",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--seeds",
        type=str,
        default="20260717,20260718,20260719",
    )

    parser.add_argument(
        "--qk_init_std",
        type=float,
        default=0.02,
    )

    return parser.parse_args()


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = root / path

    return path.resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_state_dict(checkpoint):
    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        return checkpoint["model_state_dict"]

    return checkpoint


def rms_from_sums(
    sumsq: float,
    count: int,
) -> float:
    return math.sqrt(
        sumsq / max(count, 1)
    )


def update_statistics(
    update: torch.Tensor,
) -> dict:
    update64 = update.detach().double()

    total_sumsq = float(
        update64.square().sum().cpu().item()
    )
    total_count = int(update64.numel())

    per_field_sumsq = (
        update64.square()
        .sum(dim=(0, 2, 3, 4))
        .cpu()
        .tolist()
    )

    per_field_count = int(
        update64.shape[0]
        * update64.shape[2]
        * update64.shape[3]
        * update64.shape[4]
    )

    return {
        "total_sumsq": total_sumsq,
        "total_count": total_count,
        "per_field_sumsq": [
            float(value)
            for value in per_field_sumsq
        ],
        "per_field_count": per_field_count,
    }


def merge_statistics(
    accumulator: dict,
    current: dict,
) -> None:
    accumulator["total_sumsq"] += (
        current["total_sumsq"]
    )
    accumulator["total_count"] += (
        current["total_count"]
    )

    for field_idx, value in enumerate(
        current["per_field_sumsq"]
    ):
        accumulator["per_field_sumsq"][
            field_idx
        ] += value

    accumulator["per_field_count"] += (
        current["per_field_count"]
    )


def empty_statistics(num_fields: int) -> dict:
    return {
        "total_sumsq": 0.0,
        "total_count": 0,
        "per_field_sumsq": [
            0.0 for _ in range(num_fields)
        ],
        "per_field_count": 0,
    }


def collect_features(
    name: str,
    split_path: Path,
    stats_path: Path,
    checkpoint_path: Path,
    batch_size: int,
    num_batches: int,
    device: torch.device,
) -> list[torch.Tensor]:
    with split_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        split_config = json.load(file)

    dataset = RBCDataset(
        split_config=split_config["train"],
        normalize=True,
        stats_path=str(stats_path),
        return_sequence=True,
        target_steps=4,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )

    model = FieldWiseFNO2d(
        in_channels=16,
        out_channels=4,
        modes1=16,
        modes2=16,
        width=32,
        context_length=4,
        num_fields=4,
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        extract_state_dict(checkpoint),
        strict=True,
    )

    model.eval()

    print(
        f"\n[{name}] train samples: "
        f"{len(dataset)}"
    )
    print(
        f"[{name}] M6 checkpoint loaded "
        "with strict=True"
    )

    feature_batches = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            loader,
            start=1,
        ):
            x_norm = batch[0].to(
                device=device,
                dtype=torch.float32,
            )

            _, field_features = (
                model.encode_fields(x_norm)
            )

            field_stack = torch.stack(
                field_features,
                dim=1,
            )

            feature_batches.append(
                field_stack.detach().cpu()
            )

            print(
                f"[{name}] collected "
                f"{batch_idx}/{num_batches}: "
                f"{tuple(field_stack.shape)}"
            )

            if batch_idx >= num_batches:
                break

    if len(feature_batches) != num_batches:
        raise RuntimeError(
            f"{name}: expected {num_batches} "
            f"batches, got {len(feature_batches)}"
        )

    return feature_batches


def m7_branch_update(
    block: FieldCouplingBlock,
    x: torch.Tensor,
) -> torch.Tensor:
    batch_size, fields, channels, height, width = (
        x.shape
    )

    h = x.reshape(
        batch_size * fields,
        channels,
        height,
        width,
    )

    h = block.norm(h)
    h = block.pre_proj(h)
    h = block.act(h)

    h = h.reshape(
        batch_size,
        fields,
        block.hidden_channels,
        height,
        width,
    )

    weight = (
        block.effective_coupling_matrix()
        .to(dtype=h.dtype, device=h.device)
    )

    mixed = torch.einsum(
        "ij,bjchw->bichw",
        weight,
        h,
    )

    projected = mixed.reshape(
        batch_size * fields,
        block.hidden_channels,
        height,
        width,
    )

    projected = block.post_proj(projected)
    projected = block.dropout(projected)

    projected = projected.reshape(
        batch_size,
        fields,
        channels,
        height,
        width,
    )

    gate = block.gate_values().to(
        dtype=x.dtype,
        device=x.device,
    )

    gate = gate.view(
        1,
        fields,
        1,
        1,
        1,
    )

    return gate * projected


def create_paired_blocks(
    seed: int,
    channels: int,
    num_fields: int,
    qk_init_std: float,
    device: torch.device,
):
    set_seed(seed)

    m7 = FieldCouplingBlock(
        channels=channels,
        num_fields=num_fields,
        hidden_channels=channels,
        dropout=0.0,
        init_gate=-4.0,
        use_norm=True,
    ).to(device)

    static = StaticSoftmaxFieldAttentionBlock(
        channels=channels,
        num_fields=num_fields,
        hidden_channels=channels,
        dropout=0.0,
        init_gate=-4.0,
        use_norm=True,
        attention_output_scale=1.0,
        score_temperature=TEMPERATURE,
    ).to(device)

    dynamic = StateDependentFieldAttentionBlock(
        channels=channels,
        num_fields=num_fields,
        hidden_channels=channels,
        qk_channels=channels,
        dropout=0.0,
        init_gate=-4.0,
        use_norm=True,
        attention_output_scale=1.0,
        score_scale_mode=SCORE_SCALE_MODE,
        score_temperature=TEMPERATURE,
        qk_init_std=qk_init_std,
    ).to(device)

    # Exact paired initialization for the shared branch.
    static.copy_shared_branch_from(m7)
    dynamic.copy_shared_branch_from(m7)

    m7.eval()
    static.eval()
    dynamic.eval()

    return m7, static, dynamic


def evaluate_one_seed_split(
    seed: int,
    split_name: str,
    feature_batches: list[torch.Tensor],
    channels: int,
    num_fields: int,
    qk_init_std: float,
    device: torch.device,
):
    m7, static, dynamic = create_paired_blocks(
        seed=seed,
        channels=channels,
        num_fields=num_fields,
        qk_init_std=qk_init_std,
        device=device,
    )

    m7_stats = empty_statistics(num_fields)
    static_stats = empty_statistics(num_fields)
    dynamic_stats = empty_statistics(num_fields)
    feature_stats = empty_statistics(num_fields)

    with torch.no_grad():
        for features_cpu in feature_batches:
            features = features_cpu.to(
                device=device,
                dtype=torch.float32,
            )

            m7_update = m7_branch_update(
                m7,
                features,
            )

            static_info = static(
                features,
                return_diagnostics=True,
            )

            dynamic_info = dynamic(
                features,
                return_diagnostics=True,
            )

            merge_statistics(
                m7_stats,
                update_statistics(m7_update),
            )
            merge_statistics(
                static_stats,
                update_statistics(
                    static_info["branch_update"]
                ),
            )
            merge_statistics(
                dynamic_stats,
                update_statistics(
                    dynamic_info["branch_update"]
                ),
            )
            merge_statistics(
                feature_stats,
                update_statistics(features),
            )

    m7_rms = rms_from_sums(
        m7_stats["total_sumsq"],
        m7_stats["total_count"],
    )
    static_rms = rms_from_sums(
        static_stats["total_sumsq"],
        static_stats["total_count"],
    )
    dynamic_rms = rms_from_sums(
        dynamic_stats["total_sumsq"],
        dynamic_stats["total_count"],
    )
    feature_rms = rms_from_sums(
        feature_stats["total_sumsq"],
        feature_stats["total_count"],
    )

    return {
        "seed": seed,
        "split": split_name,
        "m7_rms": m7_rms,
        "static_scale1_rms": static_rms,
        "dynamic_scale1_rms": dynamic_rms,
        "feature_rms": feature_rms,
        "m7_relative_branch": (
            m7_rms / (feature_rms + EPS)
        ),
        "static_scale1_relative_branch": (
            static_rms / (feature_rms + EPS)
        ),
        "dynamic_scale1_relative_branch": (
            dynamic_rms / (feature_rms + EPS)
        ),
        "lambda_static_candidate": (
            m7_rms / (static_rms + EPS)
        ),
        "lambda_dynamic_candidate_diagnostic": (
            m7_rms / (dynamic_rms + EPS)
        ),
        "_m7_stats": m7_stats,
        "_static_stats": static_stats,
        "_dynamic_stats": dynamic_stats,
        "_feature_stats": feature_stats,
    }


def strip_internal(row: dict) -> dict:
    return {
        key: value
        for key, value in row.items()
        if not key.startswith("_")
    }


def main():
    args = parse_args()

    project_root = (
        Path(__file__).resolve().parent.parent
    )

    paths = {
        "pr_split": resolve_path(
            project_root,
            args.pr_split,
        ),
        "pr_stats": resolve_path(
            project_root,
            args.pr_stats,
        ),
        "pr_ckpt": resolve_path(
            project_root,
            args.pr_m6_ckpt,
        ),
        "ra_split": resolve_path(
            project_root,
            args.ra_split,
        ),
        "ra_stats": resolve_path(
            project_root,
            args.ra_stats,
        ),
        "ra_ckpt": resolve_path(
            project_root,
            args.ra_m6_ckpt,
        ),
        "output_csv": resolve_path(
            project_root,
            args.output_csv,
        ),
        "output_json": resolve_path(
            project_root,
            args.output_json,
        ),
    }

    for key in (
        "pr_split",
        "pr_stats",
        "pr_ckpt",
        "ra_split",
        "ra_stats",
        "ra_ckpt",
    ):
        if not paths[key].exists():
            raise FileNotFoundError(paths[key])

    paths["output_csv"].parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    paths["output_json"].parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    seeds = [
        int(value.strip())
        for value in args.seeds.split(",")
        if value.strip()
    ]

    if not seeds:
        raise ValueError("No seeds supplied.")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "========== M9 SHARED LAMBDA CALIBRATION =========="
    )
    print("Device:", device)
    print("Temperature:", TEMPERATURE)
    print("Score scaling:", SCORE_SCALE_MODE)
    print("Seeds:", seeds)
    print("Batches per split:", args.num_batches)
    print("Batch size:", args.batch_size)
    print(
        "Calibration anchor: "
        "M9-0a Static Softmax -> M7 branch RMS"
    )
    print(
        "M9-0b receives exactly the same lambda."
    )
    print(
        "⚠️ Training data only; no model training."
    )

    pr_features = collect_features(
        name="unseen_pr",
        split_path=paths["pr_split"],
        stats_path=paths["pr_stats"],
        checkpoint_path=paths["pr_ckpt"],
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        device=device,
    )

    ra_features = collect_features(
        name="unseen_ra",
        split_path=paths["ra_split"],
        stats_path=paths["ra_stats"],
        checkpoint_path=paths["ra_ckpt"],
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        device=device,
    )

    channels = int(pr_features[0].shape[2])
    num_fields = int(pr_features[0].shape[1])

    rows = []

    pooled_m7 = empty_statistics(num_fields)
    pooled_static = empty_statistics(num_fields)
    pooled_dynamic = empty_statistics(num_fields)
    pooled_features = empty_statistics(num_fields)

    for seed in seeds:
        for split_name, feature_batches in (
            ("unseen_pr", pr_features),
            ("unseen_ra", ra_features),
        ):
            print(
                f"\nRunning seed={seed}, "
                f"split={split_name}"
            )

            row = evaluate_one_seed_split(
                seed=seed,
                split_name=split_name,
                feature_batches=feature_batches,
                channels=channels,
                num_fields=num_fields,
                qk_init_std=args.qk_init_std,
                device=device,
            )

            rows.append(row)

            merge_statistics(
                pooled_m7,
                row["_m7_stats"],
            )
            merge_statistics(
                pooled_static,
                row["_static_stats"],
            )
            merge_statistics(
                pooled_dynamic,
                row["_dynamic_stats"],
            )
            merge_statistics(
                pooled_features,
                row["_feature_stats"],
            )

            print(
                "  M7 RMS:             "
                f"{row['m7_rms']:.8e}"
            )
            print(
                "  Static RMS scale=1: "
                f"{row['static_scale1_rms']:.8e}"
            )
            print(
                "  Dynamic RMS scale=1:"
                f" {row['dynamic_scale1_rms']:.8e}"
            )
            print(
                "  Static λ candidate: "
                f"{row['lambda_static_candidate']:.8f}"
            )

    pooled_m7_rms = rms_from_sums(
        pooled_m7["total_sumsq"],
        pooled_m7["total_count"],
    )
    pooled_static_rms = rms_from_sums(
        pooled_static["total_sumsq"],
        pooled_static["total_count"],
    )
    pooled_dynamic_rms = rms_from_sums(
        pooled_dynamic["total_sumsq"],
        pooled_dynamic["total_count"],
    )
    pooled_feature_rms = rms_from_sums(
        pooled_features["total_sumsq"],
        pooled_features["total_count"],
    )

    shared_lambda = (
        pooled_m7_rms
        / (pooled_static_rms + EPS)
    )

    static_ratio = (
        shared_lambda
        * pooled_static_rms
        / (pooled_m7_rms + EPS)
    )

    dynamic_ratio = (
        shared_lambda
        * pooled_dynamic_rms
        / (pooled_m7_rms + EPS)
    )

    for row in rows:
        row["shared_lambda"] = shared_lambda
        row["static_to_m7_after_shared_lambda"] = (
            shared_lambda
            * row["static_scale1_rms"]
            / (row["m7_rms"] + EPS)
        )
        row["dynamic_to_m7_after_shared_lambda"] = (
            shared_lambda
            * row["dynamic_scale1_rms"]
            / (row["m7_rms"] + EPS)
        )

    field_names = [
        "buoyancy",
        "u_x",
        "u_y",
        "pressure",
    ]

    per_field = {}

    for field_idx, field_name in enumerate(
        field_names
    ):
        m7_field_rms = rms_from_sums(
            pooled_m7["per_field_sumsq"][
                field_idx
            ],
            pooled_m7["per_field_count"],
        )

        static_field_rms = rms_from_sums(
            pooled_static["per_field_sumsq"][
                field_idx
            ],
            pooled_static["per_field_count"],
        )

        dynamic_field_rms = rms_from_sums(
            pooled_dynamic["per_field_sumsq"][
                field_idx
            ],
            pooled_dynamic["per_field_count"],
        )

        per_field[field_name] = {
            "m7_rms": m7_field_rms,
            "static_rms_after_shared_lambda": (
                shared_lambda
                * static_field_rms
            ),
            "dynamic_rms_after_shared_lambda": (
                shared_lambda
                * dynamic_field_rms
            ),
            "static_to_m7_ratio": (
                shared_lambda
                * static_field_rms
                / (m7_field_rms + EPS)
            ),
            "dynamic_to_m7_ratio": (
                shared_lambda
                * dynamic_field_rms
                / (m7_field_rms + EPS)
            ),
        }

    output_rows = [
        strip_internal(row)
        for row in rows
    ]

    with paths["output_csv"].open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                output_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "experiment": (
            "M9-0 shared attention scale calibration"
        ),
        "data_scope": (
            "fixed unseen-Pr and unseen-Ra "
            "training batches only"
        ),
        "temperature": TEMPERATURE,
        "score_scale_mode": SCORE_SCALE_MODE,
        "qk_init_std": args.qk_init_std,
        "seeds": seeds,
        "batch_size": args.batch_size,
        "num_batches_per_split": (
            args.num_batches
        ),
        "calibration_anchor": (
            "pooled M9-0a static scale=1 branch RMS "
            "matched to pooled M7 branch RMS"
        ),
        "shared_lambda": shared_lambda,
        "pooled": {
            "feature_rms": pooled_feature_rms,
            "m7_branch_rms": pooled_m7_rms,
            "static_scale1_branch_rms": (
                pooled_static_rms
            ),
            "dynamic_scale1_branch_rms": (
                pooled_dynamic_rms
            ),
            "static_to_m7_after_shared_lambda": (
                static_ratio
            ),
            "dynamic_to_m7_after_shared_lambda": (
                dynamic_ratio
            ),
        },
        "per_field": per_field,
        "rows": output_rows,
    }

    with paths["output_json"].open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "\n========== SHARED LAMBDA SUMMARY =========="
    )
    print(
        f"pooled M7 RMS:              "
        f"{pooled_m7_rms:.8e}"
    )
    print(
        f"pooled Static RMS scale=1:  "
        f"{pooled_static_rms:.8e}"
    )
    print(
        f"pooled Dynamic RMS scale=1: "
        f"{pooled_dynamic_rms:.8e}"
    )
    print(
        f"shared lambda:              "
        f"{shared_lambda:.8f}"
    )
    print(
        f"Static/M7 after lambda:     "
        f"{static_ratio:.6f}"
    )
    print(
        f"Dynamic/M7 after lambda:    "
        f"{dynamic_ratio:.6f}"
    )

    print("\nPer-field ratios:")
    for field_name, values in per_field.items():
        print(
            f"  {field_name:10s} | "
            f"Static/M7="
            f"{values['static_to_m7_ratio']:.6f} | "
            f"Dynamic/M7="
            f"{values['dynamic_to_m7_ratio']:.6f}"
        )

    print(
        "\n✅ Shared lambda calibration complete."
    )
    print("CSV:", paths["output_csv"])
    print("JSON:", paths["output_json"])
    print(
        "⚠️ Do not start formal training until "
        "the shared lambda is reviewed and frozen."
    )


if __name__ == "__main__":
    main()
