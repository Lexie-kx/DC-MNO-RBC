"""
M9-0b Attention intervention evaluation.

Control experiment using one fixed M9-0b checkpoint:

1. Normal
   Each sample uses its own state-dependent Attention.

2. CyclicShuffle
   Each target sample uses Attention produced by a distant
   donor sample during the same rollout step.

3. TrainingMean
   Every test sample and rollout step uses one fixed mean
   Attention matrix estimated from the training split.

Outputs:
- aggregate field/global rollout metrics
- per-sample paired metrics
- Normal-vs-control difference and win-rate table
- Attention diagnostics and matrices

This is mechanism validation, not a new trained model.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from constants import FIELD_ORDER
from evaluation.evaluate_cross_param_rollout_m9_0 import (
    CrossParamRolloutDataset,
    calculate_per_sample_metrics,
    load_model_state,
    new_accumulator,
    resolve_path,
    update_accumulator,
)
from models.operators.fno2d_m9_state_attention import (
    M9StateAttentionFNO2d,
)
from training.normalization import FieldWiseNormalizer


INTERVENTION_ORDER = [
    "Normal",
    "CyclicShuffle",
    "TrainingMean",
]

COMPARISONS = [
    (
        "Normal",
        "CyclicShuffle",
        "Normal - CyclicShuffle",
    ),
    (
        "Normal",
        "TrainingMean",
        "Normal - TrainingMean",
    ),
]

METRICS = [
    "buoyancy",
    "u_x",
    "u_y",
    "pressure",
    "global",
]

EPS = 1e-12


class PairedInterventionDataset(Dataset):
    """
    Pair every target sample with a distant donor sample.

    Target:
        evaluated normally or using an override.

    Donor:
        rolled out normally and supplies its state-dependent
        Attention to the target during CyclicShuffle.
    """

    def __init__(
        self,
        base_dataset: CrossParamRolloutDataset,
        donor_offset: int | None = None,
    ) -> None:
        super().__init__()

        self.base_dataset = base_dataset
        self.num_samples = len(base_dataset)

        if self.num_samples < 2:
            raise ValueError(
                "At least two test samples are required."
            )

        if donor_offset is None:
            donor_offset = self.num_samples // 2

        donor_offset = int(
            donor_offset % self.num_samples
        )

        if donor_offset == 0:
            raise ValueError(
                "donor_offset must not map a sample "
                "to itself."
            )

        self.donor_offset = donor_offset

        same_group = 0
        same_trajectory = 0

        for index in range(self.num_samples):
            donor_index = (
                index + self.donor_offset
            ) % self.num_samples

            target_meta = base_dataset.index[index]
            donor_meta = base_dataset.index[donor_index]

            if (
                target_meta["group"]
                == donor_meta["group"]
            ):
                same_group += 1

                if (
                    target_meta["trajectory"]
                    == donor_meta["trajectory"]
                ):
                    same_trajectory += 1

        print(
            "📌 Cyclic donor offset:",
            self.donor_offset,
        )
        print(
            "📌 Same-group donor rate:",
            f"{100.0 * same_group / self.num_samples:.2f}%",
        )
        print(
            "📌 Same-group-and-trajectory rate:",
            f"{100.0 * same_trajectory / self.num_samples:.2f}%",
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        donor_index = (
            index + self.donor_offset
        ) % self.num_samples

        target = self.base_dataset[index]
        donor = self.base_dataset[donor_index]

        return {
            "target": target,
            "donor_x0_phys": donor["x0_phys"],
            "donor_sample_id": donor["sample_id"],
            "donor_group": donor["group"],
            "donor_trajectory": donor["trajectory"],
            "donor_t0": donor["t0"],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "M9-0b Normal / CyclicShuffle / "
            "TrainingMean Attention intervention."
        )
    )

    parser.add_argument(
        "--split",
        required=True,
    )
    parser.add_argument(
        "--stats",
        required=True,
    )
    parser.add_argument(
        "--config",
        default=(
            "configs/m9_0_attention_frozen.json"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )
    parser.add_argument(
        "--per_sample_output",
        required=True,
    )
    parser.add_argument(
        "--difference_output",
        required=True,
    )
    parser.add_argument(
        "--diagnostics_output",
        required=True,
    )

    parser.add_argument(
        "--horizons",
        default="1,4,8,16",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--mean_batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--donor_offset",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Lightweight trial only.",
    )
    parser.add_argument(
        "--max_mean_samples",
        type=int,
        default=None,
        help=(
            "Limit training samples used to estimate "
            "TrainingMean. Lightweight trial only."
        ),
    )

    return parser.parse_args()


def build_model(
    config: dict[str, Any],
    device: torch.device,
) -> M9StateAttentionFNO2d:
    return M9StateAttentionFNO2d(
        in_channels=16,
        out_channels=4,
        modes1=16,
        modes2=16,
        width=32,
        context_length=4,
        num_fields=4,
        field_width=int(
            config["field_width"]
        ),
        attention_hidden_channels=int(
            config["hidden_channels"]
        ),
        qk_channels=int(
            config["qk_channels"]
        ),
        attention_dropout=float(
            config["coupling_dropout"]
        ),
        attention_init_gate=float(
            config[
                "residual_gate_init_logit"
            ]
        ),
        attention_use_norm=bool(
            config["use_norm"]
        ),
        attention_output_scale=float(
            config[
                "attention_output_scale"
            ]
        ),
        score_scale_mode=str(
            config["score_scale_mode"]
        ),
        score_temperature=float(
            config["score_temperature"]
        ),
        qk_init_std=float(
            config["qk_init_std"]
        ),
    ).to(device)


@torch.no_grad()
def compute_training_mean_attention(
    model: M9StateAttentionFNO2d,
    split_items: list[dict[str, Any]],
    normalizer: FieldWiseNormalizer,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_samples: int | None,
) -> tuple[torch.Tensor, int]:
    print()
    print(
        "========== ESTIMATING TRAINING-MEAN "
        "ATTENTION =========="
    )

    train_dataset = CrossParamRolloutDataset(
        split_items=split_items,
        max_horizon=1,
        max_samples=max_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    attention_sum = torch.zeros(
        4,
        4,
        dtype=torch.float64,
        device=device,
    )

    num_samples = 0
    total_batches = len(train_loader)

    model.eval()

    for batch_index, batch in enumerate(
        train_loader,
        start=1,
    ):
        x0_phys = batch["x0_phys"].to(
            device,
            non_blocking=True,
        )

        x_norm = normalizer.normalize_x(
            x0_phys
        )

        (
            _fused,
            _field_features,
            _attended_features,
            diagnostics,
        ) = model.encode_fields(
            x_norm,
            return_attention=True,
        )

        if diagnostics is None:
            raise RuntimeError(
                "Attention diagnostics missing."
            )

        attention = diagnostics[
            "computed_attention"
        ].detach().double()

        attention_sum += attention.sum(dim=0)
        num_samples += attention.shape[0]

        if (
            batch_index % 50 == 0
            or batch_index == total_batches
        ):
            print(
                f"   mean-attention batch "
                f"{batch_index}/{total_batches}"
            )

    if num_samples == 0:
        raise RuntimeError(
            "No training samples were used."
        )

    mean_attention = (
        attention_sum / float(num_samples)
    ).float()

    diagonal_max = (
        mean_attention.diagonal().abs().max().item()
    )

    row_error = (
        mean_attention.sum(dim=-1) - 1.0
    ).abs().max().item()

    if diagonal_max > 1e-6:
        raise RuntimeError(
            "Training mean diagonal is nonzero."
        )

    if row_error > 1e-6:
        raise RuntimeError(
            "Training mean rows do not sum to one."
        )

    print(
        "✅ TrainingMean samples:",
        num_samples,
    )
    print("TrainingMean Attention:")
    print(mean_attention.detach().cpu().numpy())

    del train_loader
    del train_dataset
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return mean_attention, num_samples


def new_attention_accumulator(
    device: torch.device,
) -> dict[str, Any]:
    return {
        "num_samples": 0,
        "attention_sum": torch.zeros(
            4,
            4,
            dtype=torch.float64,
            device=device,
        ),
        "attention_sq_sum": torch.zeros(
            4,
            4,
            dtype=torch.float64,
            device=device,
        ),
        "entropy_sum": 0.0,
        "entropy_count": 0,
        "amax_sum": 0.0,
        "amax_count": 0,
        "drift_first_sum": 0.0,
        "drift_previous_sum": 0.0,
        "mismatch_sum": 0.0,
    }


def update_attention_accumulator(
    bucket: dict[str, Any],
    applied_attention: torch.Tensor,
    computed_attention: torch.Tensor,
    first_attention: torch.Tensor,
    previous_attention: torch.Tensor,
) -> None:
    applied = applied_attention.detach().double()
    computed = computed_attention.detach().double()
    first = first_attention.detach().double()
    previous = previous_attention.detach().double()

    batch_size = applied.shape[0]

    bucket["num_samples"] += batch_size

    bucket["attention_sum"] += (
        applied.sum(dim=0)
    )

    bucket["attention_sq_sum"] += (
        applied.square().sum(dim=0)
    )

    safe_attention = applied.clamp_min(EPS)

    entropy = -(
        safe_attention
        * torch.log(safe_attention)
    ).sum(dim=-1)

    entropy = (
        entropy
        / math.log(applied.shape[-1] - 1)
    )

    bucket["entropy_sum"] += float(
        entropy.sum().item()
    )
    bucket["entropy_count"] += int(
        entropy.numel()
    )

    attention_max = applied.max(
        dim=-1
    ).values

    bucket["amax_sum"] += float(
        attention_max.sum().item()
    )
    bucket["amax_count"] += int(
        attention_max.numel()
    )

    drift_first = (
        applied - first
    ).abs().mean(dim=(1, 2))

    drift_previous = (
        applied - previous
    ).abs().mean(dim=(1, 2))

    mismatch = (
        applied - computed
    ).abs().mean(dim=(1, 2))

    bucket["drift_first_sum"] += float(
        drift_first.sum().item()
    )

    bucket["drift_previous_sum"] += float(
        drift_previous.sum().item()
    )

    bucket["mismatch_sum"] += float(
        mismatch.sum().item()
    )


@torch.no_grad()
def rollout_intervention(
    model: M9StateAttentionFNO2d,
    mode: str,
    target_x0_phys: torch.Tensor,
    donor_x0_phys: torch.Tensor,
    normalizer: FieldWiseNormalizer,
    training_mean_attention: torch.Tensor,
    max_horizon: int,
    attention_accumulators: dict[
        tuple[str, int],
        dict[str, Any],
    ],
) -> torch.Tensor:
    target_x_norm = normalizer.normalize_x(
        target_x0_phys
    )

    donor_x_norm = normalizer.normalize_x(
        donor_x0_phys
    )

    predictions = []

    first_attention = None
    previous_attention = None

    for step_index in range(max_horizon):
        if mode == "Normal":
            target_info = model(
                target_x_norm,
                return_features=True,
            )

        elif mode == "TrainingMean":
            target_info = model(
                target_x_norm,
                return_features=True,
                attention_override=(
                    training_mean_attention
                ),
            )

        elif mode == "CyclicShuffle":
            donor_info = model(
                donor_x_norm,
                return_features=True,
            )

            donor_diagnostics = donor_info[
                "field_attention"
            ]

            donor_attention = (
                donor_diagnostics[
                    "computed_attention"
                ]
            )

            target_info = model(
                target_x_norm,
                return_features=True,
                attention_override=donor_attention,
            )

            donor_current_norm = donor_x_norm[
                :,
                -4:,
                :,
                :,
            ]

            donor_next_norm = (
                donor_current_norm
                + donor_info["out"]
            )

            donor_x_norm = torch.cat(
                [
                    donor_x_norm[:, 4:, :, :],
                    donor_next_norm,
                ],
                dim=1,
            )

        else:
            raise ValueError(
                f"Unknown mode: {mode}"
            )

        target_diagnostics = target_info[
            "field_attention"
        ]

        computed_attention = (
            target_diagnostics[
                "computed_attention"
            ]
        )

        applied_attention = (
            target_diagnostics["attention"]
        )

        if first_attention is None:
            first_attention = (
                applied_attention.detach().clone()
            )

        if previous_attention is None:
            previous_attention = (
                applied_attention.detach().clone()
            )

        step_number = step_index + 1

        update_attention_accumulator(
            bucket=attention_accumulators[
                (mode, step_number)
            ],
            applied_attention=applied_attention,
            computed_attention=computed_attention,
            first_attention=first_attention,
            previous_attention=previous_attention,
        )

        current_state_norm = target_x_norm[
            :,
            -4:,
            :,
            :,
        ]

        pred_next_norm = (
            current_state_norm
            + target_info["out"]
        )

        pred_next_phys = (
            normalizer.denormalize_y(
                pred_next_norm
            )
        )

        predictions.append(pred_next_phys)

        target_x_norm = torch.cat(
            [
                target_x_norm[:, 4:, :, :],
                pred_next_norm,
            ],
            dim=1,
        )

        previous_attention = (
            applied_attention.detach().clone()
        )

    return torch.stack(
        predictions,
        dim=0,
    )


def finalize_aggregate_rows(
    accumulators: dict[
        tuple[str, int],
        dict[str, torch.Tensor],
    ],
    horizons: list[int],
) -> list[dict[str, Any]]:
    rows = []

    for mode in INTERVENTION_ORDER:
        for horizon in horizons:
            bucket = accumulators[
                (mode, horizon)
            ]

            num_samples = int(
                bucket["num_samples"].item()
            )

            for field_index, field_name in enumerate(
                FIELD_ORDER
            ):
                relative_l2 = (
                    torch.sqrt(
                        bucket[
                            "field_sse"
                        ][field_index]
                        / (
                            bucket[
                                "field_target_sq"
                            ][field_index]
                            + EPS
                        )
                    )
                    * 100.0
                )

                mse = (
                    bucket[
                        "field_sse"
                    ][field_index]
                    / bucket[
                        "field_numel"
                    ][field_index]
                )

                rows.append(
                    {
                        "mode": mode,
                        "horizon": horizon,
                        "field": field_name,
                        "num_samples": num_samples,
                        "rel_l2_percent": float(
                            relative_l2.item()
                        ),
                        "mse": float(mse.item()),
                    }
                )

            global_relative_l2 = (
                torch.sqrt(
                    bucket["global_sse"]
                    / (
                        bucket[
                            "global_target_sq"
                        ]
                        + EPS
                    )
                )
                * 100.0
            )

            global_mse = (
                bucket["global_sse"]
                / bucket["global_numel"]
            )

            rows.append(
                {
                    "mode": mode,
                    "horizon": horizon,
                    "field": "global",
                    "num_samples": num_samples,
                    "rel_l2_percent": float(
                        global_relative_l2.item()
                    ),
                    "mse": float(
                        global_mse.item()
                    ),
                }
            )

    return rows


def build_difference_table(
    aggregate_dataframe: pd.DataFrame,
    per_sample_dataframe: pd.DataFrame,
    horizons: list[int],
) -> pd.DataFrame:
    aggregate_map = {
        (
            row.mode,
            int(row.horizon),
            row.field,
        ): float(row.rel_l2_percent)
        for row in aggregate_dataframe.itertuples(
            index=False
        )
    }

    metric_columns = {
        "buoyancy": (
            "buoyancy_rel_l2_percent"
        ),
        "u_x": "u_x_rel_l2_percent",
        "u_y": "u_y_rel_l2_percent",
        "pressure": (
            "pressure_rel_l2_percent"
        ),
        "global": "global_rel_l2_percent",
    }

    keys = [
        "sample_id",
        "group",
        "trajectory",
        "t0",
        "horizon",
    ]

    rows = []

    for mode_a, mode_b, comparison in COMPARISONS:
        for horizon in horizons:
            a = per_sample_dataframe[
                (
                    per_sample_dataframe["mode"]
                    == mode_a
                )
                & (
                    per_sample_dataframe["horizon"]
                    == horizon
                )
            ]

            b = per_sample_dataframe[
                (
                    per_sample_dataframe["mode"]
                    == mode_b
                )
                & (
                    per_sample_dataframe["horizon"]
                    == horizon
                )
            ]

            for metric, column in (
                metric_columns.items()
            ):
                merged = a[
                    keys + [column]
                ].merge(
                    b[keys + [column]],
                    on=keys,
                    how="inner",
                    suffixes=("_a", "_b"),
                    validate="one_to_one",
                )

                difference = (
                    merged[f"{column}_a"]
                    - merged[f"{column}_b"]
                ).to_numpy()

                rows.append(
                    {
                        "comparison": comparison,
                        "mode_a": mode_a,
                        "mode_b": mode_b,
                        "horizon": horizon,
                        "metric": metric,
                        "num_samples": len(
                            difference
                        ),
                        (
                            "aggregate_rel_l2_"
                            "percent_diff"
                        ): (
                            aggregate_map[
                                (
                                    mode_a,
                                    horizon,
                                    metric,
                                )
                            ]
                            - aggregate_map[
                                (
                                    mode_b,
                                    horizon,
                                    metric,
                                )
                            ]
                        ),
                        "paired_diff_mean": float(
                            difference.mean()
                        ),
                        "paired_diff_median": float(
                            np.median(difference)
                        ),
                        "win_rate_percent": float(
                            (
                                difference < 0.0
                            ).mean()
                            * 100.0
                        ),
                        "loss_rate_percent": float(
                            (
                                difference > 0.0
                            ).mean()
                            * 100.0
                        ),
                    }
                )

    return pd.DataFrame(rows)


def finalize_attention_diagnostics(
    accumulators: dict[
        tuple[str, int],
        dict[str, Any],
    ],
    horizons: list[int],
    training_mean_attention: torch.Tensor,
    training_mean_samples: int,
    gate_values: torch.Tensor,
) -> pd.DataFrame:
    rows = []

    gate_cpu = (
        gate_values.detach().cpu().numpy()
    )

    for mode in INTERVENTION_ORDER:
        for horizon in horizons:
            bucket = accumulators[
                (mode, horizon)
            ]

            num_samples = int(
                bucket["num_samples"]
            )

            mean_attention = (
                bucket["attention_sum"]
                / float(num_samples)
            )

            second_moment = (
                bucket["attention_sq_sum"]
                / float(num_samples)
            )

            variance = (
                second_moment
                - mean_attention.square()
            ).clamp_min(0.0)

            std_attention = torch.sqrt(
                variance
            )

            offdiag_mask = ~torch.eye(
                4,
                dtype=torch.bool,
                device=mean_attention.device,
            )

            sample_std_mean = float(
                std_attention[
                    offdiag_mask
                ].mean().item()
            )

            entropy_mean = (
                bucket["entropy_sum"]
                / max(
                    bucket["entropy_count"],
                    1,
                )
            )

            amax_mean = (
                bucket["amax_sum"]
                / max(
                    bucket["amax_count"],
                    1,
                )
            )

            drift_first = (
                bucket["drift_first_sum"]
                / max(num_samples, 1)
            )

            drift_previous = (
                bucket["drift_previous_sum"]
                / max(num_samples, 1)
            )

            mismatch = (
                bucket["mismatch_sum"]
                / max(num_samples, 1)
            )

            mean_cpu = (
                mean_attention.detach()
                .cpu()
                .numpy()
            )

            std_cpu = (
                std_attention.detach()
                .cpu()
                .numpy()
            )

            for target_index, target_name in enumerate(
                FIELD_ORDER
            ):
                for source_index, source_name in enumerate(
                    FIELD_ORDER
                ):
                    rows.append(
                        {
                            "mode": mode,
                            "horizon": horizon,
                            "num_samples": num_samples,
                            "target_field": (
                                target_name
                            ),
                            "source_field": (
                                source_name
                            ),
                            "attention_mean": float(
                                mean_cpu[
                                    target_index,
                                    source_index,
                                ]
                            ),
                            "attention_std": float(
                                std_cpu[
                                    target_index,
                                    source_index,
                                ]
                            ),
                            (
                                "attention_sample_"
                                "std_offdiag_mean"
                            ): sample_std_mean,
                            (
                                "normalized_attention_"
                                "entropy"
                            ): entropy_mean,
                            "attention_max_mean": (
                                amax_mean
                            ),
                            (
                                "drift_from_step1_l1"
                            ): drift_first,
                            (
                                "drift_from_previous_l1"
                            ): drift_previous,
                            (
                                "override_mismatch_l1"
                            ): mismatch,
                            "gate_buoyancy": float(
                                gate_cpu[0]
                            ),
                            "gate_u_x": float(
                                gate_cpu[1]
                            ),
                            "gate_u_y": float(
                                gate_cpu[2]
                            ),
                            "gate_pressure": float(
                                gate_cpu[3]
                            ),
                        }
                    )

    mean_cpu = (
        training_mean_attention.detach()
        .cpu()
        .numpy()
    )

    for target_index, target_name in enumerate(
        FIELD_ORDER
    ):
        for source_index, source_name in enumerate(
            FIELD_ORDER
        ):
            rows.append(
                {
                    "mode": (
                        "TrainingMeanReference"
                    ),
                    "horizon": 0,
                    "num_samples": (
                        training_mean_samples
                    ),
                    "target_field": target_name,
                    "source_field": source_name,
                    "attention_mean": float(
                        mean_cpu[
                            target_index,
                            source_index,
                        ]
                    ),
                    "attention_std": 0.0,
                    (
                        "attention_sample_"
                        "std_offdiag_mean"
                    ): 0.0,
                    (
                        "normalized_attention_"
                        "entropy"
                    ): np.nan,
                    "attention_max_mean": np.nan,
                    "drift_from_step1_l1": 0.0,
                    "drift_from_previous_l1": 0.0,
                    "override_mismatch_l1": 0.0,
                    "gate_buoyancy": float(
                        gate_cpu[0]
                    ),
                    "gate_u_x": float(
                        gate_cpu[1]
                    ),
                    "gate_u_y": float(
                        gate_cpu[2]
                    ),
                    "gate_pressure": float(
                        gate_cpu[3]
                    ),
                }
            )

    return pd.DataFrame(rows)


def print_compact_results(
    aggregate_dataframe: pd.DataFrame,
    difference_dataframe: pd.DataFrame,
    diagnostics_dataframe: pd.DataFrame,
) -> None:
    global_rows = aggregate_dataframe[
        aggregate_dataframe["field"]
        == "global"
    ]

    global_table = global_rows.pivot_table(
        index="mode",
        columns="horizon",
        values="rel_l2_percent",
    ).reindex(INTERVENTION_ORDER)

    global_table.columns = [
        f"h={int(value)}"
        for value in global_table.columns
    ]

    print()
    print("=" * 100)
    print(
        "M9-0b ATTENTION INTERVENTION "
        "GLOBAL REL-L2 (%)"
    )
    print("=" * 100)

    print(
        global_table.to_string(
            formatters={
                column: (
                    lambda value: f"{value:.6f}"
                )
                for column in global_table.columns
            }
        )
    )

    global_difference = difference_dataframe[
        difference_dataframe["metric"]
        == "global"
    ][
        [
            "comparison",
            "horizon",
            (
                "aggregate_rel_l2_"
                "percent_diff"
            ),
            "paired_diff_mean",
            "paired_diff_median",
            "win_rate_percent",
        ]
    ]

    print()
    print("=" * 110)
    print(
        "NORMAL VS ATTENTION CONTROLS"
    )
    print(
        "负数与胜率 > 50% 表示 Normal 更好"
    )
    print("=" * 110)

    print(
        global_difference.to_string(
            index=False,
            formatters={
                (
                    "aggregate_rel_l2_"
                    "percent_diff"
                ): (
                    lambda value: f"{value:+.6f}"
                ),
                "paired_diff_mean": (
                    lambda value: f"{value:+.6f}"
                ),
                "paired_diff_median": (
                    lambda value: f"{value:+.6f}"
                ),
                "win_rate_percent": (
                    lambda value: f"{value:.2f}%"
                ),
            },
        )
    )

    scalar_columns = [
        "mode",
        "horizon",
        "num_samples",
        (
            "attention_sample_"
            "std_offdiag_mean"
        ),
        (
            "normalized_attention_entropy"
        ),
        "attention_max_mean",
        "drift_from_step1_l1",
        "drift_from_previous_l1",
        "override_mismatch_l1",
    ]

    scalar_summary = (
        diagnostics_dataframe[
            diagnostics_dataframe["horizon"] > 0
        ][scalar_columns]
        .drop_duplicates(
            subset=["mode", "horizon"]
        )
        .sort_values(
            ["mode", "horizon"]
        )
    )

    print()
    print("=" * 120)
    print("ATTENTION DIAGNOSTICS")
    print("=" * 120)

    print(
        scalar_summary.to_string(
            index=False,
            formatters={
                column: (
                    lambda value: f"{value:.6e}"
                )
                for column in scalar_columns
                if column
                not in {
                    "mode",
                    "horizon",
                    "num_samples",
                }
            },
        )
    )


def main() -> None:
    args = parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    split_path = resolve_path(args.split)
    stats_path = resolve_path(args.stats)
    config_path = resolve_path(args.config)
    checkpoint_path = resolve_path(
        args.checkpoint
    )

    output_path = resolve_path(args.output)
    per_sample_output_path = resolve_path(
        args.per_sample_output
    )
    difference_output_path = resolve_path(
        args.difference_output
    )
    diagnostics_output_path = resolve_path(
        args.diagnostics_output
    )

    for path in [
        split_path,
        stats_path,
        config_path,
        checkpoint_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    horizons = sorted(
        {
            int(value)
            for value in args.horizons.split(",")
        }
    )

    max_horizon = max(horizons)

    with split_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        split_config = json.load(file)

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    print(
        "========== M9-0b ATTENTION "
        "INTERVENTION =========="
    )
    print("Device:", device)
    print("Split:", split_path)
    print("Stats:", stats_path)
    print("Checkpoint:", checkpoint_path)
    print("Horizons:", horizons)
    print("Max test samples:", args.max_samples)
    print(
        "Max mean samples:",
        args.max_mean_samples,
    )

    normalizer = FieldWiseNormalizer(
        stats_path
    ).to(device)

    model = build_model(
        config=config,
        device=device,
    )

    model = load_model_state(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    training_mean_attention, mean_samples = (
        compute_training_mean_attention(
            model=model,
            split_items=split_config["train"],
            normalizer=normalizer,
            device=device,
            batch_size=args.mean_batch_size,
            num_workers=args.num_workers,
            max_samples=args.max_mean_samples,
        )
    )

    test_base_dataset = (
        CrossParamRolloutDataset(
            split_items=split_config["test"],
            max_horizon=max_horizon,
            max_samples=args.max_samples,
        )
    )

    test_dataset = PairedInterventionDataset(
        base_dataset=test_base_dataset,
        donor_offset=args.donor_offset,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    metric_accumulators = {
        (mode, horizon): new_accumulator(
            device
        )
        for mode in INTERVENTION_ORDER
        for horizon in horizons
    }

    attention_accumulators = {
        (mode, step): (
            new_attention_accumulator(device)
        )
        for mode in INTERVENTION_ORDER
        for step in range(
            1,
            max_horizon + 1,
        )
    }

    per_sample_rows: list[
        dict[str, Any]
    ] = []

    total_batches = len(test_loader)

    print()
    print(
        "🔥 Starting Attention intervention "
        "rollout..."
    )

    with torch.no_grad():
        for batch_index, batch in enumerate(
            test_loader,
            start=1,
        ):
            target = batch["target"]

            target_x0_phys = target[
                "x0_phys"
            ].to(
                device,
                non_blocking=True,
            )

            future_phys = target[
                "future_phys"
            ].to(
                device,
                non_blocking=True,
            )

            donor_x0_phys = batch[
                "donor_x0_phys"
            ].to(
                device,
                non_blocking=True,
            )

            sample_ids = target[
                "sample_id"
            ].tolist()

            groups = list(target["group"])

            trajectories = target[
                "trajectory"
            ].tolist()

            t0_values = target["t0"].tolist()

            for mode in INTERVENTION_ORDER:
                prediction_sequence = (
                    rollout_intervention(
                        model=model,
                        mode=mode,
                        target_x0_phys=(
                            target_x0_phys
                        ),
                        donor_x0_phys=(
                            donor_x0_phys
                        ),
                        normalizer=normalizer,
                        training_mean_attention=(
                            training_mean_attention
                        ),
                        max_horizon=max_horizon,
                        attention_accumulators=(
                            attention_accumulators
                        ),
                    )
                )

                for horizon in horizons:
                    horizon_index = horizon - 1

                    pred_state = (
                        prediction_sequence[
                            horizon_index
                        ]
                    )

                    true_state = future_phys[
                        :,
                        horizon_index,
                        :,
                        :,
                        :,
                    ]

                    update_accumulator(
                        accumulator=(
                            metric_accumulators[
                                (mode, horizon)
                            ]
                        ),
                        pred_phys=pred_state,
                        true_phys=true_state,
                    )

                    sample_metrics = (
                        calculate_per_sample_metrics(
                            pred_phys=pred_state,
                            true_phys=true_state,
                        )
                    )

                    field_values = (
                        sample_metrics[
                            "field_rel_l2"
                        ]
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    global_values = (
                        sample_metrics[
                            "global_rel_l2"
                        ]
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    for sample_offset in range(
                        len(sample_ids)
                    ):
                        per_sample_rows.append(
                            {
                                "sample_id": int(
                                    sample_ids[
                                        sample_offset
                                    ]
                                ),
                                "group": groups[
                                    sample_offset
                                ],
                                "trajectory": int(
                                    trajectories[
                                        sample_offset
                                    ]
                                ),
                                "t0": int(
                                    t0_values[
                                        sample_offset
                                    ]
                                ),
                                "mode": mode,
                                "horizon": horizon,
                                (
                                    "buoyancy_"
                                    "rel_l2_percent"
                                ): float(
                                    field_values[
                                        sample_offset,
                                        0,
                                    ]
                                ),
                                (
                                    "u_x_rel_l2_percent"
                                ): float(
                                    field_values[
                                        sample_offset,
                                        1,
                                    ]
                                ),
                                (
                                    "u_y_rel_l2_percent"
                                ): float(
                                    field_values[
                                        sample_offset,
                                        2,
                                    ]
                                ),
                                (
                                    "pressure_"
                                    "rel_l2_percent"
                                ): float(
                                    field_values[
                                        sample_offset,
                                        3,
                                    ]
                                ),
                                (
                                    "global_"
                                    "rel_l2_percent"
                                ): float(
                                    global_values[
                                        sample_offset
                                    ]
                                ),
                            }
                        )

                del prediction_sequence

            if (
                batch_index % 10 == 0
                or batch_index == total_batches
            ):
                print(
                    f"   processed batch "
                    f"{batch_index}/{total_batches}"
                )

    aggregate_dataframe = pd.DataFrame(
        finalize_aggregate_rows(
            accumulators=metric_accumulators,
            horizons=horizons,
        )
    )

    per_sample_dataframe = pd.DataFrame(
        per_sample_rows
    )

    difference_dataframe = (
        build_difference_table(
            aggregate_dataframe=(
                aggregate_dataframe
            ),
            per_sample_dataframe=(
                per_sample_dataframe
            ),
            horizons=horizons,
        )
    )

    diagnostics_dataframe = (
        finalize_attention_diagnostics(
            accumulators=(
                attention_accumulators
            ),
            horizons=horizons,
            training_mean_attention=(
                training_mean_attention
            ),
            training_mean_samples=mean_samples,
            gate_values=(
                model.field_attention
                .gate_values()
            ),
        )
    )

    for path in [
        output_path,
        per_sample_output_path,
        difference_output_path,
        diagnostics_output_path,
    ]:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    aggregate_dataframe.to_csv(
        output_path,
        index=False,
    )

    per_sample_dataframe.to_csv(
        per_sample_output_path,
        index=False,
    )

    difference_dataframe.to_csv(
        difference_output_path,
        index=False,
    )

    diagnostics_dataframe.to_csv(
        diagnostics_output_path,
        index=False,
    )

    print_compact_results(
        aggregate_dataframe=aggregate_dataframe,
        difference_dataframe=(
            difference_dataframe
        ),
        diagnostics_dataframe=(
            diagnostics_dataframe
        ),
    )

    print()
    print(
        "✅ Attention intervention complete."
    )
    print("Aggregate:", output_path)
    print(
        "Per-sample:",
        per_sample_output_path,
    )
    print(
        "Differences:",
        difference_output_path,
    )
    print(
        "Diagnostics:",
        diagnostics_output_path,
    )

    if (
        args.max_samples is not None
        or args.max_mean_samples is not None
    ):
        print(
            "⚠️ Lightweight intervention trial only."
        )
        print(
            "⚠️ This is not a formal result."
        )


if __name__ == "__main__":
    main()
