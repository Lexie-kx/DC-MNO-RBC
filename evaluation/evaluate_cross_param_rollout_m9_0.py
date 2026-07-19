"""
M9-0 cross-parameter rollout evaluation.

Current-stage model screening:
- M6-FieldWiseEncoder-H4-continue
- M7-FieldCoupling-H4
- M7-ParamTokenOnly-H4
- M8-A-FullStatic-H4
- M8-B-ParamConditionedCoupling-H4
- M9-0a-StaticSoftmaxAttention-H4
- M9-0b-StateDependentAttention-H4

Outputs:
1. Aggregate field/global rollout metrics.
2. Per-sample paired rollout metrics.
3. Selected aggregate difference table.

Protocol:
- physical-space test trajectories
- split-specific train-only normalization
- normalized-delta autoregressive rollout
- horizons 1/4/8/16
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from constants import (
    CONTEXT_LENGTH,
    DATA_PATH,
    DTYPE,
    FIELD_ORDER,
)
from models.operators.fno2d_fieldwise import FieldWiseFNO2d
from models.operators.fno2d_fieldwise_m7 import (
    M7FieldCouplingFNO2d,
)
from models.operators.fno2d_fieldwise_paramtoken import (
    M7FieldWiseParamTokenFNO2d,
)
from models.operators.fno2d_m8_full_conditioned import (
    M8FullConditionedFNO2d,
)
from models.operators.fno2d_m9_static_attention import (
    M9StaticAttentionFNO2d,
)
from models.operators.fno2d_m9_state_attention import (
    M9StateAttentionFNO2d,
)
from training.normalization import FieldWiseNormalizer


MODEL_ORDER = [
    "M6-FieldWiseEncoder-H4-continue",
    "M7-FieldCoupling-H4",
    "M7-ParamTokenOnly-H4",
    "M8-A-FullStatic-H4",
    "M8-B-ParamConditionedCoupling-H4",
    "M9-0a-StaticSoftmaxAttention-H4",
    "M9-0b-StateDependentAttention-H4",
]

METRIC_ORDER = [
    "buoyancy",
    "u_x",
    "u_y",
    "pressure",
    "global",
]

COMPARISON_PAIRS = [
    (
        "M9-0b-StateDependentAttention-H4",
        "M9-0a-StaticSoftmaxAttention-H4",
        "M9-0b - M9-0a",
    ),
    (
        "M9-0b-StateDependentAttention-H4",
        "M8-A-FullStatic-H4",
        "M9-0b - M8-A",
    ),
    (
        "M9-0b-StateDependentAttention-H4",
        "M8-B-ParamConditionedCoupling-H4",
        "M9-0b - M8-B",
    ),
    (
        "M9-0b-StateDependentAttention-H4",
        "M7-FieldCoupling-H4",
        "M9-0b - M7-FieldCoupling",
    ),
    (
        "M9-0b-StateDependentAttention-H4",
        "M7-ParamTokenOnly-H4",
        "M9-0b - M7-ParamToken",
    ),
    (
        "M9-0a-StaticSoftmaxAttention-H4",
        "M7-FieldCoupling-H4",
        "M9-0a - M7-FieldCoupling",
    ),
    (
        "M9-0a-StaticSoftmaxAttention-H4",
        "M8-A-FullStatic-H4",
        "M9-0a - M8-A",
    ),
]

EPS = 1e-12


class CrossParamRolloutDataset(Dataset):
    """
    Each sample returns:

        x0_phys:
            [16, H, W]

        future_phys:
            [max_horizon, 4, H, W]

        param:
            [2] = [log10(Ra), log10(Pr)]

        metadata:
            sample_id, group, trajectory, t0
    """

    def __init__(
        self,
        split_items: list[dict[str, Any]],
        max_horizon: int = 16,
        max_samples: int | None = None,
    ) -> None:
        super().__init__()

        self.max_horizon = max_horizon
        self.data_store: dict[str, torch.Tensor] = {}
        self.index: list[dict[str, Any]] = []

        print(
            f"📦 正在加载 rollout 数据，"
            f"共 {len(split_items)} 个 group 批次"
        )

        with h5py.File(DATA_PATH, "r") as file:
            for split_item in split_items:
                group_name = split_item["group"]
                trajectory_ids = [
                    int(value)
                    for value in split_item["trajectories"]
                ]

                if group_name not in file:
                    raise KeyError(
                        f"Group 不存在: {group_name}"
                    )

                group = file[group_name]

                field_arrays = [
                    group[field][trajectory_ids]
                    for field in FIELD_ORDER
                ]

                stacked = np.stack(
                    field_arrays,
                    axis=2,
                )
                # [trajectory, time, field, H, W]

                data = torch.tensor(
                    stacked,
                    dtype=DTYPE,
                )

                self.data_store[group_name] = data

                param = self._make_param(group_name)

                (
                    num_trajectories,
                    num_steps,
                    num_fields,
                    _height,
                    _width,
                ) = data.shape

                if num_fields != len(FIELD_ORDER):
                    raise RuntimeError(
                        f"{group_name}: fields={num_fields}, "
                        f"expected={len(FIELD_ORDER)}"
                    )

                max_start = (
                    num_steps
                    - CONTEXT_LENGTH
                    - max_horizon
                    + 1
                )

                if max_start <= 0:
                    raise RuntimeError(
                        f"{group_name} 时间长度不足。"
                    )

                for local_trajectory in range(
                    num_trajectories
                ):
                    original_trajectory = (
                        trajectory_ids[local_trajectory]
                    )

                    for t0 in range(max_start):
                        self.index.append(
                            {
                                "group": group_name,
                                "local_trajectory": (
                                    local_trajectory
                                ),
                                "trajectory": (
                                    original_trajectory
                                ),
                                "t0": t0,
                                "param": param,
                            }
                        )

        if max_samples is not None:
            self.index = self.index[:max_samples]

        print(
            f"✅ Rollout 样本数: {len(self.index)}"
        )

    @staticmethod
    def _make_param(
        group_name: str,
    ) -> torch.Tensor:
        match = re.search(
            r"ra_([0-9.eE+-]+)_pr_([0-9.eE+-]+)",
            group_name,
            flags=re.IGNORECASE,
        )

        if match is None:
            raise ValueError(
                f"无法从 group 名解析 Ra/Pr: {group_name}"
            )

        ra = float(match.group(1))
        pr = float(match.group(2))

        return torch.tensor(
            [
                math.log10(ra),
                math.log10(pr),
            ],
            dtype=DTYPE,
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(
        self,
        sample_id: int,
    ) -> dict[str, Any]:
        item = self.index[sample_id]
        data = self.data_store[item["group"]]

        trajectory = item["local_trajectory"]
        t0 = item["t0"]

        history = data[
            trajectory,
            t0 : t0 + CONTEXT_LENGTH,
        ]

        future = data[
            trajectory,
            (
                t0 + CONTEXT_LENGTH
            ) : (
                t0
                + CONTEXT_LENGTH
                + self.max_horizon
            ),
        ]

        _, num_fields, height, width = history.shape

        x0_phys = history.reshape(
            CONTEXT_LENGTH * num_fields,
            height,
            width,
        )

        return {
            "sample_id": sample_id,
            "group": item["group"],
            "trajectory": item["trajectory"],
            "t0": t0,
            "x0_phys": x0_phys,
            "future_phys": future,
            "param": item["param"],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Seven-model M9-0 rollout evaluation."
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
        default="configs/m9_0_attention_frozen.json",
    )

    parser.add_argument(
        "--m6_checkpoint",
        required=True,
    )
    parser.add_argument(
        "--m7_fieldcoupling_checkpoint",
        required=True,
    )
    parser.add_argument(
        "--m7_paramtoken_checkpoint",
        required=True,
    )
    parser.add_argument(
        "--m8_full_static_checkpoint",
        required=True,
    )
    parser.add_argument(
        "--m8_param_conditioned_checkpoint",
        required=True,
    )
    parser.add_argument(
        "--m9_static_checkpoint",
        required=True,
    )
    parser.add_argument(
        "--m9_state_checkpoint",
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
        "--batch_size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--horizons",
        default="1,4,8,16",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Only for lightweight trial.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )

    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def load_model_state(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    print(
        f"✅ Loaded strict=True: "
        f"{checkpoint_path.name}"
    )

    return model


def rollout_without_param(
    model: torch.nn.Module,
    x0_phys: torch.Tensor,
    normalizer: FieldWiseNormalizer,
    max_horizon: int,
) -> torch.Tensor:
    x_norm = normalizer.normalize_x(x0_phys)
    predictions = []

    for _step in range(max_horizon):
        current_state_norm = x_norm[:, -4:, :, :]

        pred_delta_norm = model(x_norm)

        pred_next_norm = (
            current_state_norm
            + pred_delta_norm
        )

        pred_next_phys = (
            normalizer.denormalize_y(
                pred_next_norm
            )
        )

        predictions.append(pred_next_phys)

        x_norm = torch.cat(
            [
                x_norm[:, 4:, :, :],
                pred_next_norm,
            ],
            dim=1,
        )

    return torch.stack(
        predictions,
        dim=0,
    )


def rollout_with_param(
    model: torch.nn.Module,
    x0_phys: torch.Tensor,
    param: torch.Tensor,
    normalizer: FieldWiseNormalizer,
    max_horizon: int,
    coupling_mode: str | None = None,
) -> torch.Tensor:
    x_norm = normalizer.normalize_x(x0_phys)
    predictions = []

    for _step in range(max_horizon):
        current_state_norm = x_norm[:, -4:, :, :]

        if coupling_mode is None:
            pred_delta_norm = model(
                x_norm,
                param,
            )
        else:
            pred_delta_norm = model(
                x_norm,
                param,
                coupling_mode=coupling_mode,
            )

        pred_next_norm = (
            current_state_norm
            + pred_delta_norm
        )

        pred_next_phys = (
            normalizer.denormalize_y(
                pred_next_norm
            )
        )

        predictions.append(pred_next_phys)

        x_norm = torch.cat(
            [
                x_norm[:, 4:, :, :],
                pred_next_norm,
            ],
            dim=1,
        )

    return torch.stack(
        predictions,
        dim=0,
    )


def new_accumulator(
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "field_sse": torch.zeros(
            4,
            dtype=torch.float64,
            device=device,
        ),
        "field_target_sq": torch.zeros(
            4,
            dtype=torch.float64,
            device=device,
        ),
        "field_numel": torch.zeros(
            4,
            dtype=torch.float64,
            device=device,
        ),
        "global_sse": torch.tensor(
            0.0,
            dtype=torch.float64,
            device=device,
        ),
        "global_target_sq": torch.tensor(
            0.0,
            dtype=torch.float64,
            device=device,
        ),
        "global_numel": torch.tensor(
            0.0,
            dtype=torch.float64,
            device=device,
        ),
        "num_samples": torch.tensor(
            0,
            dtype=torch.int64,
            device=device,
        ),
    }


def update_accumulator(
    accumulator: dict[str, torch.Tensor],
    pred_phys: torch.Tensor,
    true_phys: torch.Tensor,
) -> None:
    difference = (
        pred_phys - true_phys
    ).double()

    target = true_phys.double()

    for field_index in range(4):
        field_difference = difference[
            :,
            field_index,
            :,
            :,
        ]

        field_target = target[
            :,
            field_index,
            :,
            :,
        ]

        accumulator[
            "field_sse"
        ][field_index] += (
            field_difference.square().sum()
        )

        accumulator[
            "field_target_sq"
        ][field_index] += (
            field_target.square().sum()
        )

        accumulator[
            "field_numel"
        ][field_index] += (
            field_difference.numel()
        )

    accumulator["global_sse"] += (
        difference.square().sum()
    )

    accumulator["global_target_sq"] += (
        target.square().sum()
    )

    accumulator["global_numel"] += (
        difference.numel()
    )

    accumulator["num_samples"] += (
        pred_phys.shape[0]
    )


def calculate_per_sample_metrics(
    pred_phys: torch.Tensor,
    true_phys: torch.Tensor,
) -> dict[str, torch.Tensor]:
    difference = (
        pred_phys - true_phys
    ).double()

    target = true_phys.double()

    field_sse = (
        difference.square()
        .sum(dim=(-2, -1))
    )

    field_target_sq = (
        target.square()
        .sum(dim=(-2, -1))
    )

    field_rel_l2 = (
        torch.sqrt(
            field_sse
            / (field_target_sq + EPS)
        )
        * 100.0
    )

    global_sse = (
        difference.square()
        .sum(dim=(1, 2, 3))
    )

    global_target_sq = (
        target.square()
        .sum(dim=(1, 2, 3))
    )

    global_rel_l2 = (
        torch.sqrt(
            global_sse
            / (global_target_sq + EPS)
        )
        * 100.0
    )

    return {
        "field_rel_l2": field_rel_l2,
        "global_rel_l2": global_rel_l2,
    }


def finalize_aggregate_rows(
    accumulators: dict[
        tuple[str, int],
        dict[str, torch.Tensor],
    ],
    horizons: list[int],
) -> list[dict[str, Any]]:
    rows = []

    for model_name in MODEL_ORDER:
        for horizon in horizons:
            bucket = accumulators[
                (model_name, horizon)
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
                        "model": model_name,
                        "horizon": horizon,
                        "field": field_name,
                        "num_samples": num_samples,
                        "rel_l2_percent": (
                            float(relative_l2.item())
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
                    "model": model_name,
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
    horizons: list[int],
) -> pd.DataFrame:
    wide = aggregate_dataframe.pivot_table(
        index=["model", "horizon"],
        columns="field",
        values="rel_l2_percent",
    )

    rows = []

    for model_a, model_b, comparison_name in (
        COMPARISON_PAIRS
    ):
        for horizon in horizons:
            row_a = wide.loc[
                (model_a, horizon)
            ]
            row_b = wide.loc[
                (model_b, horizon)
            ]

            row = {
                "comparison": comparison_name,
                "model_a": model_a,
                "model_b": model_b,
                "horizon": horizon,
            }

            for metric in METRIC_ORDER:
                row[
                    f"{metric}_rel_l2_percent_diff"
                ] = float(
                    row_a[metric]
                    - row_b[metric]
                )

            rows.append(row)

    return pd.DataFrame(rows)


def print_compact_summary(
    aggregate_dataframe: pd.DataFrame,
    difference_dataframe: pd.DataFrame,
) -> None:
    wide = aggregate_dataframe.pivot_table(
        index=["model", "horizon"],
        columns="field",
        values="rel_l2_percent",
    ).reset_index()

    wide["model"] = pd.Categorical(
        wide["model"],
        categories=MODEL_ORDER,
        ordered=True,
    )

    wide = wide.sort_values(
        ["model", "horizon"]
    )

    full_columns = [
        "model",
        "horizon",
        "buoyancy",
        "u_x",
        "u_y",
        "pressure",
        "global",
    ]

    print()
    print("=" * 118)
    print(
        "M9-0 UNSEEN PR FULL FIELD-WISE "
        "REL-L2 (%)"
    )
    print("=" * 118)

    print(
        wide[full_columns].to_string(
            index=False,
            formatters={
                metric: (
                    lambda value: f"{value:.6f}"
                )
                for metric in METRIC_ORDER
            },
        )
    )

    global_table = wide.pivot_table(
        index="model",
        columns="horizon",
        values="global",
        observed=False,
    ).reindex(MODEL_ORDER)

    global_table.columns = [
        f"h={int(value)}"
        for value in global_table.columns
    ]

    print()
    print("=" * 100)
    print("M9-0 UNSEEN PR GLOBAL REL-L2 (%)")
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

    comparison_order = [
        comparison_name
        for (
            _model_a,
            _model_b,
            comparison_name,
        ) in COMPARISON_PAIRS
    ]

    full_difference = (
        difference_dataframe.copy()
    )

    full_difference["comparison"] = (
        pd.Categorical(
            full_difference["comparison"],
            categories=comparison_order,
            ordered=True,
        )
    )

    full_difference = (
        full_difference.sort_values(
            ["comparison", "horizon"]
        )
    )

    difference_columns = [
        "comparison",
        "horizon",
        "buoyancy_rel_l2_percent_diff",
        "u_x_rel_l2_percent_diff",
        "u_y_rel_l2_percent_diff",
        "pressure_rel_l2_percent_diff",
        "global_rel_l2_percent_diff",
    ]

    display_difference = full_difference[
        difference_columns
    ].copy()

    display_difference.columns = [
        "comparison",
        "horizon",
        "buoyancy",
        "u_x",
        "u_y",
        "pressure",
        "global",
    ]

    print()
    print("=" * 118)
    print("SELECTED FULL FIELD-WISE DIFFERENCES")
    print("前者 - 后者；负数表示前者误差更低")
    print("=" * 118)

    print(
        display_difference.to_string(
            index=False,
            formatters={
                metric: (
                    lambda value: f"{value:+.6f}"
                )
                for metric in METRIC_ORDER
            },
        )
    )

    print()
    print(
        "说明：完整 Rel-L2、MSE、逐样本结果"
        "均保存在对应 CSV；"
        "终端打印完整逐场 Rel-L2 和关键差值。"
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

    checkpoint_paths = {
        "m6": resolve_path(
            args.m6_checkpoint
        ),
        "m7_field": resolve_path(
            args.m7_fieldcoupling_checkpoint
        ),
        "m7_param": resolve_path(
            args.m7_paramtoken_checkpoint
        ),
        "m8_static": resolve_path(
            args.m8_full_static_checkpoint
        ),
        "m8_dynamic": resolve_path(
            args.m8_param_conditioned_checkpoint
        ),
        "m9_static": resolve_path(
            args.m9_static_checkpoint
        ),
        "m9_state": resolve_path(
            args.m9_state_checkpoint
        ),
    }

    output_path = resolve_path(args.output)
    per_sample_output_path = resolve_path(
        args.per_sample_output
    )
    difference_output_path = resolve_path(
        args.difference_output
    )

    paths_to_check = [
        split_path,
        stats_path,
        config_path,
        *checkpoint_paths.values(),
    ]

    for path in paths_to_check:
        if not path.exists():
            raise FileNotFoundError(path)

    horizons = sorted(
        {
            int(value)
            for value in args.horizons.split(",")
        }
    )

    if not horizons:
        raise ValueError(
            "No rollout horizon was provided."
        )

    max_horizon = max(horizons)

    print(
        "========== M9-0 SEVEN-MODEL "
        "ROLLOUT EVALUATION =========="
    )
    print("Device:", device)
    print("Split:", split_path)
    print("Stats:", stats_path)
    print("Config:", config_path)
    print("Horizons:", horizons)
    print("Batch size:", args.batch_size)
    print("Max samples:", args.max_samples)
    print("Aggregate output:", output_path)
    print(
        "Per-sample output:",
        per_sample_output_path,
    )
    print(
        "Difference output:",
        difference_output_path,
    )

    with split_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        split_config = json.load(file)

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        attention_config = json.load(file)

    dataset = CrossParamRolloutDataset(
        split_items=split_config["test"],
        max_horizon=max_horizon,
        max_samples=args.max_samples,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    normalizer = FieldWiseNormalizer(
        stats_path
    ).to(device)

    m6 = load_model_state(
        FieldWiseFNO2d(
            in_channels=16,
            out_channels=4,
            modes1=16,
            modes2=16,
            width=32,
            context_length=4,
            num_fields=4,
        ).to(device),
        checkpoint_paths["m6"],
        device,
    )

    m7_field = load_model_state(
        M7FieldCouplingFNO2d(
            in_channels=16,
            out_channels=4,
            modes1=16,
            modes2=16,
            width=32,
            context_length=4,
            num_fields=4,
            coupling_hidden_channels=8,
            coupling_dropout=0.0,
            coupling_init_gate=-4.0,
            coupling_use_norm=True,
        ).to(device),
        checkpoint_paths["m7_field"],
        device,
    )

    m7_param = load_model_state(
        M7FieldWiseParamTokenFNO2d(
            in_channels=16,
            out_channels=4,
            modes1=16,
            modes2=16,
            width=32,
            context_length=4,
            num_fields=4,
            token_hidden_dim=64,
        ).to(device),
        checkpoint_paths["m7_param"],
        device,
    )

    m8_static = load_model_state(
        M8FullConditionedFNO2d(
            in_channels=16,
            out_channels=4,
            modes1=16,
            modes2=16,
            width=32,
            context_length=4,
            num_fields=4,
            coupling_mode="static",
            coupling_hidden_channels=8,
            coupling_dropout=0.0,
            coupling_init_gate=-4.0,
            coupling_use_norm=True,
            coupling_param_hidden_dim=64,
            coupling_condition_scale=0.10,
            token_hidden_dim=64,
        ).to(device),
        checkpoint_paths["m8_static"],
        device,
    )

    m8_dynamic = load_model_state(
        M8FullConditionedFNO2d(
            in_channels=16,
            out_channels=4,
            modes1=16,
            modes2=16,
            width=32,
            context_length=4,
            num_fields=4,
            coupling_mode=(
                "parameter_conditioned"
            ),
            coupling_hidden_channels=8,
            coupling_dropout=0.0,
            coupling_init_gate=-4.0,
            coupling_use_norm=True,
            coupling_param_hidden_dim=64,
            coupling_condition_scale=0.10,
            token_hidden_dim=64,
        ).to(device),
        checkpoint_paths["m8_dynamic"],
        device,
    )

    m9_static = load_model_state(
        M9StaticAttentionFNO2d(
            in_channels=16,
            out_channels=4,
            modes1=16,
            modes2=16,
            width=32,
            context_length=4,
            num_fields=4,
            field_width=int(
                attention_config["field_width"]
            ),
            attention_hidden_channels=int(
                attention_config[
                    "hidden_channels"
                ]
            ),
            attention_dropout=float(
                attention_config[
                    "coupling_dropout"
                ]
            ),
            attention_init_gate=float(
                attention_config[
                    "residual_gate_init_logit"
                ]
            ),
            attention_use_norm=bool(
                attention_config["use_norm"]
            ),
            attention_output_scale=float(
                attention_config[
                    "attention_output_scale"
                ]
            ),
            score_temperature=float(
                attention_config[
                    "score_temperature"
                ]
            ),
        ).to(device),
        checkpoint_paths["m9_static"],
        device,
    )

    m9_state = load_model_state(
        M9StateAttentionFNO2d(
            in_channels=16,
            out_channels=4,
            modes1=16,
            modes2=16,
            width=32,
            context_length=4,
            num_fields=4,
            field_width=int(
                attention_config["field_width"]
            ),
            attention_hidden_channels=int(
                attention_config[
                    "hidden_channels"
                ]
            ),
            qk_channels=int(
                attention_config["qk_channels"]
            ),
            attention_dropout=float(
                attention_config[
                    "coupling_dropout"
                ]
            ),
            attention_init_gate=float(
                attention_config[
                    "residual_gate_init_logit"
                ]
            ),
            attention_use_norm=bool(
                attention_config["use_norm"]
            ),
            attention_output_scale=float(
                attention_config[
                    "attention_output_scale"
                ]
            ),
            score_scale_mode=str(
                attention_config[
                    "score_scale_mode"
                ]
            ),
            score_temperature=float(
                attention_config[
                    "score_temperature"
                ]
            ),
            qk_init_std=float(
                attention_config["qk_init_std"]
            ),
        ).to(device),
        checkpoint_paths["m9_state"],
        device,
    )

    model_specs = [
        (
            "M6-FieldWiseEncoder-H4-continue",
            m6,
            "without_param",
            None,
        ),
        (
            "M7-FieldCoupling-H4",
            m7_field,
            "without_param",
            None,
        ),
        (
            "M7-ParamTokenOnly-H4",
            m7_param,
            "with_param",
            None,
        ),
        (
            "M8-A-FullStatic-H4",
            m8_static,
            "with_param",
            "static",
        ),
        (
            "M8-B-ParamConditionedCoupling-H4",
            m8_dynamic,
            "with_param",
            "parameter_conditioned",
        ),
        (
            "M9-0a-StaticSoftmaxAttention-H4",
            m9_static,
            "without_param",
            None,
        ),
        (
            "M9-0b-StateDependentAttention-H4",
            m9_state,
            "without_param",
            None,
        ),
    ]

    accumulators = {
        (model_name, horizon): (
            new_accumulator(device)
        )
        for model_name in MODEL_ORDER
        for horizon in horizons
    }

    per_sample_rows: list[
        dict[str, Any]
    ] = []

    total_batches = len(loader)

    print()
    print("🔥 Starting seven-model rollout...")

    with torch.no_grad():
        for batch_index, batch in enumerate(
            loader,
            start=1,
        ):
            x0_phys = batch[
                "x0_phys"
            ].to(
                device,
                non_blocking=True,
            )

            future_phys = batch[
                "future_phys"
            ].to(
                device,
                non_blocking=True,
            )

            param = batch["param"].to(
                device,
                non_blocking=True,
            )

            sample_ids = batch[
                "sample_id"
            ].tolist()

            groups = list(batch["group"])

            trajectories = batch[
                "trajectory"
            ].tolist()

            t0_values = batch["t0"].tolist()

            for (
                model_name,
                model,
                rollout_type,
                coupling_mode,
            ) in model_specs:
                if rollout_type == "without_param":
                    prediction_sequence = (
                        rollout_without_param(
                            model=model,
                            x0_phys=x0_phys,
                            normalizer=normalizer,
                            max_horizon=max_horizon,
                        )
                    )
                else:
                    prediction_sequence = (
                        rollout_with_param(
                            model=model,
                            x0_phys=x0_phys,
                            param=param,
                            normalizer=normalizer,
                            max_horizon=max_horizon,
                            coupling_mode=coupling_mode,
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
                        accumulator=accumulators[
                            (model_name, horizon)
                        ],
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
                                "sample_id": (
                                    int(
                                        sample_ids[
                                            sample_offset
                                        ]
                                    )
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
                                "model": model_name,
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

    aggregate_rows = finalize_aggregate_rows(
        accumulators=accumulators,
        horizons=horizons,
    )

    aggregate_dataframe = pd.DataFrame(
        aggregate_rows
    )

    per_sample_dataframe = pd.DataFrame(
        per_sample_rows
    )

    difference_dataframe = (
        build_difference_table(
            aggregate_dataframe=(
                aggregate_dataframe
            ),
            horizons=horizons,
        )
    )

    for path in [
        output_path,
        per_sample_output_path,
        difference_output_path,
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

    print_compact_summary(
        aggregate_dataframe=aggregate_dataframe,
        difference_dataframe=difference_dataframe,
    )

    print()
    print("✅ Seven-model rollout complete.")
    print("Aggregate:", output_path)
    print(
        "Per-sample:",
        per_sample_output_path,
    )
    print(
        "Differences:",
        difference_output_path,
    )

    if args.max_samples is not None:
        print(
            "⚠️ This was only a lightweight "
            "evaluation trial."
        )
        print(
            "⚠️ It is not a formal result."
        )


if __name__ == "__main__":
    main()
