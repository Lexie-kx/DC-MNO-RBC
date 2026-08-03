import os
import sys
import re
import csv
import json
import math
import argparse
from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import constants

from training.normalization import FieldWiseNormalizer
from models.operators.fno2d_fieldwise import FieldWiseFNO2d
from models.operators.fno2d_fieldwise_m7 import M7FieldCouplingFNO2d
from models.operators.fno2d_fieldwise_paramtoken import (
    M7FieldWiseParamTokenFNO2d,
)
from models.operators.fno2d_m8_full_conditioned import (
    M8FullConditionedFNO2d,
)


DATA_PATH = constants.DATA_PATH
FIELD_ORDER = constants.FIELD_ORDER
CONTEXT_LENGTH = getattr(
    constants,
    "CONTEXT_LENGTH",
    4,
)

FIELD_TO_IDX = {
    "buoyancy": FIELD_ORDER.index("buoyancy"),
    "u_x": FIELD_ORDER.index("u_x"),
    "u_y": FIELD_ORDER.index("u_y"),
    "pressure": FIELD_ORDER.index("pressure"),
}

UX_IDX = FIELD_TO_IDX["u_x"]
UY_IDX = FIELD_TO_IDX["u_y"]


MODEL_ORDER = [
    "M6-FieldWiseEncoder-H4-continue",
    "M7-FieldCoupling-O5-H4",
    "M7-ParamTokenOnly-O5-H4",
    "M8-A-O5-FullStatic-H4",
    "M8-B-O5-ParamConditionedCoupling-H4",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Current-stage rollout physics diagnostics "
            "for M6, M7 ablations, M8-A and M8-B."
        )
    )

    parser.add_argument(
        "--split",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--stats",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--m6_fieldwise_encoder_h4_checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--m7_fieldcoupling_checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--m7_paramtoken_checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--m8_full_static_checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--m8_param_conditioned_checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--horizons",
        type=str,
        default="1,4,8,16",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help=(
            "Debug limit. Omit for full evaluation."
        ),
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help=(
            "Keep 0 for h5py safety."
        ),
    )

    parser.add_argument(
        "--dx",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--dy",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--axis_convention",
        type=str,
        default="height_width",
        choices=[
            "height_width",
            "width_height",
        ],
    )

    return parser.parse_args()


def resolve_path(project_root, path):
    if os.path.isabs(path):
        return path

    return os.path.abspath(
        os.path.join(project_root, path)
    )


def parse_ra_pr_from_group(group_name):
    match = re.search(
        r"ra_([0-9.eE+-]+)_pr_([0-9.eE+-]+)",
        group_name,
    )

    if match is None:
        raise ValueError(
            "Cannot parse Ra/Pr from group name: "
            f"{group_name}"
        )

    ra = float(match.group(1))
    pr = float(match.group(2))

    return ra, pr


def make_param_from_group(group_name):
    ra, pr = parse_ra_pr_from_group(
        group_name
    )

    return torch.tensor(
        [
            math.log10(float(ra)),
            math.log10(float(pr)),
        ],
        dtype=torch.float32,
    )


class CrossParamRolloutDataset(Dataset):
    def __init__(
        self,
        split_path,
        split_key="test",
        max_horizon=16,
        stride=4,
        max_samples=None,
    ):
        super().__init__()

        self.split_path = split_path
        self.split_key = split_key
        self.max_horizon = max_horizon
        self.stride = stride

        with open(
            split_path,
            "r",
            encoding="utf-8",
        ) as file:
            split_config = json.load(file)

        if split_key not in split_config:
            raise KeyError(
                "Split file does not contain key: "
                f"{split_key}"
            )

        split_items = split_config[split_key]

        self.samples = []

        with h5py.File(DATA_PATH, "r") as file:
            for item in split_items:
                group_name = item["group"]
                trajectories = item["trajectories"]

                if group_name not in file:
                    raise KeyError(
                        f"Group {group_name} "
                        "not found in HDF5"
                    )

                group = file[group_name]

                first_field = FIELD_ORDER[0]

                for trajectory_index in trajectories:
                    total_steps = group[
                        first_field
                    ].shape[1]

                    max_t0 = (
                        total_steps
                        - CONTEXT_LENGTH
                        - max_horizon
                    )

                    if max_t0 < 0:
                        continue

                    for t0 in range(
                        0,
                        max_t0 + 1,
                        stride,
                    ):
                        self.samples.append(
                            {
                                "group": group_name,
                                "trajectory": int(
                                    trajectory_index
                                ),
                                "t0": int(t0),
                            }
                        )

        if max_samples is not None:
            self.samples = self.samples[
                :max_samples
            ]

        print(
            "📦 Built rollout physics dataset:"
        )
        print(f"   split = {split_key}")
        print(
            f"   samples = {len(self.samples)}"
        )
        print(
            f"   max_horizon = "
            f"{self.max_horizon}"
        )
        print(f"   stride = {self.stride}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        meta = self.samples[index]

        group_name = meta["group"]
        trajectory_index = meta["trajectory"]
        t0 = meta["t0"]

        start = t0
        end = (
            t0
            + CONTEXT_LENGTH
            + self.max_horizon
        )

        with h5py.File(DATA_PATH, "r") as file:
            group = file[group_name]

            field_arrays = []

            for field in FIELD_ORDER:
                array = group[field][
                    trajectory_index,
                    start:end,
                    :,
                    :,
                ]

                field_arrays.append(array)

        trajectory_data = torch.tensor(
            field_arrays,
            dtype=torch.float32,
        ).permute(1, 0, 2, 3)

        history = trajectory_data[
            :CONTEXT_LENGTH
        ]

        future = trajectory_data[
            CONTEXT_LENGTH:
        ]

        _, _, height, width = history.shape

        x0_phys = history.reshape(
            CONTEXT_LENGTH * len(FIELD_ORDER),
            height,
            width,
        )

        param = make_param_from_group(
            group_name
        )

        return {
            "x0_phys": x0_phys,
            "future_phys": future,
            "param": param,
        }


def load_checkpoint(
    model,
    checkpoint_path,
    device,
):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint[
            "model_state_dict"
        ]
    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    print(
        f"✅ Loaded checkpoint: "
        f"{checkpoint_path}"
    )

    return model



def build_m8_model_from_checkpoint(
    checkpoint_path,
    device,
    expected_coupling_mode,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]
        model_config = checkpoint.get("model_config")
    else:
        state_dict = checkpoint
        model_config = None

    if model_config is None:
        print(
            "⚠️ Legacy M8 checkpoint：未发现 model_config，"
            "使用原始 M8 默认结构恢复。"
        )

        model_config = {
            "in_channels": 16,
            "out_channels": 4,
            "modes1": 16,
            "modes2": 16,
            "width": 32,
            "context_length": 4,
            "num_fields": 4,
            "field_width": None,
            "coupling_mode": expected_coupling_mode,
            "coupling_hidden_channels": 8,
            "coupling_dropout": 0.0,
            "coupling_init_gate": -4.0,
            "coupling_use_norm": True,
            "coupling_param_hidden_dim": 64,
            "coupling_condition_scale": 0.10,
            "token_hidden_dim": 64,
            "alpha_token": 1.0,
        }

        config_source = "legacy defaults"

    else:
        model_config = dict(model_config)

        # 兼容早期结构化 checkpoint。
        model_config.setdefault(
            "alpha_token",
            1.0,
        )
        model_config.setdefault(
            "coupling_mode",
            expected_coupling_mode,
        )

        config_source = "checkpoint model_config"

    actual_mode = model_config["coupling_mode"]

    if actual_mode != expected_coupling_mode:
        raise ValueError(
            "❌ M8 checkpoint coupling_mode 不匹配："
            f"expected={expected_coupling_mode}, "
            f"checkpoint={actual_mode}, "
            f"path={checkpoint_path}"
        )

    model = M8FullConditionedFNO2d(
        **model_config
    ).to(device)

    model.load_state_dict(
        state_dict,
        strict=True,
    )
    model.eval()

    print(
        "✅ Restored M8 checkpoint | "
        f"source={config_source} | "
        f"mode={model_config['coupling_mode']} | "
        f"alpha_token={model_config['alpha_token']} | "
        "condition_scale="
        f"{model_config['coupling_condition_scale']}"
    )

    return model


@torch.no_grad()
def rollout_without_param(
    model,
    x0_phys,
    normalizer,
    max_horizon,
):
    predictions = []

    current_x_norm = (
        normalizer.normalize_x(x0_phys)
    )

    for _ in range(max_horizon):
        x_last_norm = current_x_norm[
            :,
            -len(FIELD_ORDER):,
            :,
            :,
        ]

        pred_delta_norm = model(
            current_x_norm
        )

        pred_y_norm = (
            x_last_norm
            + pred_delta_norm
        )

        pred_y_phys = (
            normalizer.denormalize_y(
                pred_y_norm
            )
        )

        predictions.append(pred_y_phys)

        current_x_norm = torch.cat(
            [
                current_x_norm[
                    :,
                    len(FIELD_ORDER):,
                    :,
                    :,
                ],
                pred_y_norm,
            ],
            dim=1,
        )

    return predictions


@torch.no_grad()
def rollout_with_param(
    model,
    x0_phys,
    param,
    normalizer,
    max_horizon,
    coupling_mode=None,
):
    predictions = []

    current_x_norm = (
        normalizer.normalize_x(x0_phys)
    )

    for _ in range(max_horizon):
        x_last_norm = current_x_norm[
            :,
            -len(FIELD_ORDER):,
            :,
            :,
        ]

        if coupling_mode is None:
            pred_delta_norm = model(
                current_x_norm,
                param,
            )
        else:
            pred_delta_norm = model(
                current_x_norm,
                param,
                coupling_mode=coupling_mode,
            )

        pred_y_norm = (
            x_last_norm
            + pred_delta_norm
        )

        pred_y_phys = (
            normalizer.denormalize_y(
                pred_y_norm
            )
        )

        predictions.append(pred_y_phys)

        current_x_norm = torch.cat(
            [
                current_x_norm[
                    :,
                    len(FIELD_ORDER):,
                    :,
                    :,
                ],
                pred_y_norm,
            ],
            dim=1,
        )

    return predictions


def spatial_derivatives_height_width(
    ux,
    uy,
    dx=1.0,
    dy=1.0,
):
    dux_dx = (
        ux[:, 2:, 1:-1]
        - ux[:, :-2, 1:-1]
    ) / (2.0 * dx)

    duy_dy = (
        uy[:, 1:-1, 2:]
        - uy[:, 1:-1, :-2]
    ) / (2.0 * dy)

    duy_dx = (
        uy[:, 2:, 1:-1]
        - uy[:, :-2, 1:-1]
    ) / (2.0 * dx)

    dux_dy = (
        ux[:, 1:-1, 2:]
        - ux[:, 1:-1, :-2]
    ) / (2.0 * dy)

    divergence = dux_dx + duy_dy
    vorticity = duy_dx - dux_dy

    return divergence, vorticity


def spatial_derivatives_width_height(
    ux,
    uy,
    dx=1.0,
    dy=1.0,
):
    dux_dx = (
        ux[:, 1:-1, 2:]
        - ux[:, 1:-1, :-2]
    ) / (2.0 * dx)

    duy_dy = (
        uy[:, 2:, 1:-1]
        - uy[:, :-2, 1:-1]
    ) / (2.0 * dy)

    duy_dx = (
        uy[:, 1:-1, 2:]
        - uy[:, 1:-1, :-2]
    ) / (2.0 * dx)

    dux_dy = (
        ux[:, 2:, 1:-1]
        - ux[:, :-2, 1:-1]
    ) / (2.0 * dy)

    divergence = dux_dx + duy_dy
    vorticity = duy_dx - dux_dy

    return divergence, vorticity


def compute_div_vort(
    state,
    axis_convention,
    dx,
    dy,
):
    ux = state[:, UX_IDX, :, :]
    uy = state[:, UY_IDX, :, :]

    if axis_convention == "height_width":
        return spatial_derivatives_height_width(
            ux,
            uy,
            dx=dx,
            dy=dy,
        )

    if axis_convention == "width_height":
        return spatial_derivatives_width_height(
            ux,
            uy,
            dx=dx,
            dy=dy,
        )

    raise ValueError(
        "Unknown axis_convention: "
        f"{axis_convention}"
    )


def new_accumulator():
    return {
        "num_samples": 0,
        "num_points": 0,
        "div_pred_abs_sum": 0.0,
        "div_gt_abs_sum": 0.0,
        "div_error_abs_sum": 0.0,
        "div_pred_sq_sum": 0.0,
        "div_gt_sq_sum": 0.0,
        "div_error_sq_sum": 0.0,
        "vort_error_sq_sum": 0.0,
        "vort_gt_sq_sum": 0.0,
    }


def update_metrics(
    accumulator,
    pred_state,
    gt_state,
    axis_convention,
    dx,
    dy,
):
    div_pred, vort_pred = compute_div_vort(
        pred_state,
        axis_convention=axis_convention,
        dx=dx,
        dy=dy,
    )

    div_gt, vort_gt = compute_div_vort(
        gt_state,
        axis_convention=axis_convention,
        dx=dx,
        dy=dy,
    )

    div_error = div_pred - div_gt
    vort_error = vort_pred - vort_gt

    num_samples = pred_state.shape[0]
    num_points = div_pred.numel()

    accumulator["num_samples"] += num_samples
    accumulator["num_points"] += num_points

    accumulator["div_pred_abs_sum"] += (
        torch.abs(div_pred).sum().item()
    )

    accumulator["div_gt_abs_sum"] += (
        torch.abs(div_gt).sum().item()
    )

    accumulator["div_error_abs_sum"] += (
        torch.abs(div_error).sum().item()
    )

    accumulator["div_pred_sq_sum"] += (
        div_pred.square().sum().item()
    )

    accumulator["div_gt_sq_sum"] += (
        div_gt.square().sum().item()
    )

    accumulator["div_error_sq_sum"] += (
        div_error.square().sum().item()
    )

    accumulator["vort_error_sq_sum"] += (
        vort_error.square().sum().item()
    )

    accumulator["vort_gt_sq_sum"] += (
        vort_gt.square().sum().item()
    )


def finalize_row(
    split_name,
    model_name,
    horizon,
    accumulator,
):
    eps = 1e-12

    num_points = max(
        accumulator["num_points"],
        1,
    )

    div_pred_mae = (
        accumulator["div_pred_abs_sum"]
        / num_points
    )

    div_gt_mae = (
        accumulator["div_gt_abs_sum"]
        / num_points
    )

    div_error_mae = (
        accumulator["div_error_abs_sum"]
        / num_points
    )

    div_pred_mse = (
        accumulator["div_pred_sq_sum"]
        / num_points
    )

    div_gt_mse = (
        accumulator["div_gt_sq_sum"]
        / num_points
    )

    div_error_mse = (
        accumulator["div_error_sq_sum"]
        / num_points
    )

    vorticity_mse = (
        accumulator["vort_error_sq_sum"]
        / num_points
    )

    vorticity_rel_l2_percent = math.sqrt(
        accumulator["vort_error_sq_sum"]
        / (
            accumulator["vort_gt_sq_sum"]
            + eps
        )
    ) * 100.0

    return {
        "split_name": split_name,
        "model_name": model_name,
        "horizon": horizon,
        "num_samples": accumulator[
            "num_samples"
        ],
        "div_pred_mae": div_pred_mae,
        "div_gt_mae": div_gt_mae,
        "div_error_mae": div_error_mae,
        "div_pred_mse": div_pred_mse,
        "div_gt_mse": div_gt_mse,
        "div_error_mse": div_error_mse,
        "vorticity_rel_l2_percent": (
            vorticity_rel_l2_percent
        ),
        "vorticity_mse": vorticity_mse,
    }


def infer_split_name(split_path):
    name = os.path.basename(split_path)

    name = name.replace(
        "_split.json",
        "",
    )

    name = name.replace(
        ".json",
        "",
    )

    return name


def calculate_difference(
    row_map,
    model_a,
    model_b,
    horizons,
):
    difference_rows = []

    for horizon in horizons:
        a = row_map[(model_a, horizon)]
        b = row_map[(model_b, horizon)]

        difference_rows.append(
            {
                "horizon": horizon,
                "div_pred_mae": (
                    a["div_pred_mae"]
                    - b["div_pred_mae"]
                ),
                "div_error_mae": (
                    a["div_error_mae"]
                    - b["div_error_mae"]
                ),
                "vorticity_rel_l2_percent": (
                    a[
                        "vorticity_rel_l2_percent"
                    ]
                    - b[
                        "vorticity_rel_l2_percent"
                    ]
                ),
            }
        )

    return difference_rows


def safe_name(text):
    return (
        text.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(":", "")
    )


def main():
    args = parse_args()

    project_root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
        )
    )

    split_path = resolve_path(
        project_root,
        args.split,
    )

    stats_path = resolve_path(
        project_root,
        args.stats,
    )

    m6_checkpoint = resolve_path(
        project_root,
        args.m6_fieldwise_encoder_h4_checkpoint,
    )

    m7_fieldcoupling_checkpoint = resolve_path(
        project_root,
        args.m7_fieldcoupling_checkpoint,
    )

    m7_paramtoken_checkpoint = resolve_path(
        project_root,
        args.m7_paramtoken_checkpoint,
    )

    m8_full_static_checkpoint = resolve_path(
        project_root,
        args.m8_full_static_checkpoint,
    )

    m8_param_conditioned_checkpoint = resolve_path(
        project_root,
        args.m8_param_conditioned_checkpoint,
    )

    output_path = resolve_path(
        project_root,
        args.output,
    )

    paths_to_check = [
        split_path,
        stats_path,
        m6_checkpoint,
        m7_fieldcoupling_checkpoint,
        m7_paramtoken_checkpoint,
        m8_full_static_checkpoint,
        m8_param_conditioned_checkpoint,
    ]

    for path in paths_to_check:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"❌ 文件不存在: {path}"
            )

    horizons = [
        int(value)
        for value in args.horizons.split(",")
    ]

    max_horizon = max(horizons)

    split_name = infer_split_name(
        split_path
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "🚀 [M8 Current-Stage "
        "Rollout Physics Diagnostics]"
    )

    print(f"📌 Device: {device}")
    print(f"📌 Split: {split_path}")
    print(f"📌 Stats: {stats_path}")
    print(f"📌 Split name: {split_name}")
    print(
        "📌 M6 control: "
        f"{m6_checkpoint}"
    )
    print(
        "📌 M7-FieldCoupling: "
        f"{m7_fieldcoupling_checkpoint}"
    )
    print(
        "📌 M7-ParamToken: "
        f"{m7_paramtoken_checkpoint}"
    )
    print(
        "📌 M8-A-FullStatic: "
        f"{m8_full_static_checkpoint}"
    )
    print(
        "📌 M8-B-ParamConditioned: "
        f"{m8_param_conditioned_checkpoint}"
    )
    print(f"📌 Horizons: {horizons}")
    print(f"📌 Stride: {args.stride}")
    print(f"📌 Batch size: {args.batch_size}")
    print(
        f"📌 Axis convention: "
        f"{args.axis_convention}"
    )
    print(
        f"📌 dx={args.dx}, dy={args.dy}"
    )
    print(f"📌 Output: {output_path}")

    normalizer = FieldWiseNormalizer(
        stats_path=stats_path,
        device=device,
    )

    dataset = CrossParamRolloutDataset(
        split_path=split_path,
        split_key="test",
        max_horizon=max_horizon,
        stride=args.stride,
        max_samples=args.max_samples,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    m6 = FieldWiseFNO2d(
        in_channels=(
            CONTEXT_LENGTH
            * len(FIELD_ORDER)
        ),
        out_channels=len(FIELD_ORDER),
        modes1=16,
        modes2=16,
        width=32,
    ).to(device)

    m6 = load_checkpoint(
        m6,
        m6_checkpoint,
        device,
    )

    m7_fieldcoupling = M7FieldCouplingFNO2d(
        in_channels=(
            CONTEXT_LENGTH
            * len(FIELD_ORDER)
        ),
        out_channels=len(FIELD_ORDER),
        modes1=16,
        modes2=16,
        width=32,
        context_length=CONTEXT_LENGTH,
        num_fields=len(FIELD_ORDER),
        coupling_hidden_channels=8,
        coupling_dropout=0.0,
        coupling_init_gate=-4.0,
        coupling_use_norm=True,
    ).to(device)

    m7_fieldcoupling = load_checkpoint(
        m7_fieldcoupling,
        m7_fieldcoupling_checkpoint,
        device,
    )

    m7_paramtoken = M7FieldWiseParamTokenFNO2d(
        in_channels=(
            CONTEXT_LENGTH
            * len(FIELD_ORDER)
        ),
        out_channels=len(FIELD_ORDER),
        modes1=16,
        modes2=16,
        width=32,
        context_length=CONTEXT_LENGTH,
        num_fields=len(FIELD_ORDER),
        token_hidden_dim=64,
    ).to(device)

    m7_paramtoken = load_checkpoint(
        m7_paramtoken,
        m7_paramtoken_checkpoint,
        device,
    )

    m8_full_static = build_m8_model_from_checkpoint(
        checkpoint_path=m8_full_static_checkpoint,
        device=device,
        expected_coupling_mode="static",
    )

    m8_param_conditioned = build_m8_model_from_checkpoint(
        checkpoint_path=m8_param_conditioned_checkpoint,
        device=device,
        expected_coupling_mode="parameter_conditioned",
    )

    accumulator = {
        model_name: {
            horizon: new_accumulator()
            for horizon in horizons
        }
        for model_name in MODEL_ORDER
    }

    total_batches = len(loader)

    print(
        "\n🔥 开始 M8 current-stage "
        "physics evaluation..."
    )

    for batch_index, batch in enumerate(
        loader,
        start=1,
    ):
        x0_phys = batch["x0_phys"].to(
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

        seq_m6 = rollout_without_param(
            model=m6,
            x0_phys=x0_phys,
            normalizer=normalizer,
            max_horizon=max_horizon,
        )

        seq_m7_fieldcoupling = (
            rollout_without_param(
                model=m7_fieldcoupling,
                x0_phys=x0_phys,
                normalizer=normalizer,
                max_horizon=max_horizon,
            )
        )

        seq_m7_paramtoken = rollout_with_param(
            model=m7_paramtoken,
            x0_phys=x0_phys,
            param=param,
            normalizer=normalizer,
            max_horizon=max_horizon,
        )

        seq_m8_full_static = rollout_with_param(
            model=m8_full_static,
            x0_phys=x0_phys,
            param=param,
            normalizer=normalizer,
            max_horizon=max_horizon,
            coupling_mode="static",
        )

        seq_m8_param_conditioned = (
            rollout_with_param(
                model=m8_param_conditioned,
                x0_phys=x0_phys,
                param=param,
                normalizer=normalizer,
                max_horizon=max_horizon,
                coupling_mode=(
                    "parameter_conditioned"
                ),
            )
        )

        prediction_dict = {
            (
                "M6-FieldWiseEncoder-H4-continue"
            ): seq_m6,
            (
                "M7-FieldCoupling-O5-H4"
            ): seq_m7_fieldcoupling,
            (
                "M7-ParamTokenOnly-O5-H4"
            ): seq_m7_paramtoken,
            (
                "M8-A-O5-FullStatic-H4"
            ): seq_m8_full_static,
            (
                "M8-B-O5-ParamConditionedCoupling-H4"
            ): seq_m8_param_conditioned,
        }

        for horizon in horizons:
            horizon_index = horizon - 1

            gt_state = future_phys[
                :,
                horizon_index,
                :,
                :,
                :,
            ]

            for model_name in MODEL_ORDER:
                pred_state = prediction_dict[
                    model_name
                ][horizon_index]

                update_metrics(
                    accumulator=(
                        accumulator[
                            model_name
                        ][horizon]
                    ),
                    pred_state=pred_state,
                    gt_state=gt_state,
                    axis_convention=(
                        args.axis_convention
                    ),
                    dx=args.dx,
                    dy=args.dy,
                )

        if (
            batch_index % 10 == 0
            or batch_index == total_batches
        ):
            print(
                f"   processed batch "
                f"{batch_index}/"
                f"{total_batches}"
            )

    rows = []

    for model_name in MODEL_ORDER:
        for horizon in horizons:
            rows.append(
                finalize_row(
                    split_name=split_name,
                    model_name=model_name,
                    horizon=horizon,
                    accumulator=(
                        accumulator[
                            model_name
                        ][horizon]
                    ),
                )
            )

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True,
    )

    fieldnames = [
        "split_name",
        "model_name",
        "horizon",
        "num_samples",
        "div_pred_mae",
        "div_gt_mae",
        "div_error_mae",
        "div_pred_mse",
        "div_gt_mse",
        "div_error_mse",
        "vorticity_rel_l2_percent",
        "vorticity_mse",
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print(
        "\n✅ Physics diagnostics saved:"
    )
    print(f"   {output_path}")

    row_map = {
        (
            row["model_name"],
            row["horizon"],
        ): row
        for row in rows
    }

    print()
    print("=" * 108)
    print(
        "M6–M8 UNSEEN PHYSICS SUMMARY"
    )
    print("=" * 108)

    header = (
        f"{'model':<40} "
        f"{'h':>4} "
        f"{'div_pred_mae':>14} "
        f"{'div_err_mae':>14} "
        f"{'vort_rel_l2%':>14}"
    )

    print(header)
    print("-" * len(header))

    for model_name in MODEL_ORDER:
        for horizon in horizons:
            row = row_map[
                (model_name, horizon)
            ]

            print(
                f"{model_name:<40} "
                f"{horizon:>4} "
                f"{row['div_pred_mae']:>14.6e} "
                f"{row['div_error_mae']:>14.6e} "
                f"{row['vorticity_rel_l2_percent']:>14.6f}"
            )

    comparison_pairs = [
        (
            "M7-FieldCoupling-O5-H4",
            "M6-FieldWiseEncoder-H4-continue",
            "M7-FieldCoupling - M6",
        ),
        (
            "M7-ParamTokenOnly-O5-H4",
            "M6-FieldWiseEncoder-H4-continue",
            "M7-ParamToken - M6",
        ),
        (
            "M8-A-O5-FullStatic-H4",
            "M6-FieldWiseEncoder-H4-continue",
            "M8-A - M6",
        ),
        (
            "M8-B-O5-ParamConditionedCoupling-H4",
            "M6-FieldWiseEncoder-H4-continue",
            "M8-B - M6",
        ),
        (
            "M8-A-O5-FullStatic-H4",
            "M7-FieldCoupling-O5-H4",
            "M8-A - M7-FieldCoupling",
        ),
        (
            "M8-A-O5-FullStatic-H4",
            "M7-ParamTokenOnly-O5-H4",
            "M8-A - M7-ParamToken",
        ),
        (
            "M8-B-O5-ParamConditionedCoupling-H4",
            "M7-FieldCoupling-O5-H4",
            "M8-B - M7-FieldCoupling",
        ),
        (
            "M8-B-O5-ParamConditionedCoupling-H4",
            "M7-ParamTokenOnly-O5-H4",
            "M8-B - M7-ParamToken",
        ),
        (
            "M8-B-O5-ParamConditionedCoupling-H4",
            "M8-A-O5-FullStatic-H4",
            "M8-B - M8-A",
        ),
    ]

    detailed_differences = {}

    print()
    print("=" * 108)
    print(
        "COMPLETE PHYSICS DIFFERENCE SUMMARY"
    )
    print(
        "前者 - 后者；"
        "div_error_mae 与 vort_rel_l2% "
        "负数表示前者误差更低。"
    )
    print("=" * 108)

    difference_header = (
        f"{'comparison':<38} "
        f"{'h':>4} "
        f"{'div_pred':>14} "
        f"{'div_error':>14} "
        f"{'vort_rel_l2%':>14}"
    )

    print(difference_header)
    print("-" * len(difference_header))

    output_object = Path(output_path)

    for (
        model_a,
        model_b,
        short_name,
    ) in comparison_pairs:
        difference_rows = calculate_difference(
            row_map=row_map,
            model_a=model_a,
            model_b=model_b,
            horizons=horizons,
        )

        detailed_differences[
            short_name
        ] = difference_rows

        for difference in difference_rows:
            print(
                f"{short_name:<38} "
                f"{difference['horizon']:>4} "
                f"{difference['div_pred_mae']:>14.6e} "
                f"{difference['div_error_mae']:>14.6e} "
                f"{difference['vorticity_rel_l2_percent']:>14.6f}"
            )

        difference_path = (
            output_object.parent
            / (
                output_object.stem
                + "_"
                + safe_name(short_name)
                + ".csv"
            )
        )

        with open(
            difference_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "horizon",
                    "div_pred_mae",
                    "div_error_mae",
                    "vorticity_rel_l2_percent",
                ],
            )

            writer.writeheader()

            for difference in difference_rows:
                writer.writerow(difference)

    print()
    print("=" * 108)
    print(
        "M8-A FULL PHYSICS COMPARISON"
    )
    print("=" * 108)

    for title in [
        "M8-A - M6",
        "M8-A - M7-FieldCoupling",
        "M8-A - M7-ParamToken",
    ]:
        print(f"\n--- {title} ---")

        for difference in detailed_differences[
            title
        ]:
            print(
                f"h={difference['horizon']:<2} "
                f"div_pred={difference['div_pred_mae']:+.6e} "
                f"div_error={difference['div_error_mae']:+.6e} "
                f"vort={difference['vorticity_rel_l2_percent']:+.6f}"
            )

    print()
    print("=" * 108)
    print(
        "M8-B FULL PHYSICS COMPARISON"
    )
    print("=" * 108)

    for title in [
        "M8-B - M6",
        "M8-B - M7-FieldCoupling",
        "M8-B - M7-ParamToken",
        "M8-B - M8-A",
    ]:
        print(f"\n--- {title} ---")

        for difference in detailed_differences[
            title
        ]:
            print(
                f"h={difference['horizon']:<2} "
                f"div_pred={difference['div_pred_mae']:+.6e} "
                f"div_error={difference['div_error_mae']:+.6e} "
                f"vort={difference['vorticity_rel_l2_percent']:+.6f}"
            )

    print()
    print("=" * 108)
    print(
        "说明：div_error 与 vort "
        "是主要误差比较指标；"
        "div_pred 是预测散度幅值诊断。"
    )
    print("=" * 108)


if __name__ == "__main__":
    main()
