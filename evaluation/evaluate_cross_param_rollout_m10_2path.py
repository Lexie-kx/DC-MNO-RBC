import argparse
import hashlib
import json
import math
import os
import re
import sys

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from constants import DATA_PATH, FIELD_ORDER, CONTEXT_LENGTH, DTYPE
from training.normalization import FieldWiseNormalizer
from models.operators.fno2d_fieldwise import FieldWiseFNO2d
from models.operators.fno2d_m10_2path import M10TwoPathFNO2d


DISPLAY_NAMES = {
    "static": "M10-Static",
    "param": "M10-Param",
    "state": "M10-State",
    "stateparam": "M10-StateParam",
}


class RolloutDataset(Dataset):
    def __init__(self, split_config, max_horizon=16, max_samples=None):
        self.split_config = split_config
        self.max_horizon = int(max_horizon)
        self.max_samples = max_samples
        self.data_store = {}
        self.index = []
        self._load_data()

    @staticmethod
    def _parse_ra_pr(group_name):
        ra_match = re.search(r"[Rr]a_?([0-9.eE+-]+)", group_name)
        pr_match = re.search(r"[Pp]r_?([0-9.eE+-]+)", group_name)
        if ra_match is None or pr_match is None:
            raise ValueError(f"Cannot parse Ra/Pr from group: {group_name}")
        return float(ra_match.group(1)), float(pr_match.group(1))

    @classmethod
    def _make_param(cls, group_name):
        ra, pr = cls._parse_ra_pr(group_name)
        return torch.tensor(
            [math.log10(ra), math.log10(pr)],
            dtype=DTYPE,
        )

    def _load_data(self):
        print(
            f"📦 Loading rollout test data: "
            f"{len(self.split_config)} group batches"
        )

        with h5py.File(DATA_PATH, "r") as f:
            for item in self.split_config:
                group_name = item["group"]
                traj_indices = item["trajectories"]

                if group_name not in f:
                    raise KeyError(
                        f"Group not found in dataset: {group_name}"
                    )

                group = f[group_name]

                fields_data = [
                    group[field][:]
                    for field in FIELD_ORDER
                ]

                stacked = torch.tensor(
                    np.stack(fields_data, axis=0),
                    dtype=DTYPE,
                )

                # [field, traj, time, X, Y]
                # ->
                # [traj, time, field, X, Y]
                selected = stacked[:, traj_indices]

                data = selected.permute(
                    1,
                    2,
                    0,
                    3,
                    4,
                ).contiguous()

                self.data_store[group_name] = data

                param = self._make_param(
                    group_name
                )

                (
                    num_traj,
                    num_steps,
                    _,
                    _,
                    _,
                ) = data.shape

                max_start = (
                    num_steps
                    - CONTEXT_LENGTH
                    - self.max_horizon
                    + 1
                )

                if max_start <= 0:
                    raise RuntimeError(
                        f"Not enough time steps in "
                        f"{group_name} for "
                        f"horizon={self.max_horizon}"
                    )

                for local_traj_idx in range(
                    num_traj
                ):
                    for t0 in range(
                        max_start
                    ):
                        self.index.append(
                            {
                                "group": group_name,
                                "traj": local_traj_idx,
                                "t0": t0,
                                "param": param,
                            }
                        )

        if self.max_samples is not None:
            self.index = self.index[
                : int(self.max_samples)
            ]

        print(
            f"✅ Rollout samples: "
            f"{len(self.index)}"
        )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        item = self.index[idx]

        data = self.data_store[
            item["group"]
        ]

        history = data[
            item["traj"],
            item["t0"]:
            item["t0"] + CONTEXT_LENGTH,
        ]

        future = data[
            item["traj"],
            item["t0"] + CONTEXT_LENGTH:
            item["t0"]
            + CONTEXT_LENGTH
            + self.max_horizon,
        ]

        _, _, h, w = history.shape

        x0_phys = history.reshape(
            CONTEXT_LENGTH * 4,
            h,
            w,
        )

        return (
            x0_phys,
            future,
            item["param"],
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "M10-0 2x2 cross-parameter "
            "free-autoregressive test rollout. "
            "Evaluates M6 + "
            "Static/Param/State/StateParam."
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
        "--split_label",
        required=True,
        choices=[
            "unseen_pr",
            "unseen_ra",
        ],
    )

    parser.add_argument(
        "--seed",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--m6_checkpoint",
        required=True,
    )

    parser.add_argument(
        "--static_checkpoint",
        required=True,
    )

    parser.add_argument(
        "--param_checkpoint",
        required=True,
    )

    parser.add_argument(
        "--state_checkpoint",
        required=True,
    )

    parser.add_argument(
        "--stateparam_checkpoint",
        required=True,
    )

    parser.add_argument(
        "--dx",
        required=True,
        type=float,
    )

    parser.add_argument(
        "--dy",
        required=True,
        type=float,
    )

    parser.add_argument(
        "--alpha_max",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--conditioner_hidden",
        type=int,
        default=32,
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
        "--max_samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--output_dir",
        default=(
            "outputs/tables/"
            "m10_0_rollout"
        ),
    )

    return parser.parse_args()


def resolve_path(path):
    if os.path.isabs(path):
        return path

    return os.path.abspath(
        os.path.join(
            PROJECT_ROOT,
            path,
        )
    )


def sha256_file(path):
    h = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:
        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def build_field_stats(
    stats_path,
):
    with open(
        stats_path,
        "r",
        encoding="utf-8",
    ) as f:
        stats = json.load(f)

    field_mean = [
        float(
            stats[field]["mean"]
        )
        for field
        in FIELD_ORDER
    ]

    field_std = [
        float(
            stats[field]["std"]
        )
        for field
        in FIELD_ORDER
    ]

    return (
        field_mean,
        field_std,
    )


def extract_state_dict(
    payload,
):
    if (
        isinstance(
            payload,
            dict,
        )
        and
        "model_state_dict"
        in payload
    ):
        return payload[
            "model_state_dict"
        ]

    if isinstance(
        payload,
        dict,
    ):
        return payload

    raise TypeError(
        "Unsupported checkpoint format."
    )


def load_checkpoint(
    path,
    device,
):
    payload = torch.load(
        path,
        map_location=device,
    )

    return (
        payload,
        extract_state_dict(
            payload
        ),
    )


def audit_metadata(
    payload,
    *,
    mode,
    seed,
    dx,
    dy,
    alpha_max,
    conditioner_hidden,
    path,
):
    if not isinstance(
        payload,
        dict,
    ):
        return

    checks = {
        "mode": mode,
        "seed": seed,
        "dx": dx,
        "dy": dy,
        "alpha_max": (
            alpha_max
        ),
        "conditioner_hidden": (
            conditioner_hidden
        ),
    }

    for (
        key,
        expected,
    ) in checks.items():

        if key not in payload:
            continue

        actual = payload[key]

        if isinstance(
            expected,
            float,
        ):
            if not math.isclose(
                float(actual),
                float(expected),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    "Checkpoint metadata "
                    "mismatch: "
                    f"{path}: "
                    f"{key}={actual}, "
                    f"expected={expected}"
                )

        else:
            if actual != expected:
                raise RuntimeError(
                    "Checkpoint metadata "
                    "mismatch: "
                    f"{path}: "
                    f"{key}={actual}, "
                    f"expected={expected}"
                )


def build_m6(
    checkpoint_path,
    device,
):
    model = FieldWiseFNO2d(
        in_channels=16,
        out_channels=4,
        modes1=16,
        modes2=16,
        width=32,
    ).to(device)

    (
        payload,
        state,
    ) = load_checkpoint(
        checkpoint_path,
        device,
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    model.eval()

    return (
        model,
        payload,
    )


def build_m10(
    mode,
    checkpoint_path,
    field_mean,
    field_std,
    args,
    device,
):
    model = M10TwoPathFNO2d(
        field_mean=field_mean,
        field_std=field_std,
        mode=mode,
        dx=args.dx,
        dy=args.dy,
        in_channels=16,
        out_channels=4,
        modes1=16,
        modes2=16,
        width=32,
        alpha_max=(
            args.alpha_max
        ),
        conditioner_hidden=(
            args.conditioner_hidden
        ),
        freeze_m6=True,
    ).to(device)

    (
        payload,
        state,
    ) = load_checkpoint(
        checkpoint_path,
        device,
    )

    audit_metadata(
        payload,
        mode=mode,
        seed=args.seed,
        dx=args.dx,
        dy=args.dy,
        alpha_max=(
            args.alpha_max
        ),
        conditioner_hidden=(
            args.conditioner_hidden
        ),
        path=checkpoint_path,
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    model.eval()

    expected_mean = torch.tensor(
        field_mean,
        device=device,
        dtype=(
            model.field_mean.dtype
        ),
    )

    expected_std = torch.tensor(
        field_std,
        device=device,
        dtype=(
            model.field_std.dtype
        ),
    )

    if not torch.equal(
        model.field_mean.view(-1),
        expected_mean,
    ):
        raise RuntimeError(
            "field_mean mismatch "
            "after loading "
            f"{checkpoint_path}"
        )

    if not torch.equal(
        model.field_std.view(-1),
        expected_std,
    ):
        raise RuntimeError(
            "field_std mismatch "
            "after loading "
            f"{checkpoint_path}"
        )

    return (
        model,
        payload,
    )


def assert_same_m6(
    reference_m6,
    m10_model,
    model_name,
):
    ref = (
        reference_m6
        .state_dict()
    )

    got = (
        m10_model
        .m6
        .state_dict()
    )

    if ref.keys() != got.keys():
        raise RuntimeError(
            f"{model_name}: "
            "embedded M6 state keys "
            "differ from audited M6."
        )

    for key in ref:
        if not torch.equal(
            ref[key],
            got[key],
        ):
            max_diff = float(
                (
                    ref[key]
                    - got[key]
                )
                .abs()
                .max()
                .item()
            )

            raise RuntimeError(
                f"{model_name}: "
                "embedded M6 differs "
                f"at {key}, "
                "max_abs_diff="
                f"{max_diff:.6e}"
            )

    print(
        f"✅ {model_name}: "
        "embedded frozen M6 "
        "exactly matches audited M6"
    )


def new_error_bucket(
    device,
):
    return {
        "field_sse": (
            torch.zeros(
                4,
                dtype=torch.float64,
                device=device,
            )
        ),
        "field_target_sq": (
            torch.zeros(
                4,
                dtype=torch.float64,
                device=device,
            )
        ),
        "field_numel": (
            torch.zeros(
                4,
                dtype=torch.float64,
                device=device,
            )
        ),
        "global_sse": (
            torch.tensor(
                0.0,
                dtype=torch.float64,
                device=device,
            )
        ),
        "global_target_sq": (
            torch.tensor(
                0.0,
                dtype=torch.float64,
                device=device,
            )
        ),
        "global_numel": (
            torch.tensor(
                0.0,
                dtype=torch.float64,
                device=device,
            )
        ),
    }


def update_error_stats(
    stats,
    model_name,
    step,
    pred_phys,
    true_phys,
):
    key = (
        model_name,
        step,
    )

    if key not in stats:
        stats[key] = (
            new_error_bucket(
                pred_phys.device
            )
        )

    bucket = stats[key]

    diff = (
        pred_phys
        - true_phys
    )

    for c in range(4):
        diff_c = (
            diff[
                :,
                c,
                :,
                :,
            ]
            .double()
        )

        true_c = (
            true_phys[
                :,
                c,
                :,
                :,
            ]
            .double()
        )

        bucket[
            "field_sse"
        ][c] += torch.sum(
            diff_c ** 2
        )

        bucket[
            "field_target_sq"
        ][c] += torch.sum(
            true_c ** 2
        )

        bucket[
            "field_numel"
        ][c] += diff_c.numel()

    bucket[
        "global_sse"
    ] += torch.sum(
        diff.double() ** 2
    )

    bucket[
        "global_target_sq"
    ] += torch.sum(
        true_phys.double() ** 2
    )

    bucket[
        "global_numel"
    ] += diff.numel()


def new_gate_bucket():
    return {
        "n": 0,

        "a_sum": 0.0,
        "a_sq_sum": 0.0,
        "a_min": float("inf"),
        "a_max": float("-inf"),

        "b_sum": 0.0,
        "b_sq_sum": 0.0,
        "b_min": float("inf"),
        "b_max": float("-inf"),

        "dyn_abs_sum": 0.0,
        "dyn_n": 0,
    }


def update_gate_stats(
    gate_stats,
    model_name,
    step,
    info,
):
    key = (
        model_name,
        step,
    )

    if key not in gate_stats:
        gate_stats[key] = (
            new_gate_bucket()
        )

    bucket = gate_stats[key]

    a = (
        info["alpha_a"]
        .detach()
        .double()
        .reshape(-1)
    )

    b = (
        info["alpha_b"]
        .detach()
        .double()
        .reshape(-1)
    )

    dyn = (
        info["dynamic_raw"]
        .detach()
        .double()
        .reshape(-1)
    )

    bucket["n"] += (
        a.numel()
    )

    bucket["a_sum"] += float(
        a.sum().item()
    )

    bucket["a_sq_sum"] += float(
        (a * a).sum().item()
    )

    bucket["a_min"] = min(
        bucket["a_min"],
        float(
            a.min().item()
        ),
    )

    bucket["a_max"] = max(
        bucket["a_max"],
        float(
            a.max().item()
        ),
    )

    bucket["b_sum"] += float(
        b.sum().item()
    )

    bucket["b_sq_sum"] += float(
        (b * b).sum().item()
    )

    bucket["b_min"] = min(
        bucket["b_min"],
        float(
            b.min().item()
        ),
    )

    bucket["b_max"] = max(
        bucket["b_max"],
        float(
            b.max().item()
        ),
    )

    bucket[
        "dyn_abs_sum"
    ] += float(
        dyn.abs()
        .sum()
        .item()
    )

    bucket[
        "dyn_n"
    ] += dyn.numel()


def rollout_one_model(
    model_name,
    model,
    x0_phys,
    future_phys,
    param,
    normalizer,
    max_horizon,
    error_stats,
    gate_stats,
):
    x_norm = (
        normalizer
        .normalize_x(
            x0_phys
        )
    )

    for step in range(
        1,
        max_horizon + 1,
    ):
        current_norm = (
            x_norm[
                :,
                -4:,
                :,
                :,
            ]
        )

        if model_name == "M6":
            pred_delta_norm = (
                model(
                    x_norm
                )
            )

            info = None

        else:
            (
                pred_delta_norm,
                info,
            ) = model(
                x_norm,
                params=param,
                return_components=True,
            )

        pred_next_norm = (
            current_norm
            + pred_delta_norm
        )

        pred_next_phys = (
            normalizer
            .denormalize_y(
                pred_next_norm
            )
        )

        true_phys = (
            future_phys[
                :,
                step - 1,
                :,
                :,
                :,
            ]
        )

        update_error_stats(
            error_stats,
            model_name,
            step,
            pred_next_phys,
            true_phys,
        )

        if info is not None:
            update_gate_stats(
                gate_stats,
                model_name,
                step,
                info,
            )

        x_norm = torch.cat(
            [
                x_norm[
                    :,
                    4:,
                    :,
                    :,
                ],
                pred_next_norm,
            ],
            dim=1,
        )


def error_rows(
    error_stats,
    model_order,
    max_horizon,
):
    rows = []

    for model_name in model_order:
        for step in range(
            1,
            max_horizon + 1,
        ):
            bucket = error_stats[
                (
                    model_name,
                    step,
                )
            ]

            for (
                c,
                field,
            ) in enumerate(
                FIELD_ORDER
            ):
                rel = torch.sqrt(
                    bucket[
                        "field_sse"
                    ][c]
                    /
                    (
                        bucket[
                            "field_target_sq"
                        ][c]
                        + 1e-12
                    )
                ) * 100.0

                mse = (
                    bucket[
                        "field_sse"
                    ][c]
                    /
                    bucket[
                        "field_numel"
                    ][c]
                )

                rows.append(
                    {
                        "model": (
                            model_name
                        ),
                        "horizon": (
                            step
                        ),
                        "field": (
                            field
                        ),
                        "rel_l2_percent": (
                            float(
                                rel.item()
                            )
                        ),
                        "mse": (
                            float(
                                mse.item()
                            )
                        ),
                    }
                )

            global_rel = torch.sqrt(
                bucket[
                    "global_sse"
                ]
                /
                (
                    bucket[
                        "global_target_sq"
                    ]
                    + 1e-12
                )
            ) * 100.0

            global_mse = (
                bucket[
                    "global_sse"
                ]
                /
                bucket[
                    "global_numel"
                ]
            )

            rows.append(
                {
                    "model": model_name,
                    "horizon": step,
                    "field": "global",
                    "rel_l2_percent": (
                        float(
                            global_rel
                            .item()
                        )
                    ),
                    "mse": (
                        float(
                            global_mse
                            .item()
                        )
                    ),
                }
            )

    return rows


def gate_rows(
    gate_stats,
    model_order,
    max_horizon,
):
    rows = []

    for model_name in model_order:
        if model_name == "M6":
            continue

        for step in range(
            1,
            max_horizon + 1,
        ):
            b = gate_stats[
                (
                    model_name,
                    step,
                )
            ]

            n = b["n"]

            a_mean = (
                b["a_sum"] / n
            )

            a_var = max(
                0.0,
                (
                    b["a_sq_sum"]
                    / n
                    - a_mean
                    * a_mean
                ),
            )

            b_mean = (
                b["b_sum"] / n
            )

            b_var = max(
                0.0,
                (
                    b["b_sq_sum"]
                    / n
                    - b_mean
                    * b_mean
                ),
            )

            rows.append(
                {
                    "model": (
                        model_name
                    ),
                    "horizon": (
                        step
                    ),
                    "alpha_a_mean": (
                        a_mean
                    ),
                    "alpha_a_std": (
                        math.sqrt(
                            a_var
                        )
                    ),
                    "alpha_a_min": (
                        b["a_min"]
                    ),
                    "alpha_a_max": (
                        b["a_max"]
                    ),
                    "alpha_b_mean": (
                        b_mean
                    ),
                    "alpha_b_std": (
                        math.sqrt(
                            b_var
                        )
                    ),
                    "alpha_b_min": (
                        b["b_min"]
                    ),
                    "alpha_b_max": (
                        b["b_max"]
                    ),
                    "dynamic_raw_abs_mean": (
                        b[
                            "dyn_abs_sum"
                        ]
                        /
                        b["dyn_n"]
                        if b["dyn_n"] > 0
                        else 0.0
                    ),
                }
            )

    return rows


def make_global_growth_table(
    curve_df,
    model_order,
    max_horizon,
):
    rows = []

    for model_name in model_order:
        sub = curve_df[
            (
                curve_df["model"]
                == model_name
            )
            &
            (
                curve_df["field"]
                == "global"
            )
        ].sort_values(
            "horizon"
        )

        x = (
            sub["horizon"]
            .to_numpy(
                dtype=float
            )
        )

        y = (
            sub[
                "rel_l2_percent"
            ]
            .to_numpy(
                dtype=float
            )
        )

        if len(y) != max_horizon:
            raise RuntimeError(
                "Incomplete global "
                "curve for "
                f"{model_name}"
            )

        auc = float(
            np.sum(
                0.5
                *
                (
                    y[:-1]
                    + y[1:]
                )
                *
                np.diff(x)
            )
        )

        span = float(
            x[-1]
            - x[0]
        )

        rows.append(
            {
                "model": (
                    model_name
                ),
                "global_rel_l2_h1": (
                    float(y[0])
                ),
                (
                    f"global_rel_l2_"
                    f"h{max_horizon}"
                ): (
                    float(y[-1])
                ),
                "global_error_growth_abs": (
                    float(
                        y[-1]
                        - y[0]
                    )
                ),
                "global_error_growth_ratio": (
                    float(
                        y[-1]
                        /
                        max(
                            y[0],
                            1e-12,
                        )
                    )
                ),
                "global_error_growth_slope": (
                    float(
                        (
                            y[-1]
                            - y[0]
                        )
                        /
                        max(
                            span,
                            1.0,
                        )
                    )
                ),
                "global_auc_trapz": (
                    auc
                ),
                "global_auc_mean_over_span": (
                    float(
                        auc
                        /
                        max(
                            span,
                            1.0,
                        )
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def make_difference_table(
    summary_df,
):
    pairs = [
        (
            "M10-Static",
            "M6",
        ),
        (
            "M10-Param",
            "M10-Static",
        ),
        (
            "M10-State",
            "M10-Static",
        ),
        (
            "M10-StateParam",
            "M10-Param",
        ),
        (
            "M10-StateParam",
            "M10-State",
        ),
        (
            "M10-StateParam",
            "M10-Static",
        ),
        (
            "M10-StateParam",
            "M6",
        ),
    ]

    rows = []

    for (
        model_a,
        model_b,
    ) in pairs:

        a = summary_df[
            summary_df["model"]
            == model_a
        ]

        b = summary_df[
            summary_df["model"]
            == model_b
        ]

        merged = a.merge(
            b,
            on=[
                "horizon",
                "field",
            ],
            suffixes=(
                "_a",
                "_b",
            ),
        )

        for (
            _,
            row,
        ) in merged.iterrows():

            rows.append(
                {
                    "comparison": (
                        f"{model_a} "
                        f"- {model_b}"
                    ),
                    "horizon": int(
                        row["horizon"]
                    ),
                    "field": (
                        row["field"]
                    ),
                    "rel_l2_diff_percent_point": (
                        row[
                            "rel_l2_percent_a"
                        ]
                        -
                        row[
                            "rel_l2_percent_b"
                        ]
                    ),
                    "mse_diff": (
                        row["mse_a"]
                        -
                        row["mse_b"]
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def main():
    args = parse_args()

    split_path = resolve_path(
        args.split
    )

    stats_path = resolve_path(
        args.stats
    )

    m6_path = resolve_path(
        args.m6_checkpoint
    )

    checkpoint_paths = {
        "static": (
            resolve_path(
                args.static_checkpoint
            )
        ),
        "param": (
            resolve_path(
                args.param_checkpoint
            )
        ),
        "state": (
            resolve_path(
                args.state_checkpoint
            )
        ),
        "stateparam": (
            resolve_path(
                args.stateparam_checkpoint
            )
        ),
    }

    requested_horizons = sorted(
        {
            int(x)
            for x
            in args.horizons.split(",")
            if x.strip()
        }
    )

    if (
        not requested_horizons
        or
        requested_horizons[0] < 1
    ):
        raise ValueError(
            "Invalid --horizons"
        )

    max_horizon = max(
        requested_horizons
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "🚀 "
        "[M10-0 2x2 "
        "Test Rollout Evaluation]"
    )

    print(
        f"📌 Device: "
        f"{device}"
    )

    print(
        f"📌 Split label: "
        f"{args.split_label}"
    )

    print(
        f"📌 Training seed label: "
        f"{args.seed}"
    )

    print(
        f"📌 Horizons reported: "
        f"{requested_horizons}"
    )

    print(
        f"📌 Full curve: "
        f"h=1..{max_horizon}"
    )

    print(
        f"📌 dx={args.dx}, "
        f"dy={args.dy}"
    )

    print(
        f"📌 alpha_max="
        f"{args.alpha_max}"
    )

    print(
        f"📌 conditioner_hidden="
        f"{args.conditioner_hidden}"
    )

    for path in [
        split_path,
        stats_path,
        m6_path,
        *checkpoint_paths.values(),
    ]:
        if not os.path.exists(
            path
        ):
            raise FileNotFoundError(
                path
            )

    print(
        "\n========== "
        "PROVENANCE "
        "=========="
    )

    print(
        "SPLIT_SHA256:",
        sha256_file(
            split_path
        ),
    )

    print(
        "STATS_SHA256:",
        sha256_file(
            stats_path
        ),
    )

    print(
        "M6_SHA256:",
        sha256_file(
            m6_path
        ),
    )

    for (
        mode,
        path,
    ) in checkpoint_paths.items():

        print(
            f"{mode.upper()}_SHA256:",
            sha256_file(
                path
            ),
        )

    with open(
        split_path,
        "r",
        encoding="utf-8",
    ) as f:
        split_config = json.load(
            f
        )

    dataset = RolloutDataset(
        split_config=(
            split_config["test"]
        ),
        max_horizon=(
            max_horizon
        ),
        max_samples=(
            args.max_samples
        ),
    )

    loader = DataLoader(
        dataset,
        batch_size=(
            args.batch_size
        ),
        shuffle=False,
        num_workers=0,
    )

    normalizer = (
        FieldWiseNormalizer(
            stats_path
        )
        .to(device)
    )

    (
        field_mean,
        field_std,
    ) = build_field_stats(
        stats_path
    )

    (
        m6,
        _,
    ) = build_m6(
        m6_path,
        device,
    )

    models = {
        "M6": m6
    }

    for mode in (
        "static",
        "param",
        "state",
        "stateparam",
    ):
        (
            model,
            payload,
        ) = build_m10(
            mode=mode,
            checkpoint_path=(
                checkpoint_paths[
                    mode
                ]
            ),
            field_mean=(
                field_mean
            ),
            field_std=(
                field_std
            ),
            args=args,
            device=device,
        )

        display_name = (
            DISPLAY_NAMES[
                mode
            ]
        )

        assert_same_m6(
            m6,
            model,
            display_name,
        )

        models[
            display_name
        ] = model

        if isinstance(
            payload,
            dict,
        ):
            print(
                f"📌 {display_name}: "
                f"best_val="
                f"{payload.get(
                    'best_val_loss',
                    payload.get(
                        'val_loss',
                        'NA'
                    )
                )}, "
                f"epoch="
                f"{payload.get(
                    'best_epoch',
                    payload.get(
                        'epoch',
                        'NA'
                    )
                )}"
            )

    model_order = [
        "M6",
        "M10-Static",
        "M10-Param",
        "M10-State",
        "M10-StateParam",
    ]

    error_stats = {}
    gate_stats = {}

    print(
        "\n🔥 Starting full "
        "free-autoregressive "
        "test rollout..."
    )

    with torch.no_grad():
        for (
            batch_idx,
            (
                x0_phys,
                future_phys,
                param,
            ),
        ) in enumerate(
            loader,
            start=1,
        ):
            x0_phys = (
                x0_phys.to(
                    device,
                    non_blocking=True,
                )
            )

            future_phys = (
                future_phys.to(
                    device,
                    non_blocking=True,
                )
            )

            param = param.to(
                device,
                non_blocking=True,
            )

            for model_name in (
                model_order
            ):
                rollout_one_model(
                    model_name=(
                        model_name
                    ),
                    model=(
                        models[
                            model_name
                        ]
                    ),
                    x0_phys=(
                        x0_phys
                    ),
                    future_phys=(
                        future_phys
                    ),
                    param=(
                        param
                    ),
                    normalizer=(
                        normalizer
                    ),
                    max_horizon=(
                        max_horizon
                    ),
                    error_stats=(
                        error_stats
                    ),
                    gate_stats=(
                        gate_stats
                    ),
                )

            if (
                batch_idx % 10 == 0
                or
                batch_idx == len(
                    loader
                )
            ):
                print(
                    f"  processed batch "
                    f"{batch_idx}/"
                    f"{len(loader)}"
                )

    curve_df = pd.DataFrame(
        error_rows(
            error_stats,
            model_order,
            max_horizon,
        )
    )

    summary_df = curve_df[
        curve_df["horizon"]
        .isin(
            requested_horizons
        )
    ].copy()

    gates_df = pd.DataFrame(
        gate_rows(
            gate_stats,
            model_order,
            max_horizon,
        )
    )

    growth_df = (
        make_global_growth_table(
            curve_df,
            model_order,
            max_horizon,
        )
    )

    diff_df = (
        make_difference_table(
            summary_df
        )
    )

    for df in (
        curve_df,
        summary_df,
        gates_df,
        growth_df,
        diff_df,
    ):
        df.insert(
            0,
            "seed",
            args.seed,
        )

        df.insert(
            0,
            "split",
            args.split_label,
        )

    output_dir = resolve_path(
        args.output_dir
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    prefix = (
        f"m10_0_rollout_"
        f"{args.split_label}_"
        f"seed{args.seed}"
    )

    paths = {
        "summary": os.path.join(
            output_dir,
            f"{prefix}_summary.csv",
        ),

        "curve": os.path.join(
            output_dir,
            (
                f"{prefix}_"
                f"curve_h1_"
                f"h{max_horizon}.csv"
            ),
        ),

        "growth": os.path.join(
            output_dir,
            (
                f"{prefix}_"
                "global_growth_auc.csv"
            ),
        ),

        "gates": os.path.join(
            output_dir,
            (
                f"{prefix}_"
                "gate_curve.csv"
            ),
        ),

        "diff": os.path.join(
            output_dir,
            (
                f"{prefix}_"
                "differences.csv"
            ),
        ),
    }

    summary_df.to_csv(
        paths["summary"],
        index=False,
    )

    curve_df.to_csv(
        paths["curve"],
        index=False,
    )

    growth_df.to_csv(
        paths["growth"],
        index=False,
    )

    gates_df.to_csv(
        paths["gates"],
        index=False,
    )

    diff_df.to_csv(
        paths["diff"],
        index=False,
    )

    global_wide = (
        summary_df[
            summary_df["field"]
            == "global"
        ]
        .pivot_table(
            index="model",
            columns="horizon",
            values=(
                "rel_l2_percent"
            ),
            aggfunc="first",
        )
    )

    global_wide = (
        global_wide
        .reindex(
            model_order
        )
    )

    print(
        "\n================ "
        "GLOBAL REL-L2 (%) "
        "================"
    )

    print(
        global_wide
        .to_string()
    )

    key_diff = diff_df[
        (
            diff_df["comparison"]
            ==
            (
                "M10-StateParam "
                "- M10-Param"
            )
        )
        &
        (
            diff_df["field"]
            == "global"
        )
    ][
        [
            "horizon",
            (
                "rel_l2_diff_"
                "percent_point"
            ),
        ]
    ]

    print(
        "\n========== "
        "StateParam - Param "
        "| global Rel-L2 pp "
        "=========="
    )

    print(
        "Negative = "
        "StateParam better."
    )

    print(
        key_diff
        .to_string(
            index=False
        )
    )

    print(
        "\n================ "
        "GLOBAL ERROR "
        "GROWTH / AUC "
        "================"
    )

    print(
        growth_df.drop(
            columns=[
                "split",
                "seed",
            ]
        ).to_string(
            index=False
        )
    )

    print(
        "\n✅ Saved:"
    )

    for (
        key,
        path,
    ) in paths.items():
        print(
            f"  {key}: "
            f"{path}"
        )


if __name__ == "__main__":
    main()
