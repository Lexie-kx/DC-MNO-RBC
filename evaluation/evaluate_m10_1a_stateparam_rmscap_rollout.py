import argparse
import importlib.util
import math
import os
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT,
    )


# ============================================================
# Reuse the EXACT closed M10-0 rollout evaluator.
# ============================================================

BASE_EVAL_PATH = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "evaluate_cross_param_rollout_m10_2path.py",
)

spec = importlib.util.spec_from_file_location(
    "m10_0_rollout_eval",
    BASE_EVAL_PATH,
)

base = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    base
)


from models.operators.fno2d_m10_1_rmscap import (
    M10RMSCapTwoPathFNO2d,
)


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "M10-1a focused StateParam-RMSCap rollout control: "
            "M6 vs original M10-StateParam vs M10-1a StateParam-RMSCap."
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
        "--original_stateparam_checkpoint",
        required=True,
    )

    parser.add_argument(
        "--rmscap_stateparam_checkpoint",
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
        "--path_b_rms_cap",
        required=True,
        type=float,
    )

    parser.add_argument(
        "--path_b_rms_eps",
        type=float,
        default=1e-12,
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
        default="outputs/tables/m10_1a_rollout",
    )

    return parser.parse_args()


# ============================================================
# M10-1a checkpoint
# ============================================================

def build_rmscap_stateparam(
    checkpoint_path,
    field_mean,
    field_std,
    args,
    device,
):
    payload, state = base.load_checkpoint(
        checkpoint_path,
        device,
    )

    if isinstance(payload, dict):

        if (
            payload.get(
                "factorial_mode",
                "stateparam",
            )
            != "stateparam"
        ):
            raise RuntimeError(
                "M10-1a checkpoint is not StateParam mode."
            )

        if (
            "seed" in payload
            and int(payload["seed"])
            != args.seed
        ):
            raise RuntimeError(
                "M10-1a checkpoint seed mismatch."
            )

        for key, expected in [
            ("dx", args.dx),
            ("dy", args.dy),
            ("alpha_max", args.alpha_max),
        ]:
            if (
                key in payload
                and not math.isclose(
                    float(payload[key]),
                    float(expected),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise RuntimeError(
                    f"M10-1a checkpoint "
                    f"{key} mismatch."
                )

        stabilization = payload.get(
            "stabilization",
            {},
        )

        if stabilization:

            stored_cap = float(
                stabilization[
                    "path_b_rms_cap"
                ]
            )

            if not math.isclose(
                stored_cap,
                args.path_b_rms_cap,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    "M10-1a Path-B RMS cap "
                    "metadata mismatch."
                )

            if not stabilization.get(
                "state_conditioner_uses_raw_path_b",
                False,
            ):
                raise RuntimeError(
                    "Unexpected M10-1a "
                    "conditioner contract."
                )

    model = M10RMSCapTwoPathFNO2d(
        field_mean=field_mean,
        field_std=field_std,
        mode="stateparam",
        dx=args.dx,
        dy=args.dy,
        path_b_rms_cap=(
            args.path_b_rms_cap
        ),
        path_b_rms_eps=(
            args.path_b_rms_eps
        ),
        in_channels=16,
        out_channels=4,
        modes1=16,
        modes2=16,
        width=32,
        alpha_max=args.alpha_max,
        conditioner_hidden=(
            args.conditioner_hidden
        ),
        freeze_m6=True,
    ).to(device)

    model.load_state_dict(
        state,
        strict=True,
    )

    model.eval()

    return (
        model,
        payload,
    )


# ============================================================
# Cap diagnostics
# ============================================================

def new_cap_bucket():
    return {
        "n": 0,
        "active": 0,
        "raw_sum": 0.0,
        "raw_max": float("-inf"),
        "safe_sum": 0.0,
        "safe_max": float("-inf"),
        "scale_sum": 0.0,
        "scale_min": float("inf"),
    }


def update_cap_stats(
    cap_stats,
    step,
    info,
):
    if step not in cap_stats:
        cap_stats[step] = (
            new_cap_bucket()
        )

    bucket = cap_stats[step]

    raw = (
        info["path_b_rms_raw"]
        .detach()
        .double()
        .reshape(-1)
    )

    safe = (
        info["path_b_rms_safe"]
        .detach()
        .double()
        .reshape(-1)
    )

    scale = (
        info["path_b_scale"]
        .detach()
        .double()
        .reshape(-1)
    )

    active = (
        info["path_b_cap_active"]
        .detach()
        .reshape(-1)
    )

    bucket["n"] += raw.numel()

    bucket["active"] += int(
        active.sum().item()
    )

    bucket["raw_sum"] += float(
        raw.sum().item()
    )

    bucket["raw_max"] = max(
        bucket["raw_max"],
        float(
            raw.max().item()
        ),
    )

    bucket["safe_sum"] += float(
        safe.sum().item()
    )

    bucket["safe_max"] = max(
        bucket["safe_max"],
        float(
            safe.max().item()
        ),
    )

    bucket["scale_sum"] += float(
        scale.sum().item()
    )

    bucket["scale_min"] = min(
        bucket["scale_min"],
        float(
            scale.min().item()
        ),
    )


def cap_rows(
    cap_stats,
    max_horizon,
):
    rows = []

    for step in range(
        1,
        max_horizon + 1,
    ):
        b = cap_stats[step]
        n = b["n"]

        rows.append(
            {
                "model": (
                    "M10-1a-StateParam-RMSCap"
                ),
                "horizon": step,
                "n": n,
                "cap_active_fraction": (
                    b["active"] / n
                ),
                "path_b_rms_raw_mean": (
                    b["raw_sum"] / n
                ),
                "path_b_rms_raw_max": (
                    b["raw_max"]
                ),
                "path_b_rms_safe_mean": (
                    b["safe_sum"] / n
                ),
                "path_b_rms_safe_max": (
                    b["safe_max"]
                ),
                "path_b_scale_mean": (
                    b["scale_sum"] / n
                ),
                "path_b_scale_min": (
                    b["scale_min"]
                ),
            }
        )

    return rows


# ============================================================
# Exact same rollout logic + cap diagnostics
# ============================================================

def rollout_rmscap(
    model,
    x0_phys,
    future_phys,
    param,
    normalizer,
    max_horizon,
    error_stats,
    gate_stats,
    cap_stats,
):
    model_name = (
        "M10-1a-StateParam-RMSCap"
    )

    x_norm = normalizer.normalize_x(
        x0_phys
    )

    for step in range(
        1,
        max_horizon + 1,
    ):

        current_norm = x_norm[
            :,
            -4:,
            :,
            :,
        ]

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
            normalizer.denormalize_y(
                pred_next_norm
            )
        )

        true_phys = future_phys[
            :,
            step - 1,
        ]

        base.update_error_stats(
            error_stats,
            model_name,
            step,
            pred_next_phys,
            true_phys,
        )

        base.update_gate_stats(
            gate_stats,
            model_name,
            step,
            info,
        )

        update_cap_stats(
            cap_stats,
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


# ============================================================
# Controlled difference table
# ============================================================

def make_difference_table(
    summary_df,
):
    pairs = [
        (
            "M10-1a-StateParam-RMSCap",
            "M10-StateParam",
        ),
        (
            "M10-1a-StateParam-RMSCap",
            "M6",
        ),
        (
            "M10-StateParam",
            "M6",
        ),
    ]

    rows = []

    for model_a, model_b in pairs:

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

        for _, row in merged.iterrows():

            rows.append(
                {
                    "comparison": (
                        f"{model_a} - "
                        f"{model_b}"
                    ),
                    "horizon": int(
                        row["horizon"]
                    ),
                    "field": row[
                        "field"
                    ],
                    "rel_l2_diff_percent_point": (
                        row[
                            "rel_l2_percent_a"
                        ]
                        - row[
                            "rel_l2_percent_b"
                        ]
                    ),
                    "mse_diff": (
                        row["mse_a"]
                        - row["mse_b"]
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    requested_horizons = sorted(
        {
            int(x)
            for x
            in args.horizons.split(",")
            if x.strip()
        }
    )

    max_horizon = max(
        requested_horizons
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    split_path = base.resolve_path(
        args.split
    )

    stats_path = base.resolve_path(
        args.stats
    )

    m6_path = base.resolve_path(
        args.m6_checkpoint
    )

    original_stateparam_path = (
        base.resolve_path(
            args.original_stateparam_checkpoint
        )
    )

    rmscap_path = base.resolve_path(
        args.rmscap_stateparam_checkpoint
    )

    for path in [
        split_path,
        stats_path,
        m6_path,
        original_stateparam_path,
        rmscap_path,
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                path
            )

    print(
        "🚀 [M10-1a Param RMSCap "
        "Focused Test Rollout]"
    )

    print(
        "📌 Device:",
        device,
    )

    print(
        "📌 Split:",
        args.split_label,
    )

    print(
        "📌 Seed:",
        args.seed,
    )

    print(
        "📌 Horizons:",
        requested_horizons,
    )

    print(
        "📌 Path-B RMS cap:",
        args.path_b_rms_cap,
    )

    print()
    print(
        "========== PROVENANCE =========="
    )

    print(
        "SPLIT_SHA256:",
        base.sha256_file(
            split_path
        ),
    )

    print(
        "STATS_SHA256:",
        base.sha256_file(
            stats_path
        ),
    )

    print(
        "M6_SHA256:",
        base.sha256_file(
            m6_path
        ),
    )

    print(
        "ORIGINAL_STATEPARAM_SHA256:",
        base.sha256_file(
            original_stateparam_path
        ),
    )

    print(
        "RMSCAP_STATEPARAM_SHA256:",
        base.sha256_file(
            rmscap_path
        ),
    )

    with open(
        split_path,
        "r",
        encoding="utf-8",
    ) as f:
        import json
        split_config = json.load(f)

    dataset = base.RolloutDataset(
        split_config=(
            split_config["test"]
        ),
        max_horizon=max_horizon,
        max_samples=args.max_samples,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    normalizer = (
        base.FieldWiseNormalizer(
            stats_path
        ).to(device)
    )

    (
        field_mean,
        field_std,
    ) = base.build_field_stats(
        stats_path
    )

    # --------------------------------------------------------
    # M6
    # --------------------------------------------------------

    (
        m6,
        _,
    ) = base.build_m6(
        m6_path,
        device,
    )

    # --------------------------------------------------------
    # Original M10-StateParam
    # --------------------------------------------------------

    (
        original_stateparam,
        original_payload,
    ) = base.build_m10(
        mode="stateparam",
        checkpoint_path=(
            original_stateparam_path
        ),
        field_mean=field_mean,
        field_std=field_std,
        args=args,
        device=device,
    )

    base.assert_same_m6(
        m6,
        original_stateparam,
        "M10-StateParam",
    )

    # --------------------------------------------------------
    # M10-1a Param RMSCap
    # --------------------------------------------------------

    (
        rmscap_stateparam,
        rmscap_payload,
    ) = build_rmscap_stateparam(
        checkpoint_path=(
            rmscap_path
        ),
        field_mean=field_mean,
        field_std=field_std,
        args=args,
        device=device,
    )

    base.assert_same_m6(
        m6,
        rmscap_stateparam,
        "M10-1a-StateParam-RMSCap",
    )

    print(
        "📌 Original StateParam best_val:",
        original_payload.get(
            "best_val_loss",
            original_payload.get(
                "val_loss",
                "NA",
            ),
        )
        if isinstance(
            original_payload,
            dict,
        )
        else "NA",
    )

    print(
        "📌 RMSCap StateParam best_val:",
        rmscap_payload.get(
            "best_val_loss",
            rmscap_payload.get(
                "val_loss",
                "NA",
            ),
        )
        if isinstance(
            rmscap_payload,
            dict,
        )
        else "NA",
    )

    model_order = [
        "M6",
        "M10-StateParam",
        "M10-1a-StateParam-RMSCap",
    ]

    error_stats = {}
    gate_stats = {}
    cap_stats = {}

    print()
    print(
        "🔥 Starting full "
        "free-autoregressive rollout..."
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

            x0_phys = x0_phys.to(
                device,
                non_blocking=True,
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

            base.rollout_one_model(
                model_name="M6",
                model=m6,
                x0_phys=x0_phys,
                future_phys=future_phys,
                param=param,
                normalizer=normalizer,
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

            base.rollout_one_model(
                model_name="M10-StateParam",
                model=original_stateparam,
                x0_phys=x0_phys,
                future_phys=future_phys,
                param=param,
                normalizer=normalizer,
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

            rollout_rmscap(
                model=rmscap_stateparam,
                x0_phys=x0_phys,
                future_phys=future_phys,
                param=param,
                normalizer=normalizer,
                max_horizon=(
                    max_horizon
                ),
                error_stats=(
                    error_stats
                ),
                gate_stats=(
                    gate_stats
                ),
                cap_stats=(
                    cap_stats
                ),
            )

            if (
                batch_idx % 10 == 0
                or batch_idx
                == len(loader)
            ):
                print(
                    f"  processed batch "
                    f"{batch_idx}/"
                    f"{len(loader)}"
                )

    # ========================================================
    # Tables
    # ========================================================

    curve_df = pd.DataFrame(
        base.error_rows(
            error_stats,
            model_order,
            max_horizon,
        )
    )

    summary_df = curve_df[
        curve_df["horizon"].isin(
            requested_horizons
        )
    ].copy()

    gates_df = pd.DataFrame(
        base.gate_rows(
            gate_stats,
            model_order,
            max_horizon,
        )
    )

    cap_df = pd.DataFrame(
        cap_rows(
            cap_stats,
            max_horizon,
        )
    )

    growth_df = (
        base.make_global_growth_table(
            curve_df,
            model_order,
            max_horizon,
        )
    )

    diff_df = make_difference_table(
        summary_df
    )

    for df in [
        curve_df,
        summary_df,
        gates_df,
        cap_df,
        growth_df,
        diff_df,
    ]:
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

    output_dir = (
        base.resolve_path(
            args.output_dir
        )
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    prefix = (
        f"m10_1a_stateparam_rmscap_"
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
            f"{prefix}_curve_h1_h"
            f"{max_horizon}.csv",
        ),
        "growth": os.path.join(
            output_dir,
            f"{prefix}_growth_auc.csv",
        ),
        "gates": os.path.join(
            output_dir,
            f"{prefix}_gates.csv",
        ),
        "cap": os.path.join(
            output_dir,
            f"{prefix}_cap_curve.csv",
        ),
        "diff": os.path.join(
            output_dir,
            f"{prefix}_differences.csv",
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

    cap_df.to_csv(
        paths["cap"],
        index=False,
    )

    diff_df.to_csv(
        paths["diff"],
        index=False,
    )

    # ========================================================
    # Compact terminal output
    # ========================================================

    global_wide = summary_df[
        summary_df["field"]
        == "global"
    ].pivot_table(
        index="model",
        columns="horizon",
        values="rel_l2_percent",
        aggfunc="first",
    )

    global_wide = (
        global_wide.reindex(
            model_order
        )
    )

    print()
    print(
        "================ "
        "GLOBAL REL-L2 (%) "
        "================"
    )

    print(
        global_wide.to_string()
    )

    key_diff = diff_df[
        (
            diff_df["comparison"]
            ==
            "M10-1a-StateParam-RMSCap "
            "- M10-StateParam"
        )
        &
        (
            diff_df["field"]
            == "global"
        )
    ][
        [
            "horizon",
            "rel_l2_diff_percent_point",
        ]
    ]

    print()
    print(
        "========== RMSCap - "
        "Original StateParam | "
        "global Rel-L2 pp =========="
    )

    print(
        "Negative = RMSCap better."
    )

    print(
        key_diff.to_string(
            index=False
        )
    )

    cap_key = cap_df[
        cap_df["horizon"].isin(
            requested_horizons
        )
    ][
        [
            "horizon",
            "cap_active_fraction",
            "path_b_rms_raw_mean",
            "path_b_rms_raw_max",
            "path_b_rms_safe_max",
            "path_b_scale_min",
        ]
    ]

    print()
    print(
        "================ "
        "RMS CAP ACTIVITY "
        "================"
    )

    print(
        cap_key.to_string(
            index=False
        )
    )

    print()
    print(
        "================ "
        "GLOBAL ERROR GROWTH / AUC "
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

    print()
    print("✅ Saved:")

    for key, path in paths.items():
        print(
            f"  {key}: {path}"
        )


if __name__ == "__main__":
    main()
