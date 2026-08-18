"""
R2-2B: Empirical Equation-Convention Diagnosis
==============================================

中文阶段：
    R2-2B 真实数据方程形式诊断

Purpose
-------
Diagnose why real RBC ground-truth states show substantial
discrete PDE residuals, especially in vertical momentum.

This script compares diagnostics only.

It does NOT:
    - modify the canonical PDE implementation;
    - train any model;
    - select a final governing equation;
    - tune PDE-loss weights;
    - access TEST data.

Diagnostics
-----------
1. Vertical buoyancy forcing:
       full b
   versus
       b' = b - <b>_x

2. Pressure/buoyancy cancellation:
       -dp/dy + b
   versus
       -dp/dy + b'

3. Divergence:
       full domain
       interior y
       y-boundary only

4. Delta/RHS directional consistency and fitted effective dt.

Important
---------
A better diagnostic score for one candidate is evidence only.
It is NOT proof that the original DNS used that convention.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
from typing import List

import torch


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


from datasets.rbc_dataset import RBCDataset
from physics.canonical_metadata import (
    build_rbc_canonical_metadata,
)
from physics.rbc_residuals import (
    CanonicalRBCResidual,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "R2-2B empirical RBC equation-convention diagnosis."
        )
    )

    parser.add_argument(
        "--split",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--stats",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--partition",
        required=True,
        choices=("train", "val"),
    )

    parser.add_argument(
        "--max_groups",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--samples_per_group",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--output",
        required=True,
        type=str,
    )

    return parser.parse_args()


def resolve(path: str) -> str:
    if os.path.isabs(path):
        return path

    return os.path.abspath(
        os.path.join(
            PROJECT_ROOT,
            path,
        )
    )


def rms(x: torch.Tensor) -> float:
    x = x.double()

    return torch.sqrt(
        torch.mean(
            x * x
        )
    ).item()


def cosine(
    a: torch.Tensor,
    b: torch.Tensor,
) -> float:
    a = a.double().reshape(-1)
    b = b.double().reshape(-1)

    na = torch.linalg.vector_norm(a)
    nb = torch.linalg.vector_norm(b)

    denom = na * nb

    if denom.item() < 1.0e-30:
        return float("nan")

    return (
        torch.dot(a, b)
        /
        denom
    ).item()


def fitted_dt(
    delta: torch.Tensor,
    rhs_rate: torch.Tensor,
) -> float:
    """
    Least-squares scalar dt satisfying

        delta ~= dt * rhs_rate.

    Diagnostic only.
    """

    delta = delta.double().reshape(-1)
    rhs_rate = rhs_rate.double().reshape(-1)

    denominator = torch.dot(
        rhs_rate,
        rhs_rate,
    )

    if denominator.item() < 1.0e-30:
        return float("nan")

    return (
        torch.dot(
            delta,
            rhs_rate,
        )
        /
        denominator
    ).item()


def balance_ratio(
    temporal: torch.Tensor,
    rhs: torch.Tensor,
) -> float:
    residual = temporal - rhs

    return (
        rms(residual)
        /
        (
            rms(temporal)
            +
            rms(rhs)
            +
            1.0e-30
        )
    )


def cancellation_ratio(
    a: torch.Tensor,
    b: torch.Tensor,
) -> float:
    """
    Small value means stronger cancellation between a and b.
    """

    return (
        rms(a + b)
        /
        (
            rms(a)
            +
            rms(b)
            +
            1.0e-30
        )
    )


def choose_indices(
    n: int,
    k: int,
) -> List[int]:

    if n <= 0:
        raise ValueError(
            "Dataset is empty."
        )

    if k <= 0:
        raise ValueError(
            "samples_per_group must be positive."
        )

    if k == 1:
        return [0]

    k = min(
        k,
        n,
    )

    indices = []

    for i in range(k):
        index = round(
            i
            *
            (n - 1)
            /
            (k - 1)
        )

        if index not in indices:
            indices.append(
                index
            )

    return indices


def compile_terms(
    residual_builder,
    x_norm,
    param,
):
    compiler = residual_builder.compiler

    current = (
        compiler
        .canonicalizer
        .latest_state_canonical(
            x_norm
        )
    )

    adv_b = compiler.compile(
        "buoyancy_advection",
        current,
    ).canonical_value

    diff_b = compiler.compile(
        "buoyancy_diffusion",
        current,
        param=param,
    ).canonical_value

    adv_u = compiler.compile(
        "momentum_advection",
        current,
    ).canonical_value

    visc_u = compiler.compile(
        "viscosity",
        current,
        param=param,
    ).canonical_value

    pressure = compiler.compile(
        "pressure_gradient",
        current,
    ).canonical_value

    full_b_force = compiler.compile(
        "buoyancy_forcing",
        current,
    ).canonical_value

    # Current canonical buoyancy field.
    b = current[:, 0]

    # x is the horizontal / periodic axis, tensor dim = -2.
    b_mean_x = torch.mean(
        b,
        dim=-2,
        keepdim=True,
    )

    b_anomaly = (
        b
        -
        b_mean_x
    )

    return {
        "current": current,

        "rhs_b": (
            adv_b
            +
            diff_b
        ),

        "rhs_ux": (
            adv_u[:, 0]
            +
            visc_u[:, 0]
            +
            pressure[:, 0]
        ),

        "adv_uy": adv_u[:, 1],
        "visc_uy": visc_u[:, 1],
        "pressure_y": pressure[:, 1],

        "full_b_force": full_b_force,
        "b_anomaly_force": b_anomaly,
    }


def diagnose_sample(
    *,
    residual_builder,
    x_norm,
    y_norm,
    param,
    group,
    trajectory,
    sample_index,
):

    result = residual_builder.compute_from_history(
        x_norm,
        y_norm,
        param,
    )

    terms = compile_terms(
        residual_builder,
        x_norm,
        param,
    )

    dt_nominal = (
        residual_builder
        .metadata
        .time
        .dt
    )

    temporal_b = (
        result.temporal_rate_b
    )

    temporal_ux = (
        result.temporal_rate_u[:, 0]
    )

    temporal_uy = (
        result.temporal_rate_u[:, 1]
    )

    delta_b = (
        dt_nominal
        *
        temporal_b
    )

    delta_ux = (
        dt_nominal
        *
        temporal_ux
    )

    delta_uy = (
        dt_nominal
        *
        temporal_uy
    )

    rhs_b = terms[
        "rhs_b"
    ]

    rhs_ux = terms[
        "rhs_ux"
    ]

    vertical_core = (
        terms["adv_uy"]
        +
        terms["visc_uy"]
        +
        terms["pressure_y"]
    )

    rhs_uy_full = (
        vertical_core
        +
        terms["full_b_force"]
    )

    rhs_uy_anomaly = (
        vertical_core
        +
        terms["b_anomaly_force"]
    )

    residual_uy_full = (
        temporal_uy
        -
        rhs_uy_full
    )

    residual_uy_anomaly = (
        temporal_uy
        -
        rhs_uy_anomaly
    )

    full_residual_rms = rms(
        residual_uy_full
    )

    anomaly_residual_rms = rms(
        residual_uy_anomaly
    )

    anomaly_improvement = (
        (
            full_residual_rms
            -
            anomaly_residual_rms
        )
        /
        (
            full_residual_rms
            +
            1.0e-30
        )
    )

    # ============================================================
    # Divergence localization
    # ============================================================

    div = result.divergence

    if div.shape[-1] < 3:
        raise RuntimeError(
            "Need at least 3 y-points."
        )

    div_interior = (
        div[..., 1:-1]
    )

    div_boundary = torch.cat(
        [
            div[..., :1],
            div[..., -1:],
        ],
        dim=-1,
    )

    full_div_energy = torch.sum(
        div.double() ** 2
    ).item()

    boundary_div_energy = torch.sum(
        div_boundary.double() ** 2
    ).item()

    boundary_energy_fraction = (
        boundary_div_energy
        /
        (
            full_div_energy
            +
            1.0e-30
        )
    )

    # ============================================================
    # Pressure / buoyancy cancellation
    # ============================================================

    pressure_y = terms[
        "pressure_y"
    ]

    full_b_force = terms[
        "full_b_force"
    ]

    anomaly_b_force = terms[
        "b_anomaly_force"
    ]

    # ============================================================
    # Parameter values
    # ============================================================

    log_ra = float(
        param[0, 0].item()
    )

    log_pr = float(
        param[0, 1].item()
    )

    return {
        "group": group,
        "trajectory": trajectory,
        "sample_index": sample_index,

        "ra": 10.0 ** log_ra,
        "pr": 10.0 ** log_pr,

        # --------------------------------------------------------
        # b / ux baseline consistency
        # --------------------------------------------------------

        "b_delta_rhs_cosine": cosine(
            delta_b,
            rhs_b,
        ),

        "b_fitted_dt": fitted_dt(
            delta_b,
            rhs_b,
        ),

        "ux_delta_rhs_cosine": cosine(
            delta_ux,
            rhs_ux,
        ),

        "ux_fitted_dt": fitted_dt(
            delta_ux,
            rhs_ux,
        ),

        # --------------------------------------------------------
        # Vertical momentum
        # --------------------------------------------------------

        "uy_temporal_rms": rms(
            temporal_uy
        ),

        "uy_rhs_full_rms": rms(
            rhs_uy_full
        ),

        "uy_rhs_anomaly_rms": rms(
            rhs_uy_anomaly
        ),

        "uy_residual_full_rms": (
            full_residual_rms
        ),

        "uy_residual_anomaly_rms": (
            anomaly_residual_rms
        ),

        "uy_balance_full": balance_ratio(
            temporal_uy,
            rhs_uy_full,
        ),

        "uy_balance_anomaly": balance_ratio(
            temporal_uy,
            rhs_uy_anomaly,
        ),

        "uy_anomaly_improvement": (
            anomaly_improvement
        ),

        "uy_full_delta_rhs_cosine": cosine(
            delta_uy,
            rhs_uy_full,
        ),

        "uy_anomaly_delta_rhs_cosine": cosine(
            delta_uy,
            rhs_uy_anomaly,
        ),

        "uy_full_fitted_dt": fitted_dt(
            delta_uy,
            rhs_uy_full,
        ),

        "uy_anomaly_fitted_dt": fitted_dt(
            delta_uy,
            rhs_uy_anomaly,
        ),

        # --------------------------------------------------------
        # Pressure / buoyancy cancellation
        # --------------------------------------------------------

        "pressure_y_rms": rms(
            pressure_y
        ),

        "buoyancy_full_rms": rms(
            full_b_force
        ),

        "buoyancy_anomaly_rms": rms(
            anomaly_b_force
        ),

        "pressure_full_b_cosine": cosine(
            pressure_y,
            full_b_force,
        ),

        "pressure_anomaly_b_cosine": cosine(
            pressure_y,
            anomaly_b_force,
        ),

        "pressure_full_b_cancellation": (
            cancellation_ratio(
                pressure_y,
                full_b_force,
            )
        ),

        "pressure_anomaly_b_cancellation": (
            cancellation_ratio(
                pressure_y,
                anomaly_b_force,
            )
        ),

        # --------------------------------------------------------
        # Divergence localization
        # --------------------------------------------------------

        "div_full_rms": rms(
            div
        ),

        "div_interior_rms": rms(
            div_interior
        ),

        "div_boundary_rms": rms(
            div_boundary
        ),

        "div_boundary_energy_fraction": (
            boundary_energy_fraction
        ),
    }


def print_row(row):
    print()
    print(
        "-" * 78
    )

    print(
        f"{row['group']} | "
        f"trajectory={row['trajectory']} | "
        f"sample={row['sample_index']}"
    )

    print(
        f"Ra={row['ra']:.6e} | "
        f"Pr={row['pr']:.8f}"
    )

    print()
    print(
        "1) b / u_x: delta 与 PDE-RHS 的方向"
    )

    print(
        "b   : "
        f"cos={row['b_delta_rhs_cosine']:.6f} | "
        f"fitted_dt={row['b_fitted_dt']:.6e}"
    )

    print(
        "u_x : "
        f"cos={row['ux_delta_rhs_cosine']:.6f} | "
        f"fitted_dt={row['ux_fitted_dt']:.6e}"
    )

    print()
    print(
        "2) u_y: full buoyancy vs horizontal anomaly"
    )

    print(
        "full b    : "
        f"rhs={row['uy_rhs_full_rms']:.6e} | "
        f"res={row['uy_residual_full_rms']:.6e} | "
        f"balance={row['uy_balance_full']:.6e} | "
        f"cos={row['uy_full_delta_rhs_cosine']:.6f} | "
        f"fit_dt={row['uy_full_fitted_dt']:.6e}"
    )

    print(
        "b anomaly : "
        f"rhs={row['uy_rhs_anomaly_rms']:.6e} | "
        f"res={row['uy_residual_anomaly_rms']:.6e} | "
        f"balance={row['uy_balance_anomaly']:.6e} | "
        f"cos={row['uy_anomaly_delta_rhs_cosine']:.6f} | "
        f"fit_dt={row['uy_anomaly_fitted_dt']:.6e}"
    )

    print(
        "anomaly residual improvement = "
        f"{100.0 * row['uy_anomaly_improvement']:.3f}%"
    )

    print()
    print(
        "3) pressure / buoyancy cancellation"
    )

    print(
        "full b    : "
        f"p_rms={row['pressure_y_rms']:.6e} | "
        f"b_rms={row['buoyancy_full_rms']:.6e} | "
        f"cos={row['pressure_full_b_cosine']:.6f} | "
        f"cancel_ratio={row['pressure_full_b_cancellation']:.6e}"
    )

    print(
        "b anomaly : "
        f"b_rms={row['buoyancy_anomaly_rms']:.6e} | "
        f"cos={row['pressure_anomaly_b_cosine']:.6f} | "
        f"cancel_ratio={row['pressure_anomaly_b_cancellation']:.6e}"
    )

    print()
    print(
        "4) divergence: full / interior / boundary"
    )

    print(
        f"full={row['div_full_rms']:.6e} | "
        f"interior={row['div_interior_rms']:.6e} | "
        f"boundary={row['div_boundary_rms']:.6e} | "
        f"boundary_energy_fraction="
        f"{row['div_boundary_energy_fraction']:.6f}"
    )


def write_csv(
    rows,
    output_path,
):
    os.makedirs(
        os.path.dirname(
            output_path
        ),
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def finite_mean(values):
    values = [
        x
        for x in values
        if math.isfinite(x)
    ]

    if not values:
        return float("nan")

    return (
        sum(values)
        /
        len(values)
    )


def print_summary(rows):
    print()
    print(
        "=" * 78
    )
    print(
        "R2-2B AGGREGATE SUMMARY"
    )
    print(
        "=" * 78
    )

    keys = [
        "b_delta_rhs_cosine",
        "b_fitted_dt",
        "ux_delta_rhs_cosine",
        "ux_fitted_dt",

        "uy_balance_full",
        "uy_balance_anomaly",
        "uy_anomaly_improvement",

        "uy_full_delta_rhs_cosine",
        "uy_anomaly_delta_rhs_cosine",

        "uy_full_fitted_dt",
        "uy_anomaly_fitted_dt",

        "pressure_full_b_cosine",
        "pressure_anomaly_b_cosine",

        "pressure_full_b_cancellation",
        "pressure_anomaly_b_cancellation",

        "div_full_rms",
        "div_interior_rms",
        "div_boundary_rms",
        "div_boundary_energy_fraction",
    ]

    for key in keys:
        values = [
            float(
                row[key]
            )
            for row in rows
        ]

        print(
            f"{key:38s} | "
            f"mean={finite_mean(values):.6e}"
        )


def main():
    args = parse_args()

    if args.partition not in (
        "train",
        "val",
    ):
        raise RuntimeError(
            "R2-2B permits TRAIN / VAL only."
        )

    if args.max_groups <= 0:
        raise ValueError(
            "max_groups must be positive."
        )

    if args.samples_per_group <= 0:
        raise ValueError(
            "samples_per_group must be positive."
        )

    split_path = resolve(
        args.split
    )

    stats_path = resolve(
        args.stats
    )

    output_path = resolve(
        args.output
    )

    with open(
        split_path,
        "r",
        encoding="utf-8",
    ) as f:
        split = json.load(
            f
        )

    entries = split[
        args.partition
    ][
        : args.max_groups
    ]

    if not entries:
        raise RuntimeError(
            "No legal split entries selected."
        )

    metadata = build_rbc_canonical_metadata(
        stats_path
    )

    residual_builder = CanonicalRBCResidual(
        metadata
    )

    print(
        "=" * 78
    )
    print(
        "R2-2B 真实数据方程形式诊断"
    )
    print(
        "=" * 78
    )

    print(
        f"partition         : {args.partition}"
    )

    print(
        f"max_groups        : {args.max_groups}"
    )

    print(
        f"samples_per_group : {args.samples_per_group}"
    )

    print(
        f"nominal_dt        : {metadata.time.dt}"
    )

    print(
        "TEST              : FORBIDDEN / NOT INSTANTIATED"
    )

    rows = []

    for entry in entries:

        group = entry[
            "group"
        ]

        trajectories = entry[
            "trajectories"
        ]

        if not trajectories:
            raise RuntimeError(
                f"No trajectory in {group}"
            )

        trajectory = int(
            trajectories[0]
        )

        tiny_split = [
            {
                "group": group,
                "trajectories": [
                    trajectory,
                ],
            }
        ]

        dataset = RBCDataset(
            split_config=tiny_split,
            normalize=True,
            stats_path=stats_path,
            return_params=True,
        )

        indices = choose_indices(
            len(dataset),
            args.samples_per_group,
        )

        for sample_index in indices:

            x_norm, y_norm, param = dataset[
                sample_index
            ]

            x_norm = torch.as_tensor(
                x_norm,
                dtype=torch.float64,
            ).unsqueeze(0)

            y_norm = torch.as_tensor(
                y_norm,
                dtype=torch.float64,
            ).unsqueeze(0)

            param = torch.as_tensor(
                param,
                dtype=torch.float64,
            ).unsqueeze(0)

            row = diagnose_sample(
                residual_builder=residual_builder,
                x_norm=x_norm,
                y_norm=y_norm,
                param=param,
                group=group,
                trajectory=trajectory,
                sample_index=sample_index,
            )

            rows.append(
                row
            )

            print_row(
                row
            )

        del dataset
        gc.collect()

    write_csv(
        rows,
        output_path,
    )

    print_summary(
        rows
    )

    print()
    print(
        f"CSV saved: {output_path}"
    )

    print()
    print(
        "=" * 78
    )
    print(
        "✅ R2-2B DIAGNOSTIC RUN COMPLETED"
    )
    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()
