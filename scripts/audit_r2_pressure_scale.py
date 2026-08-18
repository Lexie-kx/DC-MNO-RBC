"""
R2-2D: Pressure-Term Scale Consistency Diagnosis
================================================

中文阶段：
    R2-2D 压力项尺度一致性诊断

Purpose
-------
Test whether the real-RBC momentum mismatch can plausibly be
explained by ONE scalar pressure convention factor.

Current canonical equations remain unchanged.

Horizontal momentum:
    du_x/dt =
        -u·grad(u_x)
        + nu lap(u_x)
        + alpha_p (-dp/dx)

Vertical momentum:
    du_y/dt =
        -u·grad(u_y)
        + nu lap(u_y)
        + b
        + alpha_p (-dp/dy)

We estimate alpha_p diagnostically.

Important
---------
The fitted alpha is NOT:
    - a PDE correction;
    - a trainable parameter;
    - a future loss weight;
    - evidence by itself of the original DNS convention.

TEST data are forbidden.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys

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
            "R2-2D pressure-term scale consistency diagnosis."
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


def resolve(path):

    if os.path.isabs(path):
        return path

    return os.path.abspath(
        os.path.join(
            PROJECT_ROOT,
            path,
        )
    )


def rms(x):

    x = x.double()

    return torch.sqrt(
        torch.mean(
            x * x
        )
    ).item()


def cosine(a, b):

    a = a.double().reshape(-1)
    b = b.double().reshape(-1)

    denom = (
        torch.linalg.vector_norm(a)
        *
        torch.linalg.vector_norm(b)
    )

    if denom.item() < 1.0e-30:
        return float("nan")

    return (
        torch.dot(a, b)
        /
        denom
    ).item()


def fit_scalar(
    target,
    basis,
):
    """
    Solve:

        target ~= alpha * basis

    in least-squares sense.
    """

    target = target.double().reshape(-1)
    basis = basis.double().reshape(-1)

    denominator = torch.dot(
        basis,
        basis,
    )

    if denominator.item() < 1.0e-30:
        return float("nan")

    return (
        torch.dot(
            target,
            basis,
        )
        /
        denominator
    ).item()


def balance(
    temporal,
    rhs,
):

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


def choose_indices(n, k):

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

    k = min(k, n)

    result = []

    for i in range(k):

        index = round(
            i
            *
            (n - 1)
            /
            (k - 1)
        )

        if index not in result:
            result.append(index)

    return result


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

    buoyancy = compiler.compile(
        "buoyancy_forcing",
        current,
    ).canonical_value

    return {
        "adv_x": adv_u[:, 0],
        "adv_y": adv_u[:, 1],

        "visc_x": visc_u[:, 0],
        "visc_y": visc_u[:, 1],

        "pressure_x": pressure[:, 0],
        "pressure_y": pressure[:, 1],

        "buoyancy": buoyancy,
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

    temporal_x = (
        result.temporal_rate_u[:, 0]
    )

    temporal_y = (
        result.temporal_rate_u[:, 1]
    )

    nonpressure_x = (
        terms["adv_x"]
        +
        terms["visc_x"]
    )

    nonpressure_y = (
        terms["adv_y"]
        +
        terms["visc_y"]
        +
        terms["buoyancy"]
    )

    pressure_x = terms[
        "pressure_x"
    ]

    pressure_y = terms[
        "pressure_y"
    ]

    needed_pressure_x = (
        temporal_x
        -
        nonpressure_x
    )

    needed_pressure_y = (
        temporal_y
        -
        nonpressure_y
    )

    alpha_x = fit_scalar(
        needed_pressure_x,
        pressure_x,
    )

    alpha_y = fit_scalar(
        needed_pressure_y,
        pressure_y,
    )

    rhs_x_alpha1 = (
        nonpressure_x
        +
        pressure_x
    )

    rhs_y_alpha1 = (
        nonpressure_y
        +
        pressure_y
    )

    rhs_x_fit = (
        nonpressure_x
        +
        alpha_x
        *
        pressure_x
    )

    rhs_y_fit = (
        nonpressure_y
        +
        alpha_y
        *
        pressure_y
    )

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

        "alpha_pressure_x": alpha_x,
        "alpha_pressure_y": alpha_y,
        "alpha_xy_abs_difference": abs(
            alpha_x
            -
            alpha_y
        ),

        "needed_pressure_x_rms": rms(
            needed_pressure_x
        ),

        "actual_pressure_x_rms": rms(
            pressure_x
        ),

        "needed_actual_pressure_x_cosine": cosine(
            needed_pressure_x,
            pressure_x,
        ),

        "needed_pressure_y_rms": rms(
            needed_pressure_y
        ),

        "actual_pressure_y_rms": rms(
            pressure_y
        ),

        "needed_actual_pressure_y_cosine": cosine(
            needed_pressure_y,
            pressure_y,
        ),

        "ux_balance_alpha1": balance(
            temporal_x,
            rhs_x_alpha1,
        ),

        "ux_balance_fit": balance(
            temporal_x,
            rhs_x_fit,
        ),

        "ux_rhs_cosine_alpha1": cosine(
            temporal_x,
            rhs_x_alpha1,
        ),

        "ux_rhs_cosine_fit": cosine(
            temporal_x,
            rhs_x_fit,
        ),

        "uy_balance_alpha1": balance(
            temporal_y,
            rhs_y_alpha1,
        ),

        "uy_balance_fit": balance(
            temporal_y,
            rhs_y_fit,
        ),

        "uy_rhs_cosine_alpha1": cosine(
            temporal_y,
            rhs_y_alpha1,
        ),

        "uy_rhs_cosine_fit": cosine(
            temporal_y,
            rhs_y_fit,
        ),
    }


def print_row(row):

    print()
    print(
        "-" * 86
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
        "压力系数拟合"
    )

    print(
        f"u_x : alpha_p="
        f"{row['alpha_pressure_x']:.6e} | "
        f"needed/actual cos="
        f"{row['needed_actual_pressure_x_cosine']:.6f}"
    )

    print(
        f"u_y : alpha_p="
        f"{row['alpha_pressure_y']:.6e} | "
        f"needed/actual cos="
        f"{row['needed_actual_pressure_y_cosine']:.6f}"
    )

    print(
        f"|alpha_x-alpha_y| = "
        f"{row['alpha_xy_abs_difference']:.6e}"
    )

    print()
    print(
        "固定 alpha=1 vs 每个方程最优 alpha"
    )

    print(
        "u_x : "
        f"balance 1={row['ux_balance_alpha1']:.6f} -> "
        f"fit={row['ux_balance_fit']:.6f} | "
        f"cos 1={row['ux_rhs_cosine_alpha1']:.4f} -> "
        f"fit={row['ux_rhs_cosine_fit']:.4f}"
    )

    print(
        "u_y : "
        f"balance 1={row['uy_balance_alpha1']:.6f} -> "
        f"fit={row['uy_balance_fit']:.6f} | "
        f"cos 1={row['uy_rhs_cosine_alpha1']:.4f} -> "
        f"fit={row['uy_rhs_cosine_fit']:.4f}"
    )


def finite_mean(values):

    vals = [
        float(x)
        for x in values
        if math.isfinite(float(x))
    ]

    if not vals:
        return float("nan")

    return sum(vals) / len(vals)


def finite_std(values):

    vals = [
        float(x)
        for x in values
        if math.isfinite(float(x))
    ]

    if len(vals) <= 1:
        return float("nan")

    mean = sum(vals) / len(vals)

    return math.sqrt(
        sum(
            (x - mean) ** 2
            for x in vals
        )
        /
        (len(vals) - 1)
    )


def print_summary(rows):

    print()
    print(
        "=" * 92
    )

    print(
        "R2-2D 汇总：压力项尺度一致性"
    )

    print(
        "=" * 92
    )

    keys = [
        "alpha_pressure_x",
        "alpha_pressure_y",
        "alpha_xy_abs_difference",

        "needed_actual_pressure_x_cosine",
        "needed_actual_pressure_y_cosine",

        "ux_balance_alpha1",
        "ux_balance_fit",

        "uy_balance_alpha1",
        "uy_balance_fit",

        "ux_rhs_cosine_alpha1",
        "ux_rhs_cosine_fit",

        "uy_rhs_cosine_alpha1",
        "uy_rhs_cosine_fit",
    ]

    for key in keys:

        values = [
            row[key]
            for row in rows
        ]

        print(
            f"{key:38s} | "
            f"mean={finite_mean(values):.6e} | "
            f"std={finite_std(values):.6e}"
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


def main():

    args = parse_args()

    if args.partition not in (
        "train",
        "val",
    ):
        raise RuntimeError(
            "R2-2D permits TRAIN / VAL only."
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

        split = json.load(f)

    entries = split[
        args.partition
    ][
        :args.max_groups
    ]

    if not entries:
        raise RuntimeError(
            "No legal split entries."
        )

    metadata = build_rbc_canonical_metadata(
        stats_path
    )

    residual_builder = CanonicalRBCResidual(
        metadata
    )

    print(
        "=" * 92
    )

    print(
        "R2-2D 压力项尺度一致性诊断"
    )

    print(
        "=" * 92
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
        "formal alpha_p    : 1.0"
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
                f"No trajectories in {group}"
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

            rows.append(row)

            print_row(row)

        del dataset
        gc.collect()

    if not rows:
        raise RuntimeError(
            "No rows were produced."
        )

    write_csv(
        rows,
        output_path,
    )

    print_summary(rows)

    print()
    print(
        f"CSV saved: {output_path}"
    )

    print()
    print(
        "=" * 92
    )

    print(
        "✅ R2-2D 压力项尺度一致性诊断运行完成"
    )

    print(
        "=" * 92
    )


if __name__ == "__main__":
    main()
