"""
R2-2C: Spatial Direction and Discrete-Operator Diagnosis
========================================================

中文阶段：
    R2-2C 空间方向与离散算子诊断

Purpose
-------
Diagnose whether the real-RBC residual mismatch can be explained by:

1. first-derivative coordinate-direction convention;
2. relative sign between x/y divergence components;
3. possible velocity-component / spatial-axis mismatch.

Important
---------
This is a diagnostic experiment only.

It does NOT:
    - modify physics/derivatives.py;
    - modify the canonical PDE;
    - train a model;
    - choose a governing equation;
    - access TEST data.

The existing R1 derivative backend is reused unchanged.
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

from physics.derivatives import (
    grad_x_periodic,
    grad_y_nonperiodic,
    laplacian,
)


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "R2-2C spatial direction / discrete operator diagnosis."
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
        choices=(
            "train",
            "val",
        ),
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


def cosine(
    a,
    b,
):

    a = a.double().reshape(-1)
    b = b.double().reshape(-1)

    na = torch.linalg.vector_norm(
        a
    )

    nb = torch.linalg.vector_norm(
        b
    )

    denominator = (
        na
        *
        nb
    )

    if denominator.item() < 1.0e-30:
        return float("nan")

    return (
        torch.dot(
            a,
            b,
        )
        /
        denominator
    ).item()


def balance(
    temporal,
    rhs,
):

    residual = (
        temporal
        -
        rhs
    )

    return (
        rms(
            residual
        )
        /
        (
            rms(
                temporal
            )
            +
            rms(
                rhs
            )
            +
            1.0e-30
        )
    )


def fitted_relative_coefficient(
    a,
    b,
):
    """
    Find c minimizing

        || a + c b ||_2.

    For true incompressibility under the current convention:

        a = d(ux)/dx
        b = d(uy)/dy

    we expect roughly:

        c ~= 1

    if the relative sign and relative scale are correct.

    c ~= -1:
        relative derivative sign may be reversed.

    c far from +/-1:
        simple sign flip is insufficient and relative scaling /
        discretization may be inconsistent.
    """

    a = a.double().reshape(-1)
    b = b.double().reshape(-1)

    denominator = torch.dot(
        b,
        b,
    )

    if denominator.item() < 1.0e-30:
        return float("nan")

    return (
        -
        torch.dot(
            a,
            b,
        )
        /
        denominator
    ).item()


def choose_indices(
    n,
    k,
):

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
            result.append(
                index
            )

    return result


def parameter_coefficients(
    param,
):

    log_ra = param[
        0,
        0,
    ].double()

    log_pr = param[
        0,
        1,
    ].double()

    ra = (
        10.0
        **
        log_ra
    )

    pr = (
        10.0
        **
        log_pr
    )

    nu_nd = torch.sqrt(
        pr
        /
        ra
    )

    kappa_nd = (
        1.0
        /
        torch.sqrt(
            ra
            *
            pr
        )
    )

    return (
        float(
            ra.item()
        ),
        float(
            pr.item()
        ),
        nu_nd,
        kappa_nd,
    )


def get_current_state(
    residual_builder,
    x_norm,
):

    return (
        residual_builder
        .compiler
        .canonicalizer
        .latest_state_canonical(
            x_norm
        )
    )


def spatial_derivatives(
    current,
    grid,
):

    b = current[:, 0]
    ux = current[:, 1]
    uy = current[:, 2]
    p = current[:, 3]

    return {
        "b": b,
        "ux": ux,
        "uy": uy,
        "p": p,

        "db_dx": grad_x_periodic(
            b,
            grid,
        ),
        "db_dy": grad_y_nonperiodic(
            b,
            grid,
        ),

        "dux_dx": grad_x_periodic(
            ux,
            grid,
        ),
        "dux_dy": grad_y_nonperiodic(
            ux,
            grid,
        ),

        "duy_dx": grad_x_periodic(
            uy,
            grid,
        ),
        "duy_dy": grad_y_nonperiodic(
            uy,
            grid,
        ),

        "dp_dx": grad_x_periodic(
            p,
            grid,
        ),
        "dp_dy": grad_y_nonperiodic(
            p,
            grid,
        ),

        "lap_b": laplacian(
            b,
            grid,
        ),
        "lap_ux": laplacian(
            ux,
            grid,
        ),
        "lap_uy": laplacian(
            uy,
            grid,
        ),
    }


def make_rhs(
    d,
    nu_nd,
    kappa_nd,
    sx,
    sy,
):
    """
    Diagnostic first-derivative sign convention.

    sx = +1:
        current x first-derivative orientation.

    sx = -1:
        reversed x first-derivative orientation.

    sy analogous.

    Laplacians are unchanged because second derivatives are
    orientation-invariant under coordinate reversal.
    """

    db_dx = sx * d["db_dx"]
    db_dy = sy * d["db_dy"]

    dux_dx = sx * d["dux_dx"]
    dux_dy = sy * d["dux_dy"]

    duy_dx = sx * d["duy_dx"]
    duy_dy = sy * d["duy_dy"]

    dp_dx = sx * d["dp_dx"]
    dp_dy = sy * d["dp_dy"]

    rhs_b = (
        kappa_nd
        *
        d["lap_b"]
        -
        d["ux"]
        *
        db_dx
        -
        d["uy"]
        *
        db_dy
    )

    rhs_ux = (
        nu_nd
        *
        d["lap_ux"]
        -
        d["ux"]
        *
        dux_dx
        -
        d["uy"]
        *
        dux_dy
        -
        dp_dx
    )

    rhs_uy = (
        nu_nd
        *
        d["lap_uy"]
        -
        d["ux"]
        *
        duy_dx
        -
        d["uy"]
        *
        duy_dy
        -
        dp_dy
        +
        d["b"]
    )

    div = (
        dux_dx
        +
        duy_dy
    )

    return (
        rhs_b,
        rhs_ux,
        rhs_uy,
        div,
    )


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

    result = (
        residual_builder
        .compute_from_history(
            x_norm,
            y_norm,
            param,
        )
    )

    current = get_current_state(
        residual_builder,
        x_norm,
    )

    grid = (
        residual_builder
        .metadata
        .grid
    )

    d = spatial_derivatives(
        current,
        grid,
    )

    (
        ra,
        pr,
        nu_nd,
        kappa_nd,
    ) = parameter_coefficients(
        param
    )

    # ============================================================
    # Structural divergence diagnosis
    # ============================================================

    dx_ux = d[
        "dux_dx"
    ]

    dy_uy = d[
        "duy_dy"
    ]

    dx_uy = d[
        "duy_dx"
    ]

    dy_ux = d[
        "dux_dy"
    ]

    div_standard = (
        dx_ux
        +
        dy_uy
    )

    div_relative_sign_flip = (
        dx_ux
        -
        dy_uy
    )

    div_swapped_standard = (
        dx_uy
        +
        dy_ux
    )

    div_swapped_relative_sign_flip = (
        dx_uy
        -
        dy_ux
    )

    fitted_c_standard = (
        fitted_relative_coefficient(
            dx_ux,
            dy_uy,
        )
    )

    fitted_c_swapped = (
        fitted_relative_coefficient(
            dx_uy,
            dy_ux,
        )
    )

    temporal_b = (
        result
        .temporal_rate_b
    )

    temporal_ux = (
        result
        .temporal_rate_u[:, 0]
    )

    temporal_uy = (
        result
        .temporal_rate_u[:, 1]
    )

    variants = [
        (
            "baseline",
            +1.0,
            +1.0,
        ),
        (
            "flip_y_first_derivative",
            +1.0,
            -1.0,
        ),
        (
            "flip_x_first_derivative",
            -1.0,
            +1.0,
        ),
        (
            "flip_xy_first_derivative",
            -1.0,
            -1.0,
        ),
    ]

    rows = []

    for (
        variant,
        sx,
        sy,
    ) in variants:

        (
            rhs_b,
            rhs_ux,
            rhs_uy,
            div,
        ) = make_rhs(
            d,
            nu_nd,
            kappa_nd,
            sx,
            sy,
        )

        row = {
            "group": group,
            "trajectory": trajectory,
            "sample_index": sample_index,

            "ra": ra,
            "pr": pr,

            "variant": variant,
            "sx": sx,
            "sy": sy,

            # ------------------------------------------
            # PDE consistency
            # ------------------------------------------

            "b_balance": balance(
                temporal_b,
                rhs_b,
            ),

            "b_temporal_rhs_cosine": cosine(
                temporal_b,
                rhs_b,
            ),

            "ux_balance": balance(
                temporal_ux,
                rhs_ux,
            ),

            "ux_temporal_rhs_cosine": cosine(
                temporal_ux,
                rhs_ux,
            ),

            "uy_balance": balance(
                temporal_uy,
                rhs_uy,
            ),

            "uy_temporal_rhs_cosine": cosine(
                temporal_uy,
                rhs_uy,
            ),

            "div_variant_rms": rms(
                div
            ),

            # ------------------------------------------
            # Structural derivative diagnosis
            # Repeated across variants intentionally.
            # ------------------------------------------

            "dx_ux_rms": rms(
                dx_ux
            ),

            "dy_uy_rms": rms(
                dy_uy
            ),

            "dxux_dyuy_cosine": cosine(
                dx_ux,
                dy_uy,
            ),

            "div_standard_rms": rms(
                div_standard
            ),

            "div_relative_sign_flip_rms": rms(
                div_relative_sign_flip
            ),

            "fitted_c_standard": (
                fitted_c_standard
            ),

            "dx_uy_rms": rms(
                dx_uy
            ),

            "dy_ux_rms": rms(
                dy_ux
            ),

            "dxuy_dyux_cosine": cosine(
                dx_uy,
                dy_ux,
            ),

            "div_swapped_standard_rms": rms(
                div_swapped_standard
            ),

            "div_swapped_sign_flip_rms": rms(
                div_swapped_relative_sign_flip
            ),

            "fitted_c_swapped": (
                fitted_c_swapped
            ),
        }

        rows.append(
            row
        )

    return rows


def print_sample_rows(
    rows,
):

    first = rows[0]

    print()
    print(
        "-" * 90
    )

    print(
        f"{first['group']} | "
        f"trajectory={first['trajectory']} | "
        f"sample={first['sample_index']}"
    )

    print(
        f"Ra={first['ra']:.6e} | "
        f"Pr={first['pr']:.8f}"
    )

    print()
    print(
        "1) divergence 结构诊断"
    )

    print(
        "当前对应 "
        "d_x(u_x) + d_y(u_y): "
        f"{first['div_standard_rms']:.6e}"
    )

    print(
        "相对符号翻转 "
        "d_x(u_x) - d_y(u_y): "
        f"{first['div_relative_sign_flip_rms']:.6e}"
    )

    print(
        "交换分量 "
        "d_x(u_y) + d_y(u_x): "
        f"{first['div_swapped_standard_rms']:.6e}"
    )

    print(
        "交换分量+相对符号翻转: "
        f"{first['div_swapped_sign_flip_rms']:.6e}"
    )

    print(
        "d_x(u_x) vs d_y(u_y): "
        f"cos={first['dxux_dyuy_cosine']:.6f} | "
        f"fit_c={first['fitted_c_standard']:.6e}"
    )

    print(
        "d_x(u_y) vs d_y(u_x): "
        f"cos={first['dxuy_dyux_cosine']:.6f} | "
        f"fit_c={first['fitted_c_swapped']:.6e}"
    )

    print()
    print(
        "2) 一阶导数方向对完整 PDE 的影响"
    )

    for row in rows:

        print(
            f"{row['variant']:28s} | "
            f"b bal={row['b_balance']:.6f}, "
            f"cos={row['b_temporal_rhs_cosine']:.4f} | "
            f"ux bal={row['ux_balance']:.6f}, "
            f"cos={row['ux_temporal_rhs_cosine']:.4f} | "
            f"uy bal={row['uy_balance']:.6f}, "
            f"cos={row['uy_temporal_rhs_cosine']:.4f} | "
            f"div={row['div_variant_rms']:.6e}"
        )


def finite_mean(
    values,
):

    values = [
        float(x)
        for x in values
        if math.isfinite(
            float(x)
        )
    ]

    if not values:
        return float("nan")

    return (
        sum(values)
        /
        len(values)
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


def print_summary(
    rows,
):

    print()
    print(
        "=" * 100
    )

    print(
        "R2-2C 汇总：一阶导数方向候选"
    )

    print(
        "=" * 100
    )

    variants = []

    for row in rows:

        name = row[
            "variant"
        ]

        if name not in variants:
            variants.append(
                name
            )

    for variant in variants:

        subset = [
            row
            for row in rows
            if row["variant"] == variant
        ]

        print()
        print(
            variant
        )

        print(
            f"  b_balance mean   = "
            f"{finite_mean([r['b_balance'] for r in subset]):.6e}"
        )

        print(
            f"  b_cos mean       = "
            f"{finite_mean([r['b_temporal_rhs_cosine'] for r in subset]):.6e}"
        )

        print(
            f"  ux_balance mean  = "
            f"{finite_mean([r['ux_balance'] for r in subset]):.6e}"
        )

        print(
            f"  ux_cos mean      = "
            f"{finite_mean([r['ux_temporal_rhs_cosine'] for r in subset]):.6e}"
        )

        print(
            f"  uy_balance mean  = "
            f"{finite_mean([r['uy_balance'] for r in subset]):.6e}"
        )

        print(
            f"  uy_cos mean      = "
            f"{finite_mean([r['uy_temporal_rhs_cosine'] for r in subset]):.6e}"
        )

        print(
            f"  div mean         = "
            f"{finite_mean([r['div_variant_rms'] for r in subset]):.6e}"
        )

    # Structural rows are duplicated once per sign variant,
    # so only take baseline rows here.

    baseline_rows = [
        row
        for row in rows
        if row["variant"] == "baseline"
    ]

    print()
    print(
        "=" * 100
    )

    print(
        "R2-2C 汇总：divergence 结构"
    )

    print(
        "=" * 100
    )

    structural_keys = [
        "dx_ux_rms",
        "dy_uy_rms",
        "dxux_dyuy_cosine",
        "div_standard_rms",
        "div_relative_sign_flip_rms",
        "fitted_c_standard",

        "dx_uy_rms",
        "dy_ux_rms",
        "dxuy_dyux_cosine",
        "div_swapped_standard_rms",
        "div_swapped_sign_flip_rms",
        "fitted_c_swapped",
    ]

    for key in structural_keys:

        print(
            f"{key:32s} | "
            f"mean="
            f"{finite_mean([r[key] for r in baseline_rows]):.6e}"
        )


def main():

    args = parse_args()

    if args.partition not in (
        "train",
        "val",
    ):
        raise RuntimeError(
            "R2-2C permits TRAIN / VAL only."
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
        "=" * 100
    )

    print(
        "R2-2C 空间方向与离散算子诊断"
    )

    print(
        "=" * 100
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
        f"grid              : "
        f"Nx={metadata.grid.nx}, "
        f"Ny={metadata.grid.ny}, "
        f"dx={metadata.grid.dx}, "
        f"dy={metadata.grid.dy}"
    )

    print(
        "TEST              : FORBIDDEN / NOT INSTANTIATED"
    )

    all_rows = []

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

            rows = diagnose_sample(
                residual_builder=residual_builder,
                x_norm=x_norm,
                y_norm=y_norm,
                param=param,
                group=group,
                trajectory=trajectory,
                sample_index=sample_index,
            )

            all_rows.extend(
                rows
            )

            print_sample_rows(
                rows
            )

        del dataset
        gc.collect()

    if not all_rows:
        raise RuntimeError(
            "No diagnostic rows."
        )

    write_csv(
        all_rows,
        output_path,
    )

    print_summary(
        all_rows
    )

    print()
    print(
        f"CSV saved: {output_path}"
    )

    print()
    print(
        "=" * 100
    )

    print(
        "✅ R2-2C 空间方向与离散算子诊断运行完成"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()
