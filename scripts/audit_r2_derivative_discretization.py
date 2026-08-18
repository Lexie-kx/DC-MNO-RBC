"""
R2-2E: Spatial-Derivative Discretization Diagnosis
==================================================

中文阶段：
    R2-2E 空间导数离散方式诊断

Purpose
-------
Test whether the real-RBC PDE mismatch is strongly sensitive to the
spatial derivative backend used on stored snapshots.

Four diagnostic variants
------------------------
1. fd2_fd2
       x: current second-order periodic finite difference
       y: current second-order finite difference

2. spectral_fd2
       x: periodic FFT spectral derivative
       y: current second-order finite difference

3. fd4_fd4
       x: fourth-order periodic finite difference
       y: fourth-order centered finite difference on interior

4. spectral_fd4
       x: periodic FFT spectral derivative
       y: fourth-order centered finite difference on interior

Fair-comparison rule
--------------------
ALL metrics are evaluated only on the common y interior:

    y = 2 : Ny-2

i.e. tensor slice:

    [..., 2:-2]

This removes boundary-stencil differences from the comparison.

Important
---------
This script does NOT:
    - modify physics/derivatives.py;
    - modify the canonical PDE;
    - choose a new production derivative backend;
    - train a model;
    - access TEST data.

A lower residual under one backend is diagnostic evidence only.
It is NOT proof that the original DNS used that numerical method.
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
    rbc_transport_coefficients,
)

from physics.rbc_residuals import (
    CanonicalRBCResidual,
)

from physics.derivatives import (
    grad_x_periodic,
    grad_y_nonperiodic,
    second_x_periodic,
    second_y_nonperiodic,
)


# ============================================================
# Basic helpers
# ============================================================


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "R2-2E derivative-discretization diagnosis."
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


def crop_y(x):

    if x.shape[-1] < 5:
        raise ValueError(
            "Need at least 5 y points for fourth-order interior audit."
        )

    return x[..., 2:-2]


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

    denominator = (
        torch.linalg.vector_norm(a)
        *
        torch.linalg.vector_norm(b)
    )

    if denominator.item() < 1.0e-30:
        return float("nan")

    return (
        torch.dot(a, b)
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


# ============================================================
# x derivatives
# ============================================================


def grad_x_fd4_periodic(
    f,
    grid,
):

    return (
        -
        torch.roll(
            f,
            shifts=-2,
            dims=-2,
        )
        +
        8.0
        *
        torch.roll(
            f,
            shifts=-1,
            dims=-2,
        )
        -
        8.0
        *
        torch.roll(
            f,
            shifts=1,
            dims=-2,
        )
        +
        torch.roll(
            f,
            shifts=2,
            dims=-2,
        )
    ) / (
        12.0
        *
        grid.dx
    )


def second_x_fd4_periodic(
    f,
    grid,
):

    return (
        -
        torch.roll(
            f,
            shifts=-2,
            dims=-2,
        )
        +
        16.0
        *
        torch.roll(
            f,
            shifts=-1,
            dims=-2,
        )
        -
        30.0
        *
        f
        +
        16.0
        *
        torch.roll(
            f,
            shifts=1,
            dims=-2,
        )
        -
        torch.roll(
            f,
            shifts=2,
            dims=-2,
        )
    ) / (
        12.0
        *
        grid.dx ** 2
    )


def _spectral_kx(
    f,
    grid,
):

    k = (
        2.0
        *
        math.pi
        *
        torch.fft.fftfreq(
            grid.nx,
            d=grid.dx,
            dtype=f.dtype,
            device=f.device,
        )
    )

    view_shape = (
        [1]
        *
        (f.ndim - 2)
        +
        [
            grid.nx,
            1,
        ]
    )

    return k.view(
        *view_shape
    )


def grad_x_spectral(
    f,
    grid,
):

    k = _spectral_kx(
        f,
        grid,
    )

    f_hat = torch.fft.fft(
        f,
        dim=-2,
    )

    derivative_hat = (
        1j
        *
        k
        *
        f_hat
    )

    return torch.fft.ifft(
        derivative_hat,
        dim=-2,
    ).real


def second_x_spectral(
    f,
    grid,
):

    k = _spectral_kx(
        f,
        grid,
    )

    f_hat = torch.fft.fft(
        f,
        dim=-2,
    )

    derivative_hat = (
        -
        k ** 2
        *
        f_hat
    )

    return torch.fft.ifft(
        derivative_hat,
        dim=-2,
    ).real


# ============================================================
# y fourth-order interior derivatives
# ============================================================


def grad_y_fd4_interior(
    f,
    grid,
):
    """
    Returns only y=2:-2.
    """

    return (
        f[..., :-4]
        -
        8.0
        *
        f[..., 1:-3]
        +
        8.0
        *
        f[..., 3:-1]
        -
        f[..., 4:]
    ) / (
        12.0
        *
        grid.dy
    )


def second_y_fd4_interior(
    f,
    grid,
):
    """
    Returns only y=2:-2.
    """

    return (
        -
        f[..., 4:]
        +
        16.0
        *
        f[..., 3:-1]
        -
        30.0
        *
        f[..., 2:-2]
        +
        16.0
        *
        f[..., 1:-3]
        -
        f[..., :-4]
    ) / (
        12.0
        *
        grid.dy ** 2
    )


# ============================================================
# Derivative variant construction
# ============================================================


def derivatives_for_variant(
    *,
    f,
    grid,
    x_backend,
    y_backend,
):

    # --------------------------------------------------------
    # x first / second
    # --------------------------------------------------------

    if x_backend == "fd2":

        dx = grad_x_periodic(
            f,
            grid,
        )

        dxx = second_x_periodic(
            f,
            grid,
        )

    elif x_backend == "fd4":

        dx = grad_x_fd4_periodic(
            f,
            grid,
        )

        dxx = second_x_fd4_periodic(
            f,
            grid,
        )

    elif x_backend == "spectral":

        dx = grad_x_spectral(
            f,
            grid,
        )

        dxx = second_x_spectral(
            f,
            grid,
        )

    else:
        raise ValueError(
            f"Unknown x backend: {x_backend}"
        )

    # --------------------------------------------------------
    # y first / second
    # All returned values are cropped to common interior.
    # --------------------------------------------------------

    if y_backend == "fd2":

        dy = crop_y(
            grad_y_nonperiodic(
                f,
                grid,
            )
        )

        dyy = crop_y(
            second_y_nonperiodic(
                f,
                grid,
            )
        )

    elif y_backend == "fd4":

        dy = grad_y_fd4_interior(
            f,
            grid,
        )

        dyy = second_y_fd4_interior(
            f,
            grid,
        )

    else:
        raise ValueError(
            f"Unknown y backend: {y_backend}"
        )

    return {
        "value": crop_y(
            f
        ),

        "dx": crop_y(
            dx
        ),

        "dy": dy,

        "dxx": crop_y(
            dxx
        ),

        "dyy": dyy,
    }


def compile_variant_rhs(
    *,
    current,
    param,
    grid,
    x_backend,
    y_backend,
):

    b = current[:, 0]
    ux = current[:, 1]
    uy = current[:, 2]
    p = current[:, 3]

    db = derivatives_for_variant(
        f=b,
        grid=grid,
        x_backend=x_backend,
        y_backend=y_backend,
    )

    dux = derivatives_for_variant(
        f=ux,
        grid=grid,
        x_backend=x_backend,
        y_backend=y_backend,
    )

    duy = derivatives_for_variant(
        f=uy,
        grid=grid,
        x_backend=x_backend,
        y_backend=y_backend,
    )

    dp = derivatives_for_variant(
        f=p,
        grid=grid,
        x_backend=x_backend,
        y_backend=y_backend,
    )

    nu_nd, kappa_nd = (
        rbc_transport_coefficients(
            param
        )
    )

    nu_nd = nu_nd[
        :,
        None,
        None,
    ]

    kappa_nd = kappa_nd[
        :,
        None,
        None,
    ]

    b_i = db["value"]
    ux_i = dux["value"]
    uy_i = duy["value"]

    lap_b = (
        db["dxx"]
        +
        db["dyy"]
    )

    lap_ux = (
        dux["dxx"]
        +
        dux["dyy"]
    )

    lap_uy = (
        duy["dxx"]
        +
        duy["dyy"]
    )

    rhs_b = (
        kappa_nd
        *
        lap_b
        -
        ux_i
        *
        db["dx"]
        -
        uy_i
        *
        db["dy"]
    )

    rhs_ux = (
        nu_nd
        *
        lap_ux
        -
        ux_i
        *
        dux["dx"]
        -
        uy_i
        *
        dux["dy"]
        -
        dp["dx"]
    )

    rhs_uy = (
        nu_nd
        *
        lap_uy
        -
        ux_i
        *
        duy["dx"]
        -
        uy_i
        *
        duy["dy"]
        -
        dp["dy"]
        +
        b_i
    )

    div = (
        dux["dx"]
        +
        duy["dy"]
    )

    return {
        "rhs_b": rhs_b,
        "rhs_ux": rhs_ux,
        "rhs_uy": rhs_uy,
        "div": div,
    }


# ============================================================
# Sample diagnosis
# ============================================================


VARIANTS = [
    (
        "fd2_fd2",
        "fd2",
        "fd2",
    ),
    (
        "spectral_fd2",
        "spectral",
        "fd2",
    ),
    (
        "fd4_fd4",
        "fd4",
        "fd4",
    ),
    (
        "spectral_fd4",
        "spectral",
        "fd4",
    ),
]


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

    current = (
        residual_builder
        .compiler
        .canonicalizer
        .latest_state_canonical(
            x_norm
        )
    )

    grid = (
        residual_builder
        .metadata
        .grid
    )

    temporal_b = crop_y(
        result.temporal_rate_b
    )

    temporal_ux = crop_y(
        result.temporal_rate_u[:, 0]
    )

    temporal_uy = crop_y(
        result.temporal_rate_u[:, 1]
    )

    log_ra = float(
        param[0, 0].item()
    )

    log_pr = float(
        param[0, 1].item()
    )

    rows = []

    baseline_values = None

    for (
        variant,
        x_backend,
        y_backend,
    ) in VARIANTS:

        rhs = compile_variant_rhs(
            current=current,
            param=param,
            grid=grid,
            x_backend=x_backend,
            y_backend=y_backend,
        )

        row = {
            "group": group,
            "trajectory": trajectory,
            "sample_index": sample_index,

            "ra": 10.0 ** log_ra,
            "pr": 10.0 ** log_pr,

            "variant": variant,
            "x_backend": x_backend,
            "y_backend": y_backend,

            "b_balance": balance(
                temporal_b,
                rhs["rhs_b"],
            ),

            "b_cosine": cosine(
                temporal_b,
                rhs["rhs_b"],
            ),

            "ux_balance": balance(
                temporal_ux,
                rhs["rhs_ux"],
            ),

            "ux_cosine": cosine(
                temporal_ux,
                rhs["rhs_ux"],
            ),

            "uy_balance": balance(
                temporal_uy,
                rhs["rhs_uy"],
            ),

            "uy_cosine": cosine(
                temporal_uy,
                rhs["rhs_uy"],
            ),

            "div_rms": rms(
                rhs["div"]
            ),
        }

        if variant == "fd2_fd2":

            baseline_values = {
                "b_balance": row[
                    "b_balance"
                ],

                "ux_balance": row[
                    "ux_balance"
                ],

                "uy_balance": row[
                    "uy_balance"
                ],

                "div_rms": row[
                    "div_rms"
                ],
            }

        if baseline_values is not None:

            row[
                "delta_b_balance_vs_fd2"
            ] = (
                row["b_balance"]
                -
                baseline_values[
                    "b_balance"
                ]
            )

            row[
                "delta_ux_balance_vs_fd2"
            ] = (
                row["ux_balance"]
                -
                baseline_values[
                    "ux_balance"
                ]
            )

            row[
                "delta_uy_balance_vs_fd2"
            ] = (
                row["uy_balance"]
                -
                baseline_values[
                    "uy_balance"
                ]
            )

            row[
                "div_ratio_vs_fd2"
            ] = (
                row["div_rms"]
                /
                (
                    baseline_values[
                        "div_rms"
                    ]
                    +
                    1.0e-30
                )
            )

        rows.append(
            row
        )

    return rows


# ============================================================
# Output
# ============================================================


def print_sample_rows(
    rows,
):

    first = rows[0]

    print()
    print(
        "-" * 112
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
        "共同比较区域：y = 2:-2"
    )

    for row in rows:

        print(
            f"{row['variant']:16s} | "
            f"b bal={row['b_balance']:.6f}, "
            f"cos={row['b_cosine']:.4f} | "
            f"ux bal={row['ux_balance']:.6f}, "
            f"cos={row['ux_cosine']:.4f} | "
            f"uy bal={row['uy_balance']:.6f}, "
            f"cos={row['uy_cosine']:.4f} | "
            f"div={row['div_rms']:.6e}"
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


def print_summary(
    rows,
):

    print()
    print(
        "=" * 112
    )

    print(
        "R2-2E 汇总：空间导数离散方式"
    )

    print(
        "=" * 112
    )

    for (
        variant,
        _,
        _,
    ) in VARIANTS:

        subset = [
            row
            for row in rows
            if row[
                "variant"
            ] == variant
        ]

        print()
        print(
            variant
        )

        print(
            f"  b_balance mean  = "
            f"{finite_mean([r['b_balance'] for r in subset]):.6e}"
        )

        print(
            f"  b_cos mean      = "
            f"{finite_mean([r['b_cosine'] for r in subset]):.6e}"
        )

        print(
            f"  ux_balance mean = "
            f"{finite_mean([r['ux_balance'] for r in subset]):.6e}"
        )

        print(
            f"  ux_cos mean     = "
            f"{finite_mean([r['ux_cosine'] for r in subset]):.6e}"
        )

        print(
            f"  uy_balance mean = "
            f"{finite_mean([r['uy_balance'] for r in subset]):.6e}"
        )

        print(
            f"  uy_cos mean     = "
            f"{finite_mean([r['uy_cosine'] for r in subset]):.6e}"
        )

        print(
            f"  div mean        = "
            f"{finite_mean([r['div_rms'] for r in subset]):.6e}"
        )

        print(
            f"  Δb balance      = "
            f"{finite_mean([r['delta_b_balance_vs_fd2'] for r in subset]):+.6e}"
        )

        print(
            f"  Δux balance     = "
            f"{finite_mean([r['delta_ux_balance_vs_fd2'] for r in subset]):+.6e}"
        )

        print(
            f"  Δuy balance     = "
            f"{finite_mean([r['delta_uy_balance_vs_fd2'] for r in subset]):+.6e}"
        )

        print(
            f"  div/fd2 ratio   = "
            f"{finite_mean([r['div_ratio_vs_fd2'] for r in subset]):.6e}"
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


# ============================================================
# Main
# ============================================================


def main():

    args = parse_args()

    if args.partition not in (
        "train",
        "val",
    ):
        raise RuntimeError(
            "R2-2E permits TRAIN / VAL only."
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
        "=" * 112
    )

    print(
        "R2-2E 空间导数离散方式诊断"
    )

    print(
        "=" * 112
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
        "comparison region : y=2:-2"
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
        "=" * 112
    )

    print(
        "✅ R2-2E 空间导数离散方式诊断运行完成"
    )

    print(
        "=" * 112
    )


if __name__ == "__main__":
    main()
