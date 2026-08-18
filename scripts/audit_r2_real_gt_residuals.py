"""
R2-2: Real RBC Ground-Truth Residual Audit
==========================================

Purpose
-------
Measure equation-level discrete-time PDE residuals on REAL RBC
TRAIN / VAL data after R1 + R2-1 correctness has been established.

This is a diagnostic audit.

It does NOT:
    - train a model;
    - evaluate model predictions;
    - tune residual weights;
    - alter PDE signs or coefficients;
    - access TEST data.

Residual convention
-------------------
Buoyancy:
    R_b =
        (b_{t+dt} - b_t)/dt
        - [
            -u · grad(b)
            + kappa_nd * laplacian(b)
          ]

Momentum x:
    R_ux =
        (ux_{t+dt} - ux_t)/dt
        - [
            -u · grad(ux)
            + nu_nd * laplacian(ux)
            - dp/dx
          ]

Momentum y:
    R_uy =
        (uy_{t+dt} - uy_t)/dt
        - [
            -u · grad(uy)
            + nu_nd * laplacian(uy)
            - dp/dy
            + b
          ]

Constraint:
    R_div = div(u)

Important
---------
These are discrete-time residual proxies using a forward difference
over dt=0.25. They are NOT claimed to be exact continuous-time PDE
residuals.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
from typing import Dict, List

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
            "R2-2 real RBC ground-truth residual audit."
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
        help=(
            "TEST is intentionally unavailable."
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
        help=(
            "Evenly spaced samples from one legal trajectory "
            "inside each selected group."
        ),
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
    return torch.sqrt(
        torch.mean(
            x.double() ** 2
        )
    ).item()


def max_abs(x: torch.Tensor) -> float:
    return torch.max(
        torch.abs(
            x.double()
        )
    ).item()


def finite(x: torch.Tensor) -> bool:
    return bool(
        torch.isfinite(
            x
        ).all().item()
    )


def safe_balance_ratio(
    residual_rms: float,
    temporal_rms: float,
    rhs_rms: float,
) -> float:
    """
    Dimensionless diagnostic:

        residual RMS
        ---------------------------
        temporal RMS + RHS RMS + eps

    This is NOT a loss definition.
    """

    denominator = (
        temporal_rms
        +
        rhs_rms
        +
        1.0e-30
    )

    return (
        residual_rms
        /
        denominator
    )


def choose_indices(
    n: int,
    k: int,
) -> List[int]:

    if n <= 0:
        raise ValueError(
            "Dataset must contain at least one sample."
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


def compile_component_terms(
    residual_builder: CanonicalRBCResidual,
    history_norm: torch.Tensor,
    param: torch.Tensor,
):
    """
    Compile R1 terms from the exact same latest state used
    by the residual builder.
    """

    canonicalizer = (
        residual_builder
        .compiler
        .canonicalizer
    )

    current = (
        canonicalizer.latest_state_canonical(
            history_norm
        )
    )

    compiler = residual_builder.compiler

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

    grad_p = compiler.compile(
        "pressure_gradient",
        current,
    ).canonical_value

    force_b = compiler.compile(
        "buoyancy_forcing",
        current,
    ).canonical_value

    return {
        "adv_b": adv_b,
        "diff_b": diff_b,

        "adv_ux": adv_u[:, 0],
        "adv_uy": adv_u[:, 1],

        "visc_ux": visc_u[:, 0],
        "visc_uy": visc_u[:, 1],

        "pressure_x": grad_p[:, 0],
        "pressure_y": grad_p[:, 1],

        "buoyancy_force": force_b,
    }


def audit_sample(
    *,
    residual_builder: CanonicalRBCResidual,
    x_norm: torch.Tensor,
    y_norm: torch.Tensor,
    param: torch.Tensor,
    group: str,
    trajectory: int,
    sample_index: int,
) -> Dict[str, float]:

    residual = residual_builder.compute_from_history(
        x_norm,
        y_norm,
        param,
    )

    terms = compile_component_terms(
        residual_builder,
        x_norm,
        param,
    )

    # ============================================================
    # RMS values
    # ============================================================

    temporal_b = rms(
        residual.temporal_rate_b
    )

    temporal_ux = rms(
        residual.temporal_rate_u[:, 0]
    )

    temporal_uy = rms(
        residual.temporal_rate_u[:, 1]
    )

    rhs_b = rms(
        residual.rhs_rate_b
    )

    rhs_ux = rms(
        residual.rhs_rate_u[:, 0]
    )

    rhs_uy = rms(
        residual.rhs_rate_u[:, 1]
    )

    residual_b = rms(
        residual.residual_rate_b
    )

    residual_ux = rms(
        residual.residual_rate_u[:, 0]
    )

    residual_uy = rms(
        residual.residual_rate_u[:, 1]
    )

    delta_residual_b = rms(
        residual.residual_delta_b
    )

    delta_residual_ux = rms(
        residual.residual_delta_u[:, 0]
    )

    delta_residual_uy = rms(
        residual.residual_delta_u[:, 1]
    )

    divergence_rms = rms(
        residual.divergence
    )

    # ============================================================
    # R2-1 identity re-check on REAL data
    # ============================================================

    metadata = residual_builder.metadata

    dt = metadata.time.dt

    sb = (
        metadata.normalization.std_for(
            "buoyancy"
        )
        +
        metadata.normalization.eps
    )

    sux = (
        metadata.normalization.std_for(
            "u_x"
        )
        +
        metadata.normalization.eps
    )

    suy = (
        metadata.normalization.std_for(
            "u_y"
        )
        +
        metadata.normalization.eps
    )

    expected_delta_b = (
        dt
        /
        sb
        *
        residual.residual_rate_b
    )

    expected_delta_ux = (
        dt
        /
        sux
        *
        residual.residual_rate_u[:, 0]
    )

    expected_delta_uy = (
        dt
        /
        suy
        *
        residual.residual_rate_u[:, 1]
    )

    identity_err_b = max_abs(
        residual.residual_delta_b
        -
        expected_delta_b
    )

    identity_err_ux = max_abs(
        residual.residual_delta_u[:, 0]
        -
        expected_delta_ux
    )

    identity_err_uy = max_abs(
        residual.residual_delta_u[:, 1]
        -
        expected_delta_uy
    )

    # ============================================================
    # Finite checks
    # ============================================================

    tensors_to_check = [
        residual.temporal_rate_b,
        residual.temporal_rate_u,
        residual.rhs_rate_b,
        residual.rhs_rate_u,
        residual.residual_rate_b,
        residual.residual_rate_u,
        residual.residual_delta_b,
        residual.residual_delta_u,
        residual.divergence,
    ]

    all_finite = all(
        finite(x)
        for x in tensors_to_check
    )

    if not all_finite:
        raise RuntimeError(
            f"NaN/Inf detected for group={group}, "
            f"trajectory={trajectory}, sample={sample_index}"
        )

    if max(
        identity_err_b,
        identity_err_ux,
        identity_err_uy,
    ) > 1.0e-10:
        raise RuntimeError(
            "Real-data rate/delta residual identity failed."
        )

    log_ra = float(
        param[0, 0].item()
    )

    log_pr = float(
        param[0, 1].item()
    )

    row = {
        "group": group,
        "trajectory": trajectory,
        "sample_index": sample_index,

        "log10_ra": log_ra,
        "log10_pr": log_pr,
        "ra": 10.0 ** log_ra,
        "pr": 10.0 ** log_pr,

        "temporal_b_rms": temporal_b,
        "rhs_b_rms": rhs_b,
        "residual_b_rate_rms": residual_b,
        "residual_b_delta_rms": delta_residual_b,
        "balance_b": safe_balance_ratio(
            residual_b,
            temporal_b,
            rhs_b,
        ),

        "temporal_ux_rms": temporal_ux,
        "rhs_ux_rms": rhs_ux,
        "residual_ux_rate_rms": residual_ux,
        "residual_ux_delta_rms": delta_residual_ux,
        "balance_ux": safe_balance_ratio(
            residual_ux,
            temporal_ux,
            rhs_ux,
        ),

        "temporal_uy_rms": temporal_uy,
        "rhs_uy_rms": rhs_uy,
        "residual_uy_rate_rms": residual_uy,
        "residual_uy_delta_rms": delta_residual_uy,
        "balance_uy": safe_balance_ratio(
            residual_uy,
            temporal_uy,
            rhs_uy,
        ),

        "divergence_rms": divergence_rms,

        "adv_b_rms": rms(
            terms["adv_b"]
        ),
        "diff_b_rms": rms(
            terms["diff_b"]
        ),

        "adv_ux_rms": rms(
            terms["adv_ux"]
        ),
        "visc_ux_rms": rms(
            terms["visc_ux"]
        ),
        "pressure_x_rms": rms(
            terms["pressure_x"]
        ),

        "adv_uy_rms": rms(
            terms["adv_uy"]
        ),
        "visc_uy_rms": rms(
            terms["visc_uy"]
        ),
        "pressure_y_rms": rms(
            terms["pressure_y"]
        ),
        "buoyancy_force_rms": rms(
            terms["buoyancy_force"]
        ),

        "identity_error_b": identity_err_b,
        "identity_error_ux": identity_err_ux,
        "identity_error_uy": identity_err_uy,

        "all_finite": all_finite,
    }

    return row


def print_sample(row):
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
        "Equation balance "
        "(rate-space RMS)"
    )

    print(
        "b   : "
        f"temporal={row['temporal_b_rms']:.6e} | "
        f"rhs={row['rhs_b_rms']:.6e} | "
        f"residual={row['residual_b_rate_rms']:.6e} | "
        f"balance={row['balance_b']:.6e}"
    )

    print(
        "u_x : "
        f"temporal={row['temporal_ux_rms']:.6e} | "
        f"rhs={row['rhs_ux_rms']:.6e} | "
        f"residual={row['residual_ux_rate_rms']:.6e} | "
        f"balance={row['balance_ux']:.6e}"
    )

    print(
        "u_y : "
        f"temporal={row['temporal_uy_rms']:.6e} | "
        f"rhs={row['rhs_uy_rms']:.6e} | "
        f"residual={row['residual_uy_rate_rms']:.6e} | "
        f"balance={row['balance_uy']:.6e}"
    )

    print(
        f"div : RMS={row['divergence_rms']:.6e}"
    )

    print()
    print(
        "PDE-term RMS"
    )

    print(
        "b   : "
        f"advection={row['adv_b_rms']:.6e} | "
        f"diffusion={row['diff_b_rms']:.6e}"
    )

    print(
        "u_x : "
        f"advection={row['adv_ux_rms']:.6e} | "
        f"viscosity={row['visc_ux_rms']:.6e} | "
        f"pressure={row['pressure_x_rms']:.6e}"
    )

    print(
        "u_y : "
        f"advection={row['adv_uy_rms']:.6e} | "
        f"viscosity={row['visc_uy_rms']:.6e} | "
        f"pressure={row['pressure_y_rms']:.6e} | "
        f"buoyancy={row['buoyancy_force_rms']:.6e}"
    )

    print()
    print(
        "Rate <-> delta residual identity"
    )

    print(
        f"b   max error = "
        f"{row['identity_error_b']:.6e}"
    )

    print(
        f"u_x max error = "
        f"{row['identity_error_ux']:.6e}"
    )

    print(
        f"u_y max error = "
        f"{row['identity_error_uy']:.6e}"
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


def print_summary(rows):
    print()
    print(
        "=" * 78
    )
    print(
        "R2-2 AGGREGATE SUMMARY"
    )
    print(
        "=" * 78
    )

    keys = (
        "balance_b",
        "balance_ux",
        "balance_uy",
        "divergence_rms",
        "residual_b_rate_rms",
        "residual_ux_rate_rms",
        "residual_uy_rate_rms",
    )

    for key in keys:
        values = [
            float(
                row[key]
            )
            for row in rows
        ]

        mean_value = (
            sum(values)
            /
            len(values)
        )

        min_value = min(
            values
        )

        max_value = max(
            values
        )

        print(
            f"{key:24s} | "
            f"mean={mean_value:.6e} | "
            f"min={min_value:.6e} | "
            f"max={max_value:.6e}"
        )

    max_identity_error = max(
        max(
            row["identity_error_b"],
            row["identity_error_ux"],
            row["identity_error_uy"],
        )
        for row in rows
    )

    print(
        f"{'max_identity_error':24s} | "
        f"{max_identity_error:.6e}"
    )


def main():
    args = parse_args()

    split_path = resolve(
        args.split
    )

    stats_path = resolve(
        args.stats
    )

    output_path = resolve(
        args.output
    )

    if args.max_groups <= 0:
        raise ValueError(
            "--max_groups must be positive."
        )

    if args.samples_per_group <= 0:
        raise ValueError(
            "--samples_per_group must be positive."
        )

    with open(
        split_path,
        "r",
        encoding="utf-8",
    ) as f:
        split = json.load(
            f
        )

    # ============================================================
    # TEST firewall
    # ============================================================

    if args.partition not in (
        "train",
        "val",
    ):
        raise RuntimeError(
            "R2-2 permits TRAIN / VAL only."
        )

    if args.partition not in split:
        raise KeyError(
            f"Partition '{args.partition}' "
            "not found in split file."
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
        "R2-2 REAL RBC GROUND-TRUTH RESIDUAL AUDIT"
    )

    print(
        "=" * 78
    )

    print(
        f"split             : {split_path}"
    )

    print(
        f"stats             : {stats_path}"
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
        f"dt                : {metadata.time.dt}"
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
                f"No trajectory in group={group}"
            )

        # One legal trajectory per group is enough for this
        # R2-2 diagnostic audit. Statistical coverage belongs later.
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

            sample = dataset[
                sample_index
            ]

            if len(sample) != 3:
                raise RuntimeError(
                    "Expected "
                    "(x_norm, y_norm, param)."
                )

            x_norm, y_norm, param = sample

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

            row = audit_sample(
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

            print_sample(
                row
            )

        del dataset
        gc.collect()

    if not rows:
        raise RuntimeError(
            "No samples were audited."
        )

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
        "✅ R2-2 REAL GT RESIDUAL AUDIT = PASS"
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()
