"""
R1-4: Real RBC Integration Audit
================================

Purpose
-------
Verify that the R1 canonical interface works on the real RBC data flow:

    RBCDataset normalized history
        ->
    latest canonical state
        ->
    canonical PDE terms
        ->
    normalized finite-step target representation

This is an integration audit only.

It does NOT:
    - train any model;
    - evaluate rollout accuracy;
    - tune any hyperparameter;
    - access TEST samples.

Only TRAIN or VAL partitions are accepted.
"""

from __future__ import annotations

import argparse
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


from constants import B_IDX, UX_IDX, UY_IDX
from datasets.rbc_dataset import RBCDataset
from physics.canonical_metadata import (
    build_rbc_canonical_metadata,
    rbc_transport_coefficients,
)
from physics.canonical_compiler import (
    CanonicalPDECompiler,
)
from physics.derivatives import (
    grad_x_periodic,
    grad_y_nonperiodic,
)


TERM_NAMES = (
    "buoyancy_advection",
    "buoyancy_forcing",
    "momentum_advection",
    "viscosity",
    "buoyancy_diffusion",
    "pressure_gradient",
    "divergence",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="R1-4 real RBC canonical-interface audit."
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
        help="TEST is intentionally unavailable.",
    )

    parser.add_argument(
        "--max_groups",
        type=int,
        default=2,
        help=(
            "Number of split groups to audit. "
            "One real trajectory and one sample are used per group."
        ),
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


def tensor_rms(x):
    x = torch.as_tensor(
        x
    )

    return torch.sqrt(
        torch.mean(
            x.double() ** 2
        )
    ).item()


def max_abs(x):
    return torch.max(
        torch.abs(
            x.double()
        )
    ).item()


def finite_status(x):
    return bool(
        torch.isfinite(
            x
        ).all().item()
    )


def fmt_shape(x):
    return str(
        tuple(
            x.shape
        )
    )


def audit_one_entry(
    *,
    entry,
    stats_path,
    metadata,
    compiler,
):
    group = entry["group"]

    trajectories = entry[
        "trajectories"
    ]

    if not trajectories:
        raise RuntimeError(
            f"Split group {group} has no trajectories."
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

    if len(dataset) <= 0:
        raise RuntimeError(
            f"No valid samples for {group}, trajectory={trajectory}"
        )

    sample = dataset[0]

    if len(sample) != 3:
        raise RuntimeError(
            "Expected RBCDataset(return_params=True) "
            "to return (x_norm, y_norm, param)."
        )

    x_norm, y_norm, param = (
        sample
    )

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

    latest_canonical = (
        compiler.canonicalizer.latest_state_canonical(
            x_norm
        )
    )

    nu_nd, kappa_nd = (
        rbc_transport_coefficients(
            param
        )
    )

    log_ra = param[
        0,
        0,
    ].item()

    log_pr = param[
        0,
        1,
    ].item()

    ra = 10.0 ** log_ra
    pr = 10.0 ** log_pr

    print()
    print(
        "=" * 78
    )
    print(
        f"GROUP={group} | trajectory={trajectory} | sample_index=0"
    )
    print(
        "=" * 78
    )

    print(
        f"param log10(Ra),log10(Pr) = "
        f"[{log_ra:.8f}, {log_pr:.8f}]"
    )

    print(
        f"Ra={ra:.6e} | Pr={pr:.8f}"
    )

    print(
        f"nu_nd={nu_nd.item():.8e} | "
        f"kappa_nd={kappa_nd.item():.8e}"
    )

    print(
        f"nu/kappa={nu_nd.item() / kappa_nd.item():.8f} "
        f"(expected Pr={pr:.8f})"
    )

    print(
        f"nu*kappa={nu_nd.item() * kappa_nd.item():.8e} "
        f"(expected 1/Ra={1.0 / ra:.8e})"
    )

    print()
    print(
        "---------- Data-flow shapes ----------"
    )

    print(
        f"x_norm             {fmt_shape(x_norm)} "
        f"finite={finite_status(x_norm)}"
    )

    print(
        f"y_norm             {fmt_shape(y_norm)} "
        f"finite={finite_status(y_norm)}"
    )

    print(
        f"latest_canonical   {fmt_shape(latest_canonical)} "
        f"finite={finite_status(latest_canonical)}"
    )

    # ============================================================
    # Compile all seven terms.
    # ============================================================

    print()
    print(
        "---------- Canonical PDE terms ----------"
    )

    all_finite = True

    results = {}

    for term_name in TERM_NAMES:

        if term_name in (
            "viscosity",
            "buoyancy_diffusion",
        ):
            result = compiler.compile_from_history(
                term_name,
                x_norm,
                param=param,
            )

        else:
            result = compiler.compile_from_history(
                term_name,
                x_norm,
            )

        results[
            term_name
        ] = result

        canonical_finite = finite_status(
            result.canonical_value
        )

        all_finite = (
            all_finite
            and canonical_finite
        )

        if result.target_delta_norm is None:

            print(
                f"{term_name:22s} | "
                f"semantic=constraint | "
                f"canonical={fmt_shape(result.canonical_value):16s} | "
                f"rms={tensor_rms(result.canonical_value):.6e} | "
                f"finite={canonical_finite} | "
                f"delta=None"
            )

        else:

            delta_finite = finite_status(
                result.target_delta_norm
            )

            all_finite = (
                all_finite
                and delta_finite
            )

            print(
                f"{term_name:22s} | "
                f"semantic=rate       | "
                f"canonical={fmt_shape(result.canonical_value):16s} | "
                f"rate_rms={tensor_rms(result.canonical_value):.6e} | "
                f"delta_rms={tensor_rms(result.target_delta_norm):.6e} | "
                f"finite={canonical_finite and delta_finite}"
            )

    # ============================================================
    # Path-B direct regression on REAL data.
    # ============================================================

    print()
    print(
        "---------- Real Path-B regression ----------"
    )

    b = latest_canonical[
        :,
        B_IDX,
        :,
        :,
    ]

    ux = latest_canonical[
        :,
        UX_IDX,
        :,
        :,
    ]

    uy = latest_canonical[
        :,
        UY_IDX,
        :,
        :,
    ]

    direct_rate = -(
        ux
        *
        grad_x_periodic(
            b,
            metadata.grid,
        )
        +
        uy
        *
        grad_y_nonperiodic(
            b,
            metadata.grid,
        )
    )

    b_scale = (
        metadata.normalization.std_for(
            "buoyancy"
        )
        +
        metadata.normalization.eps
    )

    legacy_normalized_rate = (
        direct_rate
        /
        b_scale
    )

    canonical_delta = (
        metadata.time.dt
        *
        direct_rate
        /
        b_scale
    )

    compiled = results[
        "buoyancy_advection"
    ]

    canonical_rate_error = max_abs(
        compiled.canonical_value
        -
        direct_rate
    )

    canonical_delta_error = max_abs(
        compiled.target_delta_norm
        -
        canonical_delta
    )

    legacy_rms = tensor_rms(
        legacy_normalized_rate
    )

    canonical_rms = tensor_rms(
        canonical_delta
    )

    ratio = (
        canonical_rms
        /
        legacy_rms
        if legacy_rms > 0.0
        else float("nan")
    )

    print(
        f"direct rate RMS                = "
        f"{tensor_rms(direct_rate):.8e}"
    )

    print(
        f"legacy normalized-rate RMS     = "
        f"{legacy_rms:.8e}"
    )

    print(
        f"canonical finite-delta RMS     = "
        f"{canonical_rms:.8e}"
    )

    print(
        f"canonical / legacy RMS ratio   = "
        f"{ratio:.8f}"
    )

    print(
        f"expected dt                    = "
        f"{metadata.time.dt:.8f}"
    )

    print(
        f"compiler rate max-abs error    = "
        f"{canonical_rate_error:.8e}"
    )

    print(
        f"compiler delta max-abs error   = "
        f"{canonical_delta_error:.8e}"
    )

    # ============================================================
    # Hard integration assertions.
    # ============================================================

    if not all_finite:
        raise RuntimeError(
            "At least one real-data PDE term contains NaN/Inf."
        )

    if canonical_rate_error > 1.0e-10:
        raise RuntimeError(
            "Real Path-B canonical-rate regression failed: "
            f"{canonical_rate_error}"
        )

    if canonical_delta_error > 1.0e-10:
        raise RuntimeError(
            "Real Path-B target-delta regression failed: "
            f"{canonical_delta_error}"
        )

    if not math.isclose(
        ratio,
        metadata.time.dt,
        rel_tol=1.0e-10,
        abs_tol=1.0e-10,
    ):
        raise RuntimeError(
            "Real Path-B rate->delta ratio does not equal dt. "
            f"ratio={ratio}, dt={metadata.time.dt}"
        )

    if results[
        "divergence"
    ].target_delta_norm is not None:
        raise RuntimeError(
            "Divergence illegally entered delta space."
        )

    print()
    print(
        "✅ REAL RBC SAMPLE AUDIT = PASS"
    )

    del dataset
    del x_norm
    del y_norm
    del param
    del latest_canonical
    del results

    gc.collect()


def main():
    args = parse_args()

    split_path = resolve(
        args.split
    )

    stats_path = resolve(
        args.stats
    )

    if args.max_groups <= 0:
        raise ValueError(
            "--max_groups must be positive."
        )

    with open(
        split_path,
        "r",
        encoding="utf-8",
    ) as f:
        split = json.load(
            f
        )

    partition = args.partition

    # Explicit TEST firewall.
    if partition not in (
        "train",
        "val",
    ):
        raise RuntimeError(
            "R1-4 permits TRAIN/VAL only."
        )

    if partition not in split:
        raise KeyError(
            f"Partition '{partition}' not found in split file."
        )

    entries = split[
        partition
    ][
        : args.max_groups
    ]

    if not entries:
        raise RuntimeError(
            f"No entries found for partition={partition}."
        )

    metadata = (
        build_rbc_canonical_metadata(
            stats_path
        )
    )

    compiler = CanonicalPDECompiler(
        metadata
    )

    print(
        "=" * 78
    )
    print(
        "R1-4 REAL RBC INTEGRATION AUDIT"
    )
    print(
        "=" * 78
    )

    print(
        f"split      : {split_path}"
    )

    print(
        f"stats      : {stats_path}"
    )

    print(
        f"partition  : {partition}"
    )

    print(
        f"groups     : {len(entries)}"
    )

    print(
        "TEST       : FORBIDDEN / NOT INSTANTIATED"
    )

    print(
        f"grid       : "
        f"Nx={metadata.grid.nx}, "
        f"Ny={metadata.grid.ny}, "
        f"dx={metadata.grid.dx:.12f}, "
        f"dy={metadata.grid.dy:.12f}"
    )

    print(
        f"dt         : {metadata.time.dt}"
    )

    for entry in entries:
        audit_one_entry(
            entry=entry,
            stats_path=stats_path,
            metadata=metadata,
            compiler=compiler,
        )

    print()
    print(
        "=" * 78
    )
    print(
        f"✅ R1-4 {partition.upper()} INTEGRATION AUDIT = PASS"
    )
    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()
