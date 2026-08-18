"""
R2-3A: Normalization-Representation Scale Consistency
=====================================================

中文阶段：
    R2-3A 网络归一化表示尺度一致性验证

Goal
----
Verify that the same canonical RBC physical state can be represented
under different neural-network affine normalizations without changing:

    1. recovered canonical state;
    2. canonical PDE terms;
    3. physical finite-step PDE increments.

Only the normalized network representation is allowed to change.

Important scope
---------------
This validates invariance to normalization / representation metadata.

It does NOT yet claim invariance to arbitrary physical unit systems
(e.g. m/s -> cm/s). The current Canonicalizer assumes that the
denormalized RBC dataset field is already in canonical nondimensional
field space.

No training.
No model checkpoint.
No TEST data.
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
    sys.path.insert(0, PROJECT_ROOT)


from constants import (
    CONTEXT_LENGTH,
    NUM_FIELDS,
)

from datasets.rbc_dataset import RBCDataset

from physics.canonical_metadata import (
    CanonicalMetadata,
    FieldNormalization,
    build_rbc_canonical_metadata,
)

from physics.canonicalizer import Canonicalizer

from physics.canonical_compiler import (
    CanonicalPDECompiler,
)

from physics.rbc_terms import (
    RBC_TERM_REGISTRY,
)


# ============================================================
# CLI
# ============================================================


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "R2-3A normalization-representation "
            "scale consistency audit."
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
        choices=("train", "val"),
        default="train",
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

    parser.add_argument(
        "--tol",
        type=float,
        default=1.0e-10,
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


# ============================================================
# Numerical helpers
# ============================================================


def rms(x):

    x = x.double()

    return torch.sqrt(
        torch.mean(
            x * x
        )
    ).item()


def rel_l2(
    a,
    b,
):

    a = a.double().reshape(-1)
    b = b.double().reshape(-1)

    denominator = (
        torch.linalg.vector_norm(a)
        +
        1.0e-30
    )

    return (
        torch.linalg.vector_norm(
            b - a
        )
        /
        denominator
    ).item()


def max_abs(
    a,
    b,
):

    return torch.max(
        torch.abs(
            a.double()
            -
            b.double()
        )
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

    k = min(k, n)

    if k == 1:
        return [0]

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


def finite_mean(values):

    values = [
        float(x)
        for x in values
        if math.isfinite(float(x))
    ]

    if not values:
        return float("nan")

    return sum(values) / len(values)


# ============================================================
# Alternative normalization metadata
# ============================================================


PROFILES = {
    # Same std, different centering.
    "mean_shift": {
        "std_factor": {
            "buoyancy": 1.0,
            "u_x": 1.0,
            "u_y": 1.0,
            "pressure": 1.0,
        },
        "mean_shift_in_std": {
            "buoyancy": +1.25,
            "u_x": -0.75,
            "u_y": +0.50,
            "pressure": -1.50,
        },
    },

    # Same mean, strongly different field-wise scales.
    "std_rescale": {
        "std_factor": {
            "buoyancy": 0.50,
            "u_x": 2.00,
            "u_y": 1.50,
            "pressure": 0.75,
        },
        "mean_shift_in_std": {
            "buoyancy": 0.0,
            "u_x": 0.0,
            "u_y": 0.0,
            "pressure": 0.0,
        },
    },

    # Both centering and scales change.
    "mixed_affine": {
        "std_factor": {
            "buoyancy": 1.70,
            "u_x": 0.60,
            "u_y": 2.30,
            "pressure": 1.20,
        },
        "mean_shift_in_std": {
            "buoyancy": -0.80,
            "u_x": +1.10,
            "u_y": -0.40,
            "pressure": +0.90,
        },
    },
}


def build_alternative_metadata(
    base_metadata,
    profile_name,
):

    profile = PROFILES[
        profile_name
    ]

    base_norm = (
        base_metadata
        .normalization
    )

    alt_mean = {}
    alt_std = {}

    for field in base_metadata.field_order:

        mean = (
            base_norm
            .mean_for(field)
        )

        std = (
            base_norm
            .std_for(field)
        )

        factor = (
            profile[
                "std_factor"
            ][field]
        )

        shift = (
            profile[
                "mean_shift_in_std"
            ][field]
        )

        if factor <= 0.0:
            raise ValueError(
                f"Non-positive std factor for {field}"
            )

        alt_mean[field] = (
            mean
            +
            shift
            *
            std
        )

        alt_std[field] = (
            factor
            *
            std
        )

    alt_normalization = FieldNormalization(
        mean=alt_mean,
        std=alt_std,
        eps=base_norm.eps,
    )

    return CanonicalMetadata(
        field_order=base_metadata.field_order,
        normalization=alt_normalization,
        grid=base_metadata.grid,
        time=base_metadata.time,
        parameter_order=base_metadata.parameter_order,
    )


# ============================================================
# Representation transforms
# ============================================================


def history_to_canonical(
    history_norm,
    canonicalizer,
):

    batch = history_norm.shape[0]

    nx = history_norm.shape[-2]
    ny = history_norm.shape[-1]

    history_view = history_norm.reshape(
        batch,
        CONTEXT_LENGTH,
        NUM_FIELDS,
        nx,
        ny,
    )

    return canonicalizer.denormalize_state(
        history_view
    )


def canonical_to_history_norm(
    history_canonical,
    canonicalizer,
):

    normalized = canonicalizer.normalize_state(
        history_canonical
    )

    batch = normalized.shape[0]
    nx = normalized.shape[-2]
    ny = normalized.shape[-1]

    return normalized.reshape(
        batch,
        CONTEXT_LENGTH * NUM_FIELDS,
        nx,
        ny,
    )


# ============================================================
# Target-scale transformation law
# ============================================================


def target_scale_tensor(
    metadata,
    spec,
    reference,
):

    eps = (
        metadata
        .normalization
        .eps
    )

    if len(spec.targets) == 1:

        value = (
            metadata
            .normalization
            .std_for(
                spec.targets[0]
            )
            +
            eps
        )

        return torch.as_tensor(
            value,
            dtype=reference.dtype,
            device=reference.device,
        )

    if len(spec.targets) == 2:

        values = [
            (
                metadata
                .normalization
                .std_for(target)
                +
                eps
            )
            for target in spec.targets
        ]

        return torch.as_tensor(
            values,
            dtype=reference.dtype,
            device=reference.device,
        ).view(
            1,
            2,
            1,
            1,
        )

    raise RuntimeError(
        f"Unsupported targets: {spec.targets}"
    )


# ============================================================
# Per-sample audit
# ============================================================


def audit_sample(
    *,
    base_metadata,
    base_compiler,
    x_norm,
    param,
    group,
    trajectory,
    sample_index,
):

    base_canonicalizer = (
        base_compiler
        .canonicalizer
    )

    # --------------------------------------------------------
    # Recover the SAME physical canonical history once.
    # --------------------------------------------------------

    history_canonical = history_to_canonical(
        x_norm,
        base_canonicalizer,
    )

    base_latest = (
        base_canonicalizer
        .latest_state_canonical(
            x_norm
        )
    )

    rows = []

    for profile_name in PROFILES:

        alt_metadata = (
            build_alternative_metadata(
                base_metadata,
                profile_name,
            )
        )

        alt_canonicalizer = (
            Canonicalizer(
                alt_metadata
            )
        )

        alt_compiler = (
            CanonicalPDECompiler(
                alt_metadata
            )
        )

        # ----------------------------------------------------
        # Encode SAME canonical history using a different
        # neural-network normalization representation.
        # ----------------------------------------------------

        alt_history_norm = (
            canonical_to_history_norm(
                history_canonical,
                alt_canonicalizer,
            )
        )

        alt_latest = (
            alt_canonicalizer
            .latest_state_canonical(
                alt_history_norm
            )
        )

        representation_change_rms = rms(
            alt_history_norm
            -
            x_norm
        )

        state_rel = rel_l2(
            base_latest,
            alt_latest,
        )

        state_max = max_abs(
            base_latest,
            alt_latest,
        )

        # ----------------------------------------------------
        # Canonical PDE terms
        # ----------------------------------------------------

        for (
            term_name,
            spec,
        ) in RBC_TERM_REGISTRY.items():

            term_param = (
                param
                if spec.requires_param_coefficients
                else None
            )

            base_term = (
                base_compiler
                .compile_from_history(
                    term_name,
                    x_norm,
                    param=term_param,
                )
            )

            alt_term = (
                alt_compiler
                .compile_from_history(
                    term_name,
                    alt_history_norm,
                    param=term_param,
                )
            )

            canonical_rel = rel_l2(
                base_term.canonical_value,
                alt_term.canonical_value,
            )

            canonical_max = max_abs(
                base_term.canonical_value,
                alt_term.canonical_value,
            )

            delta_rebased_rel = float("nan")
            delta_rebased_max = float("nan")

            if base_term.target_delta_norm is not None:

                base_scale = (
                    target_scale_tensor(
                        base_metadata,
                        spec,
                        base_term.target_delta_norm,
                    )
                )

                alt_scale = (
                    target_scale_tensor(
                        alt_metadata,
                        spec,
                        alt_term.target_delta_norm,
                    )
                )

                # Convert both normalized delta representations
                # back to the common physical finite-step increment:
                #
                #     delta_phys = scale_target * delta_norm
                #
                # Both must equal dt * canonical_rate.
                base_delta_phys = (
                    base_term.target_delta_norm
                    *
                    base_scale
                )

                alt_delta_phys = (
                    alt_term.target_delta_norm
                    *
                    alt_scale
                )

                delta_rebased_rel = rel_l2(
                    base_delta_phys,
                    alt_delta_phys,
                )

                delta_rebased_max = max_abs(
                    base_delta_phys,
                    alt_delta_phys,
                )

            rows.append(
                {
                    "group": group,
                    "trajectory": trajectory,
                    "sample_index": sample_index,

                    "profile": profile_name,
                    "term": term_name,

                    "representation_change_rms":
                        representation_change_rms,

                    "state_canonical_rel_l2":
                        state_rel,

                    "state_canonical_max_abs":
                        state_max,

                    "term_canonical_rel_l2":
                        canonical_rel,

                    "term_canonical_max_abs":
                        canonical_max,

                    "delta_rebased_rel_l2":
                        delta_rebased_rel,

                    "delta_rebased_max_abs":
                        delta_rebased_max,
                }
            )

    return rows


# ============================================================
# Reporting
# ============================================================


def print_profile_summary(
    rows,
    profile_name,
):

    subset = [
        row
        for row in rows
        if row["profile"] == profile_name
    ]

    print()
    print(
        f"[{profile_name}]"
    )

    print(
        "  representation_change_rms mean = "
        f"{finite_mean([r['representation_change_rms'] for r in subset]):.6e}"
    )

    print(
        "  state canonical max abs        = "
        f"{max(r['state_canonical_max_abs'] for r in subset):.6e}"
    )

    print(
        "  PDE canonical max abs          = "
        f"{max(r['term_canonical_max_abs'] for r in subset):.6e}"
    )

    delta_values = [
        r["delta_rebased_max_abs"]
        for r in subset
        if math.isfinite(
            float(
                r["delta_rebased_max_abs"]
            )
        )
    ]

    print(
        "  rebased delta max abs          = "
        f"{max(delta_values):.6e}"
    )


def write_csv(
    rows,
    path,
):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )

    with open(
        path,
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
        writer.writerows(rows)


# ============================================================
# Main
# ============================================================


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

    base_metadata = (
        build_rbc_canonical_metadata(
            stats_path
        )
    )

    base_compiler = (
        CanonicalPDECompiler(
            base_metadata
        )
    )

    print(
        "=" * 96
    )

    print(
        "R2-3A 网络归一化表示尺度一致性验证"
    )

    print(
        "=" * 96
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
        f"profiles          : {tuple(PROFILES.keys())}"
    )

    print(
        f"tolerance         : {args.tol:.3e}"
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

            x_norm, _, param = dataset[
                sample_index
            ]

            x_norm = torch.as_tensor(
                x_norm,
                dtype=torch.float64,
            ).unsqueeze(0)

            param = torch.as_tensor(
                param,
                dtype=torch.float64,
            ).unsqueeze(0)

            rows = audit_sample(
                base_metadata=base_metadata,
                base_compiler=base_compiler,
                x_norm=x_norm,
                param=param,
                group=group,
                trajectory=trajectory,
                sample_index=sample_index,
            )

            all_rows.extend(rows)

        del dataset
        gc.collect()

    if not all_rows:
        raise RuntimeError(
            "No audit rows."
        )

    write_csv(
        all_rows,
        output_path,
    )

    print()
    print(
        "=" * 96
    )

    print(
        "R2-3A 汇总"
    )

    print(
        "=" * 96
    )

    for profile_name in PROFILES:
        print_profile_summary(
            all_rows,
            profile_name,
        )

    max_state = max(
        row["state_canonical_max_abs"]
        for row in all_rows
    )

    max_term = max(
        row["term_canonical_max_abs"]
        for row in all_rows
    )

    delta_values = [
        row["delta_rebased_max_abs"]
        for row in all_rows
        if math.isfinite(
            float(
                row["delta_rebased_max_abs"]
            )
        )
    ]

    max_delta = max(
        delta_values
    )

    representation_changes = [
        row["representation_change_rms"]
        for row in all_rows
    ]

    min_representation_change = min(
        representation_changes
    )

    print()
    print(
        "-" * 96
    )

    print(
        f"min representation change RMS = "
        f"{min_representation_change:.6e}"
    )

    print(
        f"max canonical state error      = "
        f"{max_state:.6e}"
    )

    print(
        f"max canonical PDE-term error   = "
        f"{max_term:.6e}"
    )

    print(
        f"max rebased delta error        = "
        f"{max_delta:.6e}"
    )

    pass_representation = (
        min_representation_change
        >
        1.0e-6
    )

    pass_invariance = (
        max_state <= args.tol
        and
        max_term <= args.tol
        and
        max_delta <= args.tol
    )

    print()

    print(
        "representation actually changed : "
        f"{'PASS' if pass_representation else 'FAIL'}"
    )

    print(
        "canonical invariance             : "
        f"{'PASS' if pass_invariance else 'FAIL'}"
    )

    print()
    print(
        f"CSV saved: {output_path}"
    )

    print()

    if (
        pass_representation
        and
        pass_invariance
    ):

        print(
            "✅ R2-3A 网络归一化表示尺度一致性验证 = PASS"
        )

    else:

        raise RuntimeError(
            "R2-3A invariance audit failed."
        )


if __name__ == "__main__":
    main()
