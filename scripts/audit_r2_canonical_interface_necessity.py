"""
R2-3B: Necessity of the Canonical Physical Interface
=====================================================

中文：
    R2-3B 标准物理接口必要性对照

Question
--------
For the SAME physical RBC state represented under different
network normalizations:

Correct path:
    normalized representation
        -> matching Canonicalizer
        -> canonical physical state
        -> PDE terms

Naive control:
    normalized representation
        -> directly treated as physical state
        -> PDE terms

Expected:
    Correct canonical PDE terms remain invariant.
    Naive PDE terms depend on arbitrary normalization choices.

Scope
-----
This is a representation-scale necessity control.

It does NOT yet claim arbitrary physical-unit invariance.

No training.
No checkpoint.
TEST is forbidden.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
from collections import defaultdict

import torch


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from constants import CONTEXT_LENGTH, NUM_FIELDS
from datasets.rbc_dataset import RBCDataset

from physics.canonical_metadata import (
    CanonicalMetadata,
    FieldNormalization,
    build_rbc_canonical_metadata,
)

from physics.canonicalizer import Canonicalizer
from physics.canonical_compiler import CanonicalPDECompiler
from physics.rbc_terms import RBC_TERM_REGISTRY


# ============================================================
# Representation perturbations
# ============================================================

PROFILES = {
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


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "R2-3B canonical-interface necessity control."
        )
    )

    parser.add_argument("--split", required=True)
    parser.add_argument("--stats", required=True)

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
    )

    parser.add_argument(
        "--canonical_tol",
        type=float,
        default=1.0e-10,
    )

    parser.add_argument(
        "--naive_mean_floor",
        type=float,
        default=1.0e-3,
    )

    return parser.parse_args()


def resolve(path):

    if os.path.isabs(path):
        return path

    return os.path.abspath(
        os.path.join(PROJECT_ROOT, path)
    )


# ============================================================
# Numerical helpers
# ============================================================

def rms(x):

    x = x.double()

    return torch.sqrt(
        torch.mean(x * x)
    ).item()


def max_abs(a, b):

    return torch.max(
        torch.abs(
            a.double() - b.double()
        )
    ).item()


def symmetric_rel_l2(a, b):

    a = a.double().reshape(-1)
    b = b.double().reshape(-1)

    na = torch.linalg.vector_norm(a)
    nb = torch.linalg.vector_norm(b)

    denom = (
        0.5 * (na + nb)
        +
        1.0e-30
    )

    return (
        torch.linalg.vector_norm(a - b)
        /
        denom
    ).item()


def finite_mean(values):

    values = [
        float(v)
        for v in values
        if math.isfinite(float(v))
    ]

    if not values:
        return float("nan")

    return sum(values) / len(values)


def choose_indices(n, k):

    if n <= 0:
        raise RuntimeError("Empty dataset.")

    k = min(int(k), n)

    if k <= 0:
        raise ValueError(
            "samples_per_group must be positive."
        )

    if k == 1:
        return [0]

    result = []

    for i in range(k):

        idx = round(
            i * (n - 1) / (k - 1)
        )

        if idx not in result:
            result.append(idx)

    return result


# ============================================================
# Metadata
# ============================================================

def build_alternative_metadata(
    base_metadata,
    profile_name,
):

    profile = PROFILES[profile_name]
    norm = base_metadata.normalization

    means = {}
    stds = {}

    for field in base_metadata.field_order:

        base_mean = norm.mean_for(field)
        base_std = norm.std_for(field)

        means[field] = (
            base_mean
            +
            profile["mean_shift_in_std"][field]
            *
            base_std
        )

        stds[field] = (
            profile["std_factor"][field]
            *
            base_std
        )

    alt_norm = FieldNormalization(
        mean=means,
        std=stds,
        eps=norm.eps,
    )

    return CanonicalMetadata(
        field_order=base_metadata.field_order,
        normalization=alt_norm,
        grid=base_metadata.grid,
        time=base_metadata.time,
        parameter_order=base_metadata.parameter_order,
    )


def build_identity_metadata(base_metadata):
    """
    Fake metadata for the naive control.

    Canonicalizer becomes identity:
        x_canonical = x_norm

    because:
        mean = 0
        std + eps = 1
    """

    eps = base_metadata.normalization.eps

    means = {
        field: 0.0
        for field in base_metadata.field_order
    }

    stds = {
        field: 1.0 - eps
        for field in base_metadata.field_order
    }

    identity_norm = FieldNormalization(
        mean=means,
        std=stds,
        eps=eps,
    )

    return CanonicalMetadata(
        field_order=base_metadata.field_order,
        normalization=identity_norm,
        grid=base_metadata.grid,
        time=base_metadata.time,
        parameter_order=base_metadata.parameter_order,
    )


# ============================================================
# History representation
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
# One sample
# ============================================================

def audit_sample(
    *,
    base_metadata,
    base_compiler,
    naive_compiler,
    x_norm,
    param,
    group,
    trajectory,
    sample_index,
):

    base_canonicalizer = (
        base_compiler.canonicalizer
    )

    # Same actual physical history.
    history_canonical = history_to_canonical(
        x_norm,
        base_canonicalizer,
    )

    rows = []

    for profile_name in PROFILES:

        alt_metadata = build_alternative_metadata(
            base_metadata,
            profile_name,
        )

        alt_canonicalizer = Canonicalizer(
            alt_metadata
        )

        alt_compiler = CanonicalPDECompiler(
            alt_metadata
        )

        # Encode SAME physical history under
        # another network normalization.
        alt_history_norm = canonical_to_history_norm(
            history_canonical,
            alt_canonicalizer,
        )

        representation_change = rms(
            alt_history_norm - x_norm
        )

        for term_name, spec in RBC_TERM_REGISTRY.items():

            term_param = (
                param
                if spec.requires_param_coefficients
                else None
            )

            # --------------------------------------------
            # Correct canonicalized path
            # --------------------------------------------

            base_correct = (
                base_compiler.compile_from_history(
                    term_name,
                    x_norm,
                    param=term_param,
                )
            )

            alt_correct = (
                alt_compiler.compile_from_history(
                    term_name,
                    alt_history_norm,
                    param=term_param,
                )
            )

            canonical_sym_rel = symmetric_rel_l2(
                base_correct.canonical_value,
                alt_correct.canonical_value,
            )

            canonical_max = max_abs(
                base_correct.canonical_value,
                alt_correct.canonical_value,
            )

            # --------------------------------------------
            # Naive path:
            #
            # Interpret normalized numerical values
            # directly as if they were canonical physics.
            # --------------------------------------------

            base_naive = (
                naive_compiler.compile_from_history(
                    term_name,
                    x_norm,
                    param=term_param,
                )
            )

            alt_naive = (
                naive_compiler.compile_from_history(
                    term_name,
                    alt_history_norm,
                    param=term_param,
                )
            )

            naive_sym_rel = symmetric_rel_l2(
                base_naive.canonical_value,
                alt_naive.canonical_value,
            )

            naive_max = max_abs(
                base_naive.canonical_value,
                alt_naive.canonical_value,
            )

            rows.append(
                {
                    "group": group,
                    "trajectory": trajectory,
                    "sample_index": sample_index,
                    "profile": profile_name,
                    "term": term_name,

                    "representation_change_rms":
                        representation_change,

                    "canonical_sym_rel_l2":
                        canonical_sym_rel,

                    "canonical_max_abs":
                        canonical_max,

                    "naive_sym_rel_l2":
                        naive_sym_rel,

                    "naive_max_abs":
                        naive_max,
                }
            )

    return rows


# ============================================================
# Reporting
# ============================================================

def write_csv(rows, output_path):

    os.makedirs(
        os.path.dirname(output_path),
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
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def print_profile_summary(
    rows,
    profile_name,
):

    subset = [
        r for r in rows
        if r["profile"] == profile_name
    ]

    canonical_mean = finite_mean(
        [
            r["canonical_sym_rel_l2"]
            for r in subset
        ]
    )

    naive_mean = finite_mean(
        [
            r["naive_sym_rel_l2"]
            for r in subset
        ]
    )

    canonical_max = max(
        r["canonical_max_abs"]
        for r in subset
    )

    naive_max_rel = max(
        r["naive_sym_rel_l2"]
        for r in subset
    )

    print()
    print(f"[{profile_name}]")

    print(
        "  representation change RMS mean = "
        f"{finite_mean([r['representation_change_rms'] for r in subset]):.6e}"
    )

    print(
        "  canonical sym-rel mean          = "
        f"{canonical_mean:.6e}"
    )

    print(
        "  canonical max abs               = "
        f"{canonical_max:.6e}"
    )

    print(
        "  naive sym-rel mean              = "
        f"{naive_mean:.6e}"
    )

    print(
        "  naive sym-rel max               = "
        f"{naive_max_rel:.6e}"
    )

    # Term-wise mean naive sensitivity.
    by_term = defaultdict(list)

    for row in subset:
        by_term[row["term"]].append(
            row["naive_sym_rel_l2"]
        )

    ranked = sorted(
        (
            (
                term,
                finite_mean(values),
            )
            for term, values in by_term.items()
        ),
        key=lambda x: x[1],
        reverse=True,
    )

    print(
        "  most representation-sensitive naive terms:"
    )

    for term, value in ranked[:5]:
        print(
            f"    {term:<28s} {value:.6e}"
        )

    return canonical_mean, naive_mean


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    split_path = resolve(args.split)
    stats_path = resolve(args.stats)
    output_path = resolve(args.output)

    with open(
        split_path,
        "r",
        encoding="utf-8",
    ) as f:
        split = json.load(f)

    entries = split[
        args.partition
    ][:args.max_groups]

    if not entries:
        raise RuntimeError(
            "No split entries."
        )

    base_metadata = build_rbc_canonical_metadata(
        stats_path
    )

    base_compiler = CanonicalPDECompiler(
        base_metadata
    )

    identity_metadata = build_identity_metadata(
        base_metadata
    )

    naive_compiler = CanonicalPDECompiler(
        identity_metadata
    )

    print("=" * 100)
    print(
        "R2-3B 标准物理接口必要性对照"
    )
    print("=" * 100)

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
        f"canonical_tol     : {args.canonical_tol:.3e}"
    )
    print(
        f"naive_mean_floor  : {args.naive_mean_floor:.3e}"
    )
    print(
        "TEST              : FORBIDDEN / NOT INSTANTIATED"
    )

    all_rows = []

    for entry in entries:

        group = entry["group"]
        trajectories = entry["trajectories"]

        if not trajectories:
            raise RuntimeError(
                f"No trajectories for {group}"
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
                naive_compiler=naive_compiler,
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
    print("=" * 100)
    print("R2-3B 汇总")
    print("=" * 100)

    profile_naive_means = []
    profile_canonical_means = []

    for profile_name in PROFILES:

        canonical_mean, naive_mean = (
            print_profile_summary(
                all_rows,
                profile_name,
            )
        )

        profile_canonical_means.append(
            canonical_mean
        )

        profile_naive_means.append(
            naive_mean
        )

    max_canonical_abs = max(
        row["canonical_max_abs"]
        for row in all_rows
    )

    min_representation_change = min(
        row["representation_change_rms"]
        for row in all_rows
    )

    min_naive_profile_mean = min(
        profile_naive_means
    )

    overall_canonical_mean = finite_mean(
        [
            r["canonical_sym_rel_l2"]
            for r in all_rows
        ]
    )

    overall_naive_mean = finite_mean(
        [
            r["naive_sym_rel_l2"]
            for r in all_rows
        ]
    )

    print()
    print("-" * 100)

    print(
        "min representation change RMS = "
        f"{min_representation_change:.6e}"
    )

    print(
        "max canonical PDE error       = "
        f"{max_canonical_abs:.6e}"
    )

    print(
        "overall canonical sym-rel     = "
        f"{overall_canonical_mean:.6e}"
    )

    print(
        "overall naive sym-rel         = "
        f"{overall_naive_mean:.6e}"
    )

    print(
        "min profile naive sym-rel     = "
        f"{min_naive_profile_mean:.6e}"
    )

    pass_representation = (
        min_representation_change
        >
        1.0e-6
    )

    pass_canonical = (
        max_canonical_abs
        <=
        args.canonical_tol
    )

    pass_naive_dependence = (
        min_naive_profile_mean
        >=
        args.naive_mean_floor
    )

    print()

    print(
        "representation actually changed : "
        f"{'PASS' if pass_representation else 'FAIL'}"
    )

    print(
        "canonical path invariant         : "
        f"{'PASS' if pass_canonical else 'FAIL'}"
    )

    print(
        "naive path representation-bound  : "
        f"{'PASS' if pass_naive_dependence else 'FAIL'}"
    )

    print()
    print(
        f"CSV saved: {output_path}"
    )
    print()

    if (
        pass_representation
        and
        pass_canonical
        and
        pass_naive_dependence
    ):

        print(
            "✅ R2-3B 标准物理接口必要性对照 = PASS"
        )

    else:

        raise RuntimeError(
            "R2-3B necessity control failed."
        )


if __name__ == "__main__":
    main()
