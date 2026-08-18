from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from datasets.rbc_dataset import RBCDataset

from models.operators.fno2d_r3_pde_coupling import (
    R3PDECouplingFNO2d,
)

from physics.canonical_metadata import (
    build_rbc_canonical_metadata,
)

from scripts.train_r3_1_pde_coupling_h4 import (
    R3MultiStepParamDataset,
    extract_m6_state,
    set_seed,
    sha256_file,
)


# ============================================================
# PREDECLARED BEFORE FORMAL RESULTS
# ============================================================

ROLLOUT_STEPS = 4

REPRESENTATION_MODES = (
    "naive",
    "canonical",
)

ACTIVE_TERMS = (
    "buoyancy_advection",
    "buoyancy_forcing",
)

# ------------------------------------------------------------
# R3-0C:
#
# Each PDE term may have its own fixed safety ceiling.
#
# The formal bound for each term is selected independently,
# but MUST then be shared exactly by R3-1a and R3-1b.
#
# 0.25 remains only the common historical candidate ceiling.
# ------------------------------------------------------------

CANDIDATE_ALPHA_MAX_BY_TERM = {
    "buoyancy_advection": 0.25,
    "buoyancy_forcing": 0.25,
}

# At TRAIN-support p99, a fully saturated individual physics
# path should contribute at most 50% of the corresponding
# frozen-M6 target-field delta RMS.
SAFETY_TARGET_FRACTION = 0.50

RATIO_EPS = 1.0e-8


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "R3-1 TRAIN-only per-term matched-capacity "
            "safety audit. "
            "No training, no VAL tuning, TEST forbidden."
        )
    )

    p.add_argument(
        "--split",
        required=True,
    )

    p.add_argument(
        "--stats",
        required=True,
    )

    p.add_argument(
        "--m6_checkpoint",
        required=True,
    )

    p.add_argument(
        "--split_label",
        required=True,
        choices=(
            "unseen_pr",
            "unseen_ra",
        ),
    )

    p.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    p.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help=(
            "DEBUG ONLY. "
            "Formal alpha audit must omit this."
        ),
    )

    p.add_argument(
        "--output_prefix",
        required=True,
    )

    return p.parse_args()


# ============================================================
# Numerical helpers
# ============================================================

def spatial_rms(x):
    return torch.sqrt(
        torch.mean(
            x * x,
            dim=(-2, -1),
        )
    )


def summarize_values(values):
    x = np.asarray(
        values,
        dtype=np.float64,
    )

    if x.size == 0:
        raise RuntimeError(
            "Cannot summarize empty values."
        )

    return {
        "count":
            int(x.size),

        "mean":
            float(
                np.mean(x)
            ),

        "p50":
            float(
                np.quantile(
                    x,
                    0.50,
                )
            ),

        "p90":
            float(
                np.quantile(
                    x,
                    0.90,
                )
            ),

        "p95":
            float(
                np.quantile(
                    x,
                    0.95,
                )
            ),

        "p99":
            float(
                np.quantile(
                    x,
                    0.99,
                )
            ),

        "max":
            float(
                np.max(x)
            ),
    }


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    for path in (
        args.split,
        args.stats,
        args.m6_checkpoint,
    ):
        if not os.path.exists(path):
            raise FileNotFoundError(
                path
            )

    if (
        set(
            CANDIDATE_ALPHA_MAX_BY_TERM
        )
        !=
        set(ACTIVE_TERMS)
    ):
        raise RuntimeError(
            "Candidate alpha map does not match "
            "active R3 terms."
        )

    set_seed(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )

    print(
        "=" * 118
    )

    print(
        "R3-1 TRAIN-ONLY PER-TERM "
        "MATCHED-CAPACITY SAFETY AUDIT"
    )

    print(
        "=" * 118
    )

    print(
        "Split:",
        args.split_label,
    )

    print(
        "Training: OFF"
    )

    print(
        "VAL tuning: OFF"
    )

    print(
        "TEST access: FORBIDDEN"
    )

    print(
        "State source: "
        "same frozen-M6 closed-loop H4 states"
    )

    print(
        "Candidate alpha max by term:",
        CANDIDATE_ALPHA_MAX_BY_TERM,
    )

    print(
        "Safety target fraction:",
        SAFETY_TARGET_FRACTION,
    )

    print(
        "Matched rule: "
        "each term gets one bound shared by "
        "Naive + Canonical"
    )

    if args.max_batches is not None:

        print(
            "⚠️ DEBUG ONLY: max_batches =",
            args.max_batches,
        )

        print(
            "⚠️ This run cannot lock formal alpha."
        )

    # ========================================================
    # Provenance
    # ========================================================

    split_sha = sha256_file(
        args.split
    )

    stats_sha = sha256_file(
        args.stats
    )

    m6_sha = sha256_file(
        args.m6_checkpoint
    )

    print()
    print(
        "========== PROVENANCE =========="
    )

    print(
        "SPLIT_SHA256:",
        split_sha,
    )

    print(
        "STATS_SHA256:",
        stats_sha,
    )

    print(
        "M6_SHA256:",
        m6_sha,
    )

    # ========================================================
    # TRAIN-only data
    # ========================================================

    with open(
        args.split,
        "r",
        encoding="utf-8",
    ) as f:
        split = json.load(f)

    train_base = RBCDataset(
        split_config=split[
            "train"
        ],
        normalize=True,
        stats_path=args.stats,
        return_sequence=True,
        target_steps=(
            ROLLOUT_STEPS
        ),
        return_params=True,
    )

    train_dataset = (
        R3MultiStepParamDataset(
            train_base,
            context_length=4,
        )
    )

    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    print()
    print(
        "TRAIN_WINDOWS:",
        len(train_dataset),
    )

    print(
        "BATCHES:",
        len(loader),
    )

    # ========================================================
    # Build exact matched Naive / Canonical pair
    # ========================================================

    metadata = (
        build_rbc_canonical_metadata(
            args.stats
        )
    )

    set_seed(
        args.seed
    )

    canonical = R3PDECouplingFNO2d(
        canonical_metadata=metadata,
        representation_mode="canonical",
        alpha_max_by_term=(
            CANDIDATE_ALPHA_MAX_BY_TERM
        ),
        freeze_m6=True,
    ).to(device)

    naive = R3PDECouplingFNO2d(
        canonical_metadata=metadata,
        representation_mode="naive",
        alpha_max_by_term=(
            CANDIDATE_ALPHA_MAX_BY_TERM
        ),
        freeze_m6=True,
    ).to(device)

    checkpoint = torch.load(
        args.m6_checkpoint,
        map_location=device,
    )

    m6_state = extract_m6_state(
        checkpoint
    )

    canonical.load_m6_state_dict(
        m6_state
    )

    naive.load_m6_state_dict(
        m6_state
    )

    canonical.eval()
    naive.eval()

    # --------------------------------------------------------
    # Exact embedded-M6 audit
    # --------------------------------------------------------

    canonical_m6_state = (
        canonical.m6.state_dict()
    )

    naive_m6_state = (
        naive.m6.state_dict()
    )

    if (
        list(
            canonical_m6_state.keys()
        )
        !=
        list(
            naive_m6_state.keys()
        )
    ):
        raise RuntimeError(
            "Embedded M6 state keys differ."
        )

    for name in (
        canonical_m6_state
    ):

        torch.testing.assert_close(
            canonical_m6_state[name],
            naive_m6_state[name],
            rtol=0.0,
            atol=0.0,
            msg=(
                "Embedded M6 mismatch at "
                f"{name}"
            ),
        )

    print(
        "Embedded M6 exact match: PASS"
    )

    # --------------------------------------------------------
    # Exact per-term capacity-map audit
    # --------------------------------------------------------

    if (
        canonical.alpha_max_by_term
        !=
        naive.alpha_max_by_term
    ):
        raise RuntimeError(
            "Naive / Canonical capacity maps differ."
        )

    if (
        canonical.alpha_max_by_term
        !=
        CANDIDATE_ALPHA_MAX_BY_TERM
    ):
        raise RuntimeError(
            "Model capacity map differs from "
            "predeclared audit candidate."
        )

    print(
        "Per-term capacity map exact match: PASS"
    )

    # ========================================================
    # Buckets
    # ========================================================

    buckets = {}

    for mode in (
        REPRESENTATION_MODES
    ):

        for term in (
            ACTIVE_TERMS
        ):

            for step in range(
                1,
                ROLLOUT_STEPS + 1,
            ):

                buckets[
                    (
                        mode,
                        term,
                        step,
                    )
                ] = {
                    "signal_rms": [],
                    "base_delta_rms": [],
                    "ratio": [],
                }

    models = {
        "naive":
            naive,

        "canonical":
            canonical,
    }

    # ========================================================
    # Frozen-M6 closed-loop H4 support
    #
    # IMPORTANT:
    #
    # Both representation arms are evaluated on exactly the
    # SAME frozen-M6 rollout states.
    #
    # Neither Naive nor Canonical is allowed to generate its
    # own state trajectory during this safety audit.
    # ========================================================

    with torch.no_grad():

        for (
            batch_idx,
            batch,
        ) in enumerate(
            loader
        ):

            if (
                args.max_batches
                is not None
                and
                batch_idx
                >=
                args.max_batches
            ):
                break

            (
                context_norm,
                _,
                param,
            ) = batch

            context = (
                context_norm.to(
                    device
                )
            )

            param = param.to(
                device
            )

            for step in range(
                1,
                ROLLOUT_STEPS + 1,
            ):

                (
                    bsz,
                    clen,
                    channels,
                    h,
                    w,
                ) = context.shape

                model_input = (
                    context.reshape(
                        bsz,
                        clen * channels,
                        h,
                        w,
                    )
                )

                # Same frozen-M6 base delta for both arms.
                base_delta = (
                    canonical.m6(
                        model_input
                    )
                )

                for (
                    mode,
                    model,
                ) in models.items():

                    for term in (
                        ACTIVE_TERMS
                    ):

                        compiled = (
                            model.compiler
                            .compile_from_history(
                                term,
                                model_input,
                            )
                        )

                        signal = (
                            compiled
                            .target_delta_norm
                        )

                        if signal is None:
                            raise RuntimeError(
                                f"{term} returned "
                                "no target_delta_norm."
                            )

                        output_index = (
                            model
                            .TERM_TO_OUTPUT_INDEX[
                                term
                            ]
                        )

                        base_target = (
                            base_delta[
                                :,
                                output_index,
                            ]
                        )

                        signal_rms = (
                            spatial_rms(
                                signal
                            )
                        )

                        base_rms = (
                            spatial_rms(
                                base_target
                            )
                        )

                        ratio = (
                            signal_rms
                            /
                            torch.clamp(
                                base_rms,
                                min=RATIO_EPS,
                            )
                        )

                        if not (
                            torch.isfinite(
                                signal_rms
                            ).all()
                            and
                            torch.isfinite(
                                base_rms
                            ).all()
                            and
                            torch.isfinite(
                                ratio
                            ).all()
                        ):
                            raise RuntimeError(
                                "Non-finite R3 "
                                "support statistic."
                            )

                        bucket = buckets[
                            (
                                mode,
                                term,
                                step,
                            )
                        ]

                        bucket[
                            "signal_rms"
                        ].extend(
                            signal_rms
                            .double()
                            .cpu()
                            .tolist()
                        )

                        bucket[
                            "base_delta_rms"
                        ].extend(
                            base_rms
                            .double()
                            .cpu()
                            .tolist()
                        )

                        bucket[
                            "ratio"
                        ].extend(
                            ratio
                            .double()
                            .cpu()
                            .tolist()
                        )

                # Advance ONLY with frozen M6.
                current_state = (
                    context[:, -1]
                )

                pred_next = (
                    current_state
                    +
                    base_delta
                )

                context = torch.cat(
                    [
                        context[:, 1:],
                        pred_next.unsqueeze(
                            1
                        ),
                    ],
                    dim=1,
                )

            if (
                (batch_idx + 1)
                % 50
                == 0
            ):
                print(
                    "processed",
                    batch_idx + 1,
                    "/",
                    len(loader),
                )

    # ========================================================
    # Detailed summary rows
    # ========================================================

    rows = []

    for (
        mode,
        term,
        step,
    ), bucket in buckets.items():

        signal_stats = (
            summarize_values(
                bucket[
                    "signal_rms"
                ]
            )
        )

        base_stats = (
            summarize_values(
                bucket[
                    "base_delta_rms"
                ]
            )
        )

        ratio_stats = (
            summarize_values(
                bucket[
                    "ratio"
                ]
            )
        )

        candidate_alpha = float(
            CANDIDATE_ALPHA_MAX_BY_TERM[
                term
            ]
        )

        rows.append(
            {
                "split_label":
                    args.split_label,

                "representation_mode":
                    mode,

                "term":
                    term,

                "step":
                    step,

                "samples":
                    ratio_stats[
                        "count"
                    ],

                "signal_rms_mean":
                    signal_stats[
                        "mean"
                    ],

                "signal_rms_p99":
                    signal_stats[
                        "p99"
                    ],

                "signal_rms_max":
                    signal_stats[
                        "max"
                    ],

                "base_delta_rms_mean":
                    base_stats[
                        "mean"
                    ],

                "base_delta_rms_p50":
                    base_stats[
                        "p50"
                    ],

                "ratio_mean":
                    ratio_stats[
                        "mean"
                    ],

                "ratio_p50":
                    ratio_stats[
                        "p50"
                    ],

                "ratio_p90":
                    ratio_stats[
                        "p90"
                    ],

                "ratio_p95":
                    ratio_stats[
                        "p95"
                    ],

                "ratio_p99":
                    ratio_stats[
                        "p99"
                    ],

                "ratio_max":
                    ratio_stats[
                        "max"
                    ],

                "candidate_alpha":
                    candidate_alpha,

                "candidate_p99_injection_fraction":
                    (
                        candidate_alpha
                        *
                        ratio_stats[
                            "p99"
                        ]
                    ),
            }
        )

    # ========================================================
    # R3-0C per-term safety bounds
    #
    # For each PDE term separately:
    #
    # worst across:
    #   Naive + Canonical
    #   H1 + H2 + H3 + H4
    #
    # This split then produces ONE bound for that term.
    #
    # Later:
    #
    # formal_alpha_j =
    #   min(
    #       unseen_pr_bound_j,
    #       unseen_ra_bound_j
    #   )
    #
    # Same formal_alpha_j is then used by BOTH R3-1 arms.
    # ========================================================

    per_term_summary = {}

    for term in ACTIVE_TERMS:

        term_rows = [
            row
            for row in rows
            if row["term"] == term
        ]

        worst_ratio_p99 = max(
            row["ratio_p99"]
            for row in term_rows
        )

        candidate_alpha = float(
            CANDIDATE_ALPHA_MAX_BY_TERM[
                term
            ]
        )

        alpha_bound = min(
            candidate_alpha,
            (
                SAFETY_TARGET_FRACTION
                /
                max(
                    worst_ratio_p99,
                    1.0e-30,
                )
            ),
        )

        candidate_pass = (
            candidate_alpha
            *
            worst_ratio_p99
            <=
            SAFETY_TARGET_FRACTION
        )

        per_term_summary[
            term
        ] = {
            "candidate_alpha":
                candidate_alpha,

            "worst_ratio_p99_this_split":
                worst_ratio_p99,

            "candidate_pass_this_split":
                bool(
                    candidate_pass
                ),

            "alpha_bound_this_split":
                alpha_bound,
        }

    # ========================================================
    # Save
    # ========================================================

    output_dir = os.path.dirname(
        args.output_prefix
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    csv_path = (
        args.output_prefix
        +
        "_summary.csv"
    )

    term_path = (
        args.output_prefix
        +
        "_per_term_bounds.csv"
    )

    metadata_path = (
        args.output_prefix
        +
        "_metadata.json"
    )

    with open(
        csv_path,
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

    term_rows_out = []

    for term in ACTIVE_TERMS:

        info = (
            per_term_summary[
                term
            ]
        )

        term_rows_out.append(
            {
                "split_label":
                    args.split_label,

                "term":
                    term,

                "candidate_alpha":
                    info[
                        "candidate_alpha"
                    ],

                "worst_ratio_p99_this_split":
                    info[
                        "worst_ratio_p99_this_split"
                    ],

                "candidate_pass_this_split":
                    info[
                        "candidate_pass_this_split"
                    ],

                "alpha_bound_this_split":
                    info[
                        "alpha_bound_this_split"
                    ],
            }
        )

    with open(
        term_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                term_rows_out[
                    0
                ].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            term_rows_out
        )

    metadata_out = {
        "experiment":
            (
                "R3-1-TRAIN-only-"
                "per-term-matched-capacity-"
                "safety-audit"
            ),

        "split_label":
            args.split_label,

        "formal_run":
            (
                args.max_batches
                is None
            ),

        "state_source":
            (
                "same frozen-M6 "
                "closed-loop H4 states"
            ),

        "representations":
            list(
                REPRESENTATION_MODES
            ),

        "active_terms":
            list(
                ACTIVE_TERMS
            ),

        "candidate_alpha_max_by_term":
            dict(
                CANDIDATE_ALPHA_MAX_BY_TERM
            ),

        "safety_target_fraction":
            SAFETY_TARGET_FRACTION,

        "selection_rule": (
            "for each term j: "
            "formal_alpha_j = min("
            "0.25, "
            "0.50 / "
            "worst_cross_split_representation_"
            "horizon_ratio_p99_j)"
        ),

        "cross_arm_rule":
            (
                "R3-1a and R3-1b must use "
                "exactly the same formal "
                "alpha_max_by_term"
            ),

        "per_term_this_split":
            per_term_summary,

        "split_sha256":
            split_sha,

        "stats_sha256":
            stats_sha,

        "m6_checkpoint_sha256":
            m6_sha,

        "val_tuned":
            False,

        "test_accessed":
            False,
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata_out,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # Human-readable output
    # ========================================================

    print()
    print(
        "=" * 118
    )

    print(
        "R3-1 PER-TERM SUPPORT SUMMARY"
    )

    print(
        "=" * 118
    )

    for row in rows:

        print(
            f"{row['representation_mode']:<10s} "
            f"{row['term']:<22s} "
            f"H{row['step']} "
            f"ratio_p99="
            f"{row['ratio_p99']:.6e} "
            f"candidate_alpha="
            f"{row['candidate_alpha']:.6e} "
            f"candidate_dose="
            f"{row['candidate_p99_injection_fraction']:.6e}"
        )

    print()
    print(
        "=" * 118
    )

    print(
        "R3-0C PER-TERM ALPHA BOUNDS | "
        f"{args.split_label}"
    )

    print(
        "=" * 118
    )

    for term in ACTIVE_TERMS:

        info = (
            per_term_summary[
                term
            ]
        )

        print(
            f"{term:<22s} "
            f"worst_ratio_p99="
            f"{info['worst_ratio_p99_this_split']:.12e} "
            f"candidate_pass="
            f"{info['candidate_pass_this_split']} "
            f"alpha_bound="
            f"{info['alpha_bound_this_split']:.12e}"
        )

    print()
    print(
        "Formal cross-split rule:"
    )

    print(
        "For EACH term j:"
    )

    print(
        "  FORMAL_ALPHA_MAX_BY_TERM[j] ="
    )

    print(
        "      min("
        "alpha_bound_unseen_pr[j], "
        "alpha_bound_unseen_ra[j])"
    )

    print()
    print(
        "Same FORMAL_ALPHA_MAX_BY_TERM "
        "must be passed to BOTH R3-1a and R3-1b."
    )

    print()
    print(
        "Detailed CSV:",
        csv_path,
    )

    print(
        "Per-term bounds CSV:",
        term_path,
    )

    print(
        "Metadata:",
        metadata_path,
    )

    print()

    if args.max_batches is None:

        print(
            "✅ FORMAL TRAIN-ONLY "
            "PER-TERM SUPPORT AUDIT COMPLETE"
        )

    else:

        print(
            "✅ DEBUG PER-TERM SUPPORT AUDIT "
            "COMPLETE (NOT FORMAL)"
        )


if __name__ == "__main__":
    main()
