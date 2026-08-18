import argparse
import importlib.util
import json
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
# Reuse the CLOSED D2-1 vs M10-2 evaluator.
#
# D2-3A deliberately keeps the same canonical model class as
# D2-1; only alpha_max is changed by the predeclared capacity
# intervention. The old builder is safe to reuse because it
# reconstructs alpha_max from checkpoint metadata rather than
# hard-coding 0.25.
# ============================================================

BASE_EVAL_PATH = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "evaluate_d2_1_vs_m10_2_bonly_val_h16.py",
)


def load_python_module(
    name,
    path,
):
    if not os.path.exists(
        path
    ):
        raise FileNotFoundError(
            path
        )

    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            f"Cannot import module from {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )
    spec.loader.exec_module(
        module
    )

    return module


base = load_python_module(
    "d2_1_vs_m10_2_closed_eval",
    BASE_EVAL_PATH,
)


# ============================================================
# Predeclared D2-3A capacity-control constants
# ============================================================

D2_1_ALPHA_MAX = 0.25

CANONICAL_DX = 1.0 / 64.0
CANONICAL_DY = 1.0 / 63.0
CANONICAL_DT = 0.25

CAPACITY_MATCH = {
    "unseen_pr": {
        "legacy_path_b_cap":
            2.180076360160806,
        "canonical_path_b_cap":
            0.53869654465,
        "matched_alpha_max":
            1.0117367476234869,
        "matched_max_capacity":
            0.5450190900402015,
    },

    "unseen_ra": {
        "legacy_path_b_cap":
            1.5787383958464702,
        "canonical_path_b_cap":
            0.39047183691,
        "matched_alpha_max":
            1.0107889011534232,
        "matched_max_capacity":
            0.39468459896161756,
    },
}

PRIMARY_HORIZON = 16
PRIMARY_FIELDS = (
    "buoyancy",
    "global",
)

PREDECLARED_DECISION_RULE = {
    "primary_horizon":
        16,

    "primary_metrics": [
        "buoyancy_rel_l2_percent",
        "global_rel_l2_percent",
    ],

    "strong_pass": (
        "Across both splits and all three seeds, "
        "D2-3A - M10-2 is negative for BOTH h16 "
        "buoyancy and h16 global Rel-L2; equivalently 12/12 "
        "seed-level primary differences are negative."
    ),

    "pass": (
        "For EACH split separately, the three-seed mean "
        "D2-3A - M10-2 difference is negative for BOTH h16 "
        "buoyancy and global Rel-L2, AND each split/metric "
        "has at least 2/3 seeds with negative difference."
    ),

    "inconclusive": (
        "Neither PASS nor the capacity-confound failure "
        "condition is met; directions are mixed or only one "
        "primary metric is supported."
    ),

    "fail_for_capacity_confounded_causal_claim": (
        "For at least one split, BOTH three-seed mean "
        "D2-3A - M10-2 h16 buoyancy and global differences "
        "are >= 0, while the corresponding D2-1 - M10-2 "
        "three-seed means for BOTH primary metrics are < 0. "
        "Final classification is performed only after all "
        "six runs are aggregated; no per-run threshold is "
        "tuned from VAL."
    ),

    "locked_before_d2_3a_h16_results":
        True,

    "test_accessed":
        False,
}


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "D2-3A Capacity-Matched Canonical B-only "
            "causal-control evaluation on full VAL H16: "
            "M10-2 vs D2-1 vs D2-3A. "
            "No training. Utility Gate OFF. TEST forbidden."
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
        "--m10_checkpoint",
        required=True,
    )

    parser.add_argument(
        "--d2_1_checkpoint",
        required=True,
    )

    parser.add_argument(
        "--d2_3a_checkpoint",
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--max_horizon",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--horizons",
        type=str,
        default="1,4,8,16",
    )

    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help=(
            "Debug only. FORMAL evaluation must omit this "
            "so all 486 VAL H16 windows are used."
        ),
    )

    parser.add_argument(
        "--output_prefix",
        required=True,
    )

    return parser.parse_args()


# ============================================================
# Guardrails
# ============================================================

def assert_close(
    name,
    actual,
    expected,
    *,
    atol=1.0e-12,
):
    actual = float(
        actual
    )
    expected = float(
        expected
    )

    if not math.isclose(
        actual,
        expected,
        rel_tol=0.0,
        abs_tol=atol,
    ):
        raise RuntimeError(
            f"{name} mismatch: "
            f"actual={actual:.17g}, "
            f"expected={expected:.17g}"
        )


def get_alpha_max(
    payload,
    checkpoint_name,
):
    if (
        "alpha_max"
        not in payload
    ):
        raise RuntimeError(
            f"{checkpoint_name} has no alpha_max metadata."
        )

    return float(
        payload[
            "alpha_max"
        ]
    )


def get_path_b_cap(
    payload,
    checkpoint_name,
):
    stabilization = payload.get(
        "stabilization",
        {},
    )

    if (
        "path_b_rms_cap"
        not in stabilization
    ):
        raise RuntimeError(
            f"{checkpoint_name} has no "
            "stabilization.path_b_rms_cap metadata."
        )

    return float(
        stabilization[
            "path_b_rms_cap"
        ]
    )


def get_d2_interface(
    payload,
    checkpoint_name,
):
    interface = payload.get(
        "dimensional_interface",
        {},
    )

    if not interface:
        raise RuntimeError(
            f"{checkpoint_name} has no "
            "dimensional_interface metadata."
        )

    for key in [
        "dx",
        "dy",
        "dt",
        "rate_to_increment",
    ]:
        if key not in interface:
            raise RuntimeError(
                f"{checkpoint_name} dimensional_interface "
                f"missing key={key}"
            )

    if not bool(
        interface[
            "rate_to_increment"
        ]
    ):
        raise RuntimeError(
            f"{checkpoint_name} does not declare "
            "rate_to_increment=True."
        )

    return interface


def audit_checkpoint_protocol(
    *,
    split_label,
    m10_payload,
    d2_1_payload,
    d2_3a_payload,
):
    expected = CAPACITY_MATCH[
        split_label
    ]

    # --------------------------------------------------------
    # Experiment identity
    # --------------------------------------------------------

    d2_1_experiment = str(
        d2_1_payload.get(
            "experiment",
            "",
        )
    )

    d2_3a_experiment = str(
        d2_3a_payload.get(
            "experiment",
            "",
        )
    )

    if (
        "D2-1"
        not in d2_1_experiment
    ):
        raise RuntimeError(
            "D2-1 checkpoint experiment metadata "
            "does not contain 'D2-1': "
            f"{d2_1_experiment!r}"
        )

    if (
        "D2-3A"
        not in d2_3a_experiment
    ):
        raise RuntimeError(
            "D2-3A checkpoint experiment metadata "
            "does not contain 'D2-3A': "
            f"{d2_3a_experiment!r}"
        )

    # --------------------------------------------------------
    # alpha_max intervention
    # --------------------------------------------------------

    m10_alpha = get_alpha_max(
        m10_payload,
        "M10-2",
    )

    d2_1_alpha = get_alpha_max(
        d2_1_payload,
        "D2-1",
    )

    d2_3a_alpha = get_alpha_max(
        d2_3a_payload,
        "D2-3A",
    )

    assert_close(
        "M10-2 alpha_max",
        m10_alpha,
        0.25,
    )

    assert_close(
        "D2-1 alpha_max",
        d2_1_alpha,
        D2_1_ALPHA_MAX,
    )

    assert_close(
        "D2-3A matched alpha_max",
        d2_3a_alpha,
        expected[
            "matched_alpha_max"
        ],
    )

    # --------------------------------------------------------
    # Path-B RMS caps
    # --------------------------------------------------------

    m10_cap = get_path_b_cap(
        m10_payload,
        "M10-2",
    )

    d2_1_cap = get_path_b_cap(
        d2_1_payload,
        "D2-1",
    )

    d2_3a_cap = get_path_b_cap(
        d2_3a_payload,
        "D2-3A",
    )

    assert_close(
        "M10-2 legacy Path-B cap",
        m10_cap,
        expected[
            "legacy_path_b_cap"
        ],
        atol=1.0e-10,
    )

    assert_close(
        "D2-1 canonical Path-B cap",
        d2_1_cap,
        expected[
            "canonical_path_b_cap"
        ],
        atol=1.0e-10,
    )

    assert_close(
        "D2-3A canonical Path-B cap",
        d2_3a_cap,
        expected[
            "canonical_path_b_cap"
        ],
        atol=1.0e-10,
    )

    # --------------------------------------------------------
    # Canonical dimensional interface must be IDENTICAL
    # between D2-1 and D2-3A.
    # --------------------------------------------------------

    d2_1_interface = get_d2_interface(
        d2_1_payload,
        "D2-1",
    )

    d2_3a_interface = get_d2_interface(
        d2_3a_payload,
        "D2-3A",
    )

    for checkpoint_name, interface in [
        (
            "D2-1",
            d2_1_interface,
        ),
        (
            "D2-3A",
            d2_3a_interface,
        ),
    ]:
        assert_close(
            f"{checkpoint_name} dx",
            interface[
                "dx"
            ],
            CANONICAL_DX,
        )

        assert_close(
            f"{checkpoint_name} dy",
            interface[
                "dy"
            ],
            CANONICAL_DY,
        )

        assert_close(
            f"{checkpoint_name} dt",
            interface[
                "dt"
            ],
            CANONICAL_DT,
        )

    for key in [
        "dx",
        "dy",
        "dt",
    ]:
        assert_close(
            f"D2-1 vs D2-3A interface {key}",
            d2_1_interface[
                key
            ],
            d2_3a_interface[
                key
            ],
        )

    # --------------------------------------------------------
    # Frozen M6 provenance must match across all three.
    # --------------------------------------------------------

    m6_shas = {
        "M10-2":
            m10_payload.get(
                "m6_checkpoint_sha256"
            ),

        "D2-1":
            d2_1_payload.get(
                "m6_checkpoint_sha256"
            ),

        "D2-3A":
            d2_3a_payload.get(
                "m6_checkpoint_sha256"
            ),
    }

    if any(
        value is None
        for value in m6_shas.values()
    ):
        raise RuntimeError(
            "At least one checkpoint lacks "
            "m6_checkpoint_sha256 metadata: "
            f"{m6_shas}"
        )

    if len(
        set(
            m6_shas.values()
        )
    ) != 1:
        raise RuntimeError(
            "M6 provenance mismatch across "
            "M10-2 / D2-1 / D2-3A: "
            f"{m6_shas}"
        )

    # --------------------------------------------------------
    # Maximum-capacity match.
    #
    # This controls the maximum injection bound, NOT the exact
    # realized state-dependent dose.
    # --------------------------------------------------------

    m10_max_capacity = (
        m10_alpha
        *
        m10_cap
    )

    d2_1_max_capacity = (
        d2_1_alpha
        *
        d2_1_cap
    )

    d2_3a_max_capacity = (
        d2_3a_alpha
        *
        d2_3a_cap
    )

    assert_close(
        "D2-3A matched max capacity",
        d2_3a_max_capacity,
        expected[
            "matched_max_capacity"
        ],
        atol=1.0e-12,
    )

    assert_close(
        "D2-3A vs M10-2 max capacity",
        d2_3a_max_capacity,
        m10_max_capacity,
        atol=1.0e-12,
    )

    return {
        "m10_alpha_max":
            m10_alpha,

        "d2_1_alpha_max":
            d2_1_alpha,

        "d2_3a_alpha_max":
            d2_3a_alpha,

        "m10_path_b_cap":
            m10_cap,

        "d2_1_path_b_cap":
            d2_1_cap,

        "d2_3a_path_b_cap":
            d2_3a_cap,

        "m10_max_capacity":
            m10_max_capacity,

        "d2_1_max_capacity":
            d2_1_max_capacity,

        "d2_3a_max_capacity":
            d2_3a_max_capacity,

        "d2_1_interface":
            {
                key:
                    d2_1_interface[
                        key
                    ]
                for key in [
                    "dx",
                    "dy",
                    "dt",
                    "rate_to_increment",
                ]
            },

        "d2_3a_interface":
            {
                key:
                    d2_3a_interface[
                        key
                    ]
                for key in [
                    "dx",
                    "dy",
                    "dt",
                    "rate_to_increment",
                ]
            },

        "m6_checkpoint_sha256":
            next(
                iter(
                    m6_shas.values()
                )
            ),
    }


# ============================================================
# Diagnostics
# ============================================================

def new_diag_bucket():
    return {
        "samples":
            0,

        "alpha_sum":
            0.0,

        "abs_alpha_sum":
            0.0,

        "cap_active_sum":
            0.0,

        "path_b_rms_raw_sum":
            0.0,

        "path_b_rms_raw_max":
            0.0,

        "path_b_rms_safe_sum":
            0.0,

        "path_b_rms_safe_max":
            0.0,

        "path_b_scale_sum":
            0.0,

        "path_b_scale_min":
            1.0,

        "injection_rms_sum":
            0.0,
    }


def update_diag_bucket(
    bucket,
    components,
):
    alpha = (
        components[
            "alpha_b"
        ]
        .detach()
        .reshape(-1)
    )

    path_b_rms_raw = (
        components[
            "path_b_rms_raw"
        ]
        .detach()
        .reshape(-1)
    )

    path_b_rms_safe = (
        components[
            "path_b_rms_safe"
        ]
        .detach()
        .reshape(-1)
    )

    path_b_scale = (
        components[
            "path_b_scale"
        ]
        .detach()
        .reshape(-1)
    )

    cap_active = (
        components[
            "path_b_cap_active"
        ]
        .detach()
        .float()
        .reshape(-1)
    )

    residual_b = (
        components[
            "physics_residual_norm"
        ][
            :,
            base.B_IDX,
            :,
            :,
        ]
        .detach()
    )

    injection_rms = torch.sqrt(
        torch.mean(
            residual_b
            *
            residual_b,
            dim=(-2, -1),
        )
        +
        1.0e-12
    )

    n = int(
        alpha.numel()
    )

    bucket[
        "samples"
    ] += n

    bucket[
        "alpha_sum"
    ] += float(
        alpha.sum().cpu()
    )

    bucket[
        "abs_alpha_sum"
    ] += float(
        alpha.abs().sum().cpu()
    )

    bucket[
        "cap_active_sum"
    ] += float(
        cap_active.sum().cpu()
    )

    bucket[
        "path_b_rms_raw_sum"
    ] += float(
        path_b_rms_raw.sum().cpu()
    )

    bucket[
        "path_b_rms_raw_max"
    ] = max(
        bucket[
            "path_b_rms_raw_max"
        ],
        float(
            path_b_rms_raw.max().cpu()
        ),
    )

    bucket[
        "path_b_rms_safe_sum"
    ] += float(
        path_b_rms_safe.sum().cpu()
    )

    bucket[
        "path_b_rms_safe_max"
    ] = max(
        bucket[
            "path_b_rms_safe_max"
        ],
        float(
            path_b_rms_safe.max().cpu()
        ),
    )

    bucket[
        "path_b_scale_sum"
    ] += float(
        path_b_scale.sum().cpu()
    )

    bucket[
        "path_b_scale_min"
    ] = min(
        bucket[
            "path_b_scale_min"
        ],
        float(
            path_b_scale.min().cpu()
        ),
    )

    bucket[
        "injection_rms_sum"
    ] += float(
        injection_rms.sum().cpu()
    )


def make_diag_summary(
    diag_stats,
    model_names,
    horizons,
    alpha_max_by_model,
):
    rows = []

    for model_name in model_names:

        alpha_max = float(
            alpha_max_by_model[
                model_name
            ]
        )

        for horizon in horizons:

            bucket = diag_stats[
                (
                    model_name,
                    horizon,
                )
            ]

            n = bucket[
                "samples"
            ]

            if n <= 0:
                raise RuntimeError(
                    "Diagnostic bucket has zero samples."
                )

            alpha_mean = (
                bucket[
                    "alpha_sum"
                ]
                / n
            )

            abs_alpha_mean = (
                bucket[
                    "abs_alpha_sum"
                ]
                / n
            )

            rows.append(
                {
                    "model":
                        model_name,

                    "horizon":
                        horizon,

                    "samples":
                        n,

                    "alpha_max":
                        alpha_max,

                    "alpha_mean":
                        alpha_mean,

                    "abs_alpha_mean":
                        abs_alpha_mean,

                    "abs_alpha_utilization_mean":
                        (
                            abs_alpha_mean
                            /
                            alpha_max
                        ),

                    "cap_active_fraction":
                        (
                            bucket[
                                "cap_active_sum"
                            ]
                            / n
                        ),

                    "path_b_rms_raw_mean":
                        (
                            bucket[
                                "path_b_rms_raw_sum"
                            ]
                            / n
                        ),

                    "path_b_rms_raw_max":
                        bucket[
                            "path_b_rms_raw_max"
                        ],

                    "path_b_rms_safe_mean":
                        (
                            bucket[
                                "path_b_rms_safe_sum"
                            ]
                            / n
                        ),

                    "path_b_rms_safe_max":
                        bucket[
                            "path_b_rms_safe_max"
                        ],

                    "path_b_scale_mean":
                        (
                            bucket[
                                "path_b_scale_sum"
                            ]
                            / n
                        ),

                    "path_b_scale_min":
                        bucket[
                            "path_b_scale_min"
                        ],

                    "injection_rms_mean":
                        (
                            bucket[
                                "injection_rms_sum"
                            ]
                            / n
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Difference tables
# ============================================================

def make_difference_table(
    summary,
):
    pairs = [
        (
            "D2-1-DC-BOnly",
            "M10-2-BOnly",
        ),
        (
            "D2-3A-CapacityMatch",
            "M10-2-BOnly",
        ),
        (
            "D2-3A-CapacityMatch",
            "D2-1-DC-BOnly",
        ),
    ]

    rows = []

    for (
        model_a,
        model_b,
    ) in pairs:

        a = summary[
            summary[
                "model"
            ]
            ==
            model_a
        ]

        b = summary[
            summary[
                "model"
            ]
            ==
            model_b
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

        for _, row in (
            merged.iterrows()
        ):

            rows.append(
                {
                    "comparison":
                        f"{model_a} - {model_b}",

                    "horizon":
                        int(
                            row[
                                "horizon"
                            ]
                        ),

                    "field":
                        row[
                            "field"
                        ],

                    "rel_l2_a":
                        float(
                            row[
                                "rel_l2_percent_a"
                            ]
                        ),

                    "rel_l2_b":
                        float(
                            row[
                                "rel_l2_percent_b"
                            ]
                        ),

                    "rel_l2_diff_pp":
                        float(
                            row[
                                "rel_l2_percent_a"
                            ]
                            -
                            row[
                                "rel_l2_percent_b"
                            ]
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


def make_h16_primary_table(
    differences,
):
    keep = differences[
        (
            differences[
                "horizon"
            ]
            ==
            PRIMARY_HORIZON
        )
        &
        (
            differences[
                "field"
            ].isin(
                PRIMARY_FIELDS
            )
        )
        &
        (
            differences[
                "comparison"
            ]
            ==
            (
                "D2-3A-CapacityMatch "
                "- M10-2-BOnly"
            )
        )
    ].copy()

    if len(
        keep
    ) != len(
        PRIMARY_FIELDS
    ):
        raise RuntimeError(
            "Could not construct the two "
            "predeclared h16 primary metrics."
        )

    keep[
        "negative_means_d2_3a_better"
    ] = (
        keep[
            "rel_l2_diff_pp"
        ]
        <
        0.0
    )

    return keep


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    if (
        args.max_horizon
        != 16
    ):
        raise ValueError(
            "D2-3A formal causal-control evaluator "
            "is locked to max_horizon=16."
        )

    horizons = sorted(
        {
            int(
                x.strip()
            )
            for x in args.horizons.split(
                ","
            )
            if x.strip()
        }
    )

    if horizons != [
        1,
        4,
        8,
        16,
    ]:
        raise ValueError(
            "Formal D2-3A evaluator is locked to "
            "horizons=1,4,8,16."
        )

    if (
        args.batch_size
        <= 0
    ):
        raise ValueError(
            "batch_size must be positive."
        )

    for path in [
        args.split,
        args.stats,
        args.m10_checkpoint,
        args.d2_1_checkpoint,
        args.d2_3a_checkpoint,
    ]:
        if not os.path.exists(
            path
        ):
            raise FileNotFoundError(
                path
            )

    base.set_seed(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "=" * 118
    )
    print(
        "D2-3A CAPACITY-MATCHED CANONICAL B-ONLY "
        "CAUSAL-CONTROL | FULL VAL H16"
    )
    print(
        "=" * 118
    )
    print(
        "Stage: CONTROL / CAUSAL ATTRIBUTION"
    )
    print(
        "Training: OFF"
    )
    print(
        "Utility Gate: OFF"
    )
    print(
        "PDE loss: N/A"
    )
    print(
        "Closed-loop split: VAL ONLY"
    )
    print(
        "TEST access: FORBIDDEN"
    )
    print(
        "Comparison: M10-2 vs D2-1 vs D2-3A"
    )
    print(
        "Primary: h16 buoyancy + global Rel-L2"
    )
    print(
        "Important: D2-3A matches MAXIMUM capacity, "
        "not exact realized state-dependent dose."
    )
    print(
        "Split:",
        args.split_label,
    )
    print(
        "Seed:",
        args.seed,
    )
    print(
        "Horizons:",
        horizons,
    )
    print(
        "Device:",
        device,
    )

    if (
        args.max_batches
        is not None
    ):
        print(
            "⚠️ DEBUG ONLY: max_batches =",
            args.max_batches,
        )
        print(
            "⚠️ This run MUST NOT be classified "
            "as a formal D2-3A H16 result."
        )

    # ========================================================
    # Provenance
    # ========================================================

    split_sha = base.sha256_file(
        args.split
    )

    stats_sha = base.sha256_file(
        args.stats
    )

    m10_sha = base.sha256_file(
        args.m10_checkpoint
    )

    d2_1_sha = base.sha256_file(
        args.d2_1_checkpoint
    )

    d2_3a_sha = base.sha256_file(
        args.d2_3a_checkpoint
    )

    m10_payload = base.load_payload(
        args.m10_checkpoint,
        device,
    )

    d2_1_payload = base.load_payload(
        args.d2_1_checkpoint,
        device,
    )

    d2_3a_payload = base.load_payload(
        args.d2_3a_checkpoint,
        device,
    )

    base.verify_checkpoint_provenance(
        m10_payload,
        checkpoint_name="M10-2",
        expected_seed=args.seed,
        split_sha=split_sha,
        stats_sha=stats_sha,
    )

    base.verify_checkpoint_provenance(
        d2_1_payload,
        checkpoint_name="D2-1",
        expected_seed=args.seed,
        split_sha=split_sha,
        stats_sha=stats_sha,
    )

    base.verify_checkpoint_provenance(
        d2_3a_payload,
        checkpoint_name="D2-3A",
        expected_seed=args.seed,
        split_sha=split_sha,
        stats_sha=stats_sha,
    )

    protocol_audit = (
        audit_checkpoint_protocol(
            split_label=args.split_label,
            m10_payload=m10_payload,
            d2_1_payload=d2_1_payload,
            d2_3a_payload=d2_3a_payload,
        )
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
        "M10_CHECKPOINT_SHA256:",
        m10_sha,
    )
    print(
        "D2_1_CHECKPOINT_SHA256:",
        d2_1_sha,
    )
    print(
        "D2_3A_CHECKPOINT_SHA256:",
        d2_3a_sha,
    )

    print()
    print(
        "========== CHECKPOINTS =========="
    )
    print(
        "M10-2:",
        args.m10_checkpoint,
    )
    print(
        "D2-1:",
        args.d2_1_checkpoint,
    )
    print(
        "D2-3A:",
        args.d2_3a_checkpoint,
    )
    print(
        "M10-2 best_epoch:",
        m10_payload.get(
            "best_epoch"
        ),
    )
    print(
        "D2-1 best_epoch:",
        d2_1_payload.get(
            "best_epoch"
        ),
    )
    print(
        "D2-3A best_epoch:",
        d2_3a_payload.get(
            "best_epoch"
        ),
    )
    print(
        "Shared M6 SHA:",
        protocol_audit[
            "m6_checkpoint_sha256"
        ],
    )

    print()
    print(
        "========== CAPACITY AUDIT =========="
    )
    print(
        "M10-2 alpha_max:",
        protocol_audit[
            "m10_alpha_max"
        ],
    )
    print(
        "D2-1 alpha_max:",
        protocol_audit[
            "d2_1_alpha_max"
        ],
    )
    print(
        "D2-3A alpha_max:",
        protocol_audit[
            "d2_3a_alpha_max"
        ],
    )
    print(
        "M10-2 Path-B cap:",
        protocol_audit[
            "m10_path_b_cap"
        ],
    )
    print(
        "D2-1 Path-B cap:",
        protocol_audit[
            "d2_1_path_b_cap"
        ],
    )
    print(
        "D2-3A Path-B cap:",
        protocol_audit[
            "d2_3a_path_b_cap"
        ],
    )
    print(
        "M10-2 max capacity:",
        protocol_audit[
            "m10_max_capacity"
        ],
    )
    print(
        "D2-1 max capacity:",
        protocol_audit[
            "d2_1_max_capacity"
        ],
    )
    print(
        "D2-3A max capacity:",
        protocol_audit[
            "d2_3a_max_capacity"
        ],
    )
    print(
        "capacity_match_abs_error:",
        abs(
            protocol_audit[
                "d2_3a_max_capacity"
            ]
            -
            protocol_audit[
                "m10_max_capacity"
            ]
        ),
    )

    # ========================================================
    # Models
    # ========================================================

    (
        field_mean,
        field_std,
    ) = base.build_field_stats(
        args.stats
    )

    m10_model = base.build_m10_model(
        m10_payload,
        field_mean,
        field_std,
        device,
    )

    d2_1_model = base.build_d2_model(
        d2_1_payload,
        field_mean,
        field_std,
        device,
    )

    d2_3a_model = base.build_d2_model(
        d2_3a_payload,
        field_mean,
        field_std,
        device,
    )

    m6_diff_m10_d21 = (
        base.compare_embedded_m6(
            m10_model,
            d2_1_model,
        )
    )

    m6_diff_m10_d23a = (
        base.compare_embedded_m6(
            m10_model,
            d2_3a_model,
        )
    )

    m6_diff_d21_d23a = (
        base.compare_embedded_m6(
            d2_1_model,
            d2_3a_model,
        )
    )

    print()
    print(
        "========== FAIRNESS AUDIT =========="
    )
    print(
        "embedded_M6_max_abs_diff "
        "M10_vs_D2-1:",
        f"{m6_diff_m10_d21:.12e}",
    )
    print(
        "embedded_M6_max_abs_diff "
        "M10_vs_D2-3A:",
        f"{m6_diff_m10_d23a:.12e}",
    )
    print(
        "embedded_M6_max_abs_diff "
        "D2-1_vs_D2-3A:",
        f"{m6_diff_d21_d23a:.12e}",
    )

    if max(
        m6_diff_m10_d21,
        m6_diff_m10_d23a,
        m6_diff_d21_d23a,
    ) > 1.0e-7:
        raise RuntimeError(
            "Embedded M6 backbones differ."
        )

    # ========================================================
    # VAL ONLY dataset
    # ========================================================

    with open(
        args.split,
        "r",
        encoding="utf-8",
    ) as f:
        split = json.load(
            f
        )

    if (
        "val"
        not in split
    ):
        raise KeyError(
            "Split has no val key."
        )

    val_base = base.RBCDataset(
        split_config=split[
            "val"
        ],
        normalize=True,
        stats_path=args.stats,
        return_sequence=True,
        target_steps=args.max_horizon,
        return_params=True,
    )

    val_dataset = (
        base.M10MultiStepParamDataset(
            val_base,
            context_length=4,
        )
    )

    loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    print()
    print(
        "========== VAL DATA =========="
    )
    print(
        "VAL H16 windows:",
        len(
            val_dataset
        ),
    )
    print(
        "Batches:",
        len(
            loader
        ),
    )

    if (
        args.max_batches
        is None
        and len(
            val_dataset
        )
        != 486
    ):
        raise RuntimeError(
            "Formal D2-3A H16 protocol expects "
            "exactly 486 VAL windows, got "
            f"{len(val_dataset)}."
        )

    # ========================================================
    # Metric setup
    # ========================================================

    model_names = [
        "M10-2-BOnly",
        "D2-1-DC-BOnly",
        "D2-3A-CapacityMatch",
    ]

    models = {
        "M10-2-BOnly":
            m10_model,

        "D2-1-DC-BOnly":
            d2_1_model,

        "D2-3A-CapacityMatch":
            d2_3a_model,
    }

    alpha_max_by_model = {
        "M10-2-BOnly":
            protocol_audit[
                "m10_alpha_max"
            ],

        "D2-1-DC-BOnly":
            protocol_audit[
                "d2_1_alpha_max"
            ],

        "D2-3A-CapacityMatch":
            protocol_audit[
                "d2_3a_alpha_max"
            ],
    }

    error_stats = {
        (
            model_name,
            horizon,
        ):
            base.new_error_bucket()

        for model_name in model_names
        for horizon in horizons
    }

    diag_stats = {
        (
            model_name,
            horizon,
        ):
            new_diag_bucket()

        for model_name in model_names
        for horizon in horizons
    }

    field_mean_t = torch.tensor(
        field_mean,
        device=device,
        dtype=torch.float32,
    ).view(
        1,
        4,
        1,
        1,
    )

    field_std_t = torch.tensor(
        field_std,
        device=device,
        dtype=torch.float32,
    ).view(
        1,
        4,
        1,
        1,
    )

    # ========================================================
    # Closed-loop rollout
    # ========================================================

    with torch.no_grad():

        for batch_idx, batch in enumerate(
            loader
        ):

            if (
                args.max_batches
                is not None
                and batch_idx
                >= args.max_batches
            ):
                break

            (
                context_norm,
                y_seq_norm,
                param,
            ) = batch

            context_norm = (
                context_norm.to(
                    device
                )
            )

            y_seq_norm = (
                y_seq_norm.to(
                    device
                )
            )

            param = param.to(
                device
            )

            contexts = {
                model_name:
                    context_norm.clone()

                for model_name
                in model_names
            }

            for step in range(
                1,
                args.max_horizon
                + 1,
            ):

                true_next_norm = (
                    y_seq_norm[
                        :,
                        step - 1,
                        :,
                        :,
                        :,
                    ]
                )

                true_next_phys = (
                    true_next_norm
                    *
                    field_std_t
                    +
                    field_mean_t
                )

                for model_name in model_names:

                    model = models[
                        model_name
                    ]

                    context = contexts[
                        model_name
                    ]

                    (
                        batch_size,
                        context_len,
                        channels,
                        h,
                        w,
                    ) = context.shape

                    model_input = (
                        context.reshape(
                            batch_size,
                            context_len
                            * channels,
                            h,
                            w,
                        )
                    )

                    (
                        pred_delta_norm,
                        components,
                    ) = model(
                        model_input,
                        params=param,
                        return_components=True,
                    )

                    current_norm = (
                        context[
                            :,
                            -1,
                            :,
                            :,
                            :,
                        ]
                    )

                    pred_next_norm = (
                        current_norm
                        +
                        pred_delta_norm
                    )

                    if step in horizons:

                        pred_next_phys = (
                            pred_next_norm
                            *
                            field_std_t
                            +
                            field_mean_t
                        )

                        base.update_error_bucket(
                            error_stats[
                                (
                                    model_name,
                                    step,
                                )
                            ],
                            pred_next_phys,
                            true_next_phys,
                        )

                        update_diag_bucket(
                            diag_stats[
                                (
                                    model_name,
                                    step,
                                )
                            ],
                            components,
                        )

                    contexts[
                        model_name
                    ] = torch.cat(
                        [
                            context[
                                :,
                                1:,
                                :,
                                :,
                                :,
                            ],
                            pred_next_norm.unsqueeze(
                                1
                            ),
                        ],
                        dim=1,
                    )

            if (
                (batch_idx + 1)
                % 20
                == 0
            ):
                print(
                    "processed batch",
                    batch_idx + 1,
                    "/",
                    len(
                        loader
                    ),
                )

    # ========================================================
    # Results
    # ========================================================

    summary = base.make_error_summary(
        error_stats,
        model_names,
        horizons,
    )

    diagnostics = make_diag_summary(
        diag_stats,
        model_names,
        horizons,
        alpha_max_by_model,
    )

    differences = (
        make_difference_table(
            summary
        )
    )

    primary_h16 = (
        make_h16_primary_table(
            differences
        )
    )

    output_dir = os.path.dirname(
        args.output_prefix
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    summary_path = (
        args.output_prefix
        +
        "_summary.csv"
    )

    diag_path = (
        args.output_prefix
        +
        "_diagnostics.csv"
    )

    diff_path = (
        args.output_prefix
        +
        "_differences.csv"
    )

    primary_path = (
        args.output_prefix
        +
        "_h16_primary.csv"
    )

    metadata_path = (
        args.output_prefix
        +
        "_metadata.json"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    diagnostics.to_csv(
        diag_path,
        index=False,
    )

    differences.to_csv(
        diff_path,
        index=False,
    )

    primary_h16.to_csv(
        primary_path,
        index=False,
    )

    metadata = {
        "experiment":
            "D2-3A Capacity-Matched Canonical B-only Control",

        "stage":
            "control_causal_attribution",

        "training":
            False,

        "utility_gate":
            False,

        "pde_loss":
            None,

        "closed_loop_split":
            "val",

        "test_accessed":
            False,

        "split_label":
            args.split_label,

        "seed":
            int(
                args.seed
            ),

        "horizons":
            horizons,

        "val_windows":
            len(
                val_dataset
            ),

        "max_batches":
            args.max_batches,

        "formal_full_val":
            (
                args.max_batches
                is None
                and len(
                    val_dataset
                )
                == 486
            ),

        "split":
            args.split,

        "split_sha256":
            split_sha,

        "stats":
            args.stats,

        "stats_sha256":
            stats_sha,

        "checkpoints": {
            "m10_2":
                args.m10_checkpoint,

            "m10_2_sha256":
                m10_sha,

            "d2_1":
                args.d2_1_checkpoint,

            "d2_1_sha256":
                d2_1_sha,

            "d2_3a":
                args.d2_3a_checkpoint,

            "d2_3a_sha256":
                d2_3a_sha,
        },

        "protocol_audit":
            protocol_audit,

        "capacity_control_scope":
            (
                "Maximum Path-B injection capacity is matched "
                "between M10-2 and D2-3A. This does NOT match "
                "the realized state-dependent alpha/injection dose "
                "or optimization geometry."
            ),

        "predeclared_decision_rule":
            PREDECLARED_DECISION_RULE,

        "base_evaluator":
            (
                "evaluation/"
                "evaluate_d2_1_vs_m10_2_bonly_val_h16.py"
            ),
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # Compact terminal tables
    # ========================================================

    field_columns = [
        "buoyancy",
        "u_x",
        "u_y",
        "pressure",
        "global",
    ]

    wide = summary.pivot_table(
        index=[
            "model",
            "horizon",
        ],
        columns="field",
        values="rel_l2_percent",
    ).reset_index()

    wide = wide[
        [
            "model",
            "horizon",
        ]
        +
        field_columns
    ]

    print()
    print(
        "=" * 118
    )
    print(
        "ROLLOUT Rel-L2 (%)"
    )
    print(
        "=" * 118
    )
    print(
        wide.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )

    print()
    print(
        "=" * 118
    )
    print(
        "H16 PRIMARY CAUSAL CONTROL"
    )
    print(
        "D2-3A - M10-2 | NEGATIVE = D2-3A BETTER"
    )
    print(
        "=" * 118
    )
    print(
        primary_h16[
            [
                "field",
                "rel_l2_a",
                "rel_l2_b",
                "rel_l2_diff_pp",
                "negative_means_d2_3a_better",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: (
                f"{x:+.6f}"
            ),
        )
    )

    print()
    print(
        "=" * 118
    )
    print(
        "PATH-B DIAGNOSTICS"
    )
    print(
        "=" * 118
    )
    print(
        diagnostics.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.8e}"
            ),
        )
    )

    print()
    print(
        "Summary:",
        summary_path,
    )
    print(
        "Diagnostics:",
        diag_path,
    )
    print(
        "Differences:",
        diff_path,
    )
    print(
        "H16 primary:",
        primary_path,
    )
    print(
        "Metadata:",
        metadata_path,
    )

    print()
    print(
        "✅ D2-3A full VAL H16 causal-control "
        "evaluation finished."
    )


if __name__ == "__main__":
    main()
