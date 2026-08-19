from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.normalization import FieldWiseNormalizer
from physics.canonical_metadata import build_rbc_canonical_metadata


# ============================================================
# Reuse FROZEN / AUDITED R3 evaluators.
#
# R3-3 physics adds only a thin Utility-gated model wrapper.
# TEST windows, free-autoregressive feedback, finite-difference
# physics metrics and final bucket semantics remain frozen.
# ============================================================

R3_3_ROLLOUT_PATH = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "evaluate_cross_param_rollout_r3_3.py",
)

R3_2_PHYSICS_PATH = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "evaluate_cross_param_rollout_physics_r3_2.py",
)


def load_python_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


r3_3_rollout = load_python_module(
    "r3_3_frozen_rollout",
    R3_3_ROLLOUT_PATH,
)
r3_2_physics = load_python_module(
    "r3_2_frozen_physics",
    R3_2_PHYSICS_PATH,
)

# Exact frozen R3-1 / M10 Physics-Audit-v2 machinery inherited
# through the frozen R3-2 physics evaluator.
r3_1_physics = r3_2_physics.r3_1_physics
physics = r3_2_physics.physics


# ============================================================
# Frozen identities
# ============================================================

DISPLAY_CANONICAL = r3_3_rollout.DISPLAY_CANONICAL
DISPLAY_PARAMONLY = r3_3_rollout.DISPLAY_PARAMONLY
DISPLAY_STATEPARAM = r3_3_rollout.DISPLAY_STATEPARAM
DISPLAY_PARAMUTILITY = r3_3_rollout.DISPLAY_PARAMUTILITY
DISPLAY_STATEUTILITY = r3_3_rollout.DISPLAY_STATEUTILITY

REFERENCE_MODELS = r3_3_rollout.REFERENCE_MODELS
UTILITY_MODELS = r3_3_rollout.UTILITY_MODELS
MODEL_ORDER = r3_3_rollout.MODEL_ORDER
ACTIVE_TERMS = r3_3_rollout.ACTIVE_TERMS
ARM_BY_DISPLAY = r3_3_rollout.ARM_BY_DISPLAY

PRIMARY_COMPARISON = (
    f"{DISPLAY_STATEUTILITY} - {DISPLAY_PARAMUTILITY}"
)

SECONDARY_COMPARISONS = (
    f"{DISPLAY_PARAMUTILITY} - {DISPLAY_STATEPARAM}",
    f"{DISPLAY_STATEUTILITY} - {DISPLAY_STATEPARAM}",
    f"{DISPLAY_STATEUTILITY} - {DISPLAY_PARAMONLY}",
    f"{DISPLAY_STATEUTILITY} - {DISPLAY_CANONICAL}",
)

KEY_PHYSICS_METRICS = tuple(
    r3_2_physics.KEY_PHYSICS_METRICS
)

PHYSICS_DX = r3_2_physics.PHYSICS_DX
PHYSICS_DY = r3_2_physics.PHYSICS_DY
PHYSICS_DT = r3_2_physics.PHYSICS_DT
FORMAL_HORIZONS = tuple(r3_2_physics.FORMAL_HORIZONS)
FORMAL_TEST_WINDOWS_H16 = (
    r3_2_physics.FORMAL_TEST_WINDOWS_H16
)
PREDICTION_REPRO_ABS_TOL = (
    r3_2_physics.PREDICTION_REPRO_ABS_TOL
)

EXPECTED_R3_3_ROLLOUT_SHA256 = (
    "104e3db55561368d961865c16bda3857"
    "17b3fb1cdbda1e4be3dafc8fe9800987"
)

EXPECTED_R3_2_PHYSICS_SHA256 = (
    "fd56ef510b59664cede536904c26c65cb"
    "386ec99c23ef53ea4b72228a6ab0740"
)

R3_3_PRETEST_REGISTRY = (
    "configs/r3/r3_3b_pretest_registry.json"
)
EXPECTED_R3_3_PRETEST_REGISTRY_SHA256 = (
    "2c4e06cb5dd4face1de29c7a45e4fbf"
    "0b724b2bb63ed1e020393ea8545dad9cc"
)

R3_3_FORMAL_ROLLOUT_REGISTRY = (
    "configs/r3/r3_3b_formal_rollout_registry.json"
)
EXPECTED_R3_3_FORMAL_ROLLOUT_REGISTRY_SHA256 = (
    "ba13abfc77e91fa153b93a64a3c5d2ca"
    "390047c6f2f6a0bcbb1b2a86ae89dd2a"
)

EXPECTED_R3_3_PRETEST_FREEZE_COMMIT = (
    "247f49db4655b430d5142c2bbcd26f28c1acf611"
)
EXPECTED_R3_3_ROLLOUT_RESULT_FREEZE_COMMIT = (
    "2c501a98559271c69e8fa4be087dea6c41754820"
)


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "R3-3B matched formal TEST physics evaluation: "
            "ParamUtility vs StateUtility, with frozen R3-3B "
            "prediction rollout and frozen R3-1/M10 "
            "Physics-Audit-v2 definitions. No training. "
            "Physics metrics are FD-based comparative proxies, "
            "not solver-level PDE violation."
        )
    )
    parser.add_argument(
        "--split_label",
        required=True,
        choices=("unseen_pr", "unseen_ra"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizons", default="1,4,8,16")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="DEBUG TEST only. Formal evaluation must omit this.",
    )
    parser.add_argument(
        "--audit_only",
        action="store_true",
        help=(
            "Audit frozen sources, registries, rollout outputs, "
            "checkpoints and Utility wrapper interfaces only. "
            "Does NOT construct the physics TEST dataset."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/tables/r3_3_physics",
    )
    return parser.parse_args()


# ============================================================
# General utilities
# ============================================================

def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_ROOT, path))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def audit_sha(path, expected, label):
    path = resolve_path(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"{label} SHA256 mismatch.\n"
            f"Expected={expected}\n"
            f"Actual={actual}\n"
            f"Path={path}"
        )
    return actual


# ============================================================
# Frozen R3-3B prediction result audit
# ============================================================

def frozen_rollout_paths(split_label, seed):
    base = resolve_path(
        os.path.join(
            "outputs",
            "tables",
            "r3_3_rollout",
        )
    )
    prefix = (
        f"r3_3b_rollout_{split_label}_seed{seed}"
    )
    return {
        "summary": os.path.join(
            base,
            f"{prefix}_summary.csv",
        ),
        "growth": os.path.join(
            base,
            f"{prefix}_global_growth_auc.csv",
        ),
        "metadata": os.path.join(
            base,
            f"{prefix}_metadata.json",
        ),
    }


def load_and_audit_frozen_r3_3_rollout(
    *,
    split_label,
    seed,
):
    registry_path = resolve_path(
        R3_3_FORMAL_ROLLOUT_REGISTRY
    )
    audit_sha(
        registry_path,
        EXPECTED_R3_3_FORMAL_ROLLOUT_REGISTRY_SHA256,
        "Frozen R3-3B formal rollout result registry",
    )
    registry = load_json(registry_path)

    if registry.get("stage") != "R3-3B":
        raise RuntimeError(
            "Unexpected R3-3B formal rollout registry stage."
        )
    if (
        registry.get("status")
        !=
        "FORMAL_ROLLOUT_COMPLETE"
    ):
        raise RuntimeError(
            "R3-3B formal rollout registry is incomplete."
        )

    pretest = registry["pretest_freeze"]
    if (
        pretest["commit"]
        !=
        EXPECTED_R3_3_PRETEST_FREEZE_COMMIT
    ):
        raise RuntimeError(
            "R3-3B pre-TEST freeze commit changed."
        )
    if (
        pretest["evaluator_sha256"]
        !=
        EXPECTED_R3_3_ROLLOUT_SHA256
    ):
        raise RuntimeError(
            "Frozen R3-3B rollout evaluator identity changed."
        )
    if (
        pretest["pretest_registry_sha256"]
        !=
        EXPECTED_R3_3_PRETEST_REGISTRY_SHA256
    ):
        raise RuntimeError(
            "R3-3B pre-TEST registry identity changed."
        )

    formal = registry["formal_protocol"]
    if formal["seed"] != seed:
        raise RuntimeError(
            "Frozen R3-3B rollout seed mismatch."
        )
    if formal["batch_size"] != 4:
        raise RuntimeError(
            "Frozen R3-3B rollout batch size changed."
        )
    if formal["reported_horizons"] != [1, 4, 8, 16]:
        raise RuntimeError(
            "Frozen R3-3B rollout horizons changed."
        )
    if formal["full_max_horizon"] != 16:
        raise RuntimeError(
            "Frozen R3-3B rollout max horizon changed."
        )
    if (
        formal["formal_test_windows_per_split"]
        !=
        FORMAL_TEST_WINDOWS_H16
    ):
        raise RuntimeError(
            "Frozen R3-3B formal TEST window count changed."
        )
    if formal["rollout"] != "free autoregressive":
        raise RuntimeError(
            "Frozen R3-3B rollout is not free autoregressive."
        )
    if formal["test_accessed"] is not True:
        raise RuntimeError(
            "Frozen R3-3B rollout result claims TEST untouched."
        )

    primary = registry["primary_comparison"]
    if (
        primary["comparison"]
        !=
        PRIMARY_COMPARISON
    ):
        raise RuntimeError(
            "Frozen R3-3B primary comparison changed."
        )

    decision = registry["formal_decision"]
    if (
        decision["state_information_utility_value"]
        !=
        "PASS"
    ):
        raise RuntimeError(
            "Frozen R3-3B prediction decision is not PASS."
        )

    paths = frozen_rollout_paths(
        split_label,
        seed,
    )
    axis = registry[split_label]
    expected_outputs = axis["outputs"]

    expected_by_key = {
        "summary": expected_outputs["summary_sha256"],
        "growth": expected_outputs["growth_sha256"],
        "metadata": expected_outputs["metadata_sha256"],
    }

    for key, path in paths.items():
        audit_sha(
            path,
            expected_by_key[key],
            f"{split_label} frozen R3-3B {key}",
        )

    metadata = load_json(paths["metadata"])

    checks = {
        "experiment":
            "R3-3B-matched-closed-loop-utility-gating",
        "formal_run":
            True,
        "split_label":
            split_label,
        "seed":
            seed,
        "models":
            list(MODEL_ORDER),
        "primary_comparison":
            PRIMARY_COMPARISON,
        "negative_primary_difference_means":
            "StateUtility better",
        "formal_test_windows":
            FORMAL_TEST_WINDOWS_H16,
        "free_autoregressive":
            True,
        "no_step1_special_case":
            True,
        "test_accessed":
            True,
    }

    for key, expected in checks.items():
        actual = metadata.get(key)
        if actual != expected:
            raise RuntimeError(
                "Frozen R3-3B rollout metadata mismatch.\n"
                f"key={key}\n"
                f"expected={expected!r}\n"
                f"actual={actual!r}"
            )

    return {
        "registry": registry,
        "registry_path": registry_path,
        "paths": paths,
        "metadata": metadata,
    }


# ============================================================
# Thin Utility-gated model wrapper
#
# This is the ONLY new model-facing adapter in R3-3B physics.
# r3_1_physics.run_one_model(...) remains untouched.
# ============================================================

class UtilityGatedModelWrapper(nn.Module):
    def __init__(
        self,
        *,
        parent,
        classifiers,
        arm,
        feature_mean,
        feature_std,
    ):
        super().__init__()
        self.parent = parent
        self.classifiers = classifiers
        self.arm = arm
        self.feature_mean = feature_mean
        self.feature_std = feature_std

        if arm not in ("ParamUtility", "StateUtility"):
            raise ValueError(
                f"Unknown Utility arm: {arm}"
            )

    @torch.no_grad()
    def forward(
        self,
        x_norm,
        params=None,
    ):
        if params is None:
            raise RuntimeError(
                "R3-3B Utility wrapper requires params."
            )

        gated_delta, _, _, _ = (
            r3_3_rollout.utility_forward(
                parent=self.parent,
                classifiers=self.classifiers,
                arm=self.arm,
                x_norm=x_norm,
                params=params,
                feature_mean=self.feature_mean,
                feature_std=self.feature_std,
            )
        )

        if not torch.isfinite(gated_delta).all():
            raise RuntimeError(
                f"{self.arm}: non-finite gated delta."
            )

        return gated_delta


@torch.no_grad()
def synthetic_wrapper_audit(
    *,
    parent,
    classifiers,
    feature_mean,
    feature_std,
    device,
):
    # First reuse the exact frozen R3-3B interface audit.
    r3_3_rollout.synthetic_audit(
        parent=parent,
        classifiers=classifiers,
        feature_mean=feature_mean,
        feature_std=feature_std,
        device=device,
    )

    x = torch.zeros(
        2,
        16,
        256,
        64,
        dtype=torch.float32,
        device=device,
    )
    params = torch.tensor(
        [
            [6.0, -0.3010300],
            [8.0, 0.3010300],
        ],
        dtype=torch.float32,
        device=device,
    )

    for arm in ("ParamUtility", "StateUtility"):
        wrapper = UtilityGatedModelWrapper(
            parent=parent,
            classifiers=classifiers,
            arm=arm,
            feature_mean=feature_mean,
            feature_std=feature_std,
        ).to(device)
        wrapper.eval()

        wrapped = wrapper(
            x,
            params=params,
        )
        direct, _, _, _ = (
            r3_3_rollout.utility_forward(
                parent=parent,
                classifiers=classifiers,
                arm=arm,
                x_norm=x,
                params=params,
                feature_mean=feature_mean,
                feature_std=feature_std,
            )
        )

        if not torch.equal(
            wrapped,
            direct,
        ):
            max_abs = float(
                torch.max(
                    torch.abs(
                        wrapped - direct
                    )
                ).cpu()
            )
            raise RuntimeError(
                f"{arm}: wrapper/direct mismatch; "
                f"max_abs={max_abs:.12e}"
            )

    print(
        "✅ R3-3B Utility wrapper synthetic audit PASS"
    )
    print(
        "✅ wrapper output exactly equals frozen "
        "R3-3B utility_forward"
    )
    print(
        "✅ frozen R3-1 physics run_one_model can call "
        "wrapper(x_norm, params=param)"
    )


# ============================================================
# Physics difference table
# ============================================================

def make_difference_table(summary_df):
    pairs = [
        # PRIMARY.
        (
            DISPLAY_STATEUTILITY,
            DISPLAY_PARAMUTILITY,
        ),
        # Secondary: Utility effect relative to R3-2b.
        (
            DISPLAY_PARAMUTILITY,
            DISPLAY_STATEPARAM,
        ),
        (
            DISPLAY_STATEUTILITY,
            DISPLAY_STATEPARAM,
        ),
        # Final candidate relative to strongest matched R3-2
        # adaptive control.
        (
            DISPLAY_STATEUTILITY,
            DISPLAY_PARAMONLY,
        ),
        # Context relative to R3-1b canonical.
        (
            DISPLAY_STATEUTILITY,
            DISPLAY_CANONICAL,
        ),
    ]

    rows = []

    for model_a, model_b in pairs:
        a = summary_df[
            summary_df["model"] == model_a
        ]
        b = summary_df[
            summary_df["model"] == model_b
        ]

        merged = a.merge(
            b,
            on=[
                "split",
                "seed",
                "horizon",
            ],
            suffixes=("_a", "_b"),
        )

        for _, row in merged.iterrows():
            out = {
                "split": row["split"],
                "seed": int(row["seed"]),
                "comparison":
                    f"{model_a} - {model_b}",
                "horizon":
                    int(row["horizon"]),
            }

            for metric in physics.LOWER_IS_BETTER:
                out[f"{metric}_diff"] = (
                    row[f"{metric}_a"]
                    -
                    row[f"{metric}_b"]
                )

            rows.append(out)

    return pd.DataFrame(rows)


# ============================================================
# Frozen R3-3B prediction reproduction
#
# Physics evaluator must recover the SAME prediction Rel-L2
# values as the already-frozen R3-3B rollout for all 5 models.
# ============================================================

def audit_prediction_reproduction(
    physics_summary,
    rollout_summary_path,
    *,
    split_label,
    seed,
):
    frozen = pd.read_csv(
        rollout_summary_path
    )

    required_columns = {
        "split",
        "seed",
        "model",
        "horizon",
        "field",
        "rel_l2_percent",
    }
    missing = (
        required_columns
        -
        set(frozen.columns)
    )
    if missing:
        raise RuntimeError(
            "Frozen R3-3B rollout summary "
            f"is missing columns: {sorted(missing)}"
        )

    frozen = frozen[
        (
            frozen["split"]
            ==
            split_label
        )
        &
        (
            frozen["seed"].astype(int)
            ==
            int(seed)
        )
    ].copy()

    if frozen.empty:
        raise RuntimeError(
            "No matching rows in frozen R3-3B "
            "rollout summary."
        )

    metric_map = {
        "global":
            "global_rel_l2_percent",
        "buoyancy":
            "buoyancy_rel_l2_percent",
        "u_x":
            "u_x_rel_l2_percent",
        "u_y":
            "u_y_rel_l2_percent",
        "pressure":
            "pressure_rel_l2_percent",
    }

    audit_rows = []
    max_abs_diff = 0.0

    for model_name in MODEL_ORDER:
        for horizon in FORMAL_HORIZONS:
            phys_row = physics_summary[
                (
                    physics_summary["model"]
                    ==
                    model_name
                )
                &
                (
                    physics_summary["horizon"]
                    ==
                    horizon
                )
            ]

            if len(phys_row) != 1:
                raise RuntimeError(
                    "Expected exactly one physics row for "
                    f"{model_name}, h={horizon}; "
                    f"got {len(phys_row)}"
                )

            phys_row = phys_row.iloc[0]

            for field, physics_column in metric_map.items():
                frozen_row = frozen[
                    (
                        frozen["model"]
                        ==
                        model_name
                    )
                    &
                    (
                        frozen["horizon"]
                        ==
                        horizon
                    )
                    &
                    (
                        frozen["field"]
                        ==
                        field
                    )
                ]

                if len(frozen_row) != 1:
                    raise RuntimeError(
                        "Expected exactly one frozen "
                        "R3-3B rollout row for "
                        f"{model_name}, h={horizon}, "
                        f"field={field}; "
                        f"got {len(frozen_row)}"
                    )

                frozen_value = float(
                    frozen_row.iloc[0][
                        "rel_l2_percent"
                    ]
                )
                physics_value = float(
                    phys_row[
                        physics_column
                    ]
                )
                abs_diff = abs(
                    physics_value
                    -
                    frozen_value
                )
                max_abs_diff = max(
                    max_abs_diff,
                    abs_diff,
                )

                audit_rows.append(
                    {
                        "split":
                            split_label,
                        "seed":
                            int(seed),
                        "model":
                            model_name,
                        "horizon":
                            horizon,
                        "field":
                            field,
                        "frozen_rollout_rel_l2_percent":
                            frozen_value,
                        "physics_eval_rel_l2_percent":
                            physics_value,
                        "abs_diff":
                            abs_diff,
                    }
                )

    audit_df = pd.DataFrame(
        audit_rows
    )

    if (
        max_abs_diff
        >
        PREDICTION_REPRO_ABS_TOL
    ):
        raise RuntimeError(
            "R3-3B physics evaluator does NOT reproduce "
            "the frozen R3-3B prediction rollout. "
            f"max_abs_diff={max_abs_diff:.12e}, "
            f"tolerance={PREDICTION_REPRO_ABS_TOL:.12e}"
        )

    print()
    print(
        "========== FROZEN R3-3B "
        "PREDICTION REPRODUCTION =========="
    )
    print(
        "max_abs Rel-L2 difference = "
        f"{max_abs_diff:.12e}"
    )
    print(
        "✅ Physics evaluator exactly reproduces "
        "the frozen R3-3B prediction trajectory"
    )

    return audit_df, max_abs_diff


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    # --------------------------------------------------------
    # Formal protocol locks
    # --------------------------------------------------------
    if args.seed != 42:
        raise ValueError(
            "Formal R3-3B physics evaluation "
            "is locked to seed=42."
        )
    if args.batch_size != 4:
        raise ValueError(
            "Formal R3-3B physics evaluation "
            "is locked to batch_size=4."
        )

    requested_horizons = sorted(
        {
            int(x)
            for x in args.horizons.split(",")
            if x.strip()
        }
    )
    if (
        requested_horizons
        !=
        list(FORMAL_HORIZONS)
    ):
        raise ValueError(
            "R3-3B physics protocol is locked "
            "to horizons 1,4,8,16."
        )

    max_horizon = max(
        requested_horizons
    )
    if max_horizon != 16:
        raise RuntimeError(
            "R3-3B physics protocol is locked to H16."
        )

    # --------------------------------------------------------
    # Frozen source and result provenance
    # --------------------------------------------------------
    r3_3_rollout_sha = audit_sha(
        R3_3_ROLLOUT_PATH,
        EXPECTED_R3_3_ROLLOUT_SHA256,
        "Frozen R3-3B rollout evaluator",
    )
    r3_2_physics_sha = audit_sha(
        R3_2_PHYSICS_PATH,
        EXPECTED_R3_2_PHYSICS_SHA256,
        "Frozen R3-2 physics evaluator",
    )
    r3_1_physics_sha = audit_sha(
        r3_2_physics.R3_1_PHYSICS_PATH,
        r3_2_physics.EXPECTED_R3_1_PHYSICS_SHA256,
        "Frozen R3-1 physics evaluator",
    )

    pretest_path = resolve_path(
        R3_3_PRETEST_REGISTRY
    )
    pretest_sha = audit_sha(
        pretest_path,
        EXPECTED_R3_3_PRETEST_REGISTRY_SHA256,
        "Frozen R3-3B pre-TEST registry",
    )

    frozen_rollout = (
        load_and_audit_frozen_r3_3_rollout(
            split_label=args.split_label,
            seed=args.seed,
        )
    )

    # --------------------------------------------------------
    # Frozen split / parent resources
    # --------------------------------------------------------
    resource = (
        r3_3_rollout
        .LOCKED_RESOURCES[
            args.split_label
        ]
    )

    split_path = resolve_path(
        resource["split"]
    )
    stats_path = resolve_path(
        resource["stats"]
    )
    m6_path = resolve_path(
        resource["m6"]
    )

    r3_3_rollout.audit_file(
        resource["split"],
        resource["split_sha256"],
        f"{args.split_label} split",
    )
    r3_3_rollout.audit_file(
        resource["stats"],
        resource["stats_sha256"],
        f"{args.split_label} stats",
    )
    r3_3_rollout.audit_file(
        resource["m6"],
        resource["m6_sha256"],
        f"{args.split_label} M6",
    )

    (
        _,
        entries,
        r3_2_registry_path,
        _,
        r3_2_contract_sha,
    ) = (
        r3_3_rollout
        .load_and_audit_registry(
            args.split_label
        )
    )
    r3_2_registry_sha = sha256_file(
        r3_2_registry_path
    )

    parent_sha = (
        entries["R3-2b"][
            "parent_r3_1b_sha256"
        ]
    )
    if (
        entries["R3-2a"][
            "parent_r3_1b_sha256"
        ]
        !=
        parent_sha
    ):
        raise RuntimeError(
            "R3-2a/R3-2b frozen parent mismatch."
        )

    r3_1b_path, r3_1b_sha = (
        r3_3_rollout.audit_file(
            resource["r3_1b"],
            parent_sha,
            f"{args.split_label} R3-1b parent",
        )
    )

    paramonly_path = resolve_path(
        entries["R3-2a"]["checkpoint"]
    )
    stateparam_path = resolve_path(
        entries["R3-2b"]["checkpoint"]
    )
    paramonly_sha = sha256_file(
        paramonly_path
    )
    stateparam_sha = sha256_file(
        stateparam_path
    )

    # --------------------------------------------------------
    # Device / deterministic label
    # --------------------------------------------------------
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )

    torch.manual_seed(
        args.seed
    )
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            args.seed
        )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------
    print("=" * 124)
    print(
        "R3-3B MATCHED FORMAL TEST "
        "PHYSICS-SUPPORTIVE EVALUATION"
    )
    print("=" * 124)
    print(
        "Type: formal supportive evaluation; NO TRAINING"
    )
    print("Split:", args.split_label)
    print("Seed:", args.seed)
    print("Models:", MODEL_ORDER)
    print("Primary contrast:", PRIMARY_COMPARISON)
    print(
        "Negative primary difference = "
        "StateUtility better"
    )
    print(
        "Prediction conclusion was frozen BEFORE "
        "this physics evaluation"
    )
    print(
        "Physics semantics: FD-based comparative proxy; "
        "NOT solver-level PDE violation"
    )
    print("Audit only:", args.audit_only)

    print()
    print(
        "========== FROZEN PROVENANCE =========="
    )
    provenance = {
        "R3_3_ROLLOUT_EVALUATOR_SHA256":
            r3_3_rollout_sha,
        "R3_2_PHYSICS_EVALUATOR_SHA256":
            r3_2_physics_sha,
        "R3_1_PHYSICS_EVALUATOR_SHA256":
            r3_1_physics_sha,
        "R3_3_PRETEST_REGISTRY_SHA256":
            pretest_sha,
        "R3_3_FORMAL_ROLLOUT_REGISTRY_SHA256":
            EXPECTED_R3_3_FORMAL_ROLLOUT_REGISTRY_SHA256,
        "R3_3_PRETEST_FREEZE_COMMIT":
            EXPECTED_R3_3_PRETEST_FREEZE_COMMIT,
        "R3_3_ROLLOUT_RESULT_FREEZE_COMMIT":
            EXPECTED_R3_3_ROLLOUT_RESULT_FREEZE_COMMIT,
        "R3_2_REGISTRY_SHA256":
            r3_2_registry_sha,
        "R3_1B_SHA256":
            r3_1b_sha,
        "R3_2A_SHA256":
            paramonly_sha,
        "R3_2B_SHA256":
            stateparam_sha,
    }
    for key, value in provenance.items():
        print(f"{key}:", value)

    # --------------------------------------------------------
    # Build exact frozen models / Utility classifiers.
    #
    # Allowed in --audit_only.
    # Physics TEST dataset has not been constructed.
    # --------------------------------------------------------
    normalizer = FieldWiseNormalizer(
        stats_path
    ).to(device)

    metadata = build_rbc_canonical_metadata(
        stats_path
    )

    m6, _ = r3_3_rollout.build_m6(
        m6_path,
        device,
    )

    class R3Args:
        pass

    r3_args = R3Args()
    r3_args.split_label = args.split_label
    r3_args.seed = args.seed

    r3_0_path, r3_0_sha = (
        r3_3_rollout.audit_file(
            r3_3_rollout.R3_0_CONTRACT,
            r3_3_rollout.EXPECTED_R3_0_CONTRACT_SHA256,
            "R3-0 contract",
        )
    )
    _, alpha_map = (
        r3_3_rollout
        .load_locked_contract(
            r3_0_path
        )
    )

    canonical, canonical_payload = (
        r3_3_rollout.build_r3(
            representation_mode="canonical",
            checkpoint_path=r3_1b_path,
            metadata=metadata,
            alpha_map=alpha_map,
            contract_sha=r3_0_sha,
            args=r3_args,
            device=device,
        )
    )

    paramonly, paramonly_payload = (
        r3_3_rollout.build_r3_2(
            adaptation_mode="paramonly",
            checkpoint_path=paramonly_path,
            expected_entry=entries["R3-2a"],
            metadata=metadata,
            alpha_map=alpha_map,
            r3_2_contract_sha=r3_2_contract_sha,
            split_label=args.split_label,
            device=device,
        )
    )

    stateparam, stateparam_payload = (
        r3_3_rollout.build_r3_2(
            adaptation_mode="stateparam",
            checkpoint_path=stateparam_path,
            expected_entry=entries["R3-2b"],
            metadata=metadata,
            alpha_map=alpha_map,
            r3_2_contract_sha=r3_2_contract_sha,
            split_label=args.split_label,
            device=device,
        )
    )

    r3_3_rollout.assert_same_m6(
        m6,
        canonical,
        DISPLAY_CANONICAL,
    )
    r3_3_rollout.assert_same_m6(
        m6,
        paramonly,
        DISPLAY_PARAMONLY,
    )
    r3_3_rollout.assert_same_m6(
        m6,
        stateparam,
        DISPLAY_STATEPARAM,
    )
    r3_3_rollout.assert_same_parent_raw_alpha(
        canonical,
        paramonly,
        DISPLAY_PARAMONLY,
    )
    r3_3_rollout.assert_same_parent_raw_alpha(
        canonical,
        stateparam,
        DISPLAY_STATEPARAM,
    )

    utility = (
        r3_3_rollout
        .load_and_audit_r3_3_formal_registry(
            args.split_label,
            device,
        )
    )

    synthetic_wrapper_audit(
        parent=stateparam,
        classifiers=utility["classifiers"],
        feature_mean=utility["feature_mean"],
        feature_std=utility["feature_std"],
        device=device,
    )

    wrappers = {
        DISPLAY_PARAMUTILITY:
            UtilityGatedModelWrapper(
                parent=stateparam,
                classifiers=utility["classifiers"],
                arm="ParamUtility",
                feature_mean=utility["feature_mean"],
                feature_std=utility["feature_std"],
            ).to(device).eval(),

        DISPLAY_STATEUTILITY:
            UtilityGatedModelWrapper(
                parent=stateparam,
                classifiers=utility["classifiers"],
                arm="StateUtility",
                feature_mean=utility["feature_mean"],
                feature_std=utility["feature_std"],
            ).to(device).eval(),
    }

    print()
    print(
        "========== FROZEN MODEL SELECTION =========="
    )
    for model_name, payload in (
        (
            DISPLAY_CANONICAL,
            canonical_payload,
        ),
        (
            DISPLAY_PARAMONLY,
            paramonly_payload,
        ),
        (
            DISPLAY_STATEPARAM,
            stateparam_payload,
        ),
    ):
        print(
            f"{model_name}: "
            "best_val="
            f"{payload.get('best_val_loss', payload.get('val_loss'))} "
            "best_epoch="
            f"{payload.get('best_epoch', payload.get('epoch'))}"
        )

    print(
        f"{DISPLAY_PARAMUTILITY}: "
        "frozen R3-2b parent + frozen R3-3A classifiers"
    )
    print(
        f"{DISPLAY_STATEUTILITY}: "
        "frozen R3-2b parent + frozen R3-3A classifiers"
    )

    # --------------------------------------------------------
    # AUDIT-ONLY STOP.
    # --------------------------------------------------------
    if args.audit_only:
        print()
        print(
            "✅ R3-3B PHYSICS EVALUATOR "
            "AUDIT-ONLY PASS"
        )
        print(
            "✅ Frozen rollout result / sources / "
            "checkpoints / Utility wrappers verified"
        )
        print(
            "✅ Physics TEST dataset was NOT constructed"
        )
        return

    # ========================================================
    # Formal physics TEST access starts here.
    # ========================================================
    print()
    print(
        "TEST access: YES "
        "(formal post-freeze physics evaluation)"
    )

    if args.max_samples is not None:
        print(
            "⚠️ DEBUG ONLY: max_samples =",
            args.max_samples,
        )
        print(
            "⚠️ Debug physics results are NOT formal evidence."
        )

    split_config = load_json(
        split_path
    )

    dataset = (
        r3_3_rollout.RolloutDataset(
            split_config=split_config["test"],
            max_horizon=max_horizon,
            max_samples=args.max_samples,
        )
    )

    if (
        args.max_samples is None
        and
        len(dataset) != FORMAL_TEST_WINDOWS_H16
    ):
        raise RuntimeError(
            "Formal R3-3B physics evaluation must "
            "use exactly the same 2430 H16 TEST windows "
            "as the frozen R3-3B prediction rollout; "
            f"got {len(dataset)}."
        )

    print()
    print(
        "========== DATA CONTRACT =========="
    )
    print(
        "TEST_WINDOWS:",
        len(dataset),
    )
    print(
        "Temporal-start stride: 1 "
        "(all legal frozen R3-3B TEST windows)"
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # ========================================================
    # Full free-autoregressive physics rollout.
    #
    # ALL five models use the exact frozen R3-1
    # run_one_model() trajectory / FD accumulation.
    # Utility differences exist only inside thin wrappers.
    # ========================================================
    models = {
        DISPLAY_CANONICAL:
            canonical,
        DISPLAY_PARAMONLY:
            paramonly,
        DISPLAY_STATEPARAM:
            stateparam,
        DISPLAY_PARAMUTILITY:
            wrappers[DISPLAY_PARAMUTILITY],
        DISPLAY_STATEUTILITY:
            wrappers[DISPLAY_STATEUTILITY],
    }

    stats = {}

    print()
    print(
        "🔥 Starting R3-3B free-autoregressive "
        "TEST Physics Audit v2..."
    )

    with torch.no_grad():
        for batch_idx, (
            x0_phys,
            future_phys,
            param,
        ) in enumerate(
            loader,
            start=1,
        ):
            x0_phys = x0_phys.to(
                device,
                non_blocking=True,
            )
            future_phys = future_phys.to(
                device,
                non_blocking=True,
            )
            param = param.to(
                device,
                non_blocking=True,
            )

            for model_name in MODEL_ORDER:
                # Exact frozen R3-1 physics trajectory and
                # metric accumulation. For every non-M6 model,
                # run_one_model calls model(x_norm, params=param).
                r3_1_physics.run_one_model(
                    model_name=model_name,
                    model=models[model_name],
                    x0_phys=x0_phys,
                    future_phys=future_phys,
                    param=param,
                    normalizer=normalizer,
                    max_horizon=max_horizon,
                    stats=stats,
                )

            if (
                batch_idx % 25 == 0
                or
                batch_idx == len(loader)
            ):
                print(
                    "  processed batch "
                    f"{batch_idx}/{len(loader)}"
                )

    # ========================================================
    # Finalize full H1-H16 curve using frozen physics semantics.
    # ========================================================
    rows = []

    for model_name in MODEL_ORDER:
        for horizon in range(
            1,
            max_horizon + 1,
        ):
            key = (
                model_name,
                horizon,
            )
            if key not in stats:
                raise RuntimeError(
                    f"Missing physics bucket: {key}"
                )

            rows.append(
                physics.finalize_bucket(
                    split_label=args.split_label,
                    seed=args.seed,
                    model_name=model_name,
                    horizon=horizon,
                    bucket=stats[key],
                )
            )

    curve_df = pd.DataFrame(
        rows
    )

    summary_df = curve_df[
        curve_df["horizon"].isin(
            requested_horizons
        )
    ].copy()

    diff_df = make_difference_table(
        summary_df
    )

    # ========================================================
    # Frozen prediction reproduction audit.
    # ========================================================
    if args.max_samples is None:
        (
            reproduction_df,
            reproduction_max_abs_diff,
        ) = (
            audit_prediction_reproduction(
                summary_df,
                frozen_rollout["paths"]["summary"],
                split_label=args.split_label,
                seed=args.seed,
            )
        )
    else:
        reproduction_df = pd.DataFrame()
        reproduction_max_abs_diff = float("nan")
        print()
        print(
            "⚠️ Frozen prediction reproduction "
            "audit skipped for DEBUG subset."
        )

    # ========================================================
    # Save
    # ========================================================
    output_dir = resolve_path(
        args.output_dir
    )
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    prefix = (
        "r3_3b_physics_"
        f"{args.split_label}_"
        f"seed{args.seed}"
    )

    paths = {
        "curve":
            os.path.join(
                output_dir,
                f"{prefix}_curve_h1_h16.csv",
            ),
        "summary":
            os.path.join(
                output_dir,
                f"{prefix}_summary_h1_h4_h8_h16.csv",
            ),
        "differences":
            os.path.join(
                output_dir,
                f"{prefix}_differences.csv",
            ),
        "prediction_reproduction":
            os.path.join(
                output_dir,
                f"{prefix}_prediction_reproduction.csv",
            ),
        "metadata":
            os.path.join(
                output_dir,
                f"{prefix}_metadata.json",
            ),
    }

    curve_df.to_csv(
        paths["curve"],
        index=False,
    )
    summary_df.to_csv(
        paths["summary"],
        index=False,
    )
    diff_df.to_csv(
        paths["differences"],
        index=False,
    )

    if not reproduction_df.empty:
        reproduction_df.to_csv(
            paths["prediction_reproduction"],
            index=False,
        )

    metadata_out = {
        "experiment":
            "R3-3B-matched-formal-test-physics-supportive",
        "stage":
            "R3-3B-physics-supportive",
        "formal_run":
            args.max_samples is None,
        "neural_operator_training":
            False,
        "test_split_accessed":
            True,
        "prediction_conclusion_frozen_before_physics":
            True,
        "prediction_rollout_result_freeze_commit":
            EXPECTED_R3_3_ROLLOUT_RESULT_FREEZE_COMMIT,
        "split_label":
            args.split_label,
        "seed":
            args.seed,
        "models":
            list(MODEL_ORDER),
        "primary_comparison":
            PRIMARY_COMPARISON,
        "primary_interpretation":
            (
                "incremental physics behavior of current-state "
                "information specifically for Utility/state "
                "selection beyond the matched ParamUtility control"
            ),
        "negative_primary_difference_means":
            "StateUtility better",
        "secondary_comparisons":
            list(SECONDARY_COMPARISONS),
        "key_physics_metrics_predeclared":
            list(KEY_PHYSICS_METRICS),
        "physics_evidence_role":
            (
                "supportive only; does not override the already-"
                "frozen R3-3B prediction rollout decision"
            ),
        "formal_test_windows_h16":
            (
                len(dataset)
                if args.max_samples is None
                else None
            ),
        "sampling_protocol":
            (
                "same frozen R3-3B TEST rollout windows; "
                "all legal t0 starts; no physics-specific stride"
            ),
        "rollout_protocol":
            (
                "free-autoregressive; model prediction is fed "
                "back into next context"
            ),
        "horizons_reported":
            requested_horizons,
        "full_curve":
            "h=1..16",
        "physics_semantics":
            (
                "FD-based comparative proxy; "
                "not solver-level PDE violation"
            ),
        "physics_dx":
            PHYSICS_DX,
        "physics_dy":
            PHYSICS_DY,
        "physics_dt":
            PHYSICS_DT,
        "prediction_reproduction_abs_tol":
            PREDICTION_REPRO_ABS_TOL,
        "prediction_reproduction_max_abs_diff":
            (
                reproduction_max_abs_diff
                if args.max_samples is None
                else None
            ),
        "frozen_r3_3_rollout_summary":
            frozen_rollout["paths"]["summary"],
        "utility_wrapper_semantics":
            (
                "wrapper delegates exactly to frozen R3-3B "
                "utility_forward; gated correction=q*term_correction; "
                "term_correction already contains alpha"
            ),
        "utility_parent":
            "frozen split-matched R3-2b StateParam",
        "paramutility_semantics":
            (
                "normalize with frozen TRAIN-only mean/std, "
                "then normalized dims 0..7 exact zero"
            ),
        "stateutility_semantics":
            (
                "normalize with frozen TRAIN-only mean/std; "
                "all 10D visible"
            ),
        "no_threshold":
            True,
        "no_temperature":
            True,
        "no_step1_special_case":
            True,
        "provenance":
            provenance,
    }

    with open(
        paths["metadata"],
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
    # Compact terminal output
    # ========================================================
    compact_cols = [
        "model",
        "horizon",
        "global_rel_l2_percent",
        "buoyancy_rel_l2_percent",
        "u_y_rel_l2_percent",
        "adv_b_rel_l2_percent",
        "div_error_mae",
        "vorticity_rel_l2_percent",
        "r_b_normalized_mismatch",
    ]

    utility_summary = summary_df[
        summary_df["model"].isin(
            UTILITY_MODELS
        )
    ][
        compact_cols
    ].copy()

    print()
    print("=" * 142)
    print(
        "R3-3B UTILITY PHYSICS SUMMARY "
        "(supportive evidence)"
    )
    print("=" * 142)
    print(
        utility_summary.to_string(
            index=False
        )
    )

    primary = diff_df[
        diff_df["comparison"]
        ==
        PRIMARY_COMPARISON
    ].copy()

    key_diff_cols = [
        "comparison",
        "horizon",
        "global_rel_l2_percent_diff",
        "buoyancy_rel_l2_percent_diff",
        "u_y_rel_l2_percent_diff",
        "adv_b_rel_l2_percent_diff",
        "div_error_mae_diff",
        "vorticity_rel_l2_percent_diff",
        "r_b_normalized_mismatch_diff",
    ]

    print()
    print("=" * 142)
    print(
        "R3-3B-b STATEUTILITY "
        "- R3-3B-a PARAMUTILITY "
        "| PRIMARY PHYSICS DIFFERENCES"
    )
    print(
        "Negative = StateUtility better "
        "for every displayed metric."
    )
    print("=" * 142)
    print(
        primary[
            key_diff_cols
        ].to_string(
            index=False
        )
    )

    gt_calibration_cols = [
        "horizon",
        "r_b_gt_rms",
        "r_uy_gt_rms",
        "r_u_gt_rms",
        "r_b_gt_residual_to_term_scale",
        "r_uy_gt_residual_to_term_scale",
        "r_u_gt_residual_to_term_scale",
    ]

    gt_calibration = summary_df[
        summary_df["model"]
        ==
        DISPLAY_CANONICAL
    ][
        gt_calibration_cols
    ].copy()

    print()
    print(
        "========== GT FD-PROXY CALIBRATION =========="
    )
    print(
        "Non-zero GT FD residual is calibration only; "
        "it is NOT a model error."
    )
    print(
        gt_calibration.to_string(
            index=False
        )
    )

    print()
    print("✅ Saved:")
    for key, path in paths.items():
        if (
            key
            ==
            "prediction_reproduction"
            and
            reproduction_df.empty
        ):
            continue
        print(f"  {key}: {path}")

    print()
    if args.max_samples is None:
        print(
            "✅ FORMAL R3-3B TEST "
            "PHYSICS-SUPPORTIVE EVALUATION COMPLETE"
        )
    else:
        print(
            "✅ DEBUG R3-3B TEST "
            "PHYSICS-SUPPORTIVE EVALUATION COMPLETE "
            "(NOT FORMAL)"
        )


if __name__ == "__main__":
    main()
