from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.normalization import FieldWiseNormalizer
from physics.canonical_metadata import build_rbc_canonical_metadata

from evaluation.evaluate_cross_param_rollout_r3_1 import (
    RolloutDataset,
    assert_same_m6,
    build_m6,
    build_r3,
    error_rows,
    load_locked_contract,
    make_global_growth_table,
    resolve_path,
    sha256_file,
    update_error_stats,
)

from evaluation.evaluate_cross_param_rollout_r3_2 import (
    ACTIVE_TERMS,
    DISPLAY_CANONICAL,
    DISPLAY_PARAMONLY,
    DISPLAY_STATEPARAM,
    EXPECTED_R3_0_CONTRACT_SHA256,
    LOCKED_RESOURCES,
    R3_0_CONTRACT,
    assert_same_parent_raw_alpha,
    build_r3_2,
    gate_rows,
    load_and_audit_registry,
    load_json,
    rollout_one_model as rollout_r3_2_reference,
)

from evaluation.audit_r3_3a_predictability import (
    FEATURE_NAMES,
    STATE_FEATURE_COUNT,
    UtilityClassifier,
)

DISPLAY_PARAMUTILITY = "R3-3B-a-ParamUtility"
DISPLAY_STATEUTILITY = "R3-3B-b-StateUtility"

REFERENCE_MODELS = (
    DISPLAY_CANONICAL,
    DISPLAY_PARAMONLY,
    DISPLAY_STATEPARAM,
)
UTILITY_MODELS = (
    DISPLAY_PARAMUTILITY,
    DISPLAY_STATEUTILITY,
)
MODEL_ORDER = REFERENCE_MODELS + UTILITY_MODELS

R3_3_CONTRACT = "configs/r3/r3_3_utility_contract.json"
R3_3_FORMAL_REGISTRY = (
    "configs/r3/r3_3a_formal_predictability_registry.json"
)

EXPECTED_R3_3_CONTRACT_SHA256 = (
    "426299fcce724d1e74490887c39ea583"
    "07f2180bf464c4d41ee6ee170b6c4af8"
)
EXPECTED_R3_3_FORMAL_REGISTRY_SHA256 = (
    "c82513a785263083d4aed2bd9a767c0e"
    "f7bbb242ccb2f1d70e3b82eb276acfd1"
)
EXPECTED_R3_2_EVALUATOR_SHA256 = (
    "888f1205550af0caf582ce50eac2ff96"
    "d0f891f4862d6b083b64e47f9d764949"
)
EXPECTED_R3_2_MODEL_SHA256 = (
    "793ef3da43958ded2365cb1269610d42"
    "ba13187008f306151c3158f9440c7e35"
)
EXPECTED_R3_3_PREDICTABILITY_SOURCE_SHA256 = (
    "3b9de522bb462746a3071bf27508e1e3"
    "896f4b8a9a3a2a5820aa5d6108848da7"
)
EXPECTED_DATASET_REGISTRY_SHA256 = (
    "97171f60689c8fe3c32746640a109b1f"
    "e554a091073f68536550711b7f273d40"
)

R3_2_EVALUATOR_PATH = (
    "evaluation/evaluate_cross_param_rollout_r3_2.py"
)
R3_2_MODEL_PATH = (
    "models/operators/fno2d_r3_state_adaptive.py"
)
R3_3_PREDICTABILITY_SOURCE_PATH = (
    "evaluation/audit_r3_3a_predictability.py"
)

EXPECTED_CLASSIFIER_NUMEL = 897

ARM_BY_DISPLAY = {
    DISPLAY_PARAMUTILITY: "ParamUtility",
    DISPLAY_STATEUTILITY: "StateUtility",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "R3-3B matched closed-loop Utility-gating rollout: "
            "ParamUtility vs StateUtility, with frozen R3-2 references."
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
    parser.add_argument("--device", default="cuda")
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
            "Audit frozen provenance/checkpoints/interfaces only. "
            "Does not construct or access TEST rollout data."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/tables/r3_3_rollout",
    )
    return parser.parse_args()


def audit_file(path, expected_sha, label):
    path = resolve_path(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected_sha:
        raise RuntimeError(
            f"{label} SHA256 mismatch\n"
            f"expected={expected_sha}\n"
            f"actual={actual}\n"
            f"path={path}"
        )
    return path, actual


def load_and_audit_r3_3_formal_registry(split_label, device):
    contract_path, contract_sha = audit_file(
        R3_3_CONTRACT,
        EXPECTED_R3_3_CONTRACT_SHA256,
        "R3-3 contract",
    )
    registry_path, registry_sha = audit_file(
        R3_3_FORMAL_REGISTRY,
        EXPECTED_R3_3_FORMAL_REGISTRY_SHA256,
        "R3-3A formal predictability registry",
    )

    registry = load_json(registry_path)

    if registry.get("stage") != "R3-3A":
        raise RuntimeError("Unexpected formal registry stage.")
    if registry.get("status") != "FORMAL_PREDICTABILITY_COMPLETE":
        raise RuntimeError("Formal predictability registry not complete.")
    if registry["formal_protocol"]["test_accessed"] is not False:
        raise RuntimeError("Formal Utility registry claims TEST access.")

    if (
        registry["implementation_freeze"]["source_sha256"]
        != EXPECTED_R3_3_PREDICTABILITY_SOURCE_SHA256
    ):
        raise RuntimeError("Predictability source identity changed.")

    axis = registry[split_label]

    classifiers = {}
    payloads = {}
    shared_mean = None
    shared_std = None
    shared_constant_mask = None

    for term in ACTIVE_TERMS:
        for arm in ("ParamUtility", "StateUtility"):
            arm_lower = arm.lower()
            path = resolve_path(
                "checkpoints/r3_3_utility/"
                f"{split_label}/"
                f"r3_3a_{split_label}_{term}_{arm_lower}_seed42.pth"
            )

            sha_key = (
                "paramutility_checkpoint_sha256"
                if arm == "ParamUtility"
                else "stateutility_checkpoint_sha256"
            )
            expected_sha = axis[term][sha_key]
            actual_sha = sha256_file(path)
            if actual_sha != expected_sha:
                raise RuntimeError(
                    f"{split_label}/{term}/{arm} checkpoint SHA mismatch."
                )

            payload = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )

            checks = {
                "stage": "R3-3A",
                "role": "UTILITY_PREDICTABILITY_CLASSIFIER",
                "split": split_label,
                "term": term,
                "arm": arm,
                "formal": True,
                "seed": 42,
                "epochs": 100,
                "batch_size": 512,
                "learning_rate": 1.0e-3,
                "classifier_numel": EXPECTED_CLASSIFIER_NUMEL,
                "state_feature_count": 8,
                "dataset_registry_sha256": EXPECTED_DATASET_REGISTRY_SHA256,
                "r3_3_contract_sha256": EXPECTED_R3_3_CONTRACT_SHA256,
                "val_model_selection": False,
                "val_threshold_tuning": False,
                "val_temperature_tuning": False,
                "test_accessed": False,
            }
            for key, expected in checks.items():
                if payload.get(key) != expected:
                    raise RuntimeError(
                        f"{term}/{arm}: checkpoint metadata mismatch "
                        f"at {key}: {payload.get(key)!r} != {expected!r}"
                    )

            if tuple(payload["feature_names"]) != tuple(FEATURE_NAMES):
                raise RuntimeError(f"{term}/{arm}: feature names changed.")

            mean = np.asarray(payload["feature_mean"], dtype=np.float64)
            std = np.asarray(payload["feature_std"], dtype=np.float64)
            constant_mask = np.asarray(
                payload["constant_feature_mask"],
                dtype=bool,
            )

            if mean.shape != (10,) or std.shape != (10,):
                raise RuntimeError(f"{term}/{arm}: normalization shape changed.")
            if not np.isfinite(mean).all() or not np.isfinite(std).all():
                raise RuntimeError(f"{term}/{arm}: non-finite normalization.")
            if np.any(std <= 0.0):
                raise RuntimeError(f"{term}/{arm}: non-positive stored std.")

            if shared_mean is None:
                shared_mean = mean.copy()
                shared_std = std.copy()
                shared_constant_mask = constant_mask.copy()
            else:
                if not np.array_equal(mean, shared_mean):
                    raise RuntimeError("Utility checkpoints do not share mean.")
                if not np.array_equal(std, shared_std):
                    raise RuntimeError("Utility checkpoints do not share std.")
                if not np.array_equal(
                    constant_mask,
                    shared_constant_mask,
                ):
                    raise RuntimeError(
                        "Utility checkpoints do not share constant mask."
                    )

            model = UtilityClassifier().to(device)
            model.load_state_dict(
                payload["model_state_dict"],
                strict=True,
            )
            model.eval()

            if sum(p.numel() for p in model.parameters()) != 897:
                raise RuntimeError("Utility classifier numel != 897.")
            for parameter in model.parameters():
                parameter.requires_grad_(False)

            classifiers[(arm, term)] = model
            payloads[(arm, term)] = {
                "path": path,
                "sha256": actual_sha,
                "val_auc": float(payload["val_auc"]),
            }

    return {
        "contract_path": contract_path,
        "contract_sha": contract_sha,
        "registry_path": registry_path,
        "registry_sha": registry_sha,
        "registry": registry,
        "classifiers": classifiers,
        "payloads": payloads,
        "feature_mean": shared_mean,
        "feature_std": shared_std,
        "constant_feature_mask": shared_constant_mask,
    }


def exact_classifier_input(
    full_features,
    arm,
    feature_mean,
    feature_std,
    device,
):
    raw = (
        full_features
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64, copy=False)
    )

    if raw.ndim != 2 or raw.shape[1] != 10:
        raise RuntimeError(
            f"R3-2 full_conditioner_features must be [B,10], got {raw.shape}"
        )

    z = (raw - feature_mean) / feature_std
    if not np.isfinite(z).all():
        raise RuntimeError("Non-finite normalized Utility features.")

    z = z.astype(np.float32)

    if arm == "ParamUtility":
        z = z.copy()
        z[:, :STATE_FEATURE_COUNT] = 0.0
        if not np.array_equal(
            z[:, :STATE_FEATURE_COUNT],
            np.zeros_like(z[:, :STATE_FEATURE_COUNT]),
        ):
            raise RuntimeError("ParamUtility state mask is not exact zero.")
    elif arm != "StateUtility":
        raise ValueError(f"Unknown Utility arm: {arm}")

    return torch.from_numpy(z).to(
        device=device,
        dtype=torch.float32,
    )


@torch.no_grad()
def utility_forward(
    *,
    parent,
    classifiers,
    arm,
    x_norm,
    params,
    feature_mean,
    feature_std,
):
    parent_delta, info = parent(
        x_norm,
        params=params,
        return_components=True,
    )

    required = {
        "base_delta_norm",
        "full_conditioner_features",
        "term_corrections",
        "effective_gate_values",
        "feature_names",
    }
    missing = required - set(info)
    if missing:
        raise RuntimeError(f"R3-2b parent missing keys: {sorted(missing)}")
    if tuple(info["feature_names"]) != tuple(FEATURE_NAMES):
        raise RuntimeError("R3-2b parent feature order changed.")

    classifier_x = exact_classifier_input(
        info["full_conditioner_features"],
        arm,
        feature_mean,
        feature_std,
        x_norm.device,
    )

    q_by_term = {}
    gated_delta = info["base_delta_norm"].clone()

    for term in ACTIVE_TERMS:
        logits = classifiers[(arm, term)](classifier_x)
        q = torch.sigmoid(logits).reshape(-1)

        if q.numel() != x_norm.shape[0]:
            raise RuntimeError(f"{arm}/{term}: q shape mismatch.")
        if not torch.isfinite(q).all():
            raise RuntimeError(f"{arm}/{term}: non-finite q.")
        if torch.any(q < 0.0) or torch.any(q > 1.0):
            raise RuntimeError(f"{arm}/{term}: q outside [0,1].")

        correction = info["term_corrections"][term]
        if correction.ndim != 3:
            raise RuntimeError(
                f"{arm}/{term}: correction must be [B,X,Y], "
                f"got {tuple(correction.shape)}"
            )

        out_idx = parent.TERM_TO_OUTPUT_INDEX[term]
        gated_delta[:, out_idx, :, :] = (
            gated_delta[:, out_idx, :, :]
            + q[:, None, None] * correction
        )
        q_by_term[term] = q

    return gated_delta, parent_delta, info, q_by_term


def new_utility_bucket():
    return {
        "n": 0,
        "q_sum": 0.0,
        "q_sq_sum": 0.0,
        "q_min": float("inf"),
        "q_max": float("-inf"),
        "alpha_sum": 0.0,
        "alpha_sq_sum": 0.0,
        "qalpha_abs_capacity_sum": 0.0,
        "qalpha_sat99_n": 0,
    }


def update_utility_stats(
    stats,
    model_name,
    step,
    term,
    q,
    alpha,
    alpha_cap,
):
    key = (model_name, step, term)
    if key not in stats:
        stats[key] = new_utility_bucket()

    bucket = stats[key]
    q_np = q.detach().double().cpu().numpy().reshape(-1)
    alpha_np = alpha.detach().double().cpu().numpy().reshape(-1)

    if alpha_np.size == 1 and q_np.size > 1:
        alpha_np = np.repeat(alpha_np, q_np.size)
    if q_np.shape != alpha_np.shape:
        raise RuntimeError("q/alpha diagnostic shape mismatch.")

    n = int(q_np.size)
    bucket["n"] += n
    bucket["q_sum"] += float(np.sum(q_np))
    bucket["q_sq_sum"] += float(np.sum(q_np * q_np))
    bucket["q_min"] = min(bucket["q_min"], float(np.min(q_np)))
    bucket["q_max"] = max(bucket["q_max"], float(np.max(q_np)))
    bucket["alpha_sum"] += float(np.sum(alpha_np))
    bucket["alpha_sq_sum"] += float(np.sum(alpha_np * alpha_np))

    ratio = np.abs(q_np * alpha_np) / max(float(alpha_cap), 1.0e-30)
    bucket["qalpha_abs_capacity_sum"] += float(np.sum(ratio))
    bucket["qalpha_sat99_n"] += int(np.sum(ratio >= 0.99))


def utility_stat_rows(stats, max_horizon):
    rows = []

    for model_name in UTILITY_MODELS:
        for step in range(1, max_horizon + 1):
            row = {
                "model": model_name,
                "horizon": step,
            }

            for term in ACTIVE_TERMS:
                bucket = stats[(model_name, step, term)]
                n = bucket["n"]
                q_mean = bucket["q_sum"] / n
                q_var = max(
                    0.0,
                    bucket["q_sq_sum"] / n - q_mean * q_mean,
                )
                alpha_mean = bucket["alpha_sum"] / n
                alpha_var = max(
                    0.0,
                    bucket["alpha_sq_sum"] / n
                    - alpha_mean * alpha_mean,
                )

                short = (
                    "advection"
                    if term == "buoyancy_advection"
                    else "forcing"
                )
                row[f"q_{short}_mean"] = q_mean
                row[f"q_{short}_std"] = math.sqrt(q_var)
                row[f"q_{short}_min"] = bucket["q_min"]
                row[f"q_{short}_max"] = bucket["q_max"]
                row[f"parent_alpha_{short}_mean"] = alpha_mean
                row[f"parent_alpha_{short}_std"] = math.sqrt(alpha_var)
                row[
                    f"qalpha_{short}_mean_abs_capacity_ratio"
                ] = bucket["qalpha_abs_capacity_sum"] / n
                row[
                    f"qalpha_{short}_sat99_fraction"
                ] = bucket["qalpha_sat99_n"] / n

            rows.append(row)

    return rows


@torch.no_grad()
def rollout_utility_model(
    *,
    model_name,
    parent,
    classifiers,
    arm,
    x0_phys,
    future_phys,
    param,
    normalizer,
    max_horizon,
    error_stats,
    utility_stats,
    alpha_map,
    feature_mean,
    feature_std,
):
    x_norm = normalizer.normalize_x(x0_phys)

    for step in range(1, max_horizon + 1):
        current_norm = x_norm[:, -4:, :, :]

        gated_delta, _, info, q_by_term = utility_forward(
            parent=parent,
            classifiers=classifiers,
            arm=arm,
            x_norm=x_norm,
            params=param,
            feature_mean=feature_mean,
            feature_std=feature_std,
        )

        pred_next_norm = current_norm + gated_delta
        pred_next_phys = normalizer.denormalize_y(pred_next_norm)
        true_phys = future_phys[:, step - 1, :, :, :]

        update_error_stats(
            error_stats,
            model_name,
            step,
            pred_next_phys,
            true_phys,
        )

        for term in ACTIVE_TERMS:
            update_utility_stats(
                utility_stats,
                model_name,
                step,
                term,
                q_by_term[term],
                info["effective_gate_values"][term],
                alpha_map[term],
            )

        x_norm = torch.cat(
            [x_norm[:, 4:, :, :], pred_next_norm],
            dim=1,
        )


def make_difference_table(summary_df):
    pairs = [
        (DISPLAY_STATEUTILITY, DISPLAY_PARAMUTILITY),
        (DISPLAY_PARAMUTILITY, DISPLAY_STATEPARAM),
        (DISPLAY_STATEUTILITY, DISPLAY_STATEPARAM),
        (DISPLAY_STATEUTILITY, DISPLAY_PARAMONLY),
        (DISPLAY_STATEUTILITY, DISPLAY_CANONICAL),
    ]
    rows = []

    for model_a, model_b in pairs:
        a = summary_df[summary_df["model"] == model_a]
        b = summary_df[summary_df["model"] == model_b]
        merged = a.merge(
            b,
            on=["horizon", "field"],
            suffixes=("_a", "_b"),
        )
        for _, row in merged.iterrows():
            rows.append(
                {
                    "comparison": f"{model_a} - {model_b}",
                    "horizon": int(row["horizon"]),
                    "field": row["field"],
                    "rel_l2_diff_percent_point":
                        row["rel_l2_percent_a"]
                        - row["rel_l2_percent_b"],
                    "mse_diff":
                        row["mse_a"] - row["mse_b"],
                }
            )

    return pd.DataFrame(rows)


def reproduce_frozen_r3_2(curve_df, split_label, seed):
    path = resolve_path(
        "outputs/tables/r3_2_rollout/"
        f"r3_2_rollout_{split_label}_seed{seed}_curve_h1_h16.csv"
    )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Frozen R3-2 rollout curve required: {path}"
        )

    reference = pd.read_csv(path)
    ref = reference[
        reference["model"].isin(REFERENCE_MODELS)
    ][
        ["model", "horizon", "field", "rel_l2_percent", "mse"]
    ]
    cur = curve_df[
        curve_df["model"].isin(REFERENCE_MODELS)
    ][
        ["model", "horizon", "field", "rel_l2_percent", "mse"]
    ]

    merged = cur.merge(
        ref,
        on=["model", "horizon", "field"],
        suffixes=("_current", "_reference"),
        how="inner",
    )

    expected_rows = len(REFERENCE_MODELS) * 16 * 5
    if len(merged) != expected_rows:
        raise RuntimeError(
            f"R3-2 reproduction row count mismatch: "
            f"{len(merged)} != {expected_rows}"
        )

    max_rel = float(
        np.max(
            np.abs(
                merged["rel_l2_percent_current"]
                - merged["rel_l2_percent_reference"]
            )
        )
    )
    max_mse = float(
        np.max(
            np.abs(
                merged["mse_current"]
                - merged["mse_reference"]
            )
        )
    )

    if max_rel > 2.0e-9:
        raise RuntimeError(
            f"R3-2 reference Rel-L2 reproduction failed: {max_rel:.12e}"
        )
    if max_mse > 2.0e-12:
        raise RuntimeError(
            f"R3-2 reference MSE reproduction failed: {max_mse:.12e}"
        )

    return {
        "reference_path": path,
        "max_abs_rel_l2_diff": max_rel,
        "max_abs_mse_diff": max_mse,
    }


def synthetic_audit(
    *,
    parent,
    classifiers,
    feature_mean,
    feature_std,
    device,
):
    x = torch.zeros(
        2, 16, 256, 64,
        dtype=torch.float32,
        device=device,
    )
    params = torch.tensor(
        [[6.0, -0.30103], [8.0, 0.30103]],
        dtype=torch.float32,
        device=device,
    )

    parent_delta, info = parent(
        x,
        params=params,
        return_components=True,
    )

    if tuple(info["feature_names"]) != tuple(FEATURE_NAMES):
        raise RuntimeError("Synthetic parent feature order changed.")

    recon = info["base_delta_norm"].clone()
    for term in ACTIVE_TERMS:
        correction = info["term_corrections"][term]
        if correction.shape != (2, 256, 64):
            raise RuntimeError(
                f"{term}: synthetic correction shape changed: "
                f"{tuple(correction.shape)}"
            )
        out_idx = parent.TERM_TO_OUTPUT_INDEX[term]
        recon[:, out_idx, :, :] += correction

    max_abs = float(
        torch.max(torch.abs(recon - parent_delta)).detach().cpu()
    )
    if max_abs > 1.0e-7:
        raise RuntimeError(
            f"term_corrections do not reconstruct parent delta: {max_abs}"
        )

    inputs = {}
    for arm in ("ParamUtility", "StateUtility"):
        inputs[arm] = exact_classifier_input(
            info["full_conditioner_features"],
            arm,
            feature_mean,
            feature_std,
            device,
        )

    if not torch.equal(
        inputs["ParamUtility"][:, 8:],
        inputs["StateUtility"][:, 8:],
    ):
        raise RuntimeError("Utility arms parameter dims 8..9 differ.")
    if not torch.equal(
        inputs["ParamUtility"][:, :8],
        torch.zeros_like(inputs["ParamUtility"][:, :8]),
    ):
        raise RuntimeError("ParamUtility state dims are not exact zero.")

    for arm in ("ParamUtility", "StateUtility"):
        for term in ACTIVE_TERMS:
            q = torch.sigmoid(
                classifiers[(arm, term)](inputs[arm])
            ).reshape(-1)
            if q.shape != (2,) or not torch.isfinite(q).all():
                raise RuntimeError(f"{arm}/{term}: synthetic q invalid.")

    print("✅ R3-3B synthetic Utility interface audit PASS")
    print("✅ gated correction = q * term_correction")
    print("✅ term_correction already equals alpha * canonical signal")
    print("✅ ParamUtility masks normalized dims 0..7 to exact zero")
    print("✅ StateUtility sees all normalized 10D features")
    print("✅ q = sigmoid(frozen classifier logit); no threshold")
    print("✅ No special step-1 rule")


def main():
    args = parse_args()

    if args.seed != 42:
        raise ValueError("Formal R3-3B evaluation is locked to seed=42.")
    if args.batch_size != 4:
        raise ValueError("Formal R3-3B evaluation is locked to batch_size=4.")

    requested_horizons = sorted(
        {
            int(x)
            for x in args.horizons.split(",")
            if x.strip()
        }
    )
    if requested_horizons != [1, 4, 8, 16]:
        raise ValueError(
            "R3-3B formal reporting horizons are locked to 1,4,8,16."
        )
    max_horizon = 16

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    device = torch.device(args.device)

    resource = LOCKED_RESOURCES[args.split_label]

    split_path, split_sha = audit_file(
        resource["split"],
        resource["split_sha256"],
        f"{args.split_label} split",
    )
    stats_path, stats_sha = audit_file(
        resource["stats"],
        resource["stats_sha256"],
        f"{args.split_label} stats",
    )
    m6_path, m6_sha = audit_file(
        resource["m6"],
        resource["m6_sha256"],
        f"{args.split_label} M6",
    )
    r3_0_path, r3_0_sha = audit_file(
        R3_0_CONTRACT,
        EXPECTED_R3_0_CONTRACT_SHA256,
        "R3-0 contract",
    )
    _, r3_2_eval_sha = audit_file(
        R3_2_EVALUATOR_PATH,
        EXPECTED_R3_2_EVALUATOR_SHA256,
        "R3-2 rollout evaluator",
    )
    _, r3_2_model_sha = audit_file(
        R3_2_MODEL_PATH,
        EXPECTED_R3_2_MODEL_SHA256,
        "R3-2 state-adaptive model",
    )
    _, predictability_source_sha = audit_file(
        R3_3_PREDICTABILITY_SOURCE_PATH,
        EXPECTED_R3_3_PREDICTABILITY_SOURCE_SHA256,
        "R3-3A predictability source",
    )

    _, entries, r3_2_registry_path, _, r3_2_contract_sha = (
        load_and_audit_registry(args.split_label)
    )
    r3_2_registry_sha = sha256_file(r3_2_registry_path)

    parent_sha = entries["R3-2b"]["parent_r3_1b_sha256"]
    if entries["R3-2a"]["parent_r3_1b_sha256"] != parent_sha:
        raise RuntimeError("R3-2a/R3-2b parent mismatch.")

    r3_1b_path, r3_1b_sha = audit_file(
        resource["r3_1b"],
        parent_sha,
        f"{args.split_label} R3-1b parent",
    )

    paramonly_path = resolve_path(
        entries["R3-2a"]["checkpoint"]
    )
    stateparam_path = resolve_path(
        entries["R3-2b"]["checkpoint"]
    )
    paramonly_sha = sha256_file(paramonly_path)
    stateparam_sha = sha256_file(stateparam_path)

    _, alpha_map = load_locked_contract(r3_0_path)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    metadata = build_rbc_canonical_metadata(stats_path)

    m6, _ = build_m6(m6_path, device)

    class R3Args:
        pass

    r3_args = R3Args()
    r3_args.split_label = args.split_label
    r3_args.seed = args.seed

    canonical, _ = build_r3(
        representation_mode="canonical",
        checkpoint_path=r3_1b_path,
        metadata=metadata,
        alpha_map=alpha_map,
        contract_sha=r3_0_sha,
        args=r3_args,
        device=device,
    )

    paramonly, _ = build_r3_2(
        adaptation_mode="paramonly",
        checkpoint_path=paramonly_path,
        expected_entry=entries["R3-2a"],
        metadata=metadata,
        alpha_map=alpha_map,
        r3_2_contract_sha=r3_2_contract_sha,
        split_label=args.split_label,
        device=device,
    )

    stateparam, _ = build_r3_2(
        adaptation_mode="stateparam",
        checkpoint_path=stateparam_path,
        expected_entry=entries["R3-2b"],
        metadata=metadata,
        alpha_map=alpha_map,
        r3_2_contract_sha=r3_2_contract_sha,
        split_label=args.split_label,
        device=device,
    )

    assert_same_m6(m6, canonical, DISPLAY_CANONICAL)
    assert_same_m6(m6, paramonly, DISPLAY_PARAMONLY)
    assert_same_m6(m6, stateparam, DISPLAY_STATEPARAM)
    assert_same_parent_raw_alpha(
        canonical,
        paramonly,
        DISPLAY_PARAMONLY,
    )
    assert_same_parent_raw_alpha(
        canonical,
        stateparam,
        DISPLAY_STATEPARAM,
    )

    utility = load_and_audit_r3_3_formal_registry(
        args.split_label,
        device,
    )

    synthetic_audit(
        parent=stateparam,
        classifiers=utility["classifiers"],
        feature_mean=utility["feature_mean"],
        feature_std=utility["feature_std"],
        device=device,
    )

    print("=" * 110)
    print("R3-3B MATCHED CLOSED-LOOP UTILITY-GATING EVALUATOR")
    print("=" * 110)
    print("Split:", args.split_label)
    print("Models:", MODEL_ORDER)
    print(
        "PRIMARY:",
        f"{DISPLAY_STATEUTILITY} - {DISPLAY_PARAMUTILITY}",
    )
    print("Negative rollout/AUC difference = StateUtility better")
    print("Parent for both Utility arms: frozen R3-2b StateParam")
    print("Utility classifiers: frozen R3-3A formal checkpoints")
    print("No neural-operator retraining")
    print("No threshold / no temperature / no step-1 special case")
    print("Audit only:", args.audit_only)

    if args.audit_only:
        print()
        print("✅ R3-3B EVALUATOR AUDIT-ONLY PASS")
        print("✅ Frozen parent / classifiers / interfaces verified")
        print("✅ TEST rollout dataset was NOT constructed or accessed")
        return

    print()
    print("TEST access: YES (post-freeze evaluation)")

    split_config = load_json(split_path)
    dataset = RolloutDataset(
        split_config=split_config["test"],
        max_horizon=max_horizon,
        max_samples=args.max_samples,
    )
    if args.max_samples is None and len(dataset) != 2430:
        raise RuntimeError(
            f"Formal R3-3B TEST must contain 2430 windows; got {len(dataset)}"
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    normalizer = FieldWiseNormalizer(stats_path).to(device)

    reference_models = {
        DISPLAY_CANONICAL: canonical,
        DISPLAY_PARAMONLY: paramonly,
        DISPLAY_STATEPARAM: stateparam,
    }

    error_stats = {}
    reference_gate_stats = {}
    utility_stats = {}

    print("🔥 Starting free-autoregressive R3-3B TEST rollout...")

    with torch.no_grad():
        for batch_idx, (x0_phys, future_phys, param) in enumerate(
            loader,
            start=1,
        ):
            x0_phys = x0_phys.to(device, non_blocking=True)
            future_phys = future_phys.to(device, non_blocking=True)
            param = param.to(device, non_blocking=True)

            for model_name in REFERENCE_MODELS:
                rollout_r3_2_reference(
                    model_name=model_name,
                    model=reference_models[model_name],
                    x0_phys=x0_phys,
                    future_phys=future_phys,
                    param=param,
                    normalizer=normalizer,
                    max_horizon=max_horizon,
                    error_stats=error_stats,
                    gate_stats=reference_gate_stats,
                    alpha_map=alpha_map,
                )

            for model_name in UTILITY_MODELS:
                arm = ARM_BY_DISPLAY[model_name]
                rollout_utility_model(
                    model_name=model_name,
                    parent=stateparam,
                    classifiers=utility["classifiers"],
                    arm=arm,
                    x0_phys=x0_phys,
                    future_phys=future_phys,
                    param=param,
                    normalizer=normalizer,
                    max_horizon=max_horizon,
                    error_stats=error_stats,
                    utility_stats=utility_stats,
                    alpha_map=alpha_map,
                    feature_mean=utility["feature_mean"],
                    feature_std=utility["feature_std"],
                )

            if batch_idx % 25 == 0 or batch_idx == len(loader):
                print(f"  processed batch {batch_idx}/{len(loader)}")

    curve_df = pd.DataFrame(
        error_rows(error_stats, MODEL_ORDER, max_horizon)
    )
    summary_df = curve_df[
        curve_df["horizon"].isin(requested_horizons)
    ].copy()
    growth_df = make_global_growth_table(
        curve_df,
        MODEL_ORDER,
        max_horizon,
    )
    diff_df = make_difference_table(summary_df)

    reference_gates_df = pd.DataFrame(
        gate_rows(
            reference_gate_stats,
            REFERENCE_MODELS,
            max_horizon,
        )
    )
    utility_df = pd.DataFrame(
        utility_stat_rows(
            utility_stats,
            max_horizon,
        )
    )

    for df in (
        curve_df,
        summary_df,
        growth_df,
        diff_df,
        reference_gates_df,
        utility_df,
    ):
        df.insert(0, "seed", args.seed)
        df.insert(0, "split", args.split_label)

    if args.max_samples is None:
        reproduction = reproduce_frozen_r3_2(
            curve_df,
            args.split_label,
            args.seed,
        )
        print(
            "✅ Frozen R3-2 reference reproduction PASS | "
            f"Rel-L2 max={reproduction['max_abs_rel_l2_diff']:.3e} | "
            f"MSE max={reproduction['max_abs_mse_diff']:.3e}"
        )
    else:
        reproduction = {
            "skipped": True,
            "reason": "debug max_samples run",
        }

    output_dir = resolve_path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    prefix = (
        f"r3_3b_rollout_{args.split_label}_seed{args.seed}"
    )
    paths = {
        "summary": os.path.join(output_dir, f"{prefix}_summary.csv"),
        "curve": os.path.join(
            output_dir,
            f"{prefix}_curve_h1_h16.csv",
        ),
        "growth": os.path.join(
            output_dir,
            f"{prefix}_global_growth_auc.csv",
        ),
        "diff": os.path.join(
            output_dir,
            f"{prefix}_differences.csv",
        ),
        "reference_gates": os.path.join(
            output_dir,
            f"{prefix}_reference_alpha_curve.csv",
        ),
        "utility": os.path.join(
            output_dir,
            f"{prefix}_utility_q_curve.csv",
        ),
        "metadata": os.path.join(
            output_dir,
            f"{prefix}_metadata.json",
        ),
    }

    summary_df.to_csv(paths["summary"], index=False)
    curve_df.to_csv(paths["curve"], index=False)
    growth_df.to_csv(paths["growth"], index=False)
    diff_df.to_csv(paths["diff"], index=False)
    reference_gates_df.to_csv(paths["reference_gates"], index=False)
    utility_df.to_csv(paths["utility"], index=False)

    metadata_out = {
        "experiment": "R3-3B-matched-closed-loop-utility-gating",
        "formal_run": args.max_samples is None,
        "split_label": args.split_label,
        "seed": args.seed,
        "models": list(MODEL_ORDER),
        "primary_comparison":
            f"{DISPLAY_STATEUTILITY} - {DISPLAY_PARAMUTILITY}",
        "negative_primary_difference_means":
            "StateUtility better",
        "utility_parent":
            "frozen split-matched R3-2b StateParam",
        "utility_inference":
            "q=sigmoid(frozen classifier logit); "
            "gated correction=q*term_correction; "
            "term_correction=alpha*canonical_signal",
        "paramutility_semantics":
            "normalize with frozen TRAIN-only mean/std, "
            "then dims 0..7 exact zero",
        "stateutility_semantics":
            "normalize with frozen TRAIN-only mean/std; all 10D visible",
        "no_step1_special_case": True,
        "free_autoregressive": True,
        "horizons_reported": requested_horizons,
        "full_max_horizon": max_horizon,
        "formal_test_windows":
            None if args.max_samples is not None else len(dataset),
        "r3_0_contract_sha256": r3_0_sha,
        "r3_2_contract_sha256": r3_2_contract_sha,
        "r3_2_registry_sha256": r3_2_registry_sha,
        "r3_2_evaluator_sha256": r3_2_eval_sha,
        "r3_2_model_sha256": r3_2_model_sha,
        "r3_3_contract_sha256": utility["contract_sha"],
        "r3_3_formal_registry_sha256": utility["registry_sha"],
        "r3_3_predictability_source_sha256":
            predictability_source_sha,
        "split_sha256": split_sha,
        "stats_sha256": stats_sha,
        "m6_checkpoint_sha256": m6_sha,
        "r3_1b_checkpoint_sha256": r3_1b_sha,
        "r3_2a_checkpoint_sha256": paramonly_sha,
        "r3_2b_checkpoint_sha256": stateparam_sha,
        "utility_classifier_checkpoints": {
            f"{arm}/{term}": utility["payloads"][(arm, term)]
            for arm in ("ParamUtility", "StateUtility")
            for term in ACTIVE_TERMS
        },
        "reference_reproduction": reproduction,
        "diagnostics_predeclared_before_test": {
            "q_mean_std_min_max": True,
            "parent_alpha_mean_std": True,
            "qalpha_mean_abs_capacity_ratio": True,
            "qalpha_sat99_fraction": True,
        },
        "test_accessed": True,
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

    global_wide = (
        summary_df[summary_df["field"] == "global"]
        .pivot_table(
            index="model",
            columns="horizon",
            values="rel_l2_percent",
            aggfunc="first",
        )
        .reindex(MODEL_ORDER)
    )

    print()
    print("================ GLOBAL REL-L2 (%) ================")
    print(global_wide.to_string())

    primary = diff_df[
        (diff_df["comparison"] ==
         f"{DISPLAY_STATEUTILITY} - {DISPLAY_PARAMUTILITY}")
        & (diff_df["field"] == "global")
    ][["horizon", "rel_l2_diff_percent_point"]]

    print()
    print("========== PRIMARY StateUtility - ParamUtility ==========")
    print("Negative = StateUtility better.")
    print(primary.to_string(index=False))

    print()
    print("================ GLOBAL GROWTH / AUC ================")
    print(
        growth_df.drop(
            columns=["split", "seed"],
            errors="ignore",
        ).to_string(index=False)
    )

    print()
    print("Saved:")
    for key, path in paths.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
