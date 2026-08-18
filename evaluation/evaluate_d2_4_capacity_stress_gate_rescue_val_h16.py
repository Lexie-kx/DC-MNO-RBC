import argparse
import importlib.util
import json
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from constants import B_IDX


# ============================================================
# Reuse the CLOSED D2-3A evaluator for audited model builders,
# checkpoint provenance, dataset semantics, and Rel-L2 metrics.
# ============================================================

D23A_EVAL_PATH = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "evaluate_d2_3a_capacity_control_val_h16.py",
)


def load_python_module(name, path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ctl = load_python_module("d2_3a_closed_eval", D23A_EVAL_PATH)
base = ctl.base


# ============================================================
# D2-4 contract
# Capacity-Stress + Frozen Legacy10 Utility-Gate Rescue
# ============================================================

HORIZONS_LOCKED = [1, 4, 8, 16]
PRIMARY_HORIZON = 16

LEGACY_NAMES = [
    "b_mean",
    "b_rms",
    "ux_mean",
    "ux_rms",
    "uy_mean",
    "uy_rms",
    "log1p_path_a_rms",
    "log1p_path_b_rms",
    "logRa",
    "logPr",
]

# D2-2a stores Legacy10 features with the historical `legacy__` CSV prefix.
# Runtime inference still uses the same 10 features in exactly this order.
LEGACY_COLUMNS = [f"legacy__{x}" for x in LEGACY_NAMES]

# Reproduction targets from the already-closed D2-2d formal H16 runs.
# This is NOT a tuning target. It is a guardrail that the same TRAIN-only
# Legacy10 classifier is being reconstructed before applying it to D2-3A.
EXPECTED_LEGACY10_AUC = {
    ("unseen_pr", 42): 0.891766,
    ("unseen_pr", 123): 0.782281,
    ("unseen_pr", 2026): 0.845429,
    ("unseen_ra", 42): 0.842249,
    ("unseen_ra", 123): 0.836119,
    ("unseen_ra", 2026): 0.861337,
}

PREDECLARED_DECISION_RULE = {
    "primary_horizon": 16,
    "primary_metrics": ["buoyancy_rel_l2_percent", "global_rel_l2_percent"],
    "rescue_target": "D2-4 - D2-3A",
    "solution_target": "D2-4 - M10-2",
    "strong_rescue": (
        "Across both splits, all three seeds, and both h16 primary metrics, "
        "D2-4 - D2-3A < 0 AND D2-4 - M10-2 < 0 (12/12 rescue and 12/12 solution)."
    ),
    "rescue_pass": (
        "For each split and each h16 primary metric, the three-seed mean "
        "D2-4 - D2-3A is < 0 with at least 2/3 seeds improved; additionally, "
        "the three-seed mean D2-4 - M10-2 is < 0 for both primary metrics "
        "in both splits."
    ),
    "partial": (
        "Gate consistently improves D2-3A stress but does not restore both "
        "primary split means below M10-2."
    ),
    "fail": (
        "At least one split/primary metric has non-improving three-seed mean "
        "D2-4 - D2-3A >= 0."
    ),
    "locked_before_d2_4_results": True,
    "no_val_tuning": True,
    "test_accessed": False,
}


# ============================================================
# CLI
# ============================================================


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "D2-4 Capacity-Stress + Frozen Legacy10 Utility-Gate Rescue, "
            "full VAL H16. D2-3A is the high-capacity stress model; the "
            "already-locked TRAIN-only Legacy10 gate is reconstructed and "
            "applied without refit/tuning to the D2-3A closed-loop states."
        )
    )
    p.add_argument("--split", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--split_label", required=True, choices=["unseen_pr", "unseen_ra"])
    p.add_argument("--m10_checkpoint", required=True)
    p.add_argument("--d2_1_checkpoint", required=True)
    p.add_argument("--d2_3a_checkpoint", required=True)
    p.add_argument("--train_utility_csv", required=True)
    p.add_argument("--val_utility_csv", required=True)
    p.add_argument("--seed", type=int, required=True, choices=[42, 123, 2026])
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_horizon", type=int, default=16)
    p.add_argument("--horizons", type=str, default="1,4,8,16")
    p.add_argument("--classifier_epochs", type=int, default=100)
    p.add_argument("--max_batches", type=int, default=None)
    p.add_argument("--output_prefix", required=True)
    return p.parse_args()


# ============================================================
# Exact Legacy10 classifier contract used in M10/D2-2.
# ============================================================


class UtilityClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def roc_auc_score_simple(y_true, score):
    y_true = np.asarray(y_true, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(score).rank(method="average").to_numpy()
    rank_sum_pos = float(ranks[pos].sum())
    return float(
        (rank_sum_pos - n_pos * (n_pos + 1) / 2.0)
        / (n_pos * n_neg)
    )


def fit_classifier(train_df, val_df, columns, *, seed, epochs, device):
    set_seed(seed)

    train_x = train_df[columns].to_numpy(dtype=np.float32)
    val_x = val_df[columns].to_numpy(dtype=np.float32)
    train_y = train_df["path_b_helps"].to_numpy(dtype=np.float32)
    val_y = val_df["path_b_helps"].to_numpy(dtype=np.int64)

    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1.0e-6] = 1.0

    train_z = (train_x - mean) / std
    val_z = (val_x - mean) / std

    x_train = torch.tensor(train_z, dtype=torch.float32, device=device)
    x_val = torch.tensor(val_z, dtype=torch.float32, device=device)
    y_train = torch.tensor(train_y, dtype=torch.float32, device=device)

    model = UtilityClassifier(train_z.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    criterion = nn.BCEWithLogitsLoss()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    batch_size = min(512, len(train_z))

    for _ in range(epochs):
        permutation = torch.randperm(len(train_z), generator=generator)
        model.train()
        for start in range(0, len(train_z), batch_size):
            idx = permutation[start:start + batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_train[idx])
            loss = criterion(logits, y_train[idx])
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_prob = torch.sigmoid(model(x_val)).cpu().numpy()

    auc = roc_auc_score_simple(val_y, val_prob)
    accuracy = float(((val_prob >= 0.5).astype(np.int64) == val_y).mean())

    return {
        "model": model,
        "mean": torch.tensor(mean, dtype=torch.float32, device=device),
        "std": torch.tensor(std, dtype=torch.float32, device=device),
        "auc": float(auc),
        "accuracy": accuracy,
        "num_features": len(columns),
    }


@torch.no_grad()
def predict_probability(predictor, features):
    z = (features - predictor["mean"]) / predictor["std"]
    return torch.sigmoid(predictor["model"](z))


def build_legacy_features(param, comp):
    # Exact Legacy10 inference-visible feature contract:
    # 8 model state-summary features + [logRa, logPr].
    return torch.cat([comp["state_summary"], param], dim=-1)


def make_gated_delta(comp, trust):
    out = comp["base_delta_norm"].clone()
    out[:, B_IDX] = (
        out[:, B_IDX]
        + trust[:, None, None]
        * comp["alpha_b"][:, None, None]
        * comp["path_b_norm_safe"]
    )
    return out


def spatial_rms_2d(x):
    return torch.sqrt(torch.mean(x * x, dim=(-2, -1)) + 1.0e-12)


# ============================================================
# Gate / dose diagnostics
# ============================================================


def new_gate_bucket():
    return {
        "n": 0,
        "q_sum": 0.0,
        "q_sq_sum": 0.0,
        "q_min": float("inf"),
        "q_max": float("-inf"),
        "q_ge_0p5": 0,
        "ungated_inj_sum": 0.0,
        "gated_inj_sum": 0.0,
    }


def update_gate_bucket(bucket, q, comp):
    qd = q.detach().double()
    ungated = spatial_rms_2d(
        comp["alpha_b"][:, None, None] * comp["path_b_norm_safe"]
    ).detach().double()
    gated = qd * ungated

    bucket["n"] += int(qd.numel())
    bucket["q_sum"] += float(qd.sum().item())
    bucket["q_sq_sum"] += float((qd * qd).sum().item())
    bucket["q_min"] = min(bucket["q_min"], float(qd.min().item()))
    bucket["q_max"] = max(bucket["q_max"], float(qd.max().item()))
    bucket["q_ge_0p5"] += int((qd >= 0.5).sum().item())
    bucket["ungated_inj_sum"] += float(ungated.sum().item())
    bucket["gated_inj_sum"] += float(gated.sum().item())


def finalize_gate_stats(stats, horizons):
    rows = []
    for h in horizons:
        b = stats[h]
        n = b["n"]
        q_mean = b["q_sum"] / n
        q_var = max(0.0, b["q_sq_sum"] / n - q_mean * q_mean)
        ungated = b["ungated_inj_sum"] / n
        gated = b["gated_inj_sum"] / n
        rows.append({
            "horizon": h,
            "samples": n,
            "q_mean": q_mean,
            "q_std": q_var ** 0.5,
            "q_min": b["q_min"],
            "q_max": b["q_max"],
            "q_ge_0p5_fraction": b["q_ge_0p5"] / n,
            "ungated_injection_rms_mean": ungated,
            "gated_injection_rms_mean": gated,
            "gated_to_ungated_rms_ratio": gated / max(ungated, 1.0e-30),
        })
    return pd.DataFrame(rows)


# ============================================================
# Difference helpers
# ============================================================


def get_rel(summary, model, horizon, field):
    # base.make_error_summary returns LONG format:
    # one row per (model, horizon, field), with the metric in rel_l2_percent.
    row = summary[
        (summary["model"] == model)
        & (summary["horizon"] == horizon)
        & (summary["field"] == field)
    ]
    if len(row) != 1:
        raise RuntimeError(
            f"Expected one summary row for {model} h={horizon} field={field}, got {len(row)}"
        )
    return float(row.iloc[0]["rel_l2_percent"])


def make_h16_primary(summary):
    pairs = [
        ("D2-4-CapacityStress-Gate", "D2-3A-CapacityMatch", "rescue_vs_D2-3A"),
        ("D2-4-CapacityStress-Gate", "M10-2-BOnly", "solution_vs_M10-2"),
        ("D2-4-CapacityStress-Gate", "D2-1-DC-BOnly", "context_vs_D2-1"),
    ]
    fields = [
        ("buoyancy", "buoyancy"),
        ("global", "global"),
    ]
    rows = []
    for a, b, label in pairs:
        for field_label, col in fields:
            va = get_rel(summary, a, 16, col)
            vb = get_rel(summary, b, 16, col)
            rows.append({
                "comparison": label,
                "model_a": a,
                "model_b": b,
                "horizon": 16,
                "field": field_label,
                "rel_l2_a": va,
                "rel_l2_b": vb,
                "rel_l2_diff_pp": va - vb,
                "negative_means_model_a_better": (va - vb) < 0.0,
            })
    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================


def main():
    args = parse_args()

    if args.max_horizon != 16:
        raise ValueError("D2-4 formal evaluator is locked to max_horizon=16.")
    horizons = sorted({int(x.strip()) for x in args.horizons.split(",") if x.strip()})
    if horizons != HORIZONS_LOCKED:
        raise ValueError("D2-4 is locked to horizons=1,4,8,16.")
    if args.classifier_epochs != 100:
        raise ValueError("D2-4 classifier_epochs is locked to 100; no refit/tuning allowed.")

    for path in [
        args.split,
        args.stats,
        args.m10_checkpoint,
        args.d2_1_checkpoint,
        args.d2_3a_checkpoint,
        args.train_utility_csv,
        args.val_utility_csv,
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 118)
    print("D2-4 CAPACITY-STRESS + FROZEN LEGACY10 UTILITY-GATE RESCUE | FULL VAL H16")
    print("=" * 118)
    print("Stage: SOLUTION / STRESS-RESCUE CONTROL")
    print("Training of neural operator: OFF")
    print("Utility Gate: Legacy10 PRIMARY, TRAIN-only reconstruction, then FROZEN")
    print("Gate training rows: CLOSED D2-2a rows from D2-1; NOT regenerated from D2-3A")
    print("step1 q=1 | step2-16 frozen soft P(Path B helps)")
    print("Threshold tuning: NONE")
    print("D2-3A alpha/capacity: FROZEN high-capacity stress condition")
    print("Closed-loop split: VAL ONLY")
    print("TEST access: FORBIDDEN")
    print("Primary rescue: D2-4 vs D2-3A at h16 buoyancy + global")
    print("Primary solution: D2-4 vs M10-2 at h16 buoyancy + global")
    print("Split:", args.split_label)
    print("Seed:", args.seed)
    print("Device:", device)
    if args.max_batches is not None:
        print("⚠️ DEBUG ONLY: max_batches =", args.max_batches)
        print("⚠️ This run MUST NOT be classified as a formal D2-4 result.")

    # --------------------------------------------------------
    # Checkpoint provenance + capacity stress audit
    # --------------------------------------------------------
    split_sha = base.sha256_file(args.split)
    stats_sha = base.sha256_file(args.stats)

    m10_payload = base.load_payload(args.m10_checkpoint, device)
    d21_payload = base.load_payload(args.d2_1_checkpoint, device)
    d23a_payload = base.load_payload(args.d2_3a_checkpoint, device)

    for name, payload in [
        ("M10-2", m10_payload),
        ("D2-1", d21_payload),
        ("D2-3A", d23a_payload),
    ]:
        base.verify_checkpoint_provenance(
            payload,
            checkpoint_name=name,
            expected_seed=args.seed,
            split_sha=split_sha,
            stats_sha=stats_sha,
        )

    protocol = ctl.audit_checkpoint_protocol(
        split_label=args.split_label,
        m10_payload=m10_payload,
        d2_1_payload=d21_payload,
        d2_3a_payload=d23a_payload,
    )

    print()
    print("========== CAPACITY STRESS AUDIT ==========")
    print("M10 alpha_max:", protocol["m10_alpha_max"])
    print("D2-1 alpha_max:", protocol["d2_1_alpha_max"])
    print("D2-3A alpha_max:", protocol["d2_3a_alpha_max"])
    print("M10 max capacity:", protocol["m10_max_capacity"])
    print("D2-3A max capacity:", protocol["d2_3a_max_capacity"])
    print("capacity_match_abs_error:", abs(protocol["d2_3a_max_capacity"] - protocol["m10_max_capacity"]))

    field_mean, field_std = base.build_field_stats(args.stats)
    m10 = base.build_m10_model(m10_payload, field_mean, field_std, device)
    d21 = base.build_d2_model(d21_payload, field_mean, field_std, device)
    d23a = base.build_d2_model(d23a_payload, field_mean, field_std, device)

    diffs = {
        "M10_vs_D2-1": base.compare_embedded_m6(m10, d21),
        "M10_vs_D2-3A": base.compare_embedded_m6(m10, d23a),
        "D2-1_vs_D2-3A": base.compare_embedded_m6(d21, d23a),
    }
    print()
    print("========== FAIRNESS AUDIT ==========")
    for k, v in diffs.items():
        print(f"embedded_M6_max_abs_diff {k}: {v:.12e}")
    if max(diffs.values()) > 1.0e-7:
        raise RuntimeError("Embedded M6 mismatch.")

    # --------------------------------------------------------
    # Reconstruct the CLOSED Legacy10 classifier from TRAIN only.
    # VAL labels are used only for reproduction sanity, never for tuning.
    # --------------------------------------------------------
    train_df = pd.read_csv(args.train_utility_csv)
    val_df = pd.read_csv(args.val_utility_csv)

    missing = [c for c in LEGACY_COLUMNS + ["step", "path_b_helps"] if c not in train_df.columns]
    if missing:
        raise RuntimeError(f"TRAIN utility CSV missing columns: {missing}")
    missing = [c for c in LEGACY_COLUMNS + ["step", "path_b_helps"] if c not in val_df.columns]
    if missing:
        raise RuntimeError(f"VAL utility CSV missing columns: {missing}")

    train_probe = train_df[train_df["step"] >= 2].copy()
    val_probe = val_df[val_df["step"] >= 2].copy()

    predictor = fit_classifier(
        train_probe,
        val_probe,
        LEGACY_COLUMNS,
        seed=args.seed,
        epochs=args.classifier_epochs,
        device=device,
    )

    expected_auc = EXPECTED_LEGACY10_AUC[(args.split_label, args.seed)]
    auc_err = abs(predictor["auc"] - expected_auc)

    print()
    print("========== LEGACY10 REPRODUCTION ==========")
    print("TRAIN_ROWS(step>=2):", len(train_probe))
    print("VAL_SANITY_ROWS(step>=2):", len(val_probe))
    print("Legacy10 AUC:", f"{predictor['auc']:.6f}")
    print("Expected closed D2-2d AUC:", f"{expected_auc:.6f}")
    print("AUC abs error:", f"{auc_err:.6e}")
    print("Legacy10 accuracy:", f"{predictor['accuracy']:.6f}")
    if auc_err > 2.0e-3:
        raise RuntimeError("Legacy10 reproduction failed; D2-4 must not proceed.")
    print("✅ Legacy10 reproduction PASS")

    # --------------------------------------------------------
    # VAL-only H16 loader, identical dataset semantics.
    # --------------------------------------------------------
    with open(args.split, "r", encoding="utf-8") as f:
        split = json.load(f)
    val_base = base.RBCDataset(
        split_config=split["val"],
        normalize=True,
        stats_path=args.stats,
        return_sequence=True,
        target_steps=16,
        return_params=True,
    )
    val_dataset = base.M10MultiStepParamDataset(val_base, context_length=4)
    loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    print()
    print("========== VAL DATA ==========")
    print("VAL H16 windows:", len(val_dataset))
    print("Batches:", len(loader))
    if args.max_batches is None and len(val_dataset) != 486:
        raise RuntimeError(f"Formal D2-4 expects 486 VAL windows, got {len(val_dataset)}")

    field_mean_t = torch.tensor(field_mean, device=device, dtype=torch.float32).view(1, 4, 1, 1)
    field_std_t = torch.tensor(field_std, device=device, dtype=torch.float32).view(1, 4, 1, 1)

    model_names = [
        "M10-2-BOnly",
        "D2-1-DC-BOnly",
        "D2-3A-CapacityMatch",
        "D2-4-CapacityStress-Gate",
    ]
    error_stats = {
        (name, h): base.new_error_bucket()
        for name in model_names
        for h in horizons
    }
    gate_stats = {h: new_gate_bucket() for h in horizons}

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break

            context_norm, y_seq_norm, param = batch
            context_norm = context_norm.to(device)
            y_seq_norm = y_seq_norm.to(device)
            param = param.to(device)

            contexts = {name: context_norm.clone() for name in model_names}

            for step in range(1, 17):
                true_next_norm = y_seq_norm[:, step - 1]
                true_next_phys = true_next_norm * field_std_t + field_mean_t

                # M10, D2-1, D2-3A ungated controls.
                for name, model in [
                    ("M10-2-BOnly", m10),
                    ("D2-1-DC-BOnly", d21),
                    ("D2-3A-CapacityMatch", d23a),
                ]:
                    context = contexts[name]
                    bsz, clen, ch, hh, ww = context.shape
                    model_input = context.reshape(bsz, clen * ch, hh, ww)
                    pred_delta, _comp = model(
                        model_input,
                        params=param,
                        return_components=True,
                    )
                    current = context[:, -1]
                    pred_next = current + pred_delta
                    if step in horizons:
                        pred_phys = pred_next * field_std_t + field_mean_t
                        base.update_error_bucket(error_stats[(name, step)], pred_phys, true_next_phys)
                    contexts[name] = torch.cat([context[:, 1:], pred_next.unsqueeze(1)], dim=1)

                # D2-4: SAME frozen D2-3A model, but gated Path-B injection.
                name = "D2-4-CapacityStress-Gate"
                context = contexts[name]
                bsz, clen, ch, hh, ww = context.shape
                model_input = context.reshape(bsz, clen * ch, hh, ww)
                _ungated_delta, comp = d23a(
                    model_input,
                    params=param,
                    return_components=True,
                )

                if step == 1:
                    q = torch.ones(bsz, dtype=context.dtype, device=device)
                else:
                    features = build_legacy_features(param, comp)
                    q = predict_probability(predictor, features)

                gated_delta = make_gated_delta(comp, q)
                current = context[:, -1]
                pred_next = current + gated_delta

                if step in horizons:
                    pred_phys = pred_next * field_std_t + field_mean_t
                    base.update_error_bucket(error_stats[(name, step)], pred_phys, true_next_phys)
                    update_gate_bucket(gate_stats[step], q, comp)

                contexts[name] = torch.cat([context[:, 1:], pred_next.unsqueeze(1)], dim=1)

            if (batch_idx + 1) % 20 == 0:
                print("processed batch", batch_idx + 1, "/", len(loader))

    summary = base.make_error_summary(error_stats, model_names, horizons)
    gate_df = finalize_gate_stats(gate_stats, horizons)
    primary = make_h16_primary(summary)

    print()
    print("=" * 118)
    print("ROLLOUT Rel-L2 (%)")
    print("=" * 118)
    print(summary.to_string(index=False))

    print()
    print("=" * 118)
    print("H16 PRIMARY STRESS-RESCUE")
    print("NEGATIVE diff = model_a better")
    print("=" * 118)
    print(primary.to_string(index=False, float_format=lambda x: f"{x:+.6f}"))

    print()
    print("=" * 118)
    print("UTILITY-GATE / DOSE SUPPRESSION DIAGNOSTICS")
    print("=" * 118)
    print(gate_df.to_string(index=False))

    output_dir = os.path.dirname(args.output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    summary_path = args.output_prefix + "_summary.csv"
    gate_path = args.output_prefix + "_gate_diagnostics.csv"
    primary_path = args.output_prefix + "_h16_primary.csv"
    metadata_path = args.output_prefix + "_metadata.json"

    summary.to_csv(summary_path, index=False)
    gate_df.to_csv(gate_path, index=False)
    primary.to_csv(primary_path, index=False)

    metadata = {
        "experiment": "D2-4 Capacity-Stress + Frozen Legacy10 Utility-Gate Rescue",
        "stage": "solution_stress_rescue_control",
        "neural_operator_training": False,
        "utility_gate": "Legacy10",
        "utility_fit_source": "closed D2-2a TRAIN rows from D2-1",
        "utility_refit_on_d2_3a": False,
        "utility_val_tuning": False,
        "step1_q": 1.0,
        "step2_16": "frozen soft probability",
        "test_accessed": False,
        "split_label": args.split_label,
        "seed": args.seed,
        "val_windows": len(val_dataset),
        "formal_full_val": (args.max_batches is None and len(val_dataset) == 486),
        "max_batches": args.max_batches,
        "split_sha256": split_sha,
        "stats_sha256": stats_sha,
        "m10_checkpoint": args.m10_checkpoint,
        "m10_checkpoint_sha256": base.sha256_file(args.m10_checkpoint),
        "d2_1_checkpoint": args.d2_1_checkpoint,
        "d2_1_checkpoint_sha256": base.sha256_file(args.d2_1_checkpoint),
        "d2_3a_checkpoint": args.d2_3a_checkpoint,
        "d2_3a_checkpoint_sha256": base.sha256_file(args.d2_3a_checkpoint),
        "train_utility_csv": args.train_utility_csv,
        "train_utility_csv_sha256": base.sha256_file(args.train_utility_csv),
        "val_utility_csv": args.val_utility_csv,
        "val_utility_csv_sha256": base.sha256_file(args.val_utility_csv),
        "legacy10_auc": predictor["auc"],
        "legacy10_expected_closed_d2_2d_auc": expected_auc,
        "legacy10_auc_abs_error": auc_err,
        "capacity_audit": protocol,
        "predeclared_decision_rule": PREDECLARED_DECISION_RULE,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print()
    print("Summary:", summary_path)
    print("Gate diagnostics:", gate_path)
    print("H16 primary:", primary_path)
    print("Metadata:", metadata_path)
    print()
    print("✅ D2-4 capacity-stress gate-rescue VAL H16 evaluation finished.")


if __name__ == "__main__":
    main()
