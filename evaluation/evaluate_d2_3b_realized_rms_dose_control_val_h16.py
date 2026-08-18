import argparse
import importlib.util
import json
import os
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CTL_PATH = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "evaluate_d2_3a_capacity_control_val_h16.py",
)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ctl = load_module("d2_3a_closed_control", CTL_PATH)
base = ctl.base

PRIMARY_HORIZON = 16
PRIMARY_FIELDS = ("buoyancy", "global")

PREDECLARED_RULE = {
    "experiment": "D2-3B-PerState-RealizedRMS-Dose-Matched-Control",
    "role": "causal diagnostic only; NOT a final/deployable model",
    "dose_match_definition": (
        "At every rollout step and for every sample, compute M10-2 and D2-3A "
        "B-only residuals on the SAME D2-3B current state. Multiply the D2-3A "
        "canonical residual by a scalar so its spatial RMS exactly matches the "
        "M10-2 residual RMS on that same state. No ground truth is used to set the scalar."
    ),
    "primary": "h16 buoyancy + global Rel-L2",
    "strong_support": "12/12 seed-level primary D2-3B - M10-2 differences < 0",
    "support": (
        "For each split, both three-seed mean primary differences < 0 and each "
        "split/metric has at least 2/3 negative seed differences."
    ),
    "mixed": "All other cross-split outcomes.",
    "locked_before_results": True,
    "test_accessed": False,
}


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "D2-3B per-state realized-RMS-dose-matched causal control on full VAL H16. "
            "No training, no Utility Gate, TEST forbidden."
        )
    )
    p.add_argument("--split", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--split_label", required=True, choices=["unseen_pr", "unseen_ra"])
    p.add_argument("--m10_checkpoint", required=True)
    p.add_argument("--d2_1_checkpoint", required=True)
    p.add_argument("--d2_3a_checkpoint", required=True)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_horizon", type=int, default=16)
    p.add_argument("--horizons", default="1,4,8,16")
    p.add_argument("--max_batches", type=int, default=None)
    p.add_argument("--output_prefix", required=True)
    return p.parse_args()


def residual_rms(residual):
    # residual: [B, 4, X, Y], B-only residual expected.
    r = residual[:, base.B_IDX]
    return torch.sqrt(torch.mean(r * r, dim=(-2, -1)))


def new_match_bucket():
    return {
        "samples": 0,
        "scale_sum": 0.0,
        "scale_abs_dev_from_one_sum": 0.0,
        "scale_min": float("inf"),
        "scale_max": 0.0,
        "target_rms_sum": 0.0,
        "matched_rms_sum": 0.0,
        "match_abs_error_sum": 0.0,
        "match_abs_error_max": 0.0,
    }


def update_match_bucket(bucket, scale, target_rms, matched_rms):
    err = (matched_rms - target_rms).abs()
    n = int(scale.numel())
    bucket["samples"] += n
    bucket["scale_sum"] += float(scale.sum().cpu())
    bucket["scale_abs_dev_from_one_sum"] += float((scale - 1.0).abs().sum().cpu())
    bucket["scale_min"] = min(bucket["scale_min"], float(scale.min().cpu()))
    bucket["scale_max"] = max(bucket["scale_max"], float(scale.max().cpu()))
    bucket["target_rms_sum"] += float(target_rms.sum().cpu())
    bucket["matched_rms_sum"] += float(matched_rms.sum().cpu())
    bucket["match_abs_error_sum"] += float(err.sum().cpu())
    bucket["match_abs_error_max"] = max(bucket["match_abs_error_max"], float(err.max().cpu()))


def match_summary(match_stats, horizons):
    rows = []
    for h in horizons:
        b = match_stats[h]
        n = b["samples"]
        rows.append({
            "horizon": h,
            "samples": n,
            "dose_scale_mean": b["scale_sum"] / n,
            "dose_scale_abs_dev_from_one_mean": b["scale_abs_dev_from_one_sum"] / n,
            "dose_scale_min": b["scale_min"],
            "dose_scale_max": b["scale_max"],
            "m10_same_state_target_injection_rms_mean": b["target_rms_sum"] / n,
            "d2_3b_matched_injection_rms_mean": b["matched_rms_sum"] / n,
            "dose_match_abs_error_mean": b["match_abs_error_sum"] / n,
            "dose_match_abs_error_max": b["match_abs_error_max"],
        })
    return pd.DataFrame(rows)


def make_differences(summary):
    pairs = [
        ("D2-1-DC-BOnly", "M10-2-BOnly"),
        ("D2-3A-CapacityMatch", "M10-2-BOnly"),
        ("D2-3B-RMSDoseMatch", "M10-2-BOnly"),
        ("D2-3B-RMSDoseMatch", "D2-3A-CapacityMatch"),
    ]
    rows = []
    for a_name, b_name in pairs:
        a = summary[summary.model == a_name]
        b = summary[summary.model == b_name]
        m = a.merge(b, on=["horizon", "field"], suffixes=("_a", "_b"))
        for _, r in m.iterrows():
            rows.append({
                "comparison": f"{a_name} - {b_name}",
                "horizon": int(r.horizon),
                "field": r.field,
                "rel_l2_a": float(r.rel_l2_percent_a),
                "rel_l2_b": float(r.rel_l2_percent_b),
                "rel_l2_diff_pp": float(r.rel_l2_percent_a - r.rel_l2_percent_b),
            })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    if args.max_horizon != 16:
        raise ValueError("D2-3B is locked to max_horizon=16")
    horizons = sorted({int(x) for x in args.horizons.split(",") if x.strip()})
    if horizons != [1, 4, 8, 16]:
        raise ValueError("D2-3B is locked to horizons=1,4,8,16")

    for path in [args.split, args.stats, args.m10_checkpoint, args.d2_1_checkpoint, args.d2_3a_checkpoint]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    base.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 118)
    print("D2-3B PER-STATE REALIZED-RMS-DOSE-MATCHED CANONICAL B-ONLY CONTROL | FULL VAL H16")
    print("=" * 118)
    print("Stage: CONTROL / CAUSAL ATTRIBUTION")
    print("Training: OFF")
    print("Utility Gate: OFF")
    print("Closed-loop split: VAL ONLY")
    print("TEST access: FORBIDDEN")
    print("Important: D2-3B is a diagnostic control, NOT a deployable/final model.")
    print("Dose match: per-sample, per-step residual spatial RMS; M10 target computed on SAME D2-3B state.")
    print("No GT is used to choose the dose scale.")
    print("Split:", args.split_label)
    print("Seed:", args.seed)
    print("Device:", device)
    if args.max_batches is not None:
        print("⚠️ DEBUG ONLY: max_batches =", args.max_batches)
        print("⚠️ This run MUST NOT be classified as a formal D2-3B result.")

    split_sha = base.sha256_file(args.split)
    stats_sha = base.sha256_file(args.stats)
    m10_payload = base.load_payload(args.m10_checkpoint, device)
    d21_payload = base.load_payload(args.d2_1_checkpoint, device)
    d23a_payload = base.load_payload(args.d2_3a_checkpoint, device)

    for name, payload in [("M10-2", m10_payload), ("D2-1", d21_payload), ("D2-3A", d23a_payload)]:
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

    field_mean, field_std = base.build_field_stats(args.stats)
    m10 = base.build_m10_model(m10_payload, field_mean, field_std, device)
    d21 = base.build_d2_model(d21_payload, field_mean, field_std, device)
    d23a = base.build_d2_model(d23a_payload, field_mean, field_std, device)

    for label, a, b in [
        ("M10_vs_D2-1", m10, d21),
        ("M10_vs_D2-3A", m10, d23a),
        ("D2-1_vs_D2-3A", d21, d23a),
    ]:
        diff = base.compare_embedded_m6(a, b)
        print(f"embedded_M6_max_abs_diff {label}: {diff:.12e}")
        if diff > 1e-7:
            raise RuntimeError("Embedded M6 mismatch")

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
    loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    print("VAL H16 windows:", len(val_dataset))
    print("Batches:", len(loader))
    if args.max_batches is None and len(val_dataset) != 486:
        raise RuntimeError(f"Formal run expected 486 VAL H16 windows, got {len(val_dataset)}")

    model_names = ["M10-2-BOnly", "D2-1-DC-BOnly", "D2-3A-CapacityMatch", "D2-3B-RMSDoseMatch"]
    error_stats = {(name, h): base.new_error_bucket() for name in model_names for h in horizons}
    match_stats = {h: new_match_bucket() for h in horizons}

    field_mean_t = torch.tensor(field_mean, device=device, dtype=torch.float32).view(1, 4, 1, 1)
    field_std_t = torch.tensor(field_std, device=device, dtype=torch.float32).view(1, 4, 1, 1)

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

                # Three ordinary closed-loop branches.
                for name, model in [
                    ("M10-2-BOnly", m10),
                    ("D2-1-DC-BOnly", d21),
                    ("D2-3A-CapacityMatch", d23a),
                ]:
                    context = contexts[name]
                    bsz, clen, channels, hh, ww = context.shape
                    model_input = context.reshape(bsz, clen * channels, hh, ww)
                    pred_delta, _ = model(model_input, params=param, return_components=True)
                    current = context[:, -1]
                    pred_next = current + pred_delta
                    if step in horizons:
                        pred_phys = pred_next * field_std_t + field_mean_t
                        base.update_error_bucket(error_stats[(name, step)], pred_phys, true_next_phys)
                    contexts[name] = torch.cat([context[:, 1:], pred_next.unsqueeze(1)], dim=1)

                # D2-3B: on its OWN current state, compute both canonical D2-3A
                # and legacy M10 residuals, then match ONLY their realized RMS dose.
                name = "D2-3B-RMSDoseMatch"
                context = contexts[name]
                bsz, clen, channels, hh, ww = context.shape
                model_input = context.reshape(bsz, clen * channels, hh, ww)
                _, comp_dc = d23a(model_input, params=param, return_components=True)
                _, comp_legacy_same_state = m10(model_input, params=param, return_components=True)

                dc_resid = comp_dc["physics_residual_norm"]
                target_resid = comp_legacy_same_state["physics_residual_norm"]
                dc_rms = residual_rms(dc_resid)
                target_rms = residual_rms(target_resid)
                eps = torch.tensor(1e-14, device=device, dtype=dc_rms.dtype)
                scale = target_rms / torch.clamp(dc_rms, min=eps)
                if not torch.isfinite(scale).all():
                    raise RuntimeError("Non-finite D2-3B dose scale")
                matched_resid = dc_resid * scale[:, None, None, None]
                matched_rms = residual_rms(matched_resid)
                base_delta = comp_dc["base_delta_norm"]
                pred_delta = base_delta + matched_resid
                current = context[:, -1]
                pred_next = current + pred_delta

                if step in horizons:
                    pred_phys = pred_next * field_std_t + field_mean_t
                    base.update_error_bucket(error_stats[(name, step)], pred_phys, true_next_phys)
                    update_match_bucket(match_stats[step], scale, target_rms, matched_rms)

                contexts[name] = torch.cat([context[:, 1:], pred_next.unsqueeze(1)], dim=1)

            if (batch_idx + 1) % 20 == 0:
                print("processed batch", batch_idx + 1, "/", len(loader))

    summary = base.make_error_summary(error_stats, model_names, horizons)
    differences = make_differences(summary)
    dose_match = match_summary(match_stats, horizons)

    output_dir = os.path.dirname(args.output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    summary_path = args.output_prefix + "_summary.csv"
    diff_path = args.output_prefix + "_differences.csv"
    dose_path = args.output_prefix + "_dose_match.csv"
    primary_path = args.output_prefix + "_h16_primary.csv"
    metadata_path = args.output_prefix + "_metadata.json"
    summary.to_csv(summary_path, index=False)
    differences.to_csv(diff_path, index=False)
    dose_match.to_csv(dose_path, index=False)

    primary = differences[
        (differences.comparison == "D2-3B-RMSDoseMatch - M10-2-BOnly")
        & (differences.horizon == 16)
        & (differences.field.isin(PRIMARY_FIELDS))
    ].copy()
    primary["negative_means_d2_3b_better"] = primary.rel_l2_diff_pp < 0
    primary.to_csv(primary_path, index=False)

    metadata = {
        "experiment": PREDECLARED_RULE["experiment"],
        "split_label": args.split_label,
        "seed": args.seed,
        "protocol": PREDECLARED_RULE,
        "capacity_control_parent": protocol,
        "val_h16_windows": len(val_dataset),
        "max_batches": args.max_batches,
        "test_accessed": False,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    wide = summary.pivot_table(index=["model", "horizon"], columns="field", values="rel_l2_percent").reset_index()
    wide = wide[["model", "horizon", "buoyancy", "u_x", "u_y", "pressure", "global"]]
    print("\n" + "=" * 118)
    print("ROLLOUT Rel-L2 (%)")
    print("=" * 118)
    print(wide.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\n" + "=" * 118)
    print("H16 PRIMARY | D2-3B - M10-2 | NEGATIVE = D2-3B BETTER")
    print("=" * 118)
    print(primary.to_string(index=False, float_format=lambda x: f"{x:+.6f}"))

    print("\n" + "=" * 118)
    print("PER-STATE REALIZED RMS DOSE MATCH AUDIT")
    print("=" * 118)
    print(dose_match.to_string(index=False, float_format=lambda x: f"{x:.8e}"))

    if float(dose_match["dose_match_abs_error_max"].max()) > 1e-6:
        raise RuntimeError("D2-3B realized-RMS dose match audit failed")

    print("\nSummary:", summary_path)
    print("Differences:", diff_path)
    print("Dose match:", dose_path)
    print("H16 primary:", primary_path)
    print("Metadata:", metadata_path)
    print("\n✅ D2-3B realized-RMS-dose-matched VAL H16 evaluation finished.")


if __name__ == "__main__":
    main()
