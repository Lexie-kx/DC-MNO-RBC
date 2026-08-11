import hashlib
import importlib.util
import math
import os
import sys
from types import SimpleNamespace

import pandas as pd
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# Load the already-audited Physics Audit v2 implementation.
# We reuse its:
#   - dataset
#   - normalization
#   - FD physics metrics
#   - M6/M10-1a builders
#   - rollout protocol
#
# No existing file is modified.
# ============================================================

AUDIT_PATH = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "evaluate_m10_physics_audit_v2.py",
)

spec = importlib.util.spec_from_file_location(
    "m10_physics_audit_v2",
    AUDIT_PATH,
)

if spec is None or spec.loader is None:
    raise RuntimeError(
        f"Cannot import {AUDIT_PATH}"
    )

audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


# ============================================================
# Fixed protocol
# ============================================================

SPLIT = os.path.join(
    PROJECT_ROOT,
    "data/splits/unseen_ra_split.json",
)

STATS = os.path.join(
    PROJECT_ROOT,
    "data/stats/rbc_field_stats_unseen_ra.json",
)

M6_CKPT = os.path.join(
    PROJECT_ROOT,
    "checkpoints/cross_param/"
    "m6_fieldwise_encoder_h4_unseen_ra_best.pth",
)

M10_1A_CKPT = os.path.join(
    PROJECT_ROOT,
    "checkpoints/m10_1a/"
    "m10_1a_stateparam_rmscap_h4_unseen_ra_seed42_best.pth",
)

EXPECTED_M6_SHA = (
    "3d0c1571dcc57e65b1cad45fbdcae72e"
    "2b737249033d379426dc616a6c414e53"
)

EXPECTED_M10_1A_SHA = (
    "4c515ec3eb5c2e6985b68c20e112085"
    "443c58758599ad2bc41ab4c4e26ef3927"
)

SEED = 42

MODEL_DX = 1.0 / 64.0
MODEL_DY = 1.0 / 64.0

PHYSICS_DX = 1.0 / 64.0
PHYSICS_DY = 1.0 / 63.0

DT = 0.25

PATH_B_RMS_CAP = 1.5787383958464702
PATH_B_RMS_EPS = 1.0e-12

ALPHA_MAX = 0.25
CONDITIONER_HIDDEN = 32

MAX_HORIZON = 16
REQUESTED_HORIZONS = [1, 4, 8, 16]

STRIDE = 4
BATCH_SIZE = 4

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs/tables/m10_path_intervention",
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "m10_1a_stateparam_path_intervention_"
    "unseen_ra_seed42.csv",
)

DIFF_CSV = os.path.join(
    OUTPUT_DIR,
    "m10_1a_stateparam_path_intervention_"
    "unseen_ra_seed42_differences.csv",
)


# ============================================================
# Helpers
# ============================================================

def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def require_sha(path, expected, label):

    actual = sha256_file(path)

    print(
        f"{label}_SHA256:",
        actual,
    )

    if actual != expected:
        raise RuntimeError(
            f"{label} SHA mismatch:\n"
            f"  expected={expected}\n"
            f"  actual  ={actual}\n"
            f"  path    ={path}"
        )


# ============================================================
# Inference-time path intervention
#
# IMPORTANT:
#
# We call the ORIGINAL trained M10-1a forward with
# return_components=True first.
#
# Therefore:
#   - state summary is still computed from the CURRENT
#     intervention-generated state;
#   - state conditioner still sees RAW Path B exactly as trained;
#   - alpha_A / alpha_B are still produced by the trained model;
#   - RMSCap still constructs the SAFE Path B signal.
#
# We change ONLY which residual is finally injected.
#
# A-only:
#     base M6 delta + alpha_A * Path A
#
# B-only:
#     base M6 delta + alpha_B * Path B_safe
#
# A+B:
#     exact trained M10-1a output reconstruction
#
# This is NOT retraining and NOT an independently trained
# A-only/B-only model.
# ============================================================

class PathIntervention(torch.nn.Module):

    def __init__(
        self,
        trained_model,
        use_a,
        use_b,
        label,
    ):
        super().__init__()

        self.trained_model = trained_model

        self.use_a = bool(use_a)
        self.use_b = bool(use_b)

        self.label = label


    def forward(
        self,
        x_norm,
        params=None,
    ):

        (
            original_output,
            comp,
        ) = self.trained_model(
            x_norm,
            params=params,
            return_components=True,
        )

        out = comp[
            "base_delta_norm"
        ].clone()

        if self.use_a:

            out[
                :,
                audit.UY_IDX,
                :,
                :,
            ] = (
                out[
                    :,
                    audit.UY_IDX,
                    :,
                    :,
                ]
                +
                comp["alpha_a"][
                    :,
                    None,
                    None,
                ]
                *
                comp["path_a_norm"]
            )

        if self.use_b:

            # For M10-1a, compatibility key path_b_norm
            # is already the SAFE / capped Path-B signal.
            out[
                :,
                audit.B_IDX,
                :,
                :,
            ] = (
                out[
                    :,
                    audit.B_IDX,
                    :,
                    :,
                ]
                +
                comp["alpha_b"][
                    :,
                    None,
                    None,
                ]
                *
                comp["path_b_norm"]
            )

        return out


# ============================================================
# Difference table
# ============================================================

LOWER_IS_BETTER = [
    "global_rel_l2_percent",
    "buoyancy_rel_l2_percent",
    "u_y_rel_l2_percent",
    "adv_b_rel_l2_percent",
    "div_error_mae",
    "vorticity_rel_l2_percent",
    "grad_p_rel_l2_percent",
    "r_b_normalized_mismatch",
    "r_uy_normalized_mismatch",
    "r_u_normalized_mismatch",
]


def make_diff(summary):

    pairs = [
        ("A-only", "M6"),
        ("B-only", "M6"),
        ("A+B", "M6"),
        ("A+B", "A-only"),
        ("A+B", "B-only"),
    ]

    rows = []

    for model_a, model_b in pairs:

        a = summary[
            summary["model"] == model_a
        ]

        b = summary[
            summary["model"] == model_b
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

        for _, r in merged.iterrows():

            row = {
                "comparison":
                    f"{model_a} - {model_b}",
                "horizon":
                    int(r["horizon"]),
            }

            for metric in LOWER_IS_BETTER:

                row[
                    f"{metric}_diff"
                ] = (
                    r[f"{metric}_a"]
                    -
                    r[f"{metric}_b"]
                )

            rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "=" * 100
    )

    print(
        "M10-1a STATEPARAM PATH INTERVENTION"
    )

    print(
        "=" * 100
    )

    print(
        "📌 Type: inference-time control diagnostic"
    )

    print(
        "📌 NO TRAINING"
    )

    print(
        "📌 Split: unseen-Ra/test"
    )

    print(
        "📌 Seed: 42"
    )

    print(
        "📌 Models: M6 / A-only / B-only / A+B"
    )

    print(
        "📌 A+B = trained M10-1a StateParam-RMSCap"
    )

    print(
        "📌 Path B uses the TRAIN-ONLY RMS-capped "
        "signal exactly as M10-1a."
    )

    print(
        "📌 Conditioner still sees RAW Path B."
    )

    print(
        "📌 Physics PDE quantities remain "
        "FD-based comparative proxies."
    )

    print(
        f"📌 Device: {device}"
    )

    print()

    print(
        "========== PROVENANCE =========="
    )

    require_sha(
        M6_CKPT,
        EXPECTED_M6_SHA,
        "M6",
    )

    require_sha(
        M10_1A_CKPT,
        EXPECTED_M10_1A_SHA,
        "M10_1A",
    )

    print(
        "SPLIT_SHA256:",
        sha256_file(SPLIT),
    )

    print(
        "STATS_SHA256:",
        sha256_file(STATS),
    )

    print()

    # --------------------------------------------------------
    # Dataset: EXACT same 630-window audit protocol
    # --------------------------------------------------------

    dataset = audit.PhysicsAuditDataset(
        split_path=SPLIT,
        split_key="test",
        max_horizon=MAX_HORIZON,
        stride=STRIDE,
        max_samples=None,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    normalizer = (
        audit.FieldWiseNormalizer(
            STATS
        ).to(device)
    )

    (
        field_mean,
        field_std,
    ) = audit.base.build_field_stats(
        STATS
    )

    model_args = SimpleNamespace(
        seed=SEED,
        dx=MODEL_DX,
        dy=MODEL_DY,
        alpha_max=ALPHA_MAX,
        conditioner_hidden=(
            CONDITIONER_HIDDEN
        ),
        path_b_rms_cap=(
            PATH_B_RMS_CAP
        ),
        path_b_rms_eps=(
            PATH_B_RMS_EPS
        ),
    )

    # --------------------------------------------------------
    # Frozen M6
    # --------------------------------------------------------

    m6, _ = audit.base.build_m6(
        M6_CKPT,
        device,
    )

    # --------------------------------------------------------
    # Frozen trained M10-1a StateParam-RMSCap
    # --------------------------------------------------------

    (
        trained_m10,
        payload,
    ) = (
        audit.rmscap_eval.
        build_rmscap_stateparam(
            checkpoint_path=M10_1A_CKPT,
            field_mean=field_mean,
            field_std=field_std,
            args=model_args,
            device=device,
        )
    )

    audit.base.assert_same_m6(
        m6,
        trained_m10,
        "M10-1a-StateParam-RMSCap",
    )

    print(
        "✅ Embedded M6 exactly matches audited M6."
    )

    if isinstance(
        payload,
        dict,
    ):
        print(
            "M10_1A_BEST_VAL:",
            payload.get(
                "best_val_loss",
                payload.get(
                    "val_loss",
                    "NA",
                ),
            ),
        )

    # --------------------------------------------------------
    # Interventions
    # --------------------------------------------------------

    a_only = PathIntervention(
        trained_model=trained_m10,
        use_a=True,
        use_b=False,
        label="A-only",
    ).to(device).eval()

    b_only = PathIntervention(
        trained_model=trained_m10,
        use_a=False,
        use_b=True,
        label="B-only",
    ).to(device).eval()

    a_plus_b = PathIntervention(
        trained_model=trained_m10,
        use_a=True,
        use_b=True,
        label="A+B",
    ).to(device).eval()

    models = {
        "M6": m6,
        "A-only": a_only,
        "B-only": b_only,
        "A+B": a_plus_b,
    }

    model_order = [
        "M6",
        "A-only",
        "B-only",
        "A+B",
    ]

    # --------------------------------------------------------
    # Exact reconstruction sanity check
    # --------------------------------------------------------

    x0_phys, _, param = next(
        iter(loader)
    )

    x0_phys = x0_phys.to(
        device
    )

    param = param.to(
        device
    )

    x0_norm = normalizer.normalize_x(
        x0_phys
    )

    with torch.no_grad():

        original = trained_m10(
            x0_norm,
            params=param,
        )

        reconstructed = a_plus_b(
            x0_norm,
            params=param,
        )

    max_abs = (
        original
        - reconstructed
    ).abs().max().item()

    print()
    print(
        "========== RECONSTRUCTION CHECK =========="
    )

    print(
        "A+B vs original M10-1a max_abs =",
        f"{max_abs:.12e}",
    )

    if max_abs > 1.0e-6:
        raise RuntimeError(
            "A+B intervention does not exactly "
            "reconstruct trained M10-1a."
        )

    print(
        "✅ A+B exactly reconstructs trained M10-1a."
    )

    # --------------------------------------------------------
    # Full rollout
    # --------------------------------------------------------

    stats = {}

    print()
    print(
        "🔥 Starting full intervention Physics Audit..."
    )

    with torch.no_grad():

        for (
            batch_idx,
            (
                x0_phys,
                future_phys,
                param,
            ),
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

            for model_name in model_order:

                audit.run_model_rollout(
                    model_name=model_name,
                    model=models[
                        model_name
                    ],
                    x0_phys=x0_phys,
                    future_phys=future_phys,
                    param=param,
                    normalizer=normalizer,
                    max_horizon=MAX_HORIZON,
                    physics_dx=PHYSICS_DX,
                    physics_dy=PHYSICS_DY,
                    dt=DT,
                    stats=stats,
                )

            if (
                batch_idx % 10 == 0
                or batch_idx == len(loader)
            ):
                print(
                    f"  processed batch "
                    f"{batch_idx}/{len(loader)}"
                )

    # --------------------------------------------------------
    # Finalize
    # --------------------------------------------------------

    rows = []

    for model_name in model_order:

        for horizon in range(
            1,
            MAX_HORIZON + 1,
        ):

            rows.append(
                audit.finalize_bucket(
                    split_label="unseen_ra",
                    seed=SEED,
                    model_name=model_name,
                    horizon=horizon,
                    bucket=stats[
                        (
                            model_name,
                            horizon,
                        )
                    ],
                )
            )

    curve = pd.DataFrame(
        rows
    )

    summary = curve[
        curve["horizon"].isin(
            REQUESTED_HORIZONS
        )
    ].copy()

    diff = make_diff(
        summary
    )

    curve.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    diff.to_csv(
        DIFF_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # Compact output
    # --------------------------------------------------------

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
        "r_uy_normalized_mismatch",
        "r_u_normalized_mismatch",
    ]

    print()
    print(
        "=" * 100
    )

    print(
        "PATH INTERVENTION SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        summary[
            compact_cols
        ].to_string(
            index=False
        )
    )

    diff_cols = [
        "comparison",
        "horizon",
        "global_rel_l2_percent_diff",
        "buoyancy_rel_l2_percent_diff",
        "u_y_rel_l2_percent_diff",
        "adv_b_rel_l2_percent_diff",
        "div_error_mae_diff",
        "vorticity_rel_l2_percent_diff",
        "r_b_normalized_mismatch_diff",
        "r_uy_normalized_mismatch_diff",
        "r_u_normalized_mismatch_diff",
    ]

    print()
    print(
        "=" * 100
    )

    print(
        "PATH INTERVENTION DIFFERENCES"
    )

    print(
        "Negative = first model better."
    )

    print(
        "=" * 100
    )

    print(
        diff[
            diff_cols
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "========== DECISION KEY =========="
    )

    print(
        "A-only ~ M6 or better, B-only worse:"
    )

    print(
        "  -> Path B is the main structural problem."
    )

    print(
        "A-only and B-only acceptable, A+B worse:"
    )

    print(
        "  -> A/B interaction is the main problem."
    )

    print(
        "A-only and B-only both worse:"
    )

    print(
        "  -> direct residual-injection design itself "
        "needs redesign."
    )

    print()
    print(
        "✅ Saved:"
    )

    print(
        "  summary:",
        OUTPUT_CSV,
    )

    print(
        "  differences:",
        DIFF_CSV,
    )


if __name__ == "__main__":
    main()
