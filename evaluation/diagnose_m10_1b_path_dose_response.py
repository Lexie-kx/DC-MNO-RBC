import hashlib
import importlib.util
import os
import sys
from types import SimpleNamespace

import pandas as pd
import torch
from torch.utils.data import DataLoader


# ============================================================
# Project
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# Reuse audited Physics-Audit-v2 implementation
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
# M10-1b configuration
#
# CONTROL DIAGNOSTIC ONLY
#
# No training.
# No checkpoint modification.
#
# We scale only the FINAL residual injection:
#
#   Path A injection:
#       lambda_A * alpha_A * PathA
#
#   Path B injection:
#       lambda_B * alpha_B * PathB_safe
#
# State conditioner is unchanged and continues to see the
# CURRENT state + RAW Path-B summary exactly as trained.
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


EXPECTED = {
    "M6": (
        M6_CKPT,
        "3d0c1571dcc57e65b1cad45fbdcae72e"
        "2b737249033d379426dc616a6c414e53",
    ),
    "M10_1A": (
        M10_1A_CKPT,
        "4c515ec3eb5c2e6985b68c20e112085"
        "443c58758599ad2bc41ab4c4e26ef3927",
    ),
    "SPLIT": (
        SPLIT,
        "475d3092bb9d0ad16f023088446419b"
        "5651b1186d369ccfe0019c841f1fd8e36",
    ),
    "STATS": (
        STATS,
        "a96b1a01cf25d7b9910e01abd4de567"
        "2078e6ec3f6a6dda8a96a19e1a022d5af",
    ),
}


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

REQUESTED_HORIZONS = [
    1,
    4,
    8,
    16,
]

DOSES = [
    0.00,
    0.25,
    0.50,
    0.75,
    1.00,
]

STRIDE = 4
BATCH_SIZE = 4


OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs/tables/m10_1b_dose_response",
)

CURVE_PATH = os.path.join(
    OUTPUT_DIR,
    "m10_1b_path_dose_response_"
    "unseen_ra_seed42_curve.csv",
)

SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "m10_1b_path_dose_response_"
    "unseen_ra_seed42_summary.csv",
)

DIFF_PATH = os.path.join(
    OUTPUT_DIR,
    "m10_1b_path_dose_response_"
    "unseen_ra_seed42_vs_m6.csv",
)


# ============================================================
# SHA helpers
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


def provenance_check():

    print(
        "========== PROVENANCE =========="
    )

    for label, (
        path,
        expected_sha,
    ) in EXPECTED.items():

        if not os.path.exists(path):
            raise FileNotFoundError(
                path
            )

        actual_sha = sha256_file(
            path
        )

        print(
            f"{label}_SHA256:",
            actual_sha,
        )

        if actual_sha != expected_sha:

            raise RuntimeError(
                f"{label} SHA mismatch\n"
                f"expected={expected_sha}\n"
                f"actual={actual_sha}\n"
                f"path={path}"
            )


# ============================================================
# Dose intervention
# ============================================================

class DoseIntervention(torch.nn.Module):

    def __init__(
        self,
        trained_model,
        lambda_a,
        lambda_b,
    ):
        super().__init__()

        self.trained_model = trained_model

        self.lambda_a = float(
            lambda_a
        )

        self.lambda_b = float(
            lambda_b
        )


    def forward(
        self,
        x_norm,
        params=None,
    ):

        (
            _,
            comp,
        ) = self.trained_model(
            x_norm,
            params=params,
            return_components=True,
        )

        # Frozen M6 delta from inside the exact trained M10-1a.
        out = comp[
            "base_delta_norm"
        ].clone()

        # ----------------------------------------------------
        # Path A dose
        # ----------------------------------------------------

        if self.lambda_a != 0.0:

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
                self.lambda_a
                *
                comp["alpha_a"][
                    :,
                    None,
                    None,
                ]
                *
                comp[
                    "path_a_norm"
                ]
            )

        # ----------------------------------------------------
        # Path B dose
        #
        # In M10-1a:
        # comp["path_b_norm"]
        # is the SAFE / RMS-capped signal actually injected.
        # ----------------------------------------------------

        if self.lambda_b != 0.0:

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
                self.lambda_b
                *
                comp["alpha_b"][
                    :,
                    None,
                    None,
                ]
                *
                comp[
                    "path_b_norm"
                ]
            )

        return out


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
        "M10-1b PATH INJECTION DOSE-RESPONSE AUDIT"
    )

    print(
        "=" * 100
    )

    print(
        "📌 Stage: lightweight ablation / control diagnostic"
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
        "📌 Dose grid:",
        DOSES,
    )

    print(
        "📌 A-dose: lambda_A varies, lambda_B=0"
    )

    print(
        "📌 B-dose: lambda_B varies, lambda_A=0"
    )

    print(
        "📌 Path-B uses M10-1a RMSCap-safe signal"
    )

    print(
        "📌 Conditioner remains exactly as trained"
    )

    print(
        "📌 Device:",
        device,
    )

    print()

    provenance_check()

    print()

    # ========================================================
    # Dataset
    # ========================================================

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

    # ========================================================
    # Models
    # ========================================================

    m6, _ = audit.base.build_m6(
        M6_CKPT,
        device,
    )

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

    # ========================================================
    # Sanity checks
    # ========================================================

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

    zero_model = DoseIntervention(
        trained_model=trained_m10,
        lambda_a=0.0,
        lambda_b=0.0,
    ).to(device).eval()

    full_model = DoseIntervention(
        trained_model=trained_m10,
        lambda_a=1.0,
        lambda_b=1.0,
    ).to(device).eval()

    with torch.no_grad():

        m6_delta = m6(
            x0_norm
        )

        zero_delta = zero_model(
            x0_norm,
            params=param,
        )

        original_m10 = trained_m10(
            x0_norm,
            params=param,
        )

        reconstructed_m10 = full_model(
            x0_norm,
            params=param,
        )

    zero_max_abs = (
        zero_delta
        - m6_delta
    ).abs().max().item()

    full_max_abs = (
        reconstructed_m10
        - original_m10
    ).abs().max().item()

    print()
    print(
        "========== SANITY CHECK =========="
    )

    print(
        "lambda_A=0, lambda_B=0 "
        "vs M6 max_abs =",
        f"{zero_max_abs:.12e}",
    )

    print(
        "lambda_A=1, lambda_B=1 "
        "vs original M10-1a max_abs =",
        f"{full_max_abs:.12e}",
    )

    if zero_max_abs > 1.0e-6:

        raise RuntimeError(
            "Zero-dose does not reconstruct M6."
        )

    if full_max_abs > 1.0e-6:

        raise RuntimeError(
            "Full-dose does not reconstruct M10-1a."
        )

    print(
        "✅ Dose intervention reconstruction PASS"
    )

    # ========================================================
    # We do NOT evaluate duplicate dose=0 models.
    #
    # M6 is mathematically the zero-dose point for both curves.
    # ========================================================

    models = {
        "M6": m6,
    }

    model_meta = {
        "M6": (
            "BASELINE",
            0.0,
        ),
    }

    for dose in DOSES:

        if dose == 0.0:
            continue

        name_a = (
            f"A-dose-{dose:.2f}"
        )

        models[name_a] = (
            DoseIntervention(
                trained_model=trained_m10,
                lambda_a=dose,
                lambda_b=0.0,
            ).to(device).eval()
        )

        model_meta[name_a] = (
            "A",
            dose,
        )

        name_b = (
            f"B-dose-{dose:.2f}"
        )

        models[name_b] = (
            DoseIntervention(
                trained_model=trained_m10,
                lambda_a=0.0,
                lambda_b=dose,
            ).to(device).eval()
        )

        model_meta[name_b] = (
            "B",
            dose,
        )

    model_order = [
        "M6",
        "A-dose-0.25",
        "A-dose-0.50",
        "A-dose-0.75",
        "A-dose-1.00",
        "B-dose-0.25",
        "B-dose-0.50",
        "B-dose-0.75",
        "B-dose-1.00",
    ]

    # ========================================================
    # Rollout
    # ========================================================

    stats = {}

    print()
    print(
        "🔥 Starting full dose-response rollout..."
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

    # ========================================================
    # Finalize all actual model runs
    # ========================================================

    rows = []

    for model_name in model_order:

        path_name, dose = (
            model_meta[
                model_name
            ]
        )

        for horizon in range(
            1,
            MAX_HORIZON + 1,
        ):

            row = audit.finalize_bucket(
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

            row["path"] = path_name
            row["dose"] = dose

            rows.append(
                row
            )

    curve = pd.DataFrame(
        rows
    )

    curve.to_csv(
        CURVE_PATH,
        index=False,
    )

    # ========================================================
    # Build clean dose table.
    #
    # Dose=0 for BOTH A and B comes from M6.
    # ========================================================

    dose_rows = []

    for path_name in [
        "A",
        "B",
    ]:

        for horizon in REQUESTED_HORIZONS:

            m6_row = curve[
                (
                    curve["model"] == "M6"
                )
                &
                (
                    curve["horizon"]
                    == horizon
                )
            ].iloc[0].copy()

            m6_row["path"] = path_name
            m6_row["dose"] = 0.0
            m6_row["model"] = (
                f"{path_name}-dose-0.00"
            )

            dose_rows.append(
                m6_row
            )

            for dose in DOSES:

                if dose == 0.0:
                    continue

                name = (
                    f"{path_name}-dose-"
                    f"{dose:.2f}"
                )

                row = curve[
                    (
                        curve["model"]
                        == name
                    )
                    &
                    (
                        curve["horizon"]
                        == horizon
                    )
                ].iloc[0].copy()

                dose_rows.append(
                    row
                )

    summary = pd.DataFrame(
        dose_rows
    )

    summary = summary.sort_values(
        [
            "path",
            "horizon",
            "dose",
        ]
    ).reset_index(
        drop=True
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    # ========================================================
    # Difference vs M6
    # ========================================================

    metrics = [
        "global_rel_l2_percent",
        "buoyancy_rel_l2_percent",
        "u_y_rel_l2_percent",
        "adv_b_rel_l2_percent",
        "div_error_mae",
        "vorticity_rel_l2_percent",
    ]

    diff_rows = []

    for _, row in summary.iterrows():

        if row["dose"] == 0.0:
            continue

        baseline = summary[
            (
                summary["path"]
                == row["path"]
            )
            &
            (
                summary["horizon"]
                == row["horizon"]
            )
            &
            (
                summary["dose"]
                == 0.0
            )
        ].iloc[0]

        out = {
            "path":
                row["path"],
            "dose":
                float(
                    row["dose"]
                ),
            "horizon":
                int(
                    row["horizon"]
                ),
        }

        for metric in metrics:

            out[
                f"{metric}_diff_vs_m6"
            ] = (
                row[metric]
                -
                baseline[metric]
            )

        diff_rows.append(
            out
        )

    diff = pd.DataFrame(
        diff_rows
    )

    diff.to_csv(
        DIFF_PATH,
        index=False,
    )

    # ========================================================
    # Compact terminal output
    # ========================================================

    compact_cols = [
        "path",
        "dose",
        "horizon",
        "global_rel_l2_percent",
        "buoyancy_rel_l2_percent",
        "u_y_rel_l2_percent",
        "adv_b_rel_l2_percent",
        "div_error_mae",
        "vorticity_rel_l2_percent",
    ]

    for path_name in [
        "A",
        "B",
    ]:

        print()
        print(
            "=" * 110
        )

        print(
            f"{path_name}-PATH DOSE RESPONSE"
        )

        print(
            "=" * 110
        )

        print(
            summary[
                summary["path"]
                == path_name
            ][
                compact_cols
            ].to_string(
                index=False
            )
        )

    print()
    print(
        "=" * 110
    )

    print(
        "DIFFERENCE VS M6"
    )

    print(
        "Negative = dose model better than M6"
    )

    print(
        "=" * 110
    )

    print(
        diff.to_string(
            index=False
        )
    )

    # ========================================================
    # h16 mini-decision table
    # ========================================================

    h16 = summary[
        summary["horizon"] == 16
    ].copy()

    print()
    print(
        "=" * 110
    )

    print(
        "H16 DECISION TABLE"
    )

    print(
        "=" * 110
    )

    print(
        h16[
            [
                "path",
                "dose",
                "global_rel_l2_percent",
                "buoyancy_rel_l2_percent",
                "u_y_rel_l2_percent",
                "adv_b_rel_l2_percent",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "========== INTERPRETATION KEY =========="
    )

    print(
        "If small nonzero dose beats dose=0:"
    )

    print(
        "  -> path direction may contain useful information; "
        "current injection magnitude is too strong."
    )

    print(
        "If dose=0 is best and error worsens monotonically:"
    )

    print(
        "  -> additive output-residual injection is "
        "structurally unsupported for that path."
    )

    print(
        "If response is non-monotonic:"
    )

    print(
        "  -> path is state-sensitive; fixed/global scaling "
        "is inadequate."
    )

    print()
    print(
        "✅ Saved:"
    )

    print(
        "  curve:",
        CURVE_PATH,
    )

    print(
        "  summary:",
        SUMMARY_PATH,
    )

    print(
        "  diff:",
        DIFF_PATH,
    )


if __name__ == "__main__":
    main()
