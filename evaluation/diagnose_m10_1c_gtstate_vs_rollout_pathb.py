import hashlib
import importlib.util
import os
import sys
from collections import defaultdict
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
# Reuse Physics Audit v2
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

ROLLOUT_DOSE_CSV = os.path.join(
    PROJECT_ROOT,
    "outputs/tables/m10_1b_dose_response/"
    "m10_1b_path_dose_response_"
    "unseen_ra_seed42_summary.csv",
)


EXPECTED_SHA = {
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

OFFSETS = [
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
    "outputs/tables/m10_1c_gtstate_pathb",
)

GTSTATE_CSV = os.path.join(
    OUTPUT_DIR,
    "m10_1c_gtstate_pathb_"
    "unseen_ra_seed42.csv",
)

COMPARE_CSV = os.path.join(
    OUTPUT_DIR,
    "m10_1c_gtstate_vs_rollout_pathb_"
    "unseen_ra_seed42.csv",
)

DIAG_CSV = os.path.join(
    OUTPUT_DIR,
    "m10_1c_gtstate_pathb_diagnostics_"
    "unseen_ra_seed42.csv",
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


def provenance_check():

    print(
        "========== PROVENANCE =========="
    )

    for label, (
        path,
        expected,
    ) in EXPECTED_SHA.items():

        if not os.path.exists(path):
            raise FileNotFoundError(
                path
            )

        actual = sha256_file(
            path
        )

        print(
            f"{label}_SHA256:",
            actual,
        )

        if actual != expected:

            raise RuntimeError(
                f"{label} SHA mismatch\n"
                f"expected={expected}\n"
                f"actual={actual}\n"
                f"path={path}"
            )

    if not os.path.exists(
        ROLLOUT_DOSE_CSV
    ):
        raise FileNotFoundError(
            "Previous M10-1b rollout-dose CSV missing:\n"
            f"{ROLLOUT_DOSE_CSV}"
        )


def b_dose_delta(
    comp,
    dose,
):

    # Exact frozen-M6 delta from embedded audited M6.
    out = comp[
        "base_delta_norm"
    ].clone()

    if dose != 0.0:

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
            float(dose)
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


def model_name_for_dose(
    dose,
):
    return (
        f"GT-B-dose-{dose:.2f}"
        if dose > 0.0
        else "GT-M6"
    )


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
        "=" * 110
    )

    print(
        "M10-1c GT-STATE vs ROLLOUT-STATE PATH-B AUDIT"
    )

    print(
        "=" * 110
    )

    print(
        "📌 Stage: control diagnostic / NO TRAINING"
    )

    print(
        "📌 Split: unseen-Ra/test"
    )

    print(
        "📌 Seed: 42"
    )

    print(
        "📌 Path A: OFF"
    )

    print(
        "📌 Path B doses:",
        DOSES,
    )

    print(
        "📌 GT-state means:"
    )

    print(
        "   at each original rollout offset,"
    )

    print(
        "   use the four TRUE preceding frames"
    )

    print(
        "   and make ONE next-step prediction."
    )

    print(
        "📌 Therefore offset=16 is NOT a "
        "16-step teacher-forced rollout."
    )

    print(
        "📌 It is a one-step prediction at the "
        "same temporal position as rollout h16."
    )

    print(
        "📌 Path B remains M10-1a RMSCap-safe."
    )

    print(
        "📌 State conditioner still sees RAW Path B."
    )

    print(
        f"📌 Device: {device}"
    )

    print()

    provenance_check()

    # ========================================================
    # Data
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
    # Frozen models
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

    print()
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
    # Accumulators
    # ========================================================

    stats = {}

    diag = defaultdict(
        lambda: {
            "n": 0,
            "alpha_b_sum": 0.0,
            "alpha_b_abs_sum": 0.0,
            "alpha_b_min": float("inf"),
            "alpha_b_max": float("-inf"),
            "raw_rms_sum": 0.0,
            "raw_rms_max": 0.0,
            "safe_rms_sum": 0.0,
            "cap_active": 0,
            "scale_min": 1.0,
        }
    )

    embedded_m6_max_abs = 0.0

    # ========================================================
    # GT-state one-step audit
    # ========================================================

    print()
    print(
        "🔥 Starting GT-state Path-B audit..."
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

            batch_size = (
                x0_phys.shape[0]
            )

            X = x0_phys.shape[-2]
            Y = x0_phys.shape[-1]

            # ------------------------------------------------
            # Original physical history:
            #
            # x0_phys:
            #   [B, 16, X, Y]
            #
            # ->
            #
            # history:
            #   [B, 4 time, 4 field, X, Y]
            # ------------------------------------------------

            history = x0_phys.reshape(
                batch_size,
                audit.CONTEXT_LENGTH,
                len(audit.FIELD_ORDER),
                X,
                Y,
            )

            # ------------------------------------------------
            # Complete available GT sequence:
            #
            # index 0..3:
            #   original context
            #
            # index 4..19:
            #   GT future step 1..16
            # ------------------------------------------------

            all_gt = torch.cat(
                [
                    history,
                    future_phys,
                ],
                dim=1,
            )

            for offset in OFFSETS:

                # Target index:
                #
                # offset=1  -> index 4
                # offset=4  -> index 7
                # offset=8  -> index 11
                # offset=16 -> index 19

                target_idx = (
                    audit.CONTEXT_LENGTH
                    +
                    offset
                    -
                    1
                )

                context_gt = all_gt[
                    :,
                    target_idx
                    - audit.CONTEXT_LENGTH
                    :
                    target_idx,
                ]

                gt_prev = all_gt[
                    :,
                    target_idx - 1,
                ]

                gt_next = all_gt[
                    :,
                    target_idx,
                ]

                context_flat = (
                    context_gt.reshape(
                        batch_size,
                        audit.CONTEXT_LENGTH
                        *
                        len(audit.FIELD_ORDER),
                        X,
                        Y,
                    )
                )

                x_norm = (
                    normalizer.normalize_x(
                        context_flat
                    )
                )

                current_norm = x_norm[
                    :,
                    -len(audit.FIELD_ORDER):,
                    :,
                    :,
                ]

                # --------------------------------------------
                # Independent M6
                # --------------------------------------------

                m6_delta = m6(
                    x_norm
                )

                # --------------------------------------------
                # M10-1a components on the SAME GT context.
                #
                # Conditioner therefore sees GT-state summary.
                # --------------------------------------------

                (
                    _,
                    comp,
                ) = trained_m10(
                    x_norm,
                    params=param,
                    return_components=True,
                )

                embedded_diff = (
                    comp[
                        "base_delta_norm"
                    ]
                    -
                    m6_delta
                ).abs().max().item()

                embedded_m6_max_abs = max(
                    embedded_m6_max_abs,
                    embedded_diff,
                )

                # --------------------------------------------
                # Path-B diagnostics on GT state
                # --------------------------------------------

                d = diag[offset]

                alpha_b = comp[
                    "alpha_b"
                ].detach()

                raw_rms = comp[
                    "path_b_rms_raw"
                ].detach()

                safe_rms = comp[
                    "path_b_rms_safe"
                ].detach()

                scale = comp[
                    "path_b_scale"
                ].detach()

                cap_active = comp[
                    "path_b_cap_active"
                ].detach()

                n_here = alpha_b.numel()

                d["n"] += n_here

                d[
                    "alpha_b_sum"
                ] += (
                    alpha_b.double()
                    .sum()
                    .item()
                )

                d[
                    "alpha_b_abs_sum"
                ] += (
                    alpha_b.double()
                    .abs()
                    .sum()
                    .item()
                )

                d[
                    "alpha_b_min"
                ] = min(
                    d["alpha_b_min"],
                    alpha_b.min().item(),
                )

                d[
                    "alpha_b_max"
                ] = max(
                    d["alpha_b_max"],
                    alpha_b.max().item(),
                )

                d[
                    "raw_rms_sum"
                ] += (
                    raw_rms.double()
                    .sum()
                    .item()
                )

                d[
                    "raw_rms_max"
                ] = max(
                    d["raw_rms_max"],
                    raw_rms.max().item(),
                )

                d[
                    "safe_rms_sum"
                ] += (
                    safe_rms.double()
                    .sum()
                    .item()
                )

                d[
                    "cap_active"
                ] += int(
                    cap_active.sum().item()
                )

                d[
                    "scale_min"
                ] = min(
                    d["scale_min"],
                    scale.min().item(),
                )

                # --------------------------------------------
                # Dose=0 is the independent audited M6.
                #
                # Dose>0:
                # base M6 delta
                # +
                # lambda_B * alpha_B * PathB_safe
                #
                # Path A remains OFF.
                # --------------------------------------------

                for dose in DOSES:

                    if dose == 0.0:

                        pred_delta_norm = (
                            m6_delta
                        )

                    else:

                        pred_delta_norm = (
                            b_dose_delta(
                                comp,
                                dose,
                            )
                        )

                    pred_next_norm = (
                        current_norm
                        +
                        pred_delta_norm
                    )

                    pred_next_phys = (
                        normalizer.denormalize_y(
                            pred_next_norm
                        )
                    )

                    model_name = (
                        model_name_for_dose(
                            dose
                        )
                    )

                    key = (
                        model_name,
                        offset,
                    )

                    if key not in stats:
                        stats[key] = (
                            audit.new_bucket()
                        )

                    # IMPORTANT:
                    #
                    # Teacher-forced transition:
                    #
                    # pred_prev = GT previous state
                    # gt_prev   = same GT previous state
                    #
                    # Only next state is predicted.

                    audit.update_physics_metrics(
                        stats[key],
                        pred_prev=gt_prev,
                        pred_next=(
                            pred_next_phys
                        ),
                        gt_prev=gt_prev,
                        gt_next=gt_next,
                        param=param,
                        dx=PHYSICS_DX,
                        dy=PHYSICS_DY,
                        dt=DT,
                    )

            if (
                batch_idx % 10 == 0
                or batch_idx
                == len(loader)
            ):

                print(
                    f"  processed batch "
                    f"{batch_idx}/"
                    f"{len(loader)}"
                )

    # ========================================================
    # Embedded M6 sanity
    # ========================================================

    print()
    print(
        "========== EMBEDDED M6 CHECK =========="
    )

    print(
        "max_abs embedded-M6 vs independent-M6 =",
        f"{embedded_m6_max_abs:.12e}",
    )

    if embedded_m6_max_abs > 1.0e-6:

        raise RuntimeError(
            "Embedded M6 mismatch."
        )

    print(
        "✅ Embedded M6 reconstruction PASS"
    )

    # ========================================================
    # Finalize GT-state table
    # ========================================================

    rows = []

    for offset in OFFSETS:

        for dose in DOSES:

            model_name = (
                model_name_for_dose(
                    dose
                )
            )

            row = audit.finalize_bucket(
                split_label="unseen_ra",
                seed=SEED,
                model_name=model_name,
                horizon=offset,
                bucket=stats[
                    (
                        model_name,
                        offset,
                    )
                ],
            )

            row["state_source"] = (
                "GT"
            )

            row["offset"] = (
                offset
            )

            row["dose"] = (
                dose
            )

            rows.append(
                row
            )

    gt_df = pd.DataFrame(
        rows
    )

    gt_df.to_csv(
        GTSTATE_CSV,
        index=False,
    )

    # ========================================================
    # GT-state Path-B diagnostics
    # ========================================================

    diag_rows = []

    for offset in OFFSETS:

        d = diag[offset]

        n = max(
            d["n"],
            1,
        )

        diag_rows.append(
            {
                "offset":
                    offset,

                "alpha_b_mean":
                    d["alpha_b_sum"]
                    / n,

                "alpha_b_abs_mean":
                    d[
                        "alpha_b_abs_sum"
                    ]
                    / n,

                "alpha_b_min":
                    d["alpha_b_min"],

                "alpha_b_max":
                    d["alpha_b_max"],

                "path_b_raw_rms_mean":
                    d["raw_rms_sum"]
                    / n,

                "path_b_raw_rms_max":
                    d["raw_rms_max"],

                "path_b_safe_rms_mean":
                    d["safe_rms_sum"]
                    / n,

                "cap_active_fraction":
                    d["cap_active"]
                    / n,

                "path_b_scale_min":
                    d["scale_min"],
            }
        )

    diag_df = pd.DataFrame(
        diag_rows
    )

    diag_df.to_csv(
        DIAG_CSV,
        index=False,
    )

    # ========================================================
    # Load previous free-rollout dose response
    # ========================================================

    rollout = pd.read_csv(
        ROLLOUT_DOSE_CSV
    )

    rollout = rollout[
        rollout["path"] == "B"
    ].copy()

    # ========================================================
    # h1 sanity:
    #
    # GT-state offset1 MUST equal free-rollout h1 because
    # both start from exactly the same true initial context.
    # ========================================================

    sanity_diffs = []

    for dose in DOSES:

        r = rollout[
            (
                rollout["dose"]
                == dose
            )
            &
            (
                rollout["horizon"]
                == 1
            )
        ].iloc[0]

        g = gt_df[
            (
                gt_df["dose"]
                == dose
            )
            &
            (
                gt_df["offset"]
                == 1
            )
        ].iloc[0]

        for metric in [
            "global_rel_l2_percent",
            "buoyancy_rel_l2_percent",
            "u_y_rel_l2_percent",
            "adv_b_rel_l2_percent",
        ]:

            sanity_diffs.append(
                abs(
                    float(
                        r[metric]
                    )
                    -
                    float(
                        g[metric]
                    )
                )
            )

    h1_sanity_max_abs = max(
        sanity_diffs
    )

    print()
    print(
        "========== H1 STATE-SOURCE SANITY =========="
    )

    print(
        "GT-state offset1 vs rollout h1 "
        "max metric abs diff =",
        f"{h1_sanity_max_abs:.12e}",
    )

    if h1_sanity_max_abs > 1.0e-5:

        raise RuntimeError(
            "GT-state offset1 does not reproduce "
            "rollout h1."
        )

    print(
        "✅ h1 GT-state exactly reproduces rollout start."
    )

    # ========================================================
    # Build state-source comparison.
    #
    # We compare BENEFIT relative to each source's own M6:
    #
    # rollout benefit:
    #     rollout B-dose - rollout M6
    #
    # GT-state benefit:
    #     GT-state B-dose - GT-state M6
    #
    # Negative = Path B helps.
    # ========================================================

    metrics = [
        "global_rel_l2_percent",
        "buoyancy_rel_l2_percent",
        "u_y_rel_l2_percent",
        "adv_b_rel_l2_percent",
        "div_error_mae",
        "vorticity_rel_l2_percent",
    ]

    compare_rows = []

    for offset in OFFSETS:

        rollout_base = rollout[
            (
                rollout["dose"]
                == 0.0
            )
            &
            (
                rollout["horizon"]
                == offset
            )
        ].iloc[0]

        gt_base = gt_df[
            (
                gt_df["dose"]
                == 0.0
            )
            &
            (
                gt_df["offset"]
                == offset
            )
        ].iloc[0]

        for dose in DOSES:

            if dose == 0.0:
                continue

            rollout_row = rollout[
                (
                    rollout["dose"]
                    == dose
                )
                &
                (
                    rollout["horizon"]
                    == offset
                )
            ].iloc[0]

            gt_row = gt_df[
                (
                    gt_df["dose"]
                    == dose
                )
                &
                (
                    gt_df["offset"]
                    == offset
                )
            ].iloc[0]

            out = {
                "offset":
                    offset,

                "dose":
                    dose,
            }

            for metric in metrics:

                rollout_diff = (
                    float(
                        rollout_row[
                            metric
                        ]
                    )
                    -
                    float(
                        rollout_base[
                            metric
                        ]
                    )
                )

                gt_diff = (
                    float(
                        gt_row[
                            metric
                        ]
                    )
                    -
                    float(
                        gt_base[
                            metric
                        ]
                    )
                )

                out[
                    f"rollout_{metric}_diff"
                ] = rollout_diff

                out[
                    f"gtstate_{metric}_diff"
                ] = gt_diff

            compare_rows.append(
                out
            )

    compare_df = pd.DataFrame(
        compare_rows
    )

    compare_df.to_csv(
        COMPARE_CSV,
        index=False,
    )

    # ========================================================
    # Compact output 1:
    # GT-state one-step table
    # ========================================================

    print()
    print(
        "=" * 120
    )

    print(
        "GT-STATE ONE-STEP PATH-B RESPONSE"
    )

    print(
        "Each offset uses FOUR TRUE preceding frames."
    )

    print(
        "=" * 120
    )

    print(
        gt_df[
            [
                "offset",
                "dose",
                "global_rel_l2_percent",
                "buoyancy_rel_l2_percent",
                "u_y_rel_l2_percent",
                "adv_b_rel_l2_percent",
                "div_error_mae",
                "vorticity_rel_l2_percent",
            ]
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # Compact output 2:
    # GT-state signal diagnostics
    # ========================================================

    print()
    print(
        "=" * 120
    )

    print(
        "GT-STATE PATH-B DIAGNOSTICS"
    )

    print(
        "=" * 120
    )

    print(
        diag_df.to_string(
            index=False
        )
    )

    # ========================================================
    # Compact output 3:
    # direct state-source comparison
    # ========================================================

    key_cols = [
        "offset",
        "dose",

        "rollout_global_rel_l2_percent_diff",
        "gtstate_global_rel_l2_percent_diff",

        "rollout_buoyancy_rel_l2_percent_diff",
        "gtstate_buoyancy_rel_l2_percent_diff",

        "rollout_adv_b_rel_l2_percent_diff",
        "gtstate_adv_b_rel_l2_percent_diff",
    ]

    print()
    print(
        "=" * 120
    )

    print(
        "ROLLOUT-STATE vs GT-STATE PATH-B BENEFIT"
    )

    print(
        "Negative = Path B improves over M6 "
        "under that state source."
    )

    print(
        "=" * 120
    )

    print(
        compare_df[
            key_cols
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # Most important h16 table
    # ========================================================

    print()
    print(
        "=" * 120
    )

    print(
        "H16/OFFSET16 DECISION TABLE"
    )

    print(
        "=" * 120
    )

    h16 = compare_df[
        compare_df["offset"] == 16
    ].copy()

    print(
        h16[
            key_cols
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "========== DECISION KEY =========="
    )

    print(
        "Case 1:"
    )

    print(
        "  rollout diff > 0"
    )

    print(
        "  but GT-state diff < 0"
    )

    print(
        "  -> strong evidence for rollout-state "
        "contamination / distribution shift."
    )

    print()

    print(
        "Case 2:"
    )

    print(
        "  rollout diff > 0"
    )

    print(
        "  and GT-state diff > 0"
    )

    print(
        "  -> Path B is not merely corrupted by "
        "autoregressive state error;"
    )

    print(
        "     its usefulness is intrinsically "
        "state/time selective."
    )

    print()

    print(
        "Case 3:"
    )

    print(
        "  small dose GT-state helpful,"
    )

    print(
        "  large dose GT-state harmful"
    )

    print(
        "  -> Path B contains useful information "
        "but requires state-dependent trust."
    )

    print()
    print(
        "✅ Saved:"
    )

    print(
        "  GT-state:",
        GTSTATE_CSV,
    )

    print(
        "  diagnostics:",
        DIAG_CSV,
    )

    print(
        "  comparison:",
        COMPARE_CSV,
    )


if __name__ == "__main__":
    main()
