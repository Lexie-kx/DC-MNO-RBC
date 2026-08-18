import argparse
import json
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


from constants import FIELD_ORDER, B_IDX
from datasets.rbc_dataset import RBCDataset

from models.operators.fno2d_m10_2_bonly_rmscap import (
    M10BOnlyRMSCapFNO2d,
)

from models.operators.fno2d_d2_1_dimconsistent_bonly import (
    D21DimConsistentBOnlyFNO2d,
)

from scripts.train_m10_2_bonly_rmscap_h4 import (
    M10MultiStepParamDataset,
    build_field_stats,
    set_seed,
    sha256_file,
)


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "VAL-only H16 fair rollout comparison: "
            "D2-1 canonical B-only vs M10-2 old B-only."
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
        "--m10_checkpoint",
        required=True,
    )

    parser.add_argument(
        "--d2_checkpoint",
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
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
    )

    parser.add_argument(
        "--output_prefix",
        required=True,
    )

    return parser.parse_args()


# ============================================================
# Checkpoint helpers
# ============================================================

def load_payload(
    path,
    device,
):
    if not os.path.exists(
        path
    ):
        raise FileNotFoundError(
            path
        )

    payload = torch.load(
        path,
        map_location=device,
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise RuntimeError(
            f"Checkpoint is not a dict: {path}"
        )

    if (
        "model_state_dict"
        not in payload
    ):
        raise RuntimeError(
            "Checkpoint has no "
            "model_state_dict:\n"
            f"{path}"
        )

    return payload


def verify_checkpoint_provenance(
    payload,
    *,
    checkpoint_name,
    expected_seed,
    split_sha,
    stats_sha,
):
    if (
        "seed" in payload
        and int(
            payload["seed"]
        )
        != int(
            expected_seed
        )
    ):
        raise RuntimeError(
            f"{checkpoint_name} seed mismatch."
        )

    if (
        "split_sha256" in payload
        and payload[
            "split_sha256"
        ]
        != split_sha
    ):
        raise RuntimeError(
            f"{checkpoint_name} split SHA mismatch."
        )

    if (
        "stats_sha256" in payload
        and payload[
            "stats_sha256"
        ]
        != stats_sha
    ):
        raise RuntimeError(
            f"{checkpoint_name} stats SHA mismatch."
        )


# ============================================================
# Model builders
# ============================================================

def build_m10_model(
    payload,
    field_mean,
    field_std,
    device,
):
    stabilization = payload.get(
        "stabilization",
        {},
    )

    conditioner = payload.get(
        "conditioner",
        {},
    )

    mode = payload.get(
        "factorial_mode",
        "stateparam",
    )

    if mode != "stateparam":
        raise RuntimeError(
            "Expected M10-2 StateParam checkpoint, "
            f"got mode={mode}"
        )

    model = M10BOnlyRMSCapFNO2d(
        field_mean=field_mean,
        field_std=field_std,
        mode="stateparam",
        dx=float(
            payload.get(
                "dx",
                1.0 / 64.0,
            )
        ),
        dy=float(
            payload.get(
                "dy",
                1.0 / 64.0,
            )
        ),
        path_b_rms_cap=float(
            stabilization[
                "path_b_rms_cap"
            ]
        ),
        path_b_rms_eps=float(
            stabilization.get(
                "path_b_rms_eps",
                1.0e-12,
            )
        ),
        alpha_max=float(
            payload.get(
                "alpha_max",
                0.25,
            )
        ),
        conditioner_hidden=int(
            conditioner.get(
                "hidden_dim",
                32,
            )
        ),
        freeze_m6=True,
    ).to(
        device
    )

    model.load_state_dict(
        payload[
            "model_state_dict"
        ],
        strict=True,
    )

    model.eval()
    model.m6.eval()

    return model


def build_d2_model(
    payload,
    field_mean,
    field_std,
    device,
):
    stabilization = payload.get(
        "stabilization",
        {},
    )

    conditioner = payload.get(
        "conditioner",
        {},
    )

    interface = payload.get(
        "dimensional_interface",
        {},
    )

    if not interface:
        raise RuntimeError(
            "D2 checkpoint has no "
            "dimensional_interface metadata."
        )

    if not interface.get(
        "rate_to_increment",
        False,
    ):
        raise RuntimeError(
            "D2 checkpoint does not declare "
            "rate_to_increment=True."
        )

    model = D21DimConsistentBOnlyFNO2d(
        field_mean=field_mean,
        field_std=field_std,
        dx=float(
            interface[
                "dx"
            ]
        ),
        dy=float(
            interface[
                "dy"
            ]
        ),
        dt=float(
            interface[
                "dt"
            ]
        ),
        path_b_rms_cap=float(
            stabilization[
                "path_b_rms_cap"
            ]
        ),
        path_b_rms_eps=float(
            stabilization.get(
                "path_b_rms_eps",
                1.0e-12,
            )
        ),
        alpha_max=float(
            payload.get(
                "alpha_max",
                0.25,
            )
        ),
        conditioner_hidden=int(
            conditioner.get(
                "hidden_dim",
                32,
            )
        ),
        freeze_m6=True,
    ).to(
        device
    )

    model.load_state_dict(
        payload[
            "model_state_dict"
        ],
        strict=True,
    )

    model.eval()
    model.m6.eval()

    return model


# ============================================================
# Fairness audit
# ============================================================

def compare_embedded_m6(
    m10_model,
    d2_model,
):
    state_a = (
        m10_model.m6.state_dict()
    )

    state_b = (
        d2_model.m6.state_dict()
    )

    if (
        set(
            state_a.keys()
        )
        != set(
            state_b.keys()
        )
    ):
        raise RuntimeError(
            "Embedded M6 state_dict keys differ."
        )

    max_abs = 0.0

    for key in state_a:

        a = state_a[
            key
        ]

        b = state_b[
            key
        ]

        if a.shape != b.shape:
            raise RuntimeError(
                "Embedded M6 tensor shape mismatch: "
                f"{key}"
            )

        if (
            torch.is_floating_point(
                a
            )
            or torch.is_complex(
                a
            )
        ):
            diff = float(
                (
                    a
                    -
                    b
                )
                .abs()
                .max()
                .detach()
                .cpu()
            )

            max_abs = max(
                max_abs,
                diff,
            )

        else:
            if not torch.equal(
                a,
                b,
            ):
                raise RuntimeError(
                    "Non-floating M6 buffer mismatch: "
                    f"{key}"
                )

    return max_abs


# ============================================================
# Metric buckets
# ============================================================

def new_error_bucket():
    return {
        "field_sse":
            [
                0.0
                for _ in FIELD_ORDER
            ],

        "field_target_sq":
            [
                0.0
                for _ in FIELD_ORDER
            ],

        "field_numel":
            [
                0
                for _ in FIELD_ORDER
            ],

        "global_sse":
            0.0,

        "global_target_sq":
            0.0,

        "global_numel":
            0,
    }


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

        "injection_rms_sum":
            0.0,
    }


def update_error_bucket(
    bucket,
    pred_phys,
    true_phys,
):
    diff = (
        pred_phys
        -
        true_phys
    )

    for c in range(
        len(
            FIELD_ORDER
        )
    ):
        d = diff[
            :,
            c,
            :,
            :,
        ]

        y = true_phys[
            :,
            c,
            :,
            :,
        ]

        bucket[
            "field_sse"
        ][c] += float(
            (
                d.double()
                *
                d.double()
            )
            .sum()
            .cpu()
        )

        bucket[
            "field_target_sq"
        ][c] += float(
            (
                y.double()
                *
                y.double()
            )
            .sum()
            .cpu()
        )

        bucket[
            "field_numel"
        ][c] += int(
            d.numel()
        )

    bucket[
        "global_sse"
    ] += float(
        (
            diff.double()
            *
            diff.double()
        )
        .sum()
        .cpu()
    )

    bucket[
        "global_target_sq"
    ] += float(
        (
            true_phys.double()
            *
            true_phys.double()
        )
        .sum()
        .cpu()
    )

    bucket[
        "global_numel"
    ] += int(
        diff.numel()
    )


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

    path_b_rms = (
        components[
            "path_b_rms_raw"
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
            B_IDX,
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
        path_b_rms.sum().cpu()
    )

    bucket[
        "path_b_rms_raw_max"
    ] = max(
        bucket[
            "path_b_rms_raw_max"
        ],
        float(
            path_b_rms.max().cpu()
        ),
    )

    bucket[
        "injection_rms_sum"
    ] += float(
        injection_rms.sum().cpu()
    )


# ============================================================
# Summary
# ============================================================

def make_error_summary(
    error_stats,
    model_names,
    horizons,
):
    rows = []

    for model_name in model_names:

        for horizon in horizons:

            bucket = error_stats[
                (
                    model_name,
                    horizon,
                )
            ]

            for c, field in enumerate(
                FIELD_ORDER
            ):
                rel_l2 = (
                    (
                        bucket[
                            "field_sse"
                        ][c]
                        /
                        (
                            bucket[
                                "field_target_sq"
                            ][c]
                            +
                            1.0e-30
                        )
                    )
                    ** 0.5
                    *
                    100.0
                )

                mse = (
                    bucket[
                        "field_sse"
                    ][c]
                    /
                    bucket[
                        "field_numel"
                    ][c]
                )

                rows.append(
                    {
                        "model":
                            model_name,

                        "horizon":
                            horizon,

                        "field":
                            field,

                        "rel_l2_percent":
                            rel_l2,

                        "mse":
                            mse,
                    }
                )

            global_rel = (
                (
                    bucket[
                        "global_sse"
                    ]
                    /
                    (
                        bucket[
                            "global_target_sq"
                        ]
                        +
                        1.0e-30
                    )
                )
                ** 0.5
                *
                100.0
            )

            global_mse = (
                bucket[
                    "global_sse"
                ]
                /
                bucket[
                    "global_numel"
                ]
            )

            rows.append(
                {
                    "model":
                        model_name,

                    "horizon":
                        horizon,

                    "field":
                        "global",

                    "rel_l2_percent":
                        global_rel,

                    "mse":
                        global_mse,
                }
            )

    return pd.DataFrame(
        rows
    )


def make_diag_summary(
    diag_stats,
    model_names,
    horizons,
):
    rows = []

    for model_name in model_names:

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

            rows.append(
                {
                    "model":
                        model_name,

                    "horizon":
                        horizon,

                    "samples":
                        n,

                    "alpha_mean":
                        (
                            bucket[
                                "alpha_sum"
                            ]
                            / n
                        ),

                    "abs_alpha_mean":
                        (
                            bucket[
                                "abs_alpha_sum"
                            ]
                            / n
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


def make_difference_table(
    summary,
):
    a = summary[
        summary[
            "model"
        ]
        ==
        "D2-1-DC-BOnly"
    ]

    b = summary[
        summary[
            "model"
        ]
        ==
        "M10-2-BOnly"
    ]

    merged = a.merge(
        b,
        on=[
            "horizon",
            "field",
        ],
        suffixes=(
            "_d2",
            "_m10",
        ),
    )

    rows = []

    for _, row in merged.iterrows():

        rows.append(
            {
                "comparison":
                    "D2-1 - M10-2",

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

                "rel_l2_d2":
                    float(
                        row[
                            "rel_l2_percent_d2"
                        ]
                    ),

                "rel_l2_m10":
                    float(
                        row[
                            "rel_l2_percent_m10"
                        ]
                    ),

                "rel_l2_diff_pp":
                    float(
                        row[
                            "rel_l2_percent_d2"
                        ]
                        -
                        row[
                            "rel_l2_percent_m10"
                        ]
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    if args.max_horizon != 16:
        raise ValueError(
            "This evaluator is locked "
            "to max_horizon=16."
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

    if any(
        h < 1
        or h > args.max_horizon
        for h in horizons
    ):
        raise ValueError(
            "Invalid horizons."
        )

    if args.batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    for path in [
        args.split,
        args.stats,
        args.m10_checkpoint,
        args.d2_checkpoint,
    ]:
        if not os.path.exists(
            path
        ):
            raise FileNotFoundError(
                path
            )

    set_seed(
        args.seed
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
        "D2-1 vs M10-2 B-ONLY "
        "LONG-ROLLOUT VALIDATION"
    )
    print(
        "=" * 100
    )
    print(
        "Stage: CONTROL / FAIR ROLLOUT COMPARISON"
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
        "Horizons:",
        horizons,
    )
    print(
        "Device:",
        device,
    )
    print(
        "Seed:",
        args.seed,
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

    m10_payload = load_payload(
        args.m10_checkpoint,
        device,
    )

    d2_payload = load_payload(
        args.d2_checkpoint,
        device,
    )

    verify_checkpoint_provenance(
        m10_payload,
        checkpoint_name="M10-2",
        expected_seed=args.seed,
        split_sha=split_sha,
        stats_sha=stats_sha,
    )

    verify_checkpoint_provenance(
        d2_payload,
        checkpoint_name="D2-1",
        expected_seed=args.seed,
        split_sha=split_sha,
        stats_sha=stats_sha,
    )

    m10_m6_sha = m10_payload.get(
        "m6_checkpoint_sha256"
    )

    d2_m6_sha = d2_payload.get(
        "m6_checkpoint_sha256"
    )

    if (
        m10_m6_sha is not None
        and d2_m6_sha is not None
        and m10_m6_sha
        != d2_m6_sha
    ):
        raise RuntimeError(
            "M10-2 and D2-1 do not "
            "share the same M6 provenance."
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
        args.d2_checkpoint,
    )
    print(
        "M10-2 best_epoch:",
        m10_payload.get(
            "best_epoch"
        ),
    )
    print(
        "D2-1 best_epoch:",
        d2_payload.get(
            "best_epoch"
        ),
    )
    print(
        "M10-2 M6 SHA:",
        m10_m6_sha,
    )
    print(
        "D2-1 M6 SHA:",
        d2_m6_sha,
    )

    print()
    print(
        "M10 interface:"
    )
    print(
        "  dx =",
        m10_payload.get(
            "dx"
        ),
    )
    print(
        "  dy =",
        m10_payload.get(
            "dy"
        ),
    )
    print(
        "  dt = implicit / missing"
    )
    print(
        "  cap =",
        m10_payload.get(
            "stabilization",
            {},
        ).get(
            "path_b_rms_cap"
        ),
    )

    d2_interface = d2_payload[
        "dimensional_interface"
    ]

    print(
        "D2 interface:"
    )
    print(
        "  dx =",
        d2_interface[
            "dx"
        ],
    )
    print(
        "  dy =",
        d2_interface[
            "dy"
        ],
    )
    print(
        "  dt =",
        d2_interface[
            "dt"
        ],
    )
    print(
        "  cap =",
        d2_payload.get(
            "stabilization",
            {},
        ).get(
            "path_b_rms_cap"
        ),
    )

    # ========================================================
    # Stats / models
    # ========================================================

    (
        field_mean,
        field_std,
    ) = build_field_stats(
        args.stats
    )

    m10_model = build_m10_model(
        m10_payload,
        field_mean,
        field_std,
        device,
    )

    d2_model = build_d2_model(
        d2_payload,
        field_mean,
        field_std,
        device,
    )

    m6_max_abs = compare_embedded_m6(
        m10_model,
        d2_model,
    )

    print()
    print(
        "========== FAIRNESS AUDIT =========="
    )
    print(
        "embedded_M6_max_abs_diff:",
        f"{m6_max_abs:.12e}",
    )

    if m6_max_abs > 1.0e-7:
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

    if "val" not in split:
        raise KeyError(
            "Split has no val key."
        )

    val_base = RBCDataset(
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
        M10MultiStepParamDataset(
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

    # ========================================================
    # Metric setup
    # ========================================================

    model_names = [
        "M10-2-BOnly",
        "D2-1-DC-BOnly",
    ]

    models = {
        "M10-2-BOnly":
            m10_model,

        "D2-1-DC-BOnly":
            d2_model,
    }

    error_stats = {
        (
            model_name,
            horizon,
        ):
            new_error_bucket()

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

                        update_error_bucket(
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

    summary = make_error_summary(
        error_stats,
        model_names,
        horizons,
    )

    diagnostics = make_diag_summary(
        diag_stats,
        model_names,
        horizons,
    )

    differences = (
        make_difference_table(
            summary
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
        "_d2_minus_m10.csv"
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

    wide = summary.pivot_table(
        index=[
            "model",
            "horizon",
        ],
        columns="field",
        values="rel_l2_percent",
    ).reset_index()

    field_columns = [
        "buoyancy",
        "u_x",
        "u_y",
        "pressure",
        "global",
    ]

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
        "=" * 100
    )
    print(
        "ROLLOUT Rel-L2 (%)"
    )
    print(
        "=" * 100
    )

    print(
        wide.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )

    diff_wide = (
        differences.pivot_table(
            index="horizon",
            columns="field",
            values="rel_l2_diff_pp",
        )
        .reset_index()
    )

    diff_wide = diff_wide[
        [
            "horizon",
        ]
        +
        field_columns
    ]

    print()
    print(
        "=" * 100
    )
    print(
        "D2-1 - M10-2 Rel-L2 difference (percentage points)"
    )
    print(
        "NEGATIVE = D2-1 BETTER"
    )
    print(
        "POSITIVE = D2-1 WORSE"
    )
    print(
        "=" * 100
    )

    print(
        diff_wide.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:+.6f}"
            ),
        )
    )

    print()
    print(
        "=" * 100
    )
    print(
        "PATH-B DIAGNOSTICS"
    )
    print(
        "=" * 100
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
        "Difference:",
        diff_path,
    )

    print()
    print(
        "✅ D2-1 vs M10-2 "
        "VAL H16 evaluation finished."
    )


if __name__ == "__main__":
    main()
