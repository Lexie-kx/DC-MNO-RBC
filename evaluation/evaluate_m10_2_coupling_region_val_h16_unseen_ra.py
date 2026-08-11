import hashlib
import importlib.util
import os
import sys

import torch


# ============================================================
# Project
# ============================================================

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
# Reuse the CLOSED H16 evaluator exactly.
# ============================================================

ORIGINAL_PATH = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "evaluate_m10_2_coupling_region_core_h16_unseen_ra.py",
)

spec = importlib.util.spec_from_file_location(
    "m10_2_old_h16",
    ORIGINAL_PATH,
)

if spec is None or spec.loader is None:
    raise RuntimeError(
        f"Cannot load original H16 evaluator: "
        f"{ORIGINAL_PATH}"
    )

original = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    original
)


# ============================================================
# Formal fair-rebase model
# ============================================================

from models.operators.fno2d_m10_2_bonly_rmscap import (
    M10BOnlyRMSCapFNO2d,
)


EXPECTED_M10_2_SHA = (
    "450a2dcff725154e484f746d2dc883429c10aa5036fadf63944bb4fcb7bd8d76"
)

EXPECTED_TRAIN_UTILITY_SHA = (
    "ab8b938270c2840775d3f393f5015fbcaf4fb51c8445a7f9614d530a958586ad"
)

EXPECTED_VAL_UTILITY_SHA = (
    "e82d0a340d57a8a56e43f8eb498b97b35b32ab6dbc4ff38c55dd90206ccfd02d"
)


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


# ============================================================
# Formal B-only checkpoint builder
# ============================================================

def build_m10_2_bonly_stateparam(
    checkpoint_path,
    field_mean,
    field_std,
    args,
    device,
):

    payload = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if not isinstance(payload, dict):
        raise RuntimeError(
            "M10-2 checkpoint payload must be dict."
        )

    if "model_state_dict" not in payload:
        raise RuntimeError(
            "M10-2 checkpoint missing model_state_dict."
        )

    model = M10BOnlyRMSCapFNO2d(
        field_mean=field_mean,
        field_std=field_std,
        mode="stateparam",
        dx=args.dx,
        dy=args.dy,
        path_b_rms_cap=args.path_b_rms_cap,
        path_b_rms_eps=args.path_b_rms_eps,
        in_channels=16,
        out_channels=4,
        modes1=16,
        modes2=16,
        width=32,
        alpha_max=args.alpha_max,
        conditioner_hidden=args.conditioner_hidden,
        freeze_m6=True,
    ).to(
        device
    )

    model.load_state_dict(
        payload["model_state_dict"],
        strict=True,
    )

    model.eval()

    # --------------------------------------------------------
    # Formal architecture contract
    # --------------------------------------------------------

    if model.base_raw_alpha_a.requires_grad:
        raise RuntimeError(
            "Path-A gate unexpectedly trainable."
        )

    alpha_a = float(
        model.base_raw_alpha_a
        .detach()
        .cpu()
        .item()
    )

    if alpha_a != 0.0:
        raise RuntimeError(
            f"Path-A output gate != 0: {alpha_a}"
        )

    trainable_names = [
        name
        for name, p
        in model.named_parameters()
        if p.requires_grad
    ]

    expected_names = {
        "base_raw_alpha_b",
        "conditioner.0.weight",
        "conditioner.0.bias",
        "conditioner.2.weight",
        "conditioner.2.bias",
    }

    if set(trainable_names) != expected_names:
        raise RuntimeError(
            "Unexpected trainable parameters:\n"
            f"{trainable_names}"
        )

    trainable_numel = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    if trainable_numel != 386:
        raise RuntimeError(
            f"TRAINABLE_NUMEL={trainable_numel}, "
            "expected 386."
        )

    frozen_m6 = sum(
        1
        for p in model.m6.parameters()
        if not p.requires_grad
    )

    print()
    print(
        "========== FAIR B-ONLY MODEL CONTRACT =========="
    )
    print(
        "TRAINABLE_NAMES:",
        trainable_names,
    )
    print(
        "TRAINABLE_NUMEL:",
        trainable_numel,
    )
    print(
        "PATH_A_OUTPUT_GATE:",
        alpha_a,
    )
    print(
        "FROZEN_M6_PARAMETER_TENSORS:",
        frozen_m6,
    )
    print(
        "✅ Fair B-only model contract PASS"
    )

    return (
        model,
        payload,
    )


# ============================================================
# Runtime replacement:
#
# OLD H16:
#   build_rmscap_stateparam(old M10-1a)
#
# FAIR H16:
#   build_m10_2_bonly_stateparam(new B-only checkpoint)
#
# Nothing else changes.
# ============================================================

original.audit.rmscap_eval.build_rmscap_stateparam = (
    build_m10_2_bonly_stateparam
)


# ============================================================
# Provenance lock
# ============================================================

_original_require_sha = (
    original.require_sha
)


def formal_require_sha(
    path,
    expected,
    label,
):

    replacements = {
        "M10_1A":
            EXPECTED_M10_2_SHA,

        "TRAIN_UTILITY_CSV":
            EXPECTED_TRAIN_UTILITY_SHA,

        "VAL_UTILITY_CSV":
            EXPECTED_VAL_UTILITY_SHA,
    }

    if label in replacements:

        actual = sha256_file(
            path
        )

        wanted = replacements[
            label
        ]

        print(
            f"FAIR_{label}_SHA256:",
            actual,
        )

        if actual != wanted:
            raise RuntimeError(
                f"{label} SHA mismatch.\n"
                f"expected={wanted}\n"
                f"actual={actual}"
            )

        return

    return _original_require_sha(
        path,
        expected,
        label,
    )


original.require_sha = (
    formal_require_sha
)


# ============================================================
# User-facing CLI alias:
#
#   --m10_2_checkpoint
#
# maps internally to the legacy evaluator's
#
#   --m10_1a_checkpoint
#
# This is ONLY an implementation alias.
# ============================================================

def remap_cli():

    mapped = []

    for token in sys.argv:

        if token == "--m10_2_checkpoint":
            mapped.append(
                "--m10_1a_checkpoint"
            )
        else:
            mapped.append(
                token
            )

    sys.argv[:] = mapped


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 115
    )

    print(
        "M10-2 FAIR-REBASE FORMAL CANDIDATE "
        "LONG-ROLLOUT VALIDATION"
    )

    print(
        "=" * 115
    )

    print(
        "📌 Candidate: fair-trained "
        "StateParam B-only RMSCap"
    )

    print(
        "📌 Path A output injection: OFF"
    )

    print(
        "📌 Utility labels: regenerated "
        "from fair B-only checkpoint"
    )

    print(
        "📌 Utility classifier: TRAIN-only"
    )

    print(
        "📌 Closed-loop evaluation: VAL only"
    )

    print(
        "📌 H4-derived utility gate -> H16 "
        "without refit"
    )

    print(
        "📌 Threshold tuning: NO"
    )

    print(
        "📌 TEST SPLIT IS NOT ACCESSED"
    )

    print(
        "=" * 115
    )

    remap_cli()

    original.main()
