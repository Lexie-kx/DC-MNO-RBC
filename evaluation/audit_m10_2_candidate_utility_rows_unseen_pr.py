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
# Load the CLOSED M10-2a utility/reliability generator.
#
# We do NOT copy/edit its equations, features, datasets,
# utility target, or generate_rows implementation.
# ============================================================

ORIGINAL_PATH = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "audit_m10_2a_pathb_reliability.py",
)

spec = importlib.util.spec_from_file_location(
    "m10_2a_original_generator",
    ORIGINAL_PATH,
)

if spec is None or spec.loader is None:
    raise RuntimeError(
        f"Cannot load original generator: "
        f"{ORIGINAL_PATH}"
    )

original = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    original
)


# ============================================================
# Formal M10-2 B-only architecture
# ============================================================

from models.operators.fno2d_m10_2_bonly_rmscap import (
    M10BOnlyRMSCapFNO2d,
)


EXPECTED_M10_2_SHA = (
    "6854ab6bce91184c2df85e8a9e7e4793"
    "4c1475c2b926a011db6dff857c165db3"
)


# ============================================================
# Exact B-only checkpoint builder
# ============================================================

def build_m10_2_bonly_stateparam(
    checkpoint_path,
    field_mean,
    field_std,
    args,
    device,
):

    # Reuse the audited checkpoint loader already used
    # throughout the M10 evaluation chain.
    payload, state = (
        original.audit.base.load_checkpoint(
            checkpoint_path,
            device,
        )
    )

    model = M10BOnlyRMSCapFNO2d(
        field_mean=field_mean,
        field_std=field_std,
        mode="stateparam",
        dx=args.dx,
        dy=args.dy,
        path_b_rms_cap=(
            args.path_b_rms_cap
        ),
        path_b_rms_eps=(
            args.path_b_rms_eps
        ),
        in_channels=16,
        out_channels=4,
        modes1=16,
        modes2=16,
        width=32,
        alpha_max=args.alpha_max,
        conditioner_hidden=(
            args.conditioner_hidden
        ),
        freeze_m6=True,
    ).to(
        device
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    model.eval()

    # ========================================================
    # Hard formal-candidate architecture checks
    # ========================================================

    if (
        model.base_raw_alpha_a
        .requires_grad
    ):
        raise RuntimeError(
            "Path-A gate must be frozen."
        )

    alpha_a_base = float(
        model.base_raw_alpha_a
        .detach()
        .cpu()
        .item()
    )

    if alpha_a_base != 0.0:
        raise RuntimeError(
            "Path-A output gate must be "
            f"exactly zero, got {alpha_a_base}"
        )

    trainable_names = [
        name
        for name, parameter
        in model.named_parameters()
        if parameter.requires_grad
    ]

    expected_trainable = {
        "base_raw_alpha_b",
        "conditioner.0.weight",
        "conditioner.0.bias",
        "conditioner.2.weight",
        "conditioner.2.bias",
    }

    if (
        set(trainable_names)
        !=
        expected_trainable
    ):
        raise RuntimeError(
            "Unexpected trainable parameters:\n"
            f"{trainable_names}"
        )

    trainable_numel = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    if trainable_numel != 386:
        raise RuntimeError(
            "Unexpected M10-2 trainable "
            f"numel={trainable_numel}"
        )

    frozen_m6 = sum(
        1
        for parameter
        in model.m6.parameters()
        if not parameter.requires_grad
    )

    print()
    print(
        "========== M10-2 B-ONLY "
        "ARCHITECTURE CONTRACT =========="
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
        alpha_a_base,
    )

    print(
        "FROZEN_M6_PARAMETER_TENSORS:",
        frozen_m6,
    )

    print(
        "✅ M10-2 B-only architecture "
        "contract PASS"
    )

    return (
        model,
        payload,
    )


# ============================================================
# Runtime patch 1:
#
# The original generator calls:
#
#   original.audit.rmscap_eval.build_rmscap_stateparam(...)
#
# Replace ONLY that builder.
# ============================================================

original.audit.rmscap_eval.build_rmscap_stateparam = (
    build_m10_2_bonly_stateparam
)


# ============================================================
# Runtime patch 2:
#
# Original main() still calls its M10-1a SHA audit.
# Keep all split/stats audits untouched.
# Only redirect the M10 model checkpoint audit.
# ============================================================

# ============================================================
# UNSEEN-PR formal provenance lock
#
# The reused closed M10-2a generator was originally frozen
# for unseen-Ra. For this formal unseen-Pr replication we keep
# the SHA checks ACTIVE, but redirect the original module-level
# split/stats expectations to the predeclared unseen-Pr assets.
# ============================================================

original.EXPECTED_SPLIT_SHA = (
    "ca4f1707c913c880b33398bdf17906ae"
    "70ea4dba77662a76d12de5e319cf86be"
)

original.EXPECTED_STATS_SHA = (
    "299828fb4c986f54493a552fdea8871e"
    "114fc6dd0b756c45dde0117340b54233"
)


_original_require_sha = (
    original.require_sha
)


def formal_require_sha(
    path,
    expected,
    label,
):

    if label == "M10_1A":

        actual = (
            original.sha256_file(
                path
            )
        )

        print(
            "M10_2_BONLY_SHA256:",
            actual,
        )

        if (
            actual
            !=
            EXPECTED_M10_2_SHA
        ):
            raise RuntimeError(
                "M10-2 B-only checkpoint "
                "SHA mismatch.\n"
                f"expected="
                f"{EXPECTED_M10_2_SHA}\n"
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
# CLI compatibility
#
# Original generator expects:
#   --m10_1a_checkpoint
#
# User-facing formal command uses:
#   --m10_2_checkpoint
#
# Map ONLY the argument name before calling original.main().
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
        "=" * 110
    )

    print(
        "M10-2 FORMAL B-ONLY "
        "UTILITY-ROW GENERATOR"
    )

    print(
        "=" * 110
    )

    print(
        "📌 Formal candidate checkpoint: "
        "fair-trained B-only StateParam RMSCap"
    )

    print(
        "📌 Path A output injection: OFF"
    )

    print(
        "📌 Utility/reliability equations: "
        "UNCHANGED from closed M10-2a"
    )

    print(
        "📌 Feature definitions: UNCHANGED"
    )

    print(
        "📌 TRAIN + VAL only"
    )

    print(
        "📌 TEST SPLIT IS NOT ACCESSED"
    )

    remap_cli()

    original.main()
