from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from models.operators.fno2d_r3_pde_coupling import (
    R3PDECouplingFNO2d,
)
from physics.canonical_metadata import (
    build_rbc_canonical_metadata,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STATS_PATH = (
    PROJECT_ROOT
    / "data"
    / "stats"
    / "rbc_field_stats_unseen_pr.json"
)

CONTRACT_PATH = (
    PROJECT_ROOT
    / "configs"
    / "r3"
    / "r3_0_architecture_contract.json"
)


@pytest.fixture(scope="module")
def matched_pair():
    metadata = build_rbc_canonical_metadata(
        str(STATS_PATH)
    )

    torch.manual_seed(42)

    canonical = R3PDECouplingFNO2d(
        canonical_metadata=metadata,
        representation_mode="canonical",
        alpha_max=0.25,
        freeze_m6=True,
    )

    naive = R3PDECouplingFNO2d(
        canonical_metadata=metadata,
        representation_mode="naive",
        alpha_max=0.25,
        freeze_m6=True,
    )

    # Force EXACTLY the same M6 initialization.
    naive.load_m6_state_dict(
        canonical.m6.state_dict()
    )

    canonical.eval()
    naive.eval()

    return metadata, naive, canonical


def make_input(metadata):
    torch.manual_seed(1234)

    return torch.randn(
        1,
        16,
        metadata.grid.nx,
        metadata.grid.ny,
        dtype=torch.float32,
    )


def test_r3_contract_matches_model():
    with open(
        CONTRACT_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        contract = json.load(f)

    assert contract["stage"] == "R3-0"

    assert (
        contract["matched_pair"]
        ["r3_1a"]
        ["representation_mode"]
        == "naive"
    )

    assert (
        contract["matched_pair"]
        ["r3_1b"]
        ["representation_mode"]
        == "canonical"
    )

    active_from_contract = {
        name
        for name, cfg
        in contract["active_routing"].items()
        if cfg["active"]
    }

    assert active_from_contract == set(
        R3PDECouplingFNO2d.ACTIVE_TERMS
    )

    assert set(
        R3PDECouplingFNO2d.ACTIVE_TERMS
    ) == {
        "buoyancy_advection",
        "buoyancy_forcing",
    }

    disabled = contract[
        "disabled_in_r3_1"
    ]

    assert all(
        disabled.values()
    )


def test_r3_matched_parameter_topology(
    matched_pair,
):
    _, naive, canonical = matched_pair

    # --------------------------------------------------------
    # Trainable parameter names / counts must be identical.
    # --------------------------------------------------------

    naive_names = (
        naive.trainable_parameter_names()
    )

    canonical_names = (
        canonical.trainable_parameter_names()
    )

    assert naive_names == canonical_names

    assert set(naive_names) == {
        "raw_alpha.buoyancy_advection",
        "raw_alpha.buoyancy_forcing",
    }

    assert (
        naive.trainable_parameter_count()
        ==
        canonical.trainable_parameter_count()
        ==
        2
    )

    # --------------------------------------------------------
    # Frozen M6 structure must be identical.
    # --------------------------------------------------------

    assert (
        naive.frozen_m6_parameter_tensor_count()
        ==
        canonical.frozen_m6_parameter_tensor_count()
    )

    assert all(
        not p.requires_grad
        for p in naive.m6.parameters()
    )

    assert all(
        not p.requires_grad
        for p in canonical.m6.parameters()
    )

    # --------------------------------------------------------
    # Entire learnable state topology must have identical
    # keys, tensor shapes and initialization.
    #
    # Representation metadata is intentionally NOT a
    # learnable state_dict quantity.
    # --------------------------------------------------------

    naive_state = naive.state_dict()
    canonical_state = canonical.state_dict()

    assert list(
        naive_state.keys()
    ) == list(
        canonical_state.keys()
    )

    for key in naive_state:
        assert (
            naive_state[key].shape
            ==
            canonical_state[key].shape
        )

        torch.testing.assert_close(
            naive_state[key],
            canonical_state[key],
            rtol=0.0,
            atol=0.0,
        )


def test_r3_only_representation_metadata_differs(
    matched_pair,
):
    metadata, naive, canonical = (
        matched_pair
    )

    canonical_meta = (
        canonical.compiler.metadata
    )

    naive_meta = (
        naive.compiler.metadata
    )

    # Canonical arm must use the real metadata object.
    assert canonical_meta is metadata

    # Naive arm keeps SAME physical grid/time/parameter semantics.
    assert naive_meta.grid is metadata.grid
    assert naive_meta.time is metadata.time

    assert (
        naive_meta.parameter_order
        ==
        metadata.parameter_order
    )

    # But naive normalization must be identity:
    #
    # mean = 0
    # std + eps = 1
    #
    eps = naive_meta.normalization.eps

    for field in metadata.field_order:

        assert abs(
            naive_meta.normalization.mean_for(
                field
            )
        ) <= 1.0e-12

        assert abs(
            (
                naive_meta.normalization.std_for(
                    field
                )
                + eps
            )
            - 1.0
        ) <= 1.0e-12

    # Verify the canonical metadata is genuinely not
    # the same normalization interface.
    normalization_diff_found = False

    for field in metadata.field_order:

        mean_diff = abs(
            metadata.normalization.mean_for(field)
            -
            naive_meta.normalization.mean_for(field)
        )

        std_diff = abs(
            metadata.normalization.std_for(field)
            -
            naive_meta.normalization.std_for(field)
        )

        if (
            mean_diff > 1.0e-8
            or std_diff > 1.0e-8
        ):
            normalization_diff_found = True

    assert normalization_diff_found


def test_r3_epoch0_both_arms_equal_same_m6(
    matched_pair,
):
    metadata, naive, canonical = (
        matched_pair
    )

    x_norm = make_input(metadata)

    with torch.no_grad():

        naive_out, naive_comp = naive(
            x_norm,
            return_components=True,
        )

        canonical_out, canonical_comp = (
            canonical(
                x_norm,
                return_components=True,
            )
        )

        m6_out = canonical.m6(
            x_norm
        )

    # --------------------------------------------------------
    # Epoch 0:
    #
    # R3-1a == R3-1b == pure M6
    # --------------------------------------------------------

    assert float(
        (
            naive_out - m6_out
        ).abs().max()
    ) <= 1.0e-7

    assert float(
        (
            canonical_out - m6_out
        ).abs().max()
    ) <= 1.0e-7

    assert float(
        (
            naive_out - canonical_out
        ).abs().max()
    ) <= 1.0e-7

    # Physics residual must be exactly zero because
    # every raw gate starts at zero.
    assert float(
        naive_comp[
            "physics_residual_norm"
        ].abs().max()
    ) == 0.0

    assert float(
        canonical_comp[
            "physics_residual_norm"
        ].abs().max()
    ) == 0.0

    for term_name in (
        R3PDECouplingFNO2d.ACTIVE_TERMS
    ):

        assert float(
            naive_comp[
                "gate_values"
            ][term_name]
        ) == 0.0

        assert float(
            canonical_comp[
                "gate_values"
            ][term_name]
        ) == 0.0


def test_r3_representation_switch_changes_only_pde_signal(
    matched_pair,
):
    metadata, naive, canonical = (
        matched_pair
    )

    x_norm = make_input(metadata)

    difference_found = False

    for term_name in (
        R3PDECouplingFNO2d.ACTIVE_TERMS
    ):

        canonical_term = (
            canonical.compiler
            .compile_from_history(
                term_name,
                x_norm,
            )
        )

        naive_term = (
            naive.compiler
            .compile_from_history(
                term_name,
                x_norm,
            )
        )

        assert (
            canonical_term.target_delta_norm
            is not None
        )

        assert (
            naive_term.target_delta_norm
            is not None
        )

        assert (
            canonical_term.target_delta_norm.shape
            ==
            naive_term.target_delta_norm.shape
            ==
            (
                1,
                metadata.grid.nx,
                metadata.grid.ny,
            )
        )

        max_diff = float(
            (
                canonical_term.target_delta_norm
                -
                naive_term.target_delta_norm
            )
            .abs()
            .max()
        )

        if max_diff > 1.0e-7:
            difference_found = True

    # The architecture is identical, but the physical
    # representation interface must actually have an
    # observable effect on the compiled PDE signal.
    assert difference_found
