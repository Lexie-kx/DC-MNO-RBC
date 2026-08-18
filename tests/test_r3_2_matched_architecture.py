from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from models.operators.fno2d_r3_pde_coupling import (
    R3PDECouplingFNO2d,
)

from models.operators.fno2d_r3_state_adaptive import (
    R3StateAdaptiveConditioner,
    R3StateAdaptivePDECouplingFNO2d,
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
    / "r3_2_state_adaptive_contract.json"
)


FORMAL_ALPHA_MAX_BY_TERM = {
    "buoyancy_advection":
        0.021792202037947142,

    "buoyancy_forcing":
        1.2050772196677188e-05,
}


# ============================================================
# Synthetic frozen R3-1b parent
#
# IMPORTANT:
# raw_alpha is deliberately NONZERO.
#
# This makes epoch-0 R3-2 reproduction a stronger test than
# comparing three pure-M6 models.
# ============================================================

PARENT_RAW_ALPHA = {
    "buoyancy_advection": 0.31,
    "buoyancy_forcing": -0.27,
}


def build_parent_and_matched_children():
    metadata = build_rbc_canonical_metadata(
        str(STATS_PATH)
    )

    # --------------------------------------------------------
    # Frozen synthetic R3-1b parent.
    # --------------------------------------------------------

    torch.manual_seed(1001)

    parent = R3PDECouplingFNO2d(
        canonical_metadata=metadata,
        representation_mode="canonical",
        alpha_max_by_term=(
            FORMAL_ALPHA_MAX_BY_TERM
        ),
        freeze_m6=True,
    )

    with torch.no_grad():

        for (
            term_name,
            raw_value,
        ) in PARENT_RAW_ALPHA.items():

            parent.raw_alpha[
                term_name
            ].fill_(
                raw_value
            )

    parent.eval()

    parent_state = (
        parent.state_dict()
    )

    # --------------------------------------------------------
    # R3-2a and R3-2b:
    #
    # reset RNG before EACH construction so the complete child
    # initialization, especially conditioner tensors, is exactly
    # matched before loading the same parent.
    # --------------------------------------------------------

    torch.manual_seed(4242)

    paramonly = (
        R3StateAdaptivePDECouplingFNO2d(
            canonical_metadata=metadata,
            adaptation_mode="paramonly",
            alpha_max_by_term=(
                FORMAL_ALPHA_MAX_BY_TERM
            ),
        )
    )

    torch.manual_seed(4242)

    stateparam = (
        R3StateAdaptivePDECouplingFNO2d(
            canonical_metadata=metadata,
            adaptation_mode="stateparam",
            alpha_max_by_term=(
                FORMAL_ALPHA_MAX_BY_TERM
            ),
        )
    )

    paramonly.load_parent_r3_1b_state_dict(
        parent_state
    )

    stateparam.load_parent_r3_1b_state_dict(
        parent_state
    )

    parent.eval()
    paramonly.eval()
    stateparam.eval()

    return (
        metadata,
        parent,
        paramonly,
        stateparam,
    )


@pytest.fixture(scope="module")
def matched_family():

    return build_parent_and_matched_children()


def make_input(
    metadata,
    batch_size=1,
):

    torch.manual_seed(9876)

    return torch.randn(
        batch_size,
        16,
        metadata.grid.nx,
        metadata.grid.ny,
        dtype=torch.float32,
    )


def make_params(
    batch_size=1,
):

    # [log10(Ra), log10(Pr)]
    #
    # Deliberately use nonzero / nonidentical values when
    # batch_size > 1 so parameter features are actually tested.
    base = torch.tensor(
        [
            [6.0, 0.0],
            [7.0, -0.30103],
            [8.0, 0.30103],
        ],
        dtype=torch.float32,
    )

    if batch_size > len(base):
        raise ValueError(
            "Test helper supports batch_size <= 3."
        )

    return base[
        :batch_size
    ].clone()


# ============================================================
# 1. Frozen JSON contract agrees with implementation
# ============================================================

def test_r3_2_contract_matches_model():

    with open(
        CONTRACT_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        contract = json.load(
            f
        )

    assert (
        contract[
            "contract_status"
        ]
        ==
        "LOCKED_BEFORE_FORMAL_TRAINING"
    )

    assert (
        contract[
            "parent"
        ][
            "required_parent_representation"
        ]
        ==
        "canonical"
    )

    assert (
        contract[
            "causal_decomposition"
        ][
            "primary_contrast"
        ]
        ==
        "R3-2b - R3-2a"
    )

    assert (
        contract[
            "causal_decomposition"
        ][
            "primary_contrast_interpretation"
        ]
        ==
        "value of current-state information"
    )

    assert (
        contract[
            "conditioner"
        ][
            "input_dim"
        ]
        ==
        R3StateAdaptiveConditioner.INPUT_DIM
        ==
        10
    )

    assert (
        tuple(
            contract[
                "conditioner"
            ][
                "hidden_dims"
            ]
        )
        ==
        R3StateAdaptiveConditioner.HIDDEN_DIMS
        ==
        (32, 16)
    )

    assert (
        contract[
            "conditioner"
        ][
            "output_dim"
        ]
        ==
        R3StateAdaptiveConditioner.OUTPUT_DIM
        ==
        2
    )

    assert (
        tuple(
            contract[
                "conditioner_features"
            ][
                "feature_order"
            ]
        )
        ==
        R3StateAdaptivePDECouplingFNO2d
        .FEATURE_NAMES
    )

    assert (
        contract[
            "trainable_parameter_policy"
        ][
            "only_conditioner_trainable"
        ]
        is True
    )

    assert (
        contract[
            "frozen_r3_1_structure"
        ][
            "formal_alpha_max_by_term"
        ]
        ==
        FORMAL_ALPHA_MAX_BY_TERM
    )

    assert (
        contract[
            "test_embargo"
        ][
            "test_access_during_formal_training"
        ]
        is False
    )


# ============================================================
# 2. Capacity / initialization must be exactly matched
# ============================================================

def test_r3_2_matched_parameter_topology_and_initialization(
    matched_family,
):

    (
        _,
        _,
        paramonly,
        stateparam,
    ) = matched_family

    # --------------------------------------------------------
    # Trainable topology:
    # ONLY the same conditioner parameters.
    # --------------------------------------------------------

    paramonly_names = (
        paramonly.trainable_parameter_names()
    )

    stateparam_names = (
        stateparam.trainable_parameter_names()
    )

    assert (
        paramonly_names
        ==
        stateparam_names
    )

    expected_names = {
        "conditioner.net.0.weight",
        "conditioner.net.0.bias",
        "conditioner.net.2.weight",
        "conditioner.net.2.bias",
        "conditioner.net.4.weight",
        "conditioner.net.4.bias",
    }

    assert (
        set(paramonly_names)
        ==
        expected_names
    )

    assert (
        paramonly.trainable_parameter_count()
        ==
        stateparam.trainable_parameter_count()
        ==
        914
    )

    assert (
        paramonly.conditioner_parameter_count()
        ==
        stateparam.conditioner_parameter_count()
        ==
        914
    )

    # --------------------------------------------------------
    # Every conditioner tensor must start identically.
    # --------------------------------------------------------

    p_state = (
        paramonly.conditioner.state_dict()
    )

    s_state = (
        stateparam.conditioner.state_dict()
    )

    assert (
        list(p_state.keys())
        ==
        list(s_state.keys())
    )

    for key in p_state:

        assert torch.equal(
            p_state[key],
            s_state[key],
        )

    # --------------------------------------------------------
    # Final layer MUST be exact zero.
    # --------------------------------------------------------

    p_final = (
        paramonly.conditioner.net[-1]
    )

    s_final = (
        stateparam.conditioner.net[-1]
    )

    assert (
        torch.count_nonzero(
            p_final.weight
        ).item()
        ==
        0
    )

    assert (
        torch.count_nonzero(
            p_final.bias
        ).item()
        ==
        0
    )

    assert torch.equal(
        p_final.weight,
        s_final.weight,
    )

    assert torch.equal(
        p_final.bias,
        s_final.bias,
    )

    # --------------------------------------------------------
    # Frozen parent gates / M6.
    # --------------------------------------------------------

    expected_frozen_raw = {
        "raw_alpha.buoyancy_advection",
        "raw_alpha.buoyancy_forcing",
    }

    assert (
        set(
            paramonly
            .frozen_parent_raw_alpha_names()
        )
        ==
        expected_frozen_raw
    )

    assert (
        set(
            stateparam
            .frozen_parent_raw_alpha_names()
        )
        ==
        expected_frozen_raw
    )

    assert all(
        not parameter.requires_grad
        for parameter
        in paramonly.raw_alpha.parameters()
    )

    assert all(
        not parameter.requires_grad
        for parameter
        in stateparam.raw_alpha.parameters()
    )

    assert all(
        not parameter.requires_grad
        for parameter
        in paramonly.m6.parameters()
    )

    assert all(
        not parameter.requires_grad
        for parameter
        in stateparam.m6.parameters()
    )


# ============================================================
# 3. Both children inherit EXACTLY the same frozen R3-1b parent
# ============================================================

def test_r3_2_parent_state_is_loaded_exactly(
    matched_family,
):

    (
        _,
        parent,
        paramonly,
        stateparam,
    ) = matched_family

    parent_state = (
        parent.state_dict()
    )

    param_state = (
        paramonly.state_dict()
    )

    state_state = (
        stateparam.state_dict()
    )

    for (
        key,
        parent_tensor,
    ) in parent_state.items():

        assert key in param_state
        assert key in state_state

        assert torch.equal(
            param_state[key],
            parent_tensor,
        )

        assert torch.equal(
            state_state[key],
            parent_tensor,
        )

    # The fixed capacity dictionary must also remain matched.
    assert (
        parent.alpha_max_by_term
        ==
        paramonly.alpha_max_by_term
        ==
        stateparam.alpha_max_by_term
        ==
        FORMAL_ALPHA_MAX_BY_TERM
    )

    # Confirm this is a genuinely NONZERO-gate parent.
    for term_name in (
        parent.ACTIVE_TERMS
    ):

        assert float(
            parent.gate_value(
                term_name
            )
            .detach()
            .abs()
        ) > 0.0


# ============================================================
# 4. The ONLY arm input difference is features[0:8] masking
# ============================================================

def test_r3_2_only_state_features_are_masked(
    matched_family,
):

    (
        metadata,
        _,
        paramonly,
        stateparam,
    ) = matched_family

    x_norm = make_input(
        metadata,
        batch_size=2,
    )

    params = make_params(
        batch_size=2,
    )

    with torch.no_grad():

        (
            _,
            p_comp,
        ) = paramonly(
            x_norm,
            params=params,
            return_components=True,
        )

        (
            _,
            s_comp,
        ) = stateparam(
            x_norm,
            params=params,
            return_components=True,
        )

    p_full = (
        p_comp[
            "full_conditioner_features"
        ]
    )

    s_full = (
        s_comp[
            "full_conditioner_features"
        ]
    )

    p_input = (
        p_comp[
            "conditioner_input"
        ]
    )

    s_input = (
        s_comp[
            "conditioner_input"
        ]
    )

    # --------------------------------------------------------
    # Both arms first construct the SAME real feature vector.
    # --------------------------------------------------------

    assert torch.equal(
        p_full,
        s_full,
    )

    assert (
        p_full.shape
        ==
        s_full.shape
        ==
        (2, 10)
    )

    # Real state features must not accidentally all be zero,
    # otherwise masking would be a vacuous test.
    assert (
        torch.count_nonzero(
            s_full[:, :8]
        ).item()
        >
        0
    )

    # --------------------------------------------------------
    # StateParam sees the full vector exactly.
    # --------------------------------------------------------

    assert torch.equal(
        s_input,
        s_full,
    )

    # --------------------------------------------------------
    # ParamOnly:
    # first 8 features are EXACT zero.
    # --------------------------------------------------------

    assert (
        torch.count_nonzero(
            p_input[:, :8]
        ).item()
        ==
        0
    )

    # Final two features remain exactly the same.
    assert torch.equal(
        p_input[:, 8:],
        s_input[:, 8:],
    )

    assert torch.equal(
        p_input[:, 8:],
        params,
    )

    # There must be no difference outside the intended mask.
    assert torch.equal(
        p_input[:, 8:],
        p_full[:, 8:],
    )


# ============================================================
# 5. Zero-init conditioner => zero delta_raw and parent gates
# ============================================================

def test_r3_2_epoch0_delta_raw_and_gates_equal_parent(
    matched_family,
):

    (
        metadata,
        parent,
        paramonly,
        stateparam,
    ) = matched_family

    x_norm = make_input(
        metadata,
        batch_size=2,
    )

    params = make_params(
        batch_size=2,
    )

    with torch.no_grad():

        _, p_comp = paramonly(
            x_norm,
            params=params,
            return_components=True,
        )

        _, s_comp = stateparam(
            x_norm,
            params=params,
            return_components=True,
        )

    # Exact zero by construction.
    assert (
        torch.count_nonzero(
            p_comp[
                "delta_raw"
            ]
        ).item()
        ==
        0
    )

    assert (
        torch.count_nonzero(
            s_comp[
                "delta_raw"
            ]
        ).item()
        ==
        0
    )

    assert torch.equal(
        p_comp[
            "delta_raw"
        ],
        s_comp[
            "delta_raw"
        ],
    )

    for term_name in (
        parent.ACTIVE_TERMS
    ):

        parent_alpha = (
            parent.gate_value(
                term_name
            )
        )

        p_parent_alpha = (
            p_comp[
                "parent_gate_values"
            ][
                term_name
            ]
        )

        s_parent_alpha = (
            s_comp[
                "parent_gate_values"
            ][
                term_name
            ]
        )

        assert torch.equal(
            p_parent_alpha,
            parent_alpha,
        )

        assert torch.equal(
            s_parent_alpha,
            parent_alpha,
        )

        p_effective = (
            p_comp[
                "effective_gate_values"
            ][
                term_name
            ]
        )

        s_effective = (
            s_comp[
                "effective_gate_values"
            ][
                term_name
            ]
        )

        assert torch.equal(
            p_effective,
            parent_alpha.expand_as(
                p_effective
            ),
        )

        assert torch.equal(
            s_effective,
            parent_alpha.expand_as(
                s_effective
            ),
        )

        assert torch.equal(
            p_effective,
            s_effective,
        )


# ============================================================
# 6. Strong epoch-0 identity:
#
# R3-2a == R3-2b == frozen NONZERO-gate R3-1b parent
# ============================================================

def test_r3_2_epoch0_exactly_reproduces_r3_1b(
    matched_family,
):

    (
        metadata,
        parent,
        paramonly,
        stateparam,
    ) = matched_family

    x_norm = make_input(
        metadata,
        batch_size=1,
    )

    params = make_params(
        batch_size=1,
    )

    with torch.no_grad():

        (
            parent_out,
            parent_comp,
        ) = parent(
            x_norm,
            params=params,
            return_components=True,
        )

        (
            p_out,
            p_comp,
        ) = paramonly(
            x_norm,
            params=params,
            return_components=True,
        )

        (
            s_out,
            s_comp,
        ) = stateparam(
            x_norm,
            params=params,
            return_components=True,
        )

    # --------------------------------------------------------
    # Exact model-output identity.
    # --------------------------------------------------------

    assert torch.equal(
        p_out,
        parent_out,
    )

    assert torch.equal(
        s_out,
        parent_out,
    )

    assert torch.equal(
        p_out,
        s_out,
    )

    # --------------------------------------------------------
    # Exact next-state identity too.
    # --------------------------------------------------------

    current_norm = (
        x_norm[
            :,
            -4:,
            :,
            :,
        ]
    )

    parent_next = (
        current_norm
        +
        parent_out
    )

    p_next = (
        current_norm
        +
        p_out
    )

    s_next = (
        current_norm
        +
        s_out
    )

    assert torch.equal(
        p_next,
        parent_next,
    )

    assert torch.equal(
        s_next,
        parent_next,
    )

    # --------------------------------------------------------
    # Same base M6 trajectory.
    # --------------------------------------------------------

    assert torch.equal(
        p_comp[
            "base_delta_norm"
        ],
        parent_comp[
            "base_delta_norm"
        ],
    )

    assert torch.equal(
        s_comp[
            "base_delta_norm"
        ],
        parent_comp[
            "base_delta_norm"
        ],
    )

    # --------------------------------------------------------
    # Same canonical signals.
    # --------------------------------------------------------

    for term_name in (
        parent.ACTIVE_TERMS
    ):

        assert torch.equal(
            p_comp[
                "compiled_terms"
            ][
                term_name
            ],
            parent_comp[
                "compiled_terms"
            ][
                term_name
            ],
        )

        assert torch.equal(
            s_comp[
                "compiled_terms"
            ][
                term_name
            ],
            parent_comp[
                "compiled_terms"
            ][
                term_name
            ],
        )

        assert torch.equal(
            p_comp[
                "compiled_terms"
            ][
                term_name
            ],
            s_comp[
                "compiled_terms"
            ][
                term_name
            ],
        )

        # Same actual term correction.
        assert torch.equal(
            p_comp[
                "term_corrections"
            ][
                term_name
            ],
            parent_comp[
                "term_corrections"
            ][
                term_name
            ],
        )

        assert torch.equal(
            s_comp[
                "term_corrections"
            ][
                term_name
            ],
            parent_comp[
                "term_corrections"
            ][
                term_name
            ],
        )

    # Same total physics residual.
    assert torch.equal(
        p_comp[
            "physics_residual_norm"
        ],
        parent_comp[
            "physics_residual_norm"
        ],
    )

    assert torch.equal(
        s_comp[
            "physics_residual_norm"
        ],
        parent_comp[
            "physics_residual_norm"
        ],
    )


# ============================================================
# 7. Parent loader must be strict
# ============================================================

def test_r3_2_parent_loader_rejects_key_mismatch():

    (
        _,
        parent,
        paramonly,
        _,
    ) = build_parent_and_matched_children()

    good_state = dict(
        parent.state_dict()
    )

    # Extra parent key must fail.
    extra_state = dict(
        good_state
    )

    extra_state[
        "forbidden.extra_tensor"
    ] = torch.zeros(
        1
    )

    with pytest.raises(
        RuntimeError,
        match="does not match",
    ):

        paramonly.load_parent_r3_1b_state_dict(
            extra_state
        )

    # Missing parent key must fail.
    missing_state = dict(
        good_state
    )

    removed_key = next(
        iter(
            missing_state.keys()
        )
    )

    del missing_state[
        removed_key
    ]

    with pytest.raises(
        RuntimeError,
        match="does not match",
    ):

        paramonly.load_parent_r3_1b_state_dict(
            missing_state
        )
