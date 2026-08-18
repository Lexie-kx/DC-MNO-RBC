import json
import math

import pytest
import torch

from physics.canonical_metadata import (
    build_rbc_canonical_metadata,
    rbc_transport_coefficients,
)
from physics.canonical_compiler import (
    CanonicalPDECompiler,
)
from physics.derivatives import (
    grad_x_periodic,
    grad_y_nonperiodic,
    laplacian,
)
from physics.rbc_terms import (
    RBC_TERM_REGISTRY,
    TermRole,
    TermSemantic,
)


def _write_dummy_stats(path):
    stats = {
        "buoyancy": {
            "mean": 1.0,
            "std": 2.0,
        },
        "u_x": {
            "mean": 0.2,
            "std": 3.0,
        },
        "u_y": {
            "mean": -0.3,
            "std": 4.0,
        },
        "pressure": {
            "mean": 5.0,
            "std": 6.0,
        },
    }

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            stats,
            f,
            indent=2,
        )


def _build_metadata(tmp_path):
    path = tmp_path / "stats.json"

    _write_dummy_stats(
        path
    )

    return build_rbc_canonical_metadata(
        str(path)
    )


def _xy_grid(metadata):
    grid = metadata.grid

    x = (
        torch.arange(
            grid.nx,
            dtype=torch.float64,
        )
        *
        grid.dx
    )

    y = (
        torch.arange(
            grid.ny,
            dtype=torch.float64,
        )
        *
        grid.dy
    )

    return torch.meshgrid(
        x,
        y,
        indexing="ij",
    )


def _canonical_state(metadata):
    xx, yy = _xy_grid(
        metadata
    )

    k = (
        2.0
        *
        math.pi
        /
        metadata.grid.x_domain_length
    )

    b = (
        torch.sin(k * xx)
        +
        0.25 * yy ** 2
    )

    ux = (
        0.30
        +
        0.10 * torch.cos(k * xx)
        +
        0.05 * yy
    )

    uy = (
        -0.20
        +
        0.08 * torch.sin(k * xx)
        +
        0.07 * yy ** 2
    )

    p = (
        2.0 * torch.cos(k * xx)
        +
        0.5 * yy ** 2
    )

    return torch.stack(
        [
            b,
            ux,
            uy,
            p,
        ],
        dim=0,
    ).unsqueeze(0)


def test_registry_has_exactly_seven_terms():
    assert tuple(
        RBC_TERM_REGISTRY.keys()
    ) == (
        "buoyancy_advection",
        "buoyancy_forcing",
        "momentum_advection",
        "viscosity",
        "buoyancy_diffusion",
        "pressure_gradient",
        "divergence",
    )


def test_registry_semantics_and_targets():
    adv_b = RBC_TERM_REGISTRY[
        "buoyancy_advection"
    ]

    assert adv_b.semantic == TermSemantic.RATE
    assert adv_b.role == TermRole.CROSS_FIELD
    assert adv_b.targets == (
        "buoyancy",
    )
    assert adv_b.requires_dt is True

    div = RBC_TERM_REGISTRY[
        "divergence"
    ]

    assert div.semantic == TermSemantic.CONSTRAINT
    assert div.targets == ()
    assert div.requires_dt is False


def test_buoyancy_forcing_rate_to_delta(tmp_path):
    metadata = _build_metadata(
        tmp_path
    )

    compiler = CanonicalPDECompiler(
        metadata
    )

    state = torch.zeros(
        1,
        4,
        metadata.grid.nx,
        metadata.grid.ny,
        dtype=torch.float64,
    )

    state[:, 0] = 2.0

    result = compiler.compile(
        "buoyancy_forcing",
        state,
    )

    expected = (
        metadata.time.dt
        *
        2.0
        /
        (
            metadata.normalization.std_for(
                "u_y"
            )
            +
            metadata.normalization.eps
        )
    )

    assert result.is_rate
    assert result.target_delta_norm is not None

    assert torch.allclose(
        result.target_delta_norm,
        torch.full_like(
            result.target_delta_norm,
            expected,
        ),
        rtol=1e-12,
        atol=1e-12,
    )


def test_divergence_stays_constraint(tmp_path):
    metadata = _build_metadata(
        tmp_path
    )

    compiler = CanonicalPDECompiler(
        metadata
    )

    state = _canonical_state(
        metadata
    )

    result = compiler.compile(
        "divergence",
        state,
    )

    assert result.is_constraint
    assert result.target_delta_norm is None

    with pytest.raises(
        ValueError
    ):
        compiler.to_target_delta(
            "divergence",
            result.canonical_value,
        )


def test_path_b_regression(tmp_path):
    """
    New R1 compiler must exactly reproduce the validated
    canonical Path-B representation:

        dt/(s_b + eps) * (-u · grad(b))
    """

    metadata = _build_metadata(
        tmp_path
    )

    compiler = CanonicalPDECompiler(
        metadata
    )

    state = _canonical_state(
        metadata
    )

    result = compiler.compile(
        "buoyancy_advection",
        state,
    )

    b = state[:, 0]
    ux = state[:, 1]
    uy = state[:, 2]

    direct_rate = -(
        ux
        *
        grad_x_periodic(
            b,
            metadata.grid,
        )
        +
        uy
        *
        grad_y_nonperiodic(
            b,
            metadata.grid,
        )
    )

    direct_delta = (
        metadata.time.dt
        *
        direct_rate
        /
        (
            metadata.normalization.std_for(
                "buoyancy"
            )
            +
            metadata.normalization.eps
        )
    )

    assert torch.allclose(
        result.canonical_value,
        direct_rate,
        rtol=1e-12,
        atol=1e-12,
    )

    assert torch.allclose(
        result.target_delta_norm,
        direct_delta,
        rtol=1e-12,
        atol=1e-12,
    )


def test_momentum_advection_componentwise_scaling(
    tmp_path,
):
    metadata = _build_metadata(
        tmp_path
    )

    compiler = CanonicalPDECompiler(
        metadata
    )

    state = _canonical_state(
        metadata
    )

    result = compiler.compile(
        "momentum_advection",
        state,
    )

    canonical = result.canonical_value

    expected_x = (
        metadata.time.dt
        *
        canonical[:, 0]
        /
        (
            metadata.normalization.std_for(
                "u_x"
            )
            +
            metadata.normalization.eps
        )
    )

    expected_y = (
        metadata.time.dt
        *
        canonical[:, 1]
        /
        (
            metadata.normalization.std_for(
                "u_y"
            )
            +
            metadata.normalization.eps
        )
    )

    assert torch.allclose(
        result.target_delta_norm[:, 0],
        expected_x,
        rtol=1e-12,
        atol=1e-12,
    )

    assert torch.allclose(
        result.target_delta_norm[:, 1],
        expected_y,
        rtol=1e-12,
        atol=1e-12,
    )


def test_transport_terms_use_ra_pr_coefficients(
    tmp_path,
):
    metadata = _build_metadata(
        tmp_path
    )

    compiler = CanonicalPDECompiler(
        metadata
    )

    state = _canonical_state(
        metadata
    )

    param = torch.tensor(
        [
            [6.0, 0.0],
        ],
        dtype=torch.float64,
    )

    nu_nd, kappa_nd = (
        rbc_transport_coefficients(
            param
        )
    )

    assert torch.allclose(
        nu_nd,
        torch.tensor(
            [1.0e-3],
            dtype=torch.float64,
        ),
    )

    assert torch.allclose(
        kappa_nd,
        torch.tensor(
            [1.0e-3],
            dtype=torch.float64,
        ),
    )

    viscosity = compiler.compile(
        "viscosity",
        state,
        param=param,
    )

    diffusion = compiler.compile(
        "buoyancy_diffusion",
        state,
        param=param,
    )

    expected_lap_ux = laplacian(
        state[:, 1],
        metadata.grid,
    )

    expected_lap_uy = laplacian(
        state[:, 2],
        metadata.grid,
    )

    expected_viscosity = (
        1.0e-3
        *
        torch.stack(
            [
                expected_lap_ux,
                expected_lap_uy,
            ],
            dim=1,
        )
    )

    expected_diffusion = (
        1.0e-3
        *
        laplacian(
            state[:, 0],
            metadata.grid,
        )
    )

    assert torch.allclose(
        viscosity.canonical_value,
        expected_viscosity,
        rtol=1e-12,
        atol=1e-12,
    )

    assert torch.allclose(
        diffusion.canonical_value,
        expected_diffusion,
        rtol=1e-12,
        atol=1e-12,
    )


def test_pressure_gradient_gauge_invariance(
    tmp_path,
):
    metadata = _build_metadata(
        tmp_path
    )

    compiler = CanonicalPDECompiler(
        metadata
    )

    state = _canonical_state(
        metadata
    )

    shifted = state.clone()

    shifted[:, 3] = (
        shifted[:, 3]
        +
        17.25
    )

    original = compiler.compile(
        "pressure_gradient",
        state,
    )

    gauge_shifted = compiler.compile(
        "pressure_gradient",
        shifted,
    )

    assert torch.allclose(
        original.canonical_value,
        gauge_shifted.canonical_value,
        rtol=1e-11,
        atol=1e-11,
    )

    assert torch.allclose(
        original.target_delta_norm,
        gauge_shifted.target_delta_norm,
        rtol=1e-11,
        atol=1e-11,
    )


def test_compile_from_history_uses_latest_state(
    tmp_path,
):
    metadata = _build_metadata(
        tmp_path
    )

    compiler = CanonicalPDECompiler(
        metadata
    )

    latest_canonical = _canonical_state(
        metadata
    )

    latest_norm = (
        compiler.canonicalizer.normalize_state(
            latest_canonical
        )
    )

    history = torch.zeros(
        1,
        16,
        metadata.grid.nx,
        metadata.grid.ny,
        dtype=torch.float64,
    )

    history[:, -4:] = latest_norm

    direct = compiler.compile(
        "buoyancy_advection",
        latest_canonical,
    )

    from_history = (
        compiler.compile_from_history(
            "buoyancy_advection",
            history,
        )
    )

    assert torch.allclose(
        direct.canonical_value,
        from_history.canonical_value,
        rtol=1e-11,
        atol=1e-11,
    )

    assert torch.allclose(
        direct.target_delta_norm,
        from_history.target_delta_norm,
        rtol=1e-11,
        atol=1e-11,
    )


def test_unknown_term_rejected(tmp_path):
    metadata = _build_metadata(
        tmp_path
    )

    compiler = CanonicalPDECompiler(
        metadata
    )

    state = _canonical_state(
        metadata
    )

    with pytest.raises(
        KeyError
    ):
        compiler.compile(
            "not_a_real_term",
            state,
        )
