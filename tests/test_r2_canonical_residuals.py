import json
import math

import torch

from physics.canonical_metadata import (
    build_rbc_canonical_metadata,
)
from physics.canonical_compiler import (
    CanonicalPDECompiler,
)
from physics.rbc_residuals import (
    CanonicalRBCResidual,
)


def _write_dummy_stats(path):
    stats = {
        "buoyancy": {
            "mean": 1.2,
            "std": 2.0,
        },
        "u_x": {
            "mean": -0.4,
            "std": 3.0,
        },
        "u_y": {
            "mean": 0.7,
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


def _grid(metadata):
    g = metadata.grid

    x = (
        torch.arange(
            g.nx,
            dtype=torch.float64,
        )
        *
        g.dx
    )

    y = (
        torch.arange(
            g.ny,
            dtype=torch.float64,
        )
        *
        g.dy
    )

    return torch.meshgrid(
        x,
        y,
        indexing="ij",
    )


def _current_state(metadata):
    """
    Construct a nontrivial state.

    Velocity is chosen as:

        ux = ux(y)
        uy = uy(x)

    so:

        d(ux)/dx = 0
        d(uy)/dy = 0

    and the discrete divergence should be zero up to floating point.
    """

    xx, yy = _grid(
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
        0.4
        +
        0.15 * torch.sin(k * xx)
        +
        0.08 * yy ** 2
    )

    ux = (
        0.10
        +
        0.05 * yy ** 2
    )

    uy = (
        0.02
        *
        torch.sin(k * xx)
    )

    p = (
        0.3 * torch.cos(k * xx)
        +
        0.2 * yy ** 2
        +
        7.0
    )

    state = torch.stack(
        [
            b,
            ux,
            uy,
            p,
        ],
        dim=0,
    ).unsqueeze(0)

    return state


def _param():
    return torch.tensor(
        [
            [6.0, 0.0],
        ],
        dtype=torch.float64,
    )


def _manual_rhs(
    metadata,
    state,
    param,
):
    compiler = CanonicalPDECompiler(
        metadata
    )

    adv_b = compiler.compile(
        "buoyancy_advection",
        state,
    ).canonical_value

    diff_b = compiler.compile(
        "buoyancy_diffusion",
        state,
        param=param,
    ).canonical_value

    adv_u = compiler.compile(
        "momentum_advection",
        state,
    ).canonical_value

    visc_u = compiler.compile(
        "viscosity",
        state,
        param=param,
    ).canonical_value

    grad_p = compiler.compile(
        "pressure_gradient",
        state,
    ).canonical_value

    force_b = compiler.compile(
        "buoyancy_forcing",
        state,
    ).canonical_value

    rhs_b = (
        adv_b
        +
        diff_b
    )

    rhs_ux = (
        adv_u[:, 0]
        +
        visc_u[:, 0]
        +
        grad_p[:, 0]
    )

    rhs_uy = (
        adv_u[:, 1]
        +
        visc_u[:, 1]
        +
        grad_p[:, 1]
        +
        force_b
    )

    rhs_u = torch.stack(
        [
            rhs_ux,
            rhs_uy,
        ],
        dim=1,
    )

    return (
        rhs_b,
        rhs_u,
    )


def _exact_forward_euler_next(
    metadata,
    current,
    param,
):
    rhs_b, rhs_u = _manual_rhs(
        metadata,
        current,
        param,
    )

    dt = metadata.time.dt

    next_state = current.clone()

    next_state[:, 0] = (
        current[:, 0]
        +
        dt * rhs_b
    )

    next_state[:, 1] = (
        current[:, 1]
        +
        dt * rhs_u[:, 0]
    )

    next_state[:, 2] = (
        current[:, 2]
        +
        dt * rhs_u[:, 1]
    )

    # R2-1 does not introduce a pressure evolution equation.
    next_state[:, 3] = current[:, 3]

    return next_state


def test_exact_forward_euler_evolution_residual_is_zero(
    tmp_path,
):
    metadata = _build_metadata(
        tmp_path
    )

    current = _current_state(
        metadata
    )

    param = _param()

    next_state = _exact_forward_euler_next(
        metadata,
        current,
        param,
    )

    residual = CanonicalRBCResidual(
        metadata
    ).compute(
        current,
        next_state,
        param,
    )

    assert torch.allclose(
        residual.residual_rate_b,
        torch.zeros_like(
            residual.residual_rate_b
        ),
        rtol=1e-10,
        atol=1e-10,
    )

    assert torch.allclose(
        residual.residual_rate_u,
        torch.zeros_like(
            residual.residual_rate_u
        ),
        rtol=1e-10,
        atol=1e-10,
    )

    assert torch.allclose(
        residual.residual_delta_b,
        torch.zeros_like(
            residual.residual_delta_b
        ),
        rtol=1e-10,
        atol=1e-10,
    )

    assert torch.allclose(
        residual.residual_delta_u,
        torch.zeros_like(
            residual.residual_delta_u
        ),
        rtol=1e-10,
        atol=1e-10,
    )


def test_rate_delta_residual_identity(
    tmp_path,
):
    """
    Core R2-1 identity:

        R_delta
            =
        dt/(std + eps) * R_rate
    """

    metadata = _build_metadata(
        tmp_path
    )

    current = _current_state(
        metadata
    )

    param = _param()

    next_state = _exact_forward_euler_next(
        metadata,
        current,
        param,
    )

    xx, yy = _grid(
        metadata
    )

    next_state[:, 0] += (
        0.013
        *
        torch.sin(
            2.0
            *
            math.pi
            *
            yy
        )
    )

    next_state[:, 1] += (
        0.007
        *
        torch.cos(
            2.0
            *
            math.pi
            *
            yy
        )
    )

    next_state[:, 2] += (
        0.011
        *
        torch.sin(
            (
                2.0
                *
                math.pi
                /
                metadata.grid.x_domain_length
            )
            *
            xx
        )
    )

    residual = CanonicalRBCResidual(
        metadata
    ).compute(
        current,
        next_state,
        param,
    )

    dt = metadata.time.dt

    sb = (
        metadata.normalization.std_for(
            "buoyancy"
        )
        +
        metadata.normalization.eps
    )

    sux = (
        metadata.normalization.std_for(
            "u_x"
        )
        +
        metadata.normalization.eps
    )

    suy = (
        metadata.normalization.std_for(
            "u_y"
        )
        +
        metadata.normalization.eps
    )

    expected_b = (
        dt
        /
        sb
        *
        residual.residual_rate_b
    )

    expected_ux = (
        dt
        /
        sux
        *
        residual.residual_rate_u[:, 0]
    )

    expected_uy = (
        dt
        /
        suy
        *
        residual.residual_rate_u[:, 1]
    )

    assert torch.allclose(
        residual.residual_delta_b,
        expected_b,
        rtol=1e-10,
        atol=1e-10,
    )

    assert torch.allclose(
        residual.residual_delta_u[:, 0],
        expected_ux,
        rtol=1e-10,
        atol=1e-10,
    )

    assert torch.allclose(
        residual.residual_delta_u[:, 1],
        expected_uy,
        rtol=1e-10,
        atol=1e-10,
    )


def test_divergence_remains_constraint_space(
    tmp_path,
):
    metadata = _build_metadata(
        tmp_path
    )

    current = _current_state(
        metadata
    )

    param = _param()

    next_state = _exact_forward_euler_next(
        metadata,
        current,
        param,
    )

    residual = CanonicalRBCResidual(
        metadata
    ).compute(
        current,
        next_state,
        param,
    )

    assert residual.divergence.shape == (
        1,
        metadata.grid.nx,
        metadata.grid.ny,
    )

    assert torch.max(
        torch.abs(
            residual.divergence
        )
    ).item() < 1e-12


def test_pressure_gauge_invariance_at_residual_level(
    tmp_path,
):
    metadata = _build_metadata(
        tmp_path
    )

    current = _current_state(
        metadata
    )

    param = _param()

    next_state = _exact_forward_euler_next(
        metadata,
        current,
        param,
    )

    shifted_current = current.clone()

    shifted_current[:, 3] += 123.456

    original = CanonicalRBCResidual(
        metadata
    ).compute(
        current,
        next_state,
        param,
    )

    shifted = CanonicalRBCResidual(
        metadata
    ).compute(
        shifted_current,
        next_state,
        param,
    )

    assert torch.allclose(
        original.residual_rate_b,
        shifted.residual_rate_b,
        rtol=1e-10,
        atol=1e-10,
    )

    assert torch.allclose(
        original.residual_rate_u,
        shifted.residual_rate_u,
        rtol=1e-10,
        atol=1e-10,
    )

    assert torch.allclose(
        original.residual_delta_u,
        shifted.residual_delta_u,
        rtol=1e-10,
        atol=1e-10,
    )


def test_normalized_state_interface_matches_canonical(
    tmp_path,
):
    metadata = _build_metadata(
        tmp_path
    )

    residual_builder = CanonicalRBCResidual(
        metadata
    )

    current = _current_state(
        metadata
    )

    param = _param()

    next_state = _exact_forward_euler_next(
        metadata,
        current,
        param,
    )

    current_norm = (
        residual_builder
        .compiler
        .canonicalizer
        .normalize_state(
            current
        )
    )

    next_norm = (
        residual_builder
        .compiler
        .canonicalizer
        .normalize_state(
            next_state
        )
    )

    direct = residual_builder.compute(
        current,
        next_state,
        param,
    )

    normalized = (
        residual_builder.compute_from_normalized_states(
            current_norm,
            next_norm,
            param,
        )
    )

    assert torch.allclose(
        direct.residual_rate_b,
        normalized.residual_rate_b,
        rtol=1e-10,
        atol=1e-10,
    )

    assert torch.allclose(
        direct.residual_rate_u,
        normalized.residual_rate_u,
        rtol=1e-10,
        atol=1e-10,
    )

    assert torch.allclose(
        direct.residual_delta_b,
        normalized.residual_delta_b,
        rtol=1e-10,
        atol=1e-10,
    )

    assert torch.allclose(
        direct.residual_delta_u,
        normalized.residual_delta_u,
        rtol=1e-10,
        atol=1e-10,
    )


def test_history_interface_uses_latest_state(
    tmp_path,
):
    metadata = _build_metadata(
        tmp_path
    )

    residual_builder = CanonicalRBCResidual(
        metadata
    )

    current = _current_state(
        metadata
    )

    param = _param()

    next_state = _exact_forward_euler_next(
        metadata,
        current,
        param,
    )

    current_norm = (
        residual_builder
        .compiler
        .canonicalizer
        .normalize_state(
            current
        )
    )

    next_norm = (
        residual_builder
        .compiler
        .canonicalizer
        .normalize_state(
            next_state
        )
    )

    history = torch.zeros(
        1,
        16,
        metadata.grid.nx,
        metadata.grid.ny,
        dtype=torch.float64,
    )

    history[:, 0:4] = -3.0
    history[:, 4:8] = 8.0
    history[:, 8:12] = 1.5
    history[:, 12:16] = current_norm

    direct = residual_builder.compute(
        current,
        next_state,
        param,
    )

    from_history = (
        residual_builder.compute_from_history(
            history,
            next_norm,
            param,
        )
    )

    assert torch.allclose(
        direct.residual_rate_b,
        from_history.residual_rate_b,
        rtol=1e-10,
        atol=1e-10,
    )

    assert torch.allclose(
        direct.residual_rate_u,
        from_history.residual_rate_u,
        rtol=1e-10,
        atol=1e-10,
    )

    assert torch.allclose(
        direct.residual_delta_b,
        from_history.residual_delta_b,
        rtol=1e-10,
        atol=1e-10,
    )

    assert torch.allclose(
        direct.residual_delta_u,
        from_history.residual_delta_u,
        rtol=1e-10,
        atol=1e-10,
    )
