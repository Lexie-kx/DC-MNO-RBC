import json
import math

import torch

from physics.canonical_metadata import (
    build_rbc_canonical_metadata,
)
from physics.canonicalizer import (
    Canonicalizer,
)
from physics.derivatives import (
    divergence,
    grad_x_periodic,
    grad_y_nonperiodic,
    laplacian,
    second_x_periodic,
    second_y_nonperiodic,
)


def _write_dummy_stats(path):
    stats = {
        "buoyancy": {
            "mean": 1.25,
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
    stats_path = (
        tmp_path
        /
        "stats.json"
    )

    _write_dummy_stats(
        stats_path
    )

    return build_rbc_canonical_metadata(
        str(stats_path)
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

    xx, yy = torch.meshgrid(
        x,
        y,
        indexing="ij",
    )

    return (
        xx,
        yy,
    )


def test_canonicalizer_round_trip(tmp_path):
    metadata = _build_metadata(
        tmp_path
    )

    canonicalizer = Canonicalizer(
        metadata
    )

    torch.manual_seed(42)

    state_norm = torch.randn(
        3,
        4,
        metadata.grid.nx,
        metadata.grid.ny,
        dtype=torch.float64,
    )

    state_canonical = (
        canonicalizer.denormalize_state(
            state_norm
        )
    )

    recovered_norm = (
        canonicalizer.normalize_state(
            state_canonical
        )
    )

    assert torch.allclose(
        recovered_norm,
        state_norm,
        rtol=1e-12,
        atol=1e-12,
    )


def test_latest_history_extraction(tmp_path):
    metadata = _build_metadata(
        tmp_path
    )

    canonicalizer = Canonicalizer(
        metadata
    )

    history = torch.zeros(
        2,
        16,
        metadata.grid.nx,
        metadata.grid.ny,
        dtype=torch.float64,
    )

    history[:, 0:4] = 1.0
    history[:, 4:8] = 2.0
    history[:, 8:12] = 3.0
    history[:, 12:16] = 4.0

    latest = canonicalizer.latest_state_norm(
        history
    )

    assert latest.shape == (
        2,
        4,
        metadata.grid.nx,
        metadata.grid.ny,
    )

    assert torch.all(
        latest == 4.0
    )


def test_grad_x_periodic_analytic(tmp_path):
    metadata = _build_metadata(
        tmp_path
    )

    grid = metadata.grid
    xx, yy = _xy_grid(
        metadata
    )

    k = (
        2.0
        *
        math.pi
        /
        grid.x_domain_length
    )

    f = (
        torch.sin(k * xx)
        +
        yy ** 3
    )

    expected = (
        k
        *
        torch.cos(k * xx)
    )

    actual = grad_x_periodic(
        f,
        grid,
    )

    max_error = torch.max(
        torch.abs(
            actual - expected
        )
    ).item()

    assert max_error < 5.0e-4


def test_grad_y_nonperiodic_analytic(tmp_path):
    metadata = _build_metadata(
        tmp_path
    )

    grid = metadata.grid
    xx, yy = _xy_grid(
        metadata
    )

    k = (
        2.0
        *
        math.pi
        /
        grid.x_domain_length
    )

    f = (
        torch.sin(k * xx)
        +
        yy ** 3
    )

    expected = (
        3.0
        *
        yy ** 2
    )

    actual = grad_y_nonperiodic(
        f,
        grid,
    )

    max_error = torch.max(
        torch.abs(
            actual - expected
        )
    ).item()

    assert max_error < 1.0e-3


def test_second_x_periodic_analytic(tmp_path):
    metadata = _build_metadata(
        tmp_path
    )

    grid = metadata.grid
    xx, yy = _xy_grid(
        metadata
    )

    k = (
        2.0
        *
        math.pi
        /
        grid.x_domain_length
    )

    f = (
        torch.sin(k * xx)
        +
        yy ** 3
    )

    expected = (
        -(k ** 2)
        *
        torch.sin(k * xx)
    )

    actual = second_x_periodic(
        f,
        grid,
    )

    max_error = torch.max(
        torch.abs(
            actual - expected
        )
    ).item()

    assert max_error < 5.0e-4


def test_second_y_nonperiodic_analytic(tmp_path):
    metadata = _build_metadata(
        tmp_path
    )

    grid = metadata.grid
    xx, yy = _xy_grid(
        metadata
    )

    k = (
        2.0
        *
        math.pi
        /
        grid.x_domain_length
    )

    f = (
        torch.sin(k * xx)
        +
        yy ** 3
    )

    expected = (
        6.0
        *
        yy
    )

    actual = second_y_nonperiodic(
        f,
        grid,
    )

    max_error = torch.max(
        torch.abs(
            actual - expected
        )
    ).item()

    assert max_error < 1.0e-9


def test_laplacian_analytic(tmp_path):
    metadata = _build_metadata(
        tmp_path
    )

    grid = metadata.grid
    xx, yy = _xy_grid(
        metadata
    )

    k = (
        2.0
        *
        math.pi
        /
        grid.x_domain_length
    )

    f = (
        torch.sin(k * xx)
        +
        yy ** 3
    )

    expected = (
        -(k ** 2)
        *
        torch.sin(k * xx)
        +
        6.0 * yy
    )

    actual = laplacian(
        f,
        grid,
    )

    max_error = torch.max(
        torch.abs(
            actual - expected
        )
    ).item()

    assert max_error < 5.0e-4


def test_divergence_analytic(tmp_path):
    metadata = _build_metadata(
        tmp_path
    )

    grid = metadata.grid
    xx, yy = _xy_grid(
        metadata
    )

    k = (
        2.0
        *
        math.pi
        /
        grid.x_domain_length
    )

    u_x = torch.sin(
        k * xx
    )

    u_y = yy ** 2

    expected = (
        k
        *
        torch.cos(k * xx)
        +
        2.0 * yy
    )

    actual = divergence(
        u_x,
        u_y,
        grid,
    )

    max_error = torch.max(
        torch.abs(
            actual - expected
        )
    ).item()

    assert max_error < 5.0e-4
