import json
import math

import torch

from constants import (
    EPS,
    FIELD_ORDER,
    SPATIAL_RESOLUTION,
)
from physics.canonical_metadata import (
    GridSpec,
    TimeSpec,
    build_rbc_canonical_metadata,
    load_field_normalization,
    rbc_transport_coefficients,
)


def _write_dummy_stats(path):
    stats = {
        "buoyancy": {
            "mean": 1.0,
            "std": 2.0,
        },
        "u_x": {
            "mean": 0.1,
            "std": 3.0,
        },
        "u_y": {
            "mean": -0.2,
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


def test_constants_contract():
    assert tuple(FIELD_ORDER) == (
        "buoyancy",
        "u_x",
        "u_y",
        "pressure",
    )

    assert tuple(SPATIAL_RESOLUTION) == (
        256,
        64,
    )


def test_grid_contract():
    grid = GridSpec()

    assert grid.nx == 256
    assert grid.ny == 64

    assert grid.x_axis == -2
    assert grid.y_axis == -1

    assert grid.x_periodic is True
    assert grid.y_periodic is False

    assert math.isclose(
        grid.dx,
        1.0 / 64.0,
        rel_tol=0.0,
        abs_tol=1e-15,
    )

    assert math.isclose(
        grid.dy,
        1.0 / 63.0,
        rel_tol=0.0,
        abs_tol=1e-15,
    )

    assert math.isclose(
        grid.x_domain_length,
        4.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )

    assert math.isclose(
        grid.y_domain_length,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_time_contract():
    time = TimeSpec()

    assert math.isclose(
        time.dt,
        0.25,
        rel_tol=0.0,
        abs_tol=1e-15,
    )


def test_load_field_normalization(tmp_path):
    stats_path = (
        tmp_path
        /
        "stats.json"
    )

    _write_dummy_stats(
        stats_path
    )

    norm = load_field_normalization(
        str(stats_path)
    )

    assert norm.mean_for(
        "buoyancy"
    ) == 1.0

    assert norm.std_for(
        "u_y"
    ) == 4.0

    assert norm.index_for(
        "pressure"
    ) == 3

    assert norm.eps == EPS


def test_build_metadata(tmp_path):
    stats_path = (
        tmp_path
        /
        "stats.json"
    )

    _write_dummy_stats(
        stats_path
    )

    metadata = build_rbc_canonical_metadata(
        str(stats_path)
    )

    assert metadata.field_order == tuple(
        FIELD_ORDER
    )

    assert metadata.parameter_order == (
        "log10_Ra",
        "log10_Pr",
    )

    assert metadata.grid.nx == 256
    assert metadata.grid.ny == 64
    assert metadata.time.dt == 0.25


def test_rbc_transport_coefficients_ra1e6_pr1():
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

    expected = torch.tensor(
        [1.0e-3],
        dtype=torch.float64,
    )

    assert torch.allclose(
        nu_nd,
        expected,
        rtol=1e-12,
        atol=1e-15,
    )

    assert torch.allclose(
        kappa_nd,
        expected,
        rtol=1e-12,
        atol=1e-15,
    )


def test_rbc_transport_identity():
    param = torch.tensor(
        [
            [6.0, -0.3010299956639812],
            [6.0, 0.0],
            [7.0, 0.3010299956639812],
            [8.0, 0.0],
        ],
        dtype=torch.float64,
    )

    nu_nd, kappa_nd = (
        rbc_transport_coefficients(
            param
        )
    )

    ra = torch.pow(
        torch.tensor(
            10.0,
            dtype=param.dtype,
        ),
        param[:, 0],
    )

    pr = torch.pow(
        torch.tensor(
            10.0,
            dtype=param.dtype,
        ),
        param[:, 1],
    )

    assert torch.allclose(
        nu_nd / kappa_nd,
        pr,
        rtol=1e-12,
        atol=1e-15,
    )

    assert torch.allclose(
        nu_nd * kappa_nd,
        1.0 / ra,
        rtol=1e-12,
        atol=1e-15,
    )
