"""
R1-2: Canonical Spatial Derivative Backend
==========================================

Scalar field layout:
    [..., X, Y]

x:
    axis -2
    periodic

y:
    axis -1
    non-periodic

All derivatives operate in canonical RBC field space.
"""

from __future__ import annotations

import torch

from physics.canonical_metadata import GridSpec


def _validate_scalar_field(
    f: torch.Tensor,
    grid: GridSpec,
) -> None:

    if not torch.is_tensor(f):
        raise TypeError(
            f"f must be torch.Tensor, got {type(f)}"
        )

    if f.ndim < 2:
        raise ValueError(
            "Scalar field must have at least two spatial dimensions, "
            f"got shape={tuple(f.shape)}"
        )

    if f.shape[-2] != grid.nx:
        raise ValueError(
            "X resolution mismatch. "
            f"Expected {grid.nx}, got {f.shape[-2]}"
        )

    if f.shape[-1] != grid.ny:
        raise ValueError(
            "Y resolution mismatch. "
            f"Expected {grid.ny}, got {f.shape[-1]}"
        )


def grad_x_periodic(
    f: torch.Tensor,
    grid: GridSpec,
) -> torch.Tensor:
    """
    Second-order central first derivative in periodic x.
    """

    _validate_scalar_field(
        f,
        grid,
    )

    return (
        torch.roll(
            f,
            shifts=-1,
            dims=-2,
        )
        -
        torch.roll(
            f,
            shifts=1,
            dims=-2,
        )
    ) / (
        2.0 * grid.dx
    )


def grad_y_nonperiodic(
    f: torch.Tensor,
    grid: GridSpec,
) -> torch.Tensor:
    """
    Second-order central interior derivative in y.

    Boundaries use second-order one-sided formulas.
    """

    _validate_scalar_field(
        f,
        grid,
    )

    if grid.ny < 3:
        raise ValueError(
            "Need at least 3 y points."
        )

    grad = torch.empty_like(
        f
    )

    grad[..., 1:-1] = (
        f[..., 2:]
        -
        f[..., :-2]
    ) / (
        2.0 * grid.dy
    )

    grad[..., 0] = (
        -3.0 * f[..., 0]
        +
        4.0 * f[..., 1]
        -
        f[..., 2]
    ) / (
        2.0 * grid.dy
    )

    grad[..., -1] = (
        3.0 * f[..., -1]
        -
        4.0 * f[..., -2]
        +
        f[..., -3]
    ) / (
        2.0 * grid.dy
    )

    return grad


def second_x_periodic(
    f: torch.Tensor,
    grid: GridSpec,
) -> torch.Tensor:
    """
    Second derivative in periodic x.
    """

    _validate_scalar_field(
        f,
        grid,
    )

    return (
        torch.roll(
            f,
            shifts=-1,
            dims=-2,
        )
        -
        2.0 * f
        +
        torch.roll(
            f,
            shifts=1,
            dims=-2,
        )
    ) / (
        grid.dx ** 2
    )


def second_y_nonperiodic(
    f: torch.Tensor,
    grid: GridSpec,
) -> torch.Tensor:
    """
    Second derivative in non-periodic y.

    Interior:
        central second-order

    Boundaries:
        second-order one-sided
    """

    _validate_scalar_field(
        f,
        grid,
    )

    if grid.ny < 4:
        raise ValueError(
            "Need at least 4 y points."
        )

    second = torch.empty_like(
        f
    )

    second[..., 1:-1] = (
        f[..., 2:]
        -
        2.0 * f[..., 1:-1]
        +
        f[..., :-2]
    ) / (
        grid.dy ** 2
    )

    second[..., 0] = (
        2.0 * f[..., 0]
        -
        5.0 * f[..., 1]
        +
        4.0 * f[..., 2]
        -
        f[..., 3]
    ) / (
        grid.dy ** 2
    )

    second[..., -1] = (
        2.0 * f[..., -1]
        -
        5.0 * f[..., -2]
        +
        4.0 * f[..., -3]
        -
        f[..., -4]
    ) / (
        grid.dy ** 2
    )

    return second


def laplacian(
    f: torch.Tensor,
    grid: GridSpec,
) -> torch.Tensor:

    return (
        second_x_periodic(
            f,
            grid,
        )
        +
        second_y_nonperiodic(
            f,
            grid,
        )
    )


def divergence(
    u_x: torch.Tensor,
    u_y: torch.Tensor,
    grid: GridSpec,
) -> torch.Tensor:

    _validate_scalar_field(
        u_x,
        grid,
    )

    _validate_scalar_field(
        u_y,
        grid,
    )

    if u_x.shape != u_y.shape:
        raise ValueError(
            "u_x and u_y must have identical shapes, "
            f"got {tuple(u_x.shape)} "
            f"and {tuple(u_y.shape)}"
        )

    return (
        grad_x_periodic(
            u_x,
            grid,
        )
        +
        grad_y_nonperiodic(
            u_y,
            grid,
        )
    )
