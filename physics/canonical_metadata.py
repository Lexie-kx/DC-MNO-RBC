"""
R1-1: Canonical Metadata Contract
=================================

Purpose
-------
Provide one audited source of truth for the representation metadata
required by the DC-MNO canonical PDE-term interface.

This module does NOT:
    - train a model;
    - compute PDE residuals;
    - inject physics corrections;
    - perform normalization/denormalization;
    - access TEST data.

It only defines the metadata contract that later R1 compiler modules
must consume explicitly.

Current audited RBC convention
------------------------------
Field order:
    [buoyancy, u_x, u_y, pressure]

Tensor spatial layout:
    [..., X, Y]

    x-axis = -2
    y-axis = -1

Grid:
    Nx = 256
    Ny = 64

    dx = 1 / 64
    dy = 1 / 63

Therefore:
    periodic x-domain length = Nx * dx = 4
    non-periodic y-domain length = (Ny - 1) * dy = 1

Time:
    dt = 0.25

Parameter input:
    [log10(Ra), log10(Pr)]

Dimensionless RBC transport coefficients:
    nu_nd    = sqrt(Pr / Ra)
    kappa_nd = 1 / sqrt(Ra * Pr)

Equivalently:
    log10(nu_nd)
        = -0.5 * (log10(Ra) - log10(Pr))

    log10(kappa_nd)
        = -0.5 * (log10(Ra) + log10(Pr))
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Dict, Mapping, Tuple, Union

import torch

from constants import (
    EPS,
    FIELD_ORDER,
    FIELD_TO_INDEX,
    NUM_FIELDS,
    SPATIAL_RESOLUTION,
)


Number = Union[int, float]


# ============================================================
# Field-wise normalization metadata
# ============================================================

@dataclass(frozen=True)
class FieldNormalization:
    """
    Split-specific field-wise affine normalization metadata.

    Important
    ---------
    The values are deliberately loaded from an explicit stats file.
    They must NOT be hard-coded because unseen-Ra / unseen-Pr
    experiments may use different training statistics.
    """

    mean: Mapping[str, float]
    std: Mapping[str, float]
    eps: float = EPS

    def __post_init__(self) -> None:
        expected = tuple(FIELD_ORDER)

        if tuple(self.mean.keys()) != expected:
            raise ValueError(
                "Normalization mean field order mismatch.\n"
                f"Expected: {expected}\n"
                f"Got:      {tuple(self.mean.keys())}"
            )

        if tuple(self.std.keys()) != expected:
            raise ValueError(
                "Normalization std field order mismatch.\n"
                f"Expected: {expected}\n"
                f"Got:      {tuple(self.std.keys())}"
            )

        for field in FIELD_ORDER:
            value = float(self.std[field])

            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"std[{field}] must be finite and positive, "
                    f"got {value}"
                )

        if not math.isfinite(float(self.eps)) or self.eps <= 0.0:
            raise ValueError(
                f"eps must be finite and positive, got {self.eps}"
            )

    def mean_for(self, field: str) -> float:
        _validate_field_name(field)
        return float(self.mean[field])

    def std_for(self, field: str) -> float:
        _validate_field_name(field)
        return float(self.std[field])

    def index_for(self, field: str) -> int:
        _validate_field_name(field)
        return int(FIELD_TO_INDEX[field])


# ============================================================
# Grid / boundary metadata
# ============================================================

@dataclass(frozen=True)
class GridSpec:
    """
    Canonical spatial layout used by the current RBC dataset.

    Tensor convention:
        [..., X, Y]

    x:
        tensor axis -2
        periodic

    y:
        tensor axis -1
        non-periodic
    """

    nx: int = 256
    ny: int = 64

    dx: float = 1.0 / 64.0
    dy: float = 1.0 / 63.0

    x_axis: int = -2
    y_axis: int = -1

    x_periodic: bool = True
    y_periodic: bool = False

    def __post_init__(self) -> None:
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError(
                f"Grid sizes must be positive, got "
                f"nx={self.nx}, ny={self.ny}"
            )

        if self.dx <= 0.0 or self.dy <= 0.0:
            raise ValueError(
                f"Grid spacings must be positive, got "
                f"dx={self.dx}, dy={self.dy}"
            )

        if self.x_axis == self.y_axis:
            raise ValueError(
                "x_axis and y_axis must refer to different tensor axes."
            )

    @property
    def x_domain_length(self) -> float:
        """
        Periodic x convention:
            Lx = Nx * dx
        """
        return float(self.nx * self.dx)

    @property
    def y_domain_length(self) -> float:
        """
        Non-periodic y convention:
            Ly = (Ny - 1) * dy
        """
        return float((self.ny - 1) * self.dy)


# ============================================================
# Time metadata
# ============================================================

@dataclass(frozen=True)
class TimeSpec:
    """
    Finite-step prediction interval.

    This dt is a representation conversion factor for rate-type
    PDE terms. It is NOT an adaptive gate or tunable residual weight.
    """

    dt: float = 0.25

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.dt)) or self.dt <= 0.0:
            raise ValueError(
                f"dt must be finite and positive, got {self.dt}"
            )


# ============================================================
# Unified metadata package
# ============================================================

@dataclass(frozen=True)
class CanonicalMetadata:
    """
    Complete static metadata contract for one RBC experiment split.

    Sample-dependent Ra/Pr values are intentionally NOT stored here.
    They are supplied per batch to later compiler operations.
    """

    field_order: Tuple[str, ...]
    normalization: FieldNormalization
    grid: GridSpec
    time: TimeSpec

    parameter_order: Tuple[str, str] = (
        "log10_Ra",
        "log10_Pr",
    )

    def __post_init__(self) -> None:
        if tuple(self.field_order) != tuple(FIELD_ORDER):
            raise ValueError(
                "Canonical field order disagrees with constants.py.\n"
                f"Expected: {tuple(FIELD_ORDER)}\n"
                f"Got:      {tuple(self.field_order)}"
            )

        if len(self.field_order) != NUM_FIELDS:
            raise ValueError(
                f"Expected {NUM_FIELDS} fields, "
                f"got {len(self.field_order)}"
            )

        if self.parameter_order != (
            "log10_Ra",
            "log10_Pr",
        ):
            raise ValueError(
                "Current RBC parameter interface must be "
                "('log10_Ra', 'log10_Pr')."
            )


# ============================================================
# Construction helpers
# ============================================================

def _validate_field_name(field: str) -> None:
    if field not in FIELD_TO_INDEX:
        raise KeyError(
            f"Unknown field '{field}'. "
            f"Expected one of {tuple(FIELD_ORDER)}."
        )


def load_field_normalization(
    stats_path: str,
    *,
    eps: float = EPS,
) -> FieldNormalization:
    """
    Load split-specific field statistics.

    Expected JSON structure:
        {
            "buoyancy": {"mean": ..., "std": ...},
            "u_x":      {"mean": ..., "std": ...},
            "u_y":      {"mean": ..., "std": ...},
            "pressure": {"mean": ..., "std": ...}
        }
    """

    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    missing = [
        field
        for field in FIELD_ORDER
        if field not in stats
    ]

    if missing:
        raise KeyError(
            f"Stats file is missing fields: {missing}"
        )

    mean: Dict[str, float] = {}
    std: Dict[str, float] = {}

    for field in FIELD_ORDER:
        if "mean" not in stats[field]:
            raise KeyError(
                f"Stats for '{field}' missing key 'mean'."
            )

        if "std" not in stats[field]:
            raise KeyError(
                f"Stats for '{field}' missing key 'std'."
            )

        mean[field] = float(stats[field]["mean"])
        std[field] = float(stats[field]["std"])

    return FieldNormalization(
        mean=mean,
        std=std,
        eps=float(eps),
    )


def build_rbc_canonical_metadata(
    stats_path: str,
    *,
    dx: float = 1.0 / 64.0,
    dy: float = 1.0 / 63.0,
    dt: float = 0.25,
    eps: float = EPS,
) -> CanonicalMetadata:
    """
    Construct the current audited RBC canonical metadata package.

    dx/dy/dt are explicit arguments so future datasets or resolution
    conventions cannot silently reuse the present RBC values.
    """

    expected_resolution = (256, 64)

    if tuple(SPATIAL_RESOLUTION) != expected_resolution:
        raise RuntimeError(
            "constants.SPATIAL_RESOLUTION changed from the audited "
            "RBC convention.\n"
            f"Expected: {expected_resolution}\n"
            f"Got:      {tuple(SPATIAL_RESOLUTION)}"
        )

    normalization = load_field_normalization(
        stats_path,
        eps=eps,
    )

    grid = GridSpec(
        nx=SPATIAL_RESOLUTION[0],
        ny=SPATIAL_RESOLUTION[1],
        dx=float(dx),
        dy=float(dy),
    )

    time = TimeSpec(
        dt=float(dt),
    )

    return CanonicalMetadata(
        field_order=tuple(FIELD_ORDER),
        normalization=normalization,
        grid=grid,
        time=time,
    )


# ============================================================
# RBC dimensionless transport coefficients
# ============================================================

def rbc_transport_coefficients(
    param: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert:
        param [..., 2]
        =
        [log10(Ra), log10(Pr)]

    into the current dimensionless RBC transport coefficients:

        nu_nd
            = sqrt(Pr / Ra)

        kappa_nd
            = 1 / sqrt(Ra * Pr)

    Returns
    -------
    nu_nd:
        param.shape[:-1]

    kappa_nd:
        param.shape[:-1]

    Notes
    -----
    These are dimensionless PDE coefficients in the current RBC
    convention, NOT SI material-property reconstruction.
    """

    if not torch.is_tensor(param):
        raise TypeError(
            f"param must be a torch.Tensor, got {type(param)}"
        )

    if param.ndim < 1 or param.shape[-1] != 2:
        raise ValueError(
            "param must have final dimension 2 "
            "= [log10(Ra), log10(Pr)], "
            f"got shape={tuple(param.shape)}"
        )

    log_ra = param[..., 0]
    log_pr = param[..., 1]

    log_nu = -0.5 * (
        log_ra
        -
        log_pr
    )

    log_kappa = -0.5 * (
        log_ra
        +
        log_pr
    )

    ten = torch.as_tensor(
        10.0,
        dtype=param.dtype,
        device=param.device,
    )

    nu_nd = torch.pow(
        ten,
        log_nu,
    )

    kappa_nd = torch.pow(
        ten,
        log_kappa,
    )

    return (
        nu_nd,
        kappa_nd,
    )
