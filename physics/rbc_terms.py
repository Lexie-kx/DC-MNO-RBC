"""
R1-3: RBC PDE-Term Registry
===========================

This module defines the physical typing of the canonical RBC terms.

It does NOT evaluate derivatives or convert values into neural-network
target space. Numerical evaluation is handled by canonical_compiler.py.

Seven conceptual PDE terms
--------------------------
1. buoyancy_advection
       -u · grad(b)

2. buoyancy_forcing
       b e_y

3. momentum_advection
       -u · grad(u)

4. viscosity
       nu_nd * laplacian(u)

5. buoyancy_diffusion
       kappa_nd * laplacian(b)

6. pressure_gradient
       -grad(p)

7. divergence
       div(u)

Important
---------
"rate" terms belong to evolution equations and therefore require
finite-step conversion by dt before entering a delta-prediction space.

"constraint" terms do NOT go through rate -> delta conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple


class TermSemantic(str, Enum):
    RATE = "rate"
    CONSTRAINT = "constraint"


class TermRole(str, Enum):
    CROSS_FIELD = "cross_field"
    SELF_OPERATOR = "self_operator"
    PRESSURE_SPECIFIC = "pressure_specific"
    CONSTRAINT = "constraint"


@dataclass(frozen=True)
class PDETermSpec:
    name: str
    sources: Tuple[str, ...]
    targets: Tuple[str, ...]
    semantic: TermSemantic
    role: TermRole
    derivative_order: int
    requires_param_coefficients: bool
    description: str

    @property
    def requires_dt(self) -> bool:
        return self.semantic == TermSemantic.RATE


RBC_TERM_REGISTRY: Dict[str, PDETermSpec] = {
    "buoyancy_advection": PDETermSpec(
        name="buoyancy_advection",
        sources=(
            "u_x",
            "u_y",
            "buoyancy",
        ),
        targets=(
            "buoyancy",
        ),
        semantic=TermSemantic.RATE,
        role=TermRole.CROSS_FIELD,
        derivative_order=1,
        requires_param_coefficients=False,
        description="-u · grad(b)",
    ),

    "buoyancy_forcing": PDETermSpec(
        name="buoyancy_forcing",
        sources=(
            "buoyancy",
        ),
        targets=(
            "u_y",
        ),
        semantic=TermSemantic.RATE,
        role=TermRole.CROSS_FIELD,
        derivative_order=0,
        requires_param_coefficients=False,
        description="b e_y",
    ),

    "momentum_advection": PDETermSpec(
        name="momentum_advection",
        sources=(
            "u_x",
            "u_y",
        ),
        targets=(
            "u_x",
            "u_y",
        ),
        semantic=TermSemantic.RATE,
        role=TermRole.SELF_OPERATOR,
        derivative_order=1,
        requires_param_coefficients=False,
        description="-u · grad(u)",
    ),

    "viscosity": PDETermSpec(
        name="viscosity",
        sources=(
            "u_x",
            "u_y",
        ),
        targets=(
            "u_x",
            "u_y",
        ),
        semantic=TermSemantic.RATE,
        role=TermRole.SELF_OPERATOR,
        derivative_order=2,
        requires_param_coefficients=True,
        description="nu_nd * laplacian(u)",
    ),

    "buoyancy_diffusion": PDETermSpec(
        name="buoyancy_diffusion",
        sources=(
            "buoyancy",
        ),
        targets=(
            "buoyancy",
        ),
        semantic=TermSemantic.RATE,
        role=TermRole.SELF_OPERATOR,
        derivative_order=2,
        requires_param_coefficients=True,
        description="kappa_nd * laplacian(b)",
    ),

    "pressure_gradient": PDETermSpec(
        name="pressure_gradient",
        sources=(
            "pressure",
        ),
        targets=(
            "u_x",
            "u_y",
        ),
        semantic=TermSemantic.RATE,
        role=TermRole.PRESSURE_SPECIFIC,
        derivative_order=1,
        requires_param_coefficients=False,
        description="-grad(p)",
    ),

    "divergence": PDETermSpec(
        name="divergence",
        sources=(
            "u_x",
            "u_y",
        ),
        targets=(),
        semantic=TermSemantic.CONSTRAINT,
        role=TermRole.CONSTRAINT,
        derivative_order=1,
        requires_param_coefficients=False,
        description="div(u)",
    ),
}


def get_rbc_term_spec(
    name: str,
) -> PDETermSpec:
    try:
        return RBC_TERM_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown RBC PDE term '{name}'. "
            f"Available terms: {tuple(RBC_TERM_REGISTRY.keys())}"
        ) from exc
