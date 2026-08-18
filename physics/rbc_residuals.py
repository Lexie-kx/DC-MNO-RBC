"""
R2-1: Canonical RBC Residual Algebra
====================================

Purpose
-------
Assemble the individually validated R1 canonical PDE terms into
equation-level RBC residuals.

Current nondimensional RBC equations
------------------------------------

Buoyancy:
    db/dt
        =
    kappa_nd * laplacian(b)
    - u · grad(b)

Momentum x:
    du_x/dt
        =
    nu_nd * laplacian(u_x)
    - dp/dx
    - u · grad(u_x)

Momentum y:
    du_y/dt
        =
    nu_nd * laplacian(u_y)
    - dp/dy
    + b
    - u · grad(u_y)

Constraint:
    div(u) = 0

Discrete-time residual convention
---------------------------------
The temporal derivative is represented by a forward finite difference:

    (y_{t+dt} - y_t) / dt

Therefore these are discrete-time PDE residual proxies, not exact
continuous-time residuals.

Two equivalent evolution-residual spaces are maintained:

1. canonical rate space

       R_y_rate
           =
       (y_next - y_current)/dt
       - RHS_y

2. normalized finite-step delta space

       R_y_delta
           =
       Delta(y_norm)
       - canonical_compiled_RHS_delta

They must satisfy:

       R_y_delta
           =
       dt / (std_y + eps)
       * R_y_rate

No pressure residual R_p is defined here.
Pressure enters momentum only through -grad(p).

Divergence remains a constraint residual and never enters delta space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from constants import (
    B_IDX,
    UX_IDX,
    UY_IDX,
)

from physics.canonical_metadata import (
    CanonicalMetadata,
)
from physics.canonical_compiler import (
    CanonicalPDECompiler,
)


@dataclass
class RBCResidualResult:
    """
    Equation-level RBC residual result.

    Shapes
    ------
    temporal_rate_b:
        [B, X, Y]

    temporal_rate_u:
        [B, 2, X, Y]

    rhs_rate_b:
        [B, X, Y]

    rhs_rate_u:
        [B, 2, X, Y]

    residual_rate_b:
        [B, X, Y]

    residual_rate_u:
        [B, 2, X, Y]

    residual_delta_b:
        [B, X, Y]

    residual_delta_u:
        [B, 2, X, Y]

    divergence:
        [B, X, Y]
    """

    temporal_rate_b: torch.Tensor
    temporal_rate_u: torch.Tensor

    rhs_rate_b: torch.Tensor
    rhs_rate_u: torch.Tensor

    residual_rate_b: torch.Tensor
    residual_rate_u: torch.Tensor

    residual_delta_b: torch.Tensor
    residual_delta_u: torch.Tensor

    divergence: torch.Tensor


class CanonicalRBCResidual:
    """
    Assemble equation-level residuals from R1 canonical PDE terms.
    """

    def __init__(
        self,
        metadata: CanonicalMetadata,
    ) -> None:

        self.metadata = metadata

        self.compiler = CanonicalPDECompiler(
            metadata
        )

    # ============================================================
    # Validation
    # ============================================================

    def _validate_state_pair(
        self,
        current: torch.Tensor,
        next_state: torch.Tensor,
    ) -> None:

        self.compiler._validate_canonical_state(
            current
        )

        self.compiler._validate_canonical_state(
            next_state
        )

        if current.shape != next_state.shape:
            raise ValueError(
                "current and next_state must have identical shapes, "
                f"got {tuple(current.shape)} and "
                f"{tuple(next_state.shape)}"
            )

    # ============================================================
    # Target scales
    # ============================================================

    def _scalar_scale(
        self,
        field: str,
        reference: torch.Tensor,
    ) -> torch.Tensor:

        value = (
            self.metadata.normalization.std_for(
                field
            )
            +
            self.metadata.normalization.eps
        )

        return torch.as_tensor(
            value,
            dtype=reference.dtype,
            device=reference.device,
        )

    def _velocity_scales(
        self,
        reference: torch.Tensor,
    ) -> torch.Tensor:

        values = [
            (
                self.metadata.normalization.std_for(
                    "u_x"
                )
                +
                self.metadata.normalization.eps
            ),
            (
                self.metadata.normalization.std_for(
                    "u_y"
                )
                +
                self.metadata.normalization.eps
            ),
        ]

        return torch.as_tensor(
            values,
            dtype=reference.dtype,
            device=reference.device,
        ).view(
            1,
            2,
            1,
            1,
        )

    # ============================================================
    # RHS assembly
    # ============================================================

    def _compile_rhs(
        self,
        current: torch.Tensor,
        param: torch.Tensor,
    ):
        """
        Compile all equation terms at the CURRENT state.

        This uses the R1 convention:
            explicit PDE terms evaluated at t,
            temporal difference from t -> t+dt.
        """

        adv_b = self.compiler.compile(
            "buoyancy_advection",
            current,
        )

        diff_b = self.compiler.compile(
            "buoyancy_diffusion",
            current,
            param=param,
        )

        adv_u = self.compiler.compile(
            "momentum_advection",
            current,
        )

        visc_u = self.compiler.compile(
            "viscosity",
            current,
            param=param,
        )

        grad_p = self.compiler.compile(
            "pressure_gradient",
            current,
        )

        buoyancy_force = self.compiler.compile(
            "buoyancy_forcing",
            current,
        )

        div = self.compiler.compile(
            "divergence",
            current,
        )

        # --------------------------------------------------------
        # Canonical RATE-space RHS
        # --------------------------------------------------------

        rhs_rate_b = (
            adv_b.canonical_value
            +
            diff_b.canonical_value
        )

        rhs_rate_ux = (
            adv_u.canonical_value[:, 0]
            +
            visc_u.canonical_value[:, 0]
            +
            grad_p.canonical_value[:, 0]
        )

        rhs_rate_uy = (
            adv_u.canonical_value[:, 1]
            +
            visc_u.canonical_value[:, 1]
            +
            grad_p.canonical_value[:, 1]
            +
            buoyancy_force.canonical_value
        )

        rhs_rate_u = torch.stack(
            [
                rhs_rate_ux,
                rhs_rate_uy,
            ],
            dim=1,
        )

        # --------------------------------------------------------
        # Normalized finite-step DELTA-space RHS
        # --------------------------------------------------------

        rhs_delta_b = (
            adv_b.target_delta_norm
            +
            diff_b.target_delta_norm
        )

        rhs_delta_ux = (
            adv_u.target_delta_norm[:, 0]
            +
            visc_u.target_delta_norm[:, 0]
            +
            grad_p.target_delta_norm[:, 0]
        )

        rhs_delta_uy = (
            adv_u.target_delta_norm[:, 1]
            +
            visc_u.target_delta_norm[:, 1]
            +
            grad_p.target_delta_norm[:, 1]
            +
            buoyancy_force.target_delta_norm
        )

        rhs_delta_u = torch.stack(
            [
                rhs_delta_ux,
                rhs_delta_uy,
            ],
            dim=1,
        )

        return (
            rhs_rate_b,
            rhs_rate_u,
            rhs_delta_b,
            rhs_delta_u,
            div.canonical_value,
        )

    # ============================================================
    # Public residual computation
    # ============================================================

    def compute(
        self,
        current: torch.Tensor,
        next_state: torch.Tensor,
        param: torch.Tensor,
    ) -> RBCResidualResult:
        """
        Compute discrete-time RBC residuals from canonical states.

        current:
            [B,4,X,Y]

        next_state:
            [B,4,X,Y]

        param:
            [B,2] = [log10(Ra), log10(Pr)]
        """

        self._validate_state_pair(
            current,
            next_state,
        )

        dt = self.metadata.time.dt

        # --------------------------------------------------------
        # Observed finite-difference temporal rates
        # --------------------------------------------------------

        temporal_rate_b = (
            next_state[:, B_IDX]
            -
            current[:, B_IDX]
        ) / dt

        temporal_rate_u = torch.stack(
            [
                (
                    next_state[:, UX_IDX]
                    -
                    current[:, UX_IDX]
                ) / dt,
                (
                    next_state[:, UY_IDX]
                    -
                    current[:, UY_IDX]
                ) / dt,
            ],
            dim=1,
        )

        (
            rhs_rate_b,
            rhs_rate_u,
            rhs_delta_b,
            rhs_delta_u,
            divergence,
        ) = self._compile_rhs(
            current,
            param,
        )

        # --------------------------------------------------------
        # Canonical RATE-space residual
        # --------------------------------------------------------

        residual_rate_b = (
            temporal_rate_b
            -
            rhs_rate_b
        )

        residual_rate_u = (
            temporal_rate_u
            -
            rhs_rate_u
        )

        # --------------------------------------------------------
        # Observed normalized finite-step delta
        #
        # Mean cancels:
        #
        #   Delta(y_norm)
        #       =
        #   (y_next - y_current)/(std_y + eps)
        # --------------------------------------------------------

        b_scale = self._scalar_scale(
            "buoyancy",
            current,
        )

        u_scales = self._velocity_scales(
            current
        )

        observed_delta_b = (
            next_state[:, B_IDX]
            -
            current[:, B_IDX]
        ) / b_scale

        observed_delta_u = torch.stack(
            [
                (
                    next_state[:, UX_IDX]
                    -
                    current[:, UX_IDX]
                ),
                (
                    next_state[:, UY_IDX]
                    -
                    current[:, UY_IDX]
                ),
            ],
            dim=1,
        ) / u_scales

        # --------------------------------------------------------
        # Normalized finite-step DELTA-space residual
        # --------------------------------------------------------

        residual_delta_b = (
            observed_delta_b
            -
            rhs_delta_b
        )

        residual_delta_u = (
            observed_delta_u
            -
            rhs_delta_u
        )

        return RBCResidualResult(
            temporal_rate_b=temporal_rate_b,
            temporal_rate_u=temporal_rate_u,
            rhs_rate_b=rhs_rate_b,
            rhs_rate_u=rhs_rate_u,
            residual_rate_b=residual_rate_b,
            residual_rate_u=residual_rate_u,
            residual_delta_b=residual_delta_b,
            residual_delta_u=residual_delta_u,
            divergence=divergence,
        )

    # ============================================================
    # Normalized-data interfaces
    # ============================================================

    def compute_from_normalized_states(
        self,
        current_norm: torch.Tensor,
        next_norm: torch.Tensor,
        param: torch.Tensor,
    ) -> RBCResidualResult:
        """
        normalized current/next states
            ->
        canonical states
            ->
        equation residuals
        """

        current = (
            self.compiler.canonicalizer.denormalize_state(
                current_norm
            )
        )

        next_state = (
            self.compiler.canonicalizer.denormalize_state(
                next_norm
            )
        )

        return self.compute(
            current,
            next_state,
            param,
        )

    def compute_from_history(
        self,
        history_norm: torch.Tensor,
        next_norm: torch.Tensor,
        param: torch.Tensor,
    ) -> RBCResidualResult:
        """
        Interface matching the real RBCDataset data flow.

        history_norm:
            [B,16,X,Y]

        next_norm:
            [B,4,X,Y]

        param:
            [B,2]

        The current state is the latest frame of history.
        """

        current_norm = (
            self.compiler.canonicalizer.latest_state_norm(
                history_norm
            )
        )

        return self.compute_from_normalized_states(
            current_norm,
            next_norm,
            param,
        )
