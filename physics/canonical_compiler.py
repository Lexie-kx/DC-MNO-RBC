"""
R1-3: Coupling-Aware Canonical PDE-Term Compiler
================================================

Purpose
-------
Evaluate typed RBC PDE terms in canonical field space and convert
rate-type terms into the representation expected by the current
normalized finite-step delta neural operator.

Pipeline
--------
normalized history
    ->
latest normalized state
    ->
canonical nondimensional RBC state
    ->
canonical PDE term
    ->
semantic dispatch
        rate:
            multiply by dt
            divide by target field scale
        constraint:
            remain in constraint / residual space

For a rate T_y contained in d(y)/dt:

    canonical target delta
        =
        dt * T_y / (std_y + eps)

Constraint terms such as div(u) are NOT converted into field delta.

This module does NOT:
    - train a neural operator;
    - apply StateParam gates;
    - apply Utility Gate;
    - apply RMSCap;
    - open TEST data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from constants import (
    B_IDX,
    P_IDX,
    UX_IDX,
    UY_IDX,
)

from physics.canonical_metadata import (
    CanonicalMetadata,
    rbc_transport_coefficients,
)
from physics.canonicalizer import Canonicalizer
from physics.derivatives import (
    divergence,
    grad_x_periodic,
    grad_y_nonperiodic,
    laplacian,
)
from physics.rbc_terms import (
    PDETermSpec,
    TermSemantic,
    get_rbc_term_spec,
)


@dataclass
class CompiledPDETerm:
    """
    Result of compiling one typed PDE term.

    canonical_value
    ----------------
    Term evaluated in canonical RBC PDE space.

    target_delta_norm
    -----------------
    For RATE terms:
        finite-step normalized target-space representation.

    For CONSTRAINT terms:
        None.
    """

    spec: PDETermSpec
    canonical_value: torch.Tensor
    target_delta_norm: Optional[torch.Tensor]

    @property
    def is_rate(self) -> bool:
        return self.spec.semantic == TermSemantic.RATE

    @property
    def is_constraint(self) -> bool:
        return self.spec.semantic == TermSemantic.CONSTRAINT


class CanonicalPDECompiler:
    """
    Canonical compiler for the current RBC system.

    Canonical state input:
        [B, 4, X, Y]

    Field order:
        [buoyancy, u_x, u_y, pressure]
    """

    def __init__(
        self,
        metadata: CanonicalMetadata,
    ) -> None:
        self.metadata = metadata
        self.canonicalizer = Canonicalizer(
            metadata
        )

    # ============================================================
    # Validation / field access
    # ============================================================

    def _validate_canonical_state(
        self,
        state: torch.Tensor,
    ) -> None:

        if not torch.is_tensor(state):
            raise TypeError(
                f"state must be torch.Tensor, got {type(state)}"
            )

        if state.ndim != 4:
            raise ValueError(
                "Canonical compiler currently expects "
                "[B,4,X,Y], "
                f"got shape={tuple(state.shape)}"
            )

        if state.shape[1] != 4:
            raise ValueError(
                "Canonical compiler expects exactly 4 fields, "
                f"got {state.shape[1]}"
            )

        if state.shape[-2] != self.metadata.grid.nx:
            raise ValueError(
                "X resolution mismatch. "
                f"Expected {self.metadata.grid.nx}, "
                f"got {state.shape[-2]}"
            )

        if state.shape[-1] != self.metadata.grid.ny:
            raise ValueError(
                "Y resolution mismatch. "
                f"Expected {self.metadata.grid.ny}, "
                f"got {state.shape[-1]}"
            )

    def _validate_param(
        self,
        param: torch.Tensor,
        *,
        batch_size: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:

        if param is None:
            raise ValueError(
                "This PDE term requires "
                "[log10(Ra), log10(Pr)]."
            )

        if not torch.is_tensor(param):
            param = torch.as_tensor(
                param,
                dtype=reference.dtype,
                device=reference.device,
            )
        else:
            param = param.to(
                dtype=reference.dtype,
                device=reference.device,
            )

        if param.ndim != 2 or param.shape[1] != 2:
            raise ValueError(
                "param must be [B,2] "
                "= [log10(Ra), log10(Pr)], "
                f"got shape={tuple(param.shape)}"
            )

        if param.shape[0] != batch_size:
            raise ValueError(
                "Parameter batch size mismatch. "
                f"state batch={batch_size}, "
                f"param batch={param.shape[0]}"
            )

        return param

    # ============================================================
    # Rate -> normalized finite-step target conversion
    # ============================================================

    def to_target_delta(
        self,
        term_name: str,
        canonical_rate: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert canonical rate representation to normalized
        finite-step target increment.

        Scalar target:
            [B,X,Y]

        Two-component target:
            [B,2,X,Y]
        """

        spec = get_rbc_term_spec(
            term_name
        )

        if spec.semantic != TermSemantic.RATE:
            raise ValueError(
                f"Term '{term_name}' is a "
                f"{spec.semantic.value} term and cannot be "
                "converted into field delta space."
            )

        dt = self.metadata.time.dt

        if len(spec.targets) == 1:
            target = spec.targets[0]

            scale = (
                self.metadata.normalization.std_for(
                    target
                )
                +
                self.metadata.normalization.eps
            )

            return (
                dt
                *
                canonical_rate
                /
                scale
            )

        if len(spec.targets) == 2:
            if canonical_rate.ndim != 4:
                raise ValueError(
                    f"Term '{term_name}' with two targets "
                    "must have canonical rate [B,2,X,Y], "
                    f"got {tuple(canonical_rate.shape)}"
                )

            if canonical_rate.shape[1] != 2:
                raise ValueError(
                    f"Term '{term_name}' expects two rate components, "
                    f"got {canonical_rate.shape[1]}"
                )

            scales = torch.as_tensor(
                [
                    (
                        self.metadata.normalization.std_for(
                            target
                        )
                        +
                        self.metadata.normalization.eps
                    )
                    for target in spec.targets
                ],
                dtype=canonical_rate.dtype,
                device=canonical_rate.device,
            ).view(
                1,
                2,
                1,
                1,
            )

            return (
                dt
                *
                canonical_rate
                /
                scales
            )

        raise RuntimeError(
            f"Unsupported target structure for term '{term_name}': "
            f"{spec.targets}"
        )

    # ============================================================
    # Canonical PDE evaluation
    # ============================================================

    def _evaluate_canonical(
        self,
        term_name: str,
        state: torch.Tensor,
        param: Optional[torch.Tensor],
    ) -> torch.Tensor:

        self._validate_canonical_state(
            state
        )

        spec = get_rbc_term_spec(
            term_name
        )

        b = state[
            :,
            B_IDX,
            :,
            :,
        ]

        ux = state[
            :,
            UX_IDX,
            :,
            :,
        ]

        uy = state[
            :,
            UY_IDX,
            :,
            :,
        ]

        p = state[
            :,
            P_IDX,
            :,
            :,
        ]

        grid = self.metadata.grid

        # --------------------------------------------------------
        # 1. -u · grad(b)
        # --------------------------------------------------------

        if term_name == "buoyancy_advection":

            db_dx = grad_x_periodic(
                b,
                grid,
            )

            db_dy = grad_y_nonperiodic(
                b,
                grid,
            )

            return -(
                ux * db_dx
                +
                uy * db_dy
            )

        # --------------------------------------------------------
        # 2. b e_y
        #
        # Scalar representation because only u_y is targeted.
        # --------------------------------------------------------

        if term_name == "buoyancy_forcing":
            return b

        # --------------------------------------------------------
        # 3. -u · grad(u)
        # --------------------------------------------------------

        if term_name == "momentum_advection":

            dux_dx = grad_x_periodic(
                ux,
                grid,
            )

            dux_dy = grad_y_nonperiodic(
                ux,
                grid,
            )

            duy_dx = grad_x_periodic(
                uy,
                grid,
            )

            duy_dy = grad_y_nonperiodic(
                uy,
                grid,
            )

            adv_x = -(
                ux * dux_dx
                +
                uy * dux_dy
            )

            adv_y = -(
                ux * duy_dx
                +
                uy * duy_dy
            )

            return torch.stack(
                [
                    adv_x,
                    adv_y,
                ],
                dim=1,
            )

        # --------------------------------------------------------
        # 4. nu_nd * laplacian(u)
        # --------------------------------------------------------

        if term_name == "viscosity":

            param = self._validate_param(
                param,
                batch_size=state.shape[0],
                reference=state,
            )

            nu_nd, _ = (
                rbc_transport_coefficients(
                    param
                )
            )

            lap_ux = laplacian(
                ux,
                grid,
            )

            lap_uy = laplacian(
                uy,
                grid,
            )

            lap_u = torch.stack(
                [
                    lap_ux,
                    lap_uy,
                ],
                dim=1,
            )

            return (
                nu_nd[
                    :,
                    None,
                    None,
                    None,
                ]
                *
                lap_u
            )

        # --------------------------------------------------------
        # 5. kappa_nd * laplacian(b)
        # --------------------------------------------------------

        if term_name == "buoyancy_diffusion":

            param = self._validate_param(
                param,
                batch_size=state.shape[0],
                reference=state,
            )

            _, kappa_nd = (
                rbc_transport_coefficients(
                    param
                )
            )

            lap_b = laplacian(
                b,
                grid,
            )

            return (
                kappa_nd[
                    :,
                    None,
                    None,
                ]
                *
                lap_b
            )

        # --------------------------------------------------------
        # 6. -grad(p)
        # --------------------------------------------------------

        if term_name == "pressure_gradient":

            dp_dx = grad_x_periodic(
                p,
                grid,
            )

            dp_dy = grad_y_nonperiodic(
                p,
                grid,
            )

            return torch.stack(
                [
                    -dp_dx,
                    -dp_dy,
                ],
                dim=1,
            )

        # --------------------------------------------------------
        # 7. div(u)
        # --------------------------------------------------------

        if term_name == "divergence":

            return divergence(
                ux,
                uy,
                grid,
            )

        raise RuntimeError(
            f"Term '{spec.name}' exists in registry "
            "but has no canonical evaluator."
        )

    # ============================================================
    # Public compilation API
    # ============================================================

    def compile(
        self,
        term_name: str,
        state_canonical: torch.Tensor,
        param: Optional[torch.Tensor] = None,
    ) -> CompiledPDETerm:
        """
        Compile one PDE term from an already canonical state.
        """

        spec = get_rbc_term_spec(
            term_name
        )

        canonical_value = (
            self._evaluate_canonical(
                term_name,
                state_canonical,
                param,
            )
        )

        if spec.semantic == TermSemantic.RATE:
            target_delta_norm = (
                self.to_target_delta(
                    term_name,
                    canonical_value,
                )
            )
        else:
            target_delta_norm = None

        return CompiledPDETerm(
            spec=spec,
            canonical_value=canonical_value,
            target_delta_norm=target_delta_norm,
        )

    def compile_from_history(
        self,
        term_name: str,
        history_norm: torch.Tensor,
        param: Optional[torch.Tensor] = None,
    ) -> CompiledPDETerm:
        """
        Compile directly from normalized neural-operator history.

        normalized history
            ->
        latest canonical state
            ->
        canonical PDE term
            ->
        target-space representation
        """

        latest_canonical = (
            self.canonicalizer.latest_state_canonical(
                history_norm
            )
        )

        return self.compile(
            term_name,
            latest_canonical,
            param=param,
        )
