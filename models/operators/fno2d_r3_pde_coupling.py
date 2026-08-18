from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from constants import B_IDX, UY_IDX

from models.operators.fno2d_fieldwise import FieldWiseFNO2d

from physics.canonical_compiler import CanonicalPDECompiler
from physics.canonical_metadata import (
    CanonicalMetadata,
    FieldNormalization,
)
from physics.rbc_terms import (
    TermSemantic,
    get_rbc_term_spec,
)


class R3PDECouplingFNO2d(nn.Module):
    """
    R3-1 shared matched architecture.

    ------------------------------------------------------------
    Purpose
    ------------------------------------------------------------

    Build ONE common architecture for:

        R3-1a:
            naive normalized-space PDE sparse coupling

        R3-1b:
            canonical dimension-valid PDE sparse coupling

    The ONLY intended structural difference is:

        representation_mode = "naive"
            -> identity normalization metadata
            -> normalized numerical values are interpreted directly
               as canonical physical values

        representation_mode = "canonical"
            -> real canonical metadata
            -> normalized history is restored to canonical RBC space
               before PDE evaluation

    Everything else is shared:

        - same frozen M6 backbone
        - same PDE registry
        - same derivative implementation
        - same dt
        - same parameter-coefficient logic
        - same active sparse topology
        - same number of gates
        - same gate parameterization
        - same zero initialization
        - same output-delta merge rule

    ------------------------------------------------------------
    R3-1 ACTIVE physical paths
    ------------------------------------------------------------

    1. buoyancy_advection
           -u · grad(b)
           -> buoyancy normalized finite-step delta

    2. buoyancy_forcing
           b
           -> u_y normalized finite-step delta

    ------------------------------------------------------------
    Explicitly NOT included in R3-1
    ------------------------------------------------------------

        - StateParam
        - Utility Gate
        - RMSCap
        - realized-dose matching
        - PDE loss
        - learnable spatial Phi_j
        - legacy M10 buoyancy-anomaly Path A
        - pressure-gradient output injection
        - self-dynamics output injection
        - divergence output injection
    """

    VALID_REPRESENTATION_MODES = (
        "naive",
        "canonical",
    )

    ACTIVE_TERMS = (
        "buoyancy_advection",
        "buoyancy_forcing",
    )

    TERM_TO_OUTPUT_INDEX = {
        "buoyancy_advection": B_IDX,
        "buoyancy_forcing": UY_IDX,
    }

    def __init__(
        self,
        canonical_metadata: CanonicalMetadata,
        *,
        representation_mode: str,
        in_channels: int = 16,
        out_channels: int = 4,
        modes1: int = 16,
        modes2: int = 16,
        width: int = 32,
        alpha_max: float = 0.25,
        freeze_m6: bool = True,
    ):
        super().__init__()

        if (
            representation_mode
            not in self.VALID_REPRESENTATION_MODES
        ):
            raise ValueError(
                "representation_mode must be one of "
                f"{self.VALID_REPRESENTATION_MODES}, "
                f"got {representation_mode!r}"
            )

        if alpha_max <= 0.0:
            raise ValueError(
                f"alpha_max must be positive, got {alpha_max}"
            )

        self.representation_mode = (
            representation_mode
        )

        self.alpha_max = float(alpha_max)

        # ========================================================
        # 1. Audited M6 FieldWise backbone
        # ========================================================

        self.m6 = FieldWiseFNO2d(
            in_channels=in_channels,
            out_channels=out_channels,
            modes1=modes1,
            modes2=modes2,
            width=width,
        )

        # ========================================================
        # 2. Representation interface
        #
        # Canonical arm:
        #     use real metadata.
        #
        # Naive arm:
        #     use identity normalization metadata while preserving
        #     the SAME grid, dt and parameter semantics.
        #
        # Both arms then use the SAME CanonicalPDECompiler.
        # ========================================================

        self.canonical_metadata = (
            canonical_metadata
        )

        if representation_mode == "canonical":
            compiler_metadata = (
                canonical_metadata
            )

        else:
            compiler_metadata = (
                self._build_identity_metadata(
                    canonical_metadata
                )
            )

        self.compiler = CanonicalPDECompiler(
            compiler_metadata
        )

        # ========================================================
        # 3. Shared zero-init bounded scalar gates
        #
        # One scalar per active PDE term.
        #
        # alpha_j =
        #     alpha_max * tanh(raw_alpha_j)
        #
        # raw_alpha_j == 0 at initialization, therefore:
        #
        #     alpha_j == 0
        #
        # and epoch-0 output == pure M6 exactly.
        # ========================================================

        self.raw_alpha = nn.ParameterDict(
            {
                term_name: nn.Parameter(
                    torch.zeros(())
                )
                for term_name
                in self.ACTIVE_TERMS
            }
        )

        # ========================================================
        # 4. Validate routing contract immediately
        # ========================================================

        self._validate_active_routing()

        if freeze_m6:
            self.freeze_m6()

    # ============================================================
    # Metadata
    # ============================================================

    @staticmethod
    def _build_identity_metadata(
        base_metadata: CanonicalMetadata,
    ) -> CanonicalMetadata:
        """
        Reproduce the audited R2-3B naive-interface definition.

        Identity normalization:

            mean = 0
            std + eps = 1

        Therefore Canonicalizer becomes:

            x_canonical = x_norm

        while grid / dt / parameter semantics remain identical
        to the canonical arm.
        """

        eps = (
            base_metadata
            .normalization
            .eps
        )

        means = {
            field: 0.0
            for field
            in base_metadata.field_order
        }

        stds = {
            field: 1.0 - eps
            for field
            in base_metadata.field_order
        }

        identity_norm = FieldNormalization(
            mean=means,
            std=stds,
            eps=eps,
        )

        return CanonicalMetadata(
            field_order=(
                base_metadata.field_order
            ),
            normalization=identity_norm,
            grid=base_metadata.grid,
            time=base_metadata.time,
            parameter_order=(
                base_metadata.parameter_order
            ),
        )

    # ============================================================
    # Routing contract
    # ============================================================

    def _validate_active_routing(
        self,
    ) -> None:

        for term_name in self.ACTIVE_TERMS:

            spec = get_rbc_term_spec(
                term_name
            )

            if (
                spec.semantic
                != TermSemantic.RATE
            ):
                raise RuntimeError(
                    f"Active R3 term {term_name!r} "
                    "must be a RATE term."
                )

            if len(spec.targets) != 1:
                raise RuntimeError(
                    f"R3-1 active term {term_name!r} "
                    "must have exactly one target."
                )

            if (
                term_name
                not in self.TERM_TO_OUTPUT_INDEX
            ):
                raise RuntimeError(
                    f"Missing output routing for "
                    f"{term_name!r}."
                )

        if set(self.ACTIVE_TERMS) != set(
            self.TERM_TO_OUTPUT_INDEX
        ):
            raise RuntimeError(
                "ACTIVE_TERMS and "
                "TERM_TO_OUTPUT_INDEX mismatch."
            )

    # ============================================================
    # M6 control
    # ============================================================

    def freeze_m6(
        self,
    ) -> None:

        for parameter in self.m6.parameters():
            parameter.requires_grad = False

    def unfreeze_m6(
        self,
    ) -> None:

        for parameter in self.m6.parameters():
            parameter.requires_grad = True

    def load_m6_state_dict(
        self,
        state_dict,
    ) -> None:
        """
        Strictly load the audited bare-M6 checkpoint state.
        """

        self.m6.load_state_dict(
            state_dict,
            strict=True,
        )

    # ============================================================
    # Gate
    # ============================================================

    def gate_value(
        self,
        term_name: str,
    ) -> torch.Tensor:

        if term_name not in self.raw_alpha:
            raise KeyError(
                f"Inactive / unknown R3 term "
                f"{term_name!r}"
            )

        return (
            self.alpha_max
            *
            torch.tanh(
                self.raw_alpha[
                    term_name
                ]
            )
        )

    # ============================================================
    # Forward
    # ============================================================

    def forward(
        self,
        x_norm: torch.Tensor,
        params: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ):
        """
        Parameters
        ----------
        x_norm:
            [B,16,X,Y] normalized 4-frame history.

        params:
            [B,2] = [log10(Ra), log10(Pr)].

            Currently the two active R3-1 terms do not require
            parameter coefficients, but this argument is retained
            so the shared model interface already matches the
            canonical compiler contract and future registered
            PDE terms.

        Returns
        -------
        normalized finite-step delta:
            [B,4,X,Y]
        """

        # --------------------------------------------------------
        # 1. Frozen M6 base delta
        # --------------------------------------------------------

        base_delta_norm = self.m6(
            x_norm
        )

        # --------------------------------------------------------
        # 2. Empty physics residual in SAME output space
        # --------------------------------------------------------

        physics_residual_norm = (
            torch.zeros_like(
                base_delta_norm
            )
        )

        compiled_terms: Dict[
            str,
            torch.Tensor,
        ] = {}

        gate_values: Dict[
            str,
            torch.Tensor,
        ] = {}

        term_corrections: Dict[
            str,
            torch.Tensor,
        ] = {}

        # --------------------------------------------------------
        # 3. Shared typed PDE compilation + sparse routing
        # --------------------------------------------------------

        for term_name in self.ACTIVE_TERMS:

            spec = get_rbc_term_spec(
                term_name
            )

            term_param = (
                params
                if spec.requires_param_coefficients
                else None
            )

            compiled = (
                self.compiler
                .compile_from_history(
                    term_name,
                    x_norm,
                    param=term_param,
                )
            )

            if (
                compiled.target_delta_norm
                is None
            ):
                raise RuntimeError(
                    f"Active term {term_name!r} "
                    "did not produce a target delta."
                )

            signal = (
                compiled.target_delta_norm
            )

            if signal.ndim != 3:
                raise RuntimeError(
                    f"R3-1 active term {term_name!r} "
                    "must produce [B,X,Y], "
                    f"got {tuple(signal.shape)}"
                )

            alpha = self.gate_value(
                term_name
            )

            correction = (
                alpha
                *
                signal
            )

            output_index = (
                self.TERM_TO_OUTPUT_INDEX[
                    term_name
                ]
            )

            physics_residual_norm[
                :,
                output_index,
                :,
                :,
            ] = (
                physics_residual_norm[
                    :,
                    output_index,
                    :,
                    :,
                ]
                +
                correction
            )

            compiled_terms[
                term_name
            ] = signal

            gate_values[
                term_name
            ] = alpha

            term_corrections[
                term_name
            ] = correction

        # --------------------------------------------------------
        # 4. Shared output-space residual merge
        # --------------------------------------------------------

        output = (
            base_delta_norm
            +
            physics_residual_norm
        )

        if not return_components:
            return output

        return output, {
            "representation_mode":
                self.representation_mode,

            "active_terms":
                self.ACTIVE_TERMS,

            "base_delta_norm":
                base_delta_norm,

            "physics_residual_norm":
                physics_residual_norm,

            "compiled_terms":
                compiled_terms,

            "gate_values":
                gate_values,

            "term_corrections":
                term_corrections,
        }

    # ============================================================
    # Audit helpers
    # ============================================================

    def trainable_parameter_names(
        self,
    ):

        return [
            name
            for name, parameter
            in self.named_parameters()
            if parameter.requires_grad
        ]

    def trainable_parameter_count(
        self,
    ) -> int:

        return sum(
            parameter.numel()
            for parameter
            in self.parameters()
            if parameter.requires_grad
        )

    def frozen_m6_parameter_tensor_count(
        self,
    ) -> int:

        return sum(
            1
            for parameter
            in self.m6.parameters()
            if not parameter.requires_grad
        )


# Explicit aliases for experiment naming.
R31NaiveCanonicalSharedFNO2d = (
    R3PDECouplingFNO2d
)

R3DimensionallyValidSparseCouplingFNO2d = (
    R3PDECouplingFNO2d
)
