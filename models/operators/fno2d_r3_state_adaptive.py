from __future__ import annotations

from typing import Dict, Mapping

import torch
import torch.nn as nn

from constants import (
    B_IDX,
    UX_IDX,
    UY_IDX,
    FIELD_ORDER,
)

from physics.canonical_metadata import (
    CanonicalMetadata,
)

from physics.rbc_terms import (
    get_rbc_term_spec,
)

from models.operators.fno2d_r3_pde_coupling import (
    R3PDECouplingFNO2d,
)


class R3StateAdaptiveConditioner(nn.Module):
    """
    Capacity-matched R3-2 conditioner.

    Architecture is IDENTICAL for:
        R3-2a ParamOnly
        R3-2b StateParam

    Input:
        [B, 10]

    Output:
        [B, 2] delta_raw

    The final Linear layer is initialized to exact zero so:

        delta_raw == 0

    for every sample at epoch 0.

    Therefore, after loading the frozen R3-1b parent:

        R3-2a(epoch0)
        =
        R3-2b(epoch0)
        =
        R3-1b
    """

    INPUT_DIM = 10
    HIDDEN_DIMS = (32, 16)
    OUTPUT_DIM = 2

    def __init__(
        self,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                self.INPUT_DIM,
                self.HIDDEN_DIMS[0],
            ),
            nn.GELU(),
            nn.Linear(
                self.HIDDEN_DIMS[0],
                self.HIDDEN_DIMS[1],
            ),
            nn.GELU(),
            nn.Linear(
                self.HIDDEN_DIMS[1],
                self.OUTPUT_DIM,
            ),
        )

        # Exact epoch-0 parent reproduction.
        final_linear = self.net[-1]

        if not isinstance(
            final_linear,
            nn.Linear,
        ):
            raise RuntimeError(
                "R3-2 conditioner final layer "
                "must be nn.Linear."
            )

        nn.init.zeros_(
            final_linear.weight
        )

        nn.init.zeros_(
            final_linear.bias
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:

        if (
            features.ndim != 2
            or
            features.shape[-1]
            != self.INPUT_DIM
        ):
            raise ValueError(
                "R3-2 conditioner expects "
                f"[B,{self.INPUT_DIM}], got "
                f"{tuple(features.shape)}"
            )

        return self.net(
            features
        )


class R3StateAdaptivePDECouplingFNO2d(
    R3PDECouplingFNO2d
):
    """
    R3-2 State-Adaptive Dimension-Valid Coupling.

    ============================================================
    Scientific question
    ============================================================

    R3-1 already established the matched comparison:

        Naive representation
        vs
        Canonical dimension-valid representation.

    R3-2 does NOT reopen that question.

    R3-2 starts strictly from the frozen R3-1b Canonical model
    and asks:

        Does adapting coupling strength to inference-visible
        current state provide value beyond parameter-only
        adaptation?

    ============================================================
    Matched arms
    ============================================================

    R3-2a ParamOnly:
        same 10-D input tensor and same conditioner architecture,
        but features 0..7 are set to exact zero.

        Visible information:
            logRa
            logPr

    R3-2b StateParam:
        same conditioner architecture and parameter count.

        Visible information:
            current normalized-state summaries
            current canonical PDE-signal summaries
            logRa
            logPr

    The ONLY arm difference is whether features 0..7 are visible.

    ============================================================
    Adaptive gate
    ============================================================

    Frozen R3-1b parent:

        raw_alpha_parent_j

    Trainable R3-2 conditioner:

        delta_raw_j(s_t, Ra, Pr)

    Effective gate:

        alpha_j =
            alpha_max_j
            * tanh(
                raw_alpha_parent_j
                +
                delta_raw_j
            )

    raw_alpha_parent_j is frozen.

    alpha_max_j is fixed and nonlearnable.

    Only the conditioner is trainable.

    ============================================================
    Conditioner features
    ============================================================

    Feature order:

        0  state_b_mean_norm
        1  state_b_rms_norm

        2  state_ux_mean_norm
        3  state_ux_rms_norm

        4  state_uy_mean_norm
        5  state_uy_rms_norm

        6  log1p(
               canonical buoyancy-advection
               target-delta signal RMS
           )

        7  log1p(
               canonical buoyancy-forcing
               target-delta signal RMS
           )

        8  logRa
        9  logPr

    All state features are inference-visible.

    No GT state enters the conditioner.

    PDE signal summaries are computed BEFORE adaptive gate
    multiplication and already live in the normalized finite-step
    target-delta merge space produced by the canonical compiler.

    ============================================================
    Explicitly NOT included
    ============================================================

        - Utility Gate
        - utility labels
        - temporal d1 / d2 features
        - horizon index
        - RMSCap
        - PDE loss
        - new PDE terms
        - pressure pathway
        - learnable spatial Phi
    """

    VALID_ADAPTATION_MODES = (
        "paramonly",
        "stateparam",
    )

    FEATURE_NAMES = (
        "state_b_mean_norm",
        "state_b_rms_norm",
        "state_ux_mean_norm",
        "state_ux_rms_norm",
        "state_uy_mean_norm",
        "state_uy_rms_norm",
        (
            "log1p_canonical_"
            "buoyancy_advection_"
            "signal_rms_norm"
        ),
        (
            "log1p_canonical_"
            "buoyancy_forcing_"
            "signal_rms_norm"
        ),
        "logRa",
        "logPr",
    )

    TERM_TO_CONDITIONER_INDEX = {
        "buoyancy_advection": 0,
        "buoyancy_forcing": 1,
    }

    STATE_FEATURE_COUNT = 8
    PARAM_FEATURE_START = 8

    RMS_EPS = 1.0e-12

    def __init__(
        self,
        canonical_metadata: CanonicalMetadata,
        *,
        adaptation_mode: str,
        alpha_max_by_term: Mapping[
            str,
            float,
        ],
        in_channels: int = 16,
        out_channels: int = 4,
        modes1: int = 16,
        modes2: int = 16,
        width: int = 32,
    ):
        if (
            adaptation_mode
            not in self.VALID_ADAPTATION_MODES
        ):
            raise ValueError(
                "adaptation_mode must be one of "
                f"{self.VALID_ADAPTATION_MODES}, "
                f"got {adaptation_mode!r}"
            )

        # R3-2 is Canonical ONLY.
        #
        # There is deliberately no representation-mode option
        # here. The Naive-vs-Canonical question was closed by R3-1.
        super().__init__(
            canonical_metadata=(
                canonical_metadata
            ),
            representation_mode="canonical",
            alpha_max_by_term=(
                alpha_max_by_term
            ),
            in_channels=in_channels,
            out_channels=out_channels,
            modes1=modes1,
            modes2=modes2,
            width=width,
            freeze_m6=True,
        )

        self.adaptation_mode = (
            adaptation_mode
        )

        # Validate the exact project field convention required
        # by the state summaries.
        if tuple(
            canonical_metadata.field_order
        ) != tuple(
            FIELD_ORDER
        ):
            raise RuntimeError(
                "R3-2 requires canonical metadata "
                "field order to match FIELD_ORDER.\n"
                f"metadata={tuple(canonical_metadata.field_order)}\n"
                f"project={tuple(FIELD_ORDER)}"
            )

        if len(
            self.FEATURE_NAMES
        ) != 10:
            raise RuntimeError(
                "R3-2 feature contract must contain "
                "exactly 10 features."
            )

        if set(
            self.ACTIVE_TERMS
        ) != set(
            self.TERM_TO_CONDITIONER_INDEX
        ):
            raise RuntimeError(
                "R3-2 active terms and conditioner "
                "output routing mismatch."
            )

        # Same architecture / same capacity in both arms.
        self.conditioner = (
            R3StateAdaptiveConditioner()
        )

        # Parent raw_alpha MUST NOT move in R3-2.
        self.freeze_parent_raw_alpha()

        # M6 was frozen by the parent constructor.
        # Verify the R3-2 trainable-parameter contract now.
        self._validate_trainable_contract()

    # ============================================================
    # Parent R3-1b loading / freezing
    # ============================================================

    def freeze_parent_raw_alpha(
        self,
    ) -> None:

        for parameter in (
            self.raw_alpha.parameters()
        ):
            parameter.requires_grad = (
                False
            )

    def load_parent_r3_1b_state_dict(
        self,
        parent_state_dict,
    ) -> None:
        """
        Strictly load a bare R3-1b model state_dict.

        The child model contains one additional module:

            conditioner.*

        Therefore the expected parent state_dict is exactly the
        current child state_dict with all conditioner keys removed.

        No missing / extra parent keys are accepted.
        """

        current_state = (
            self.state_dict()
        )

        parent_expected_keys = {
            key
            for key
            in current_state.keys()
            if not key.startswith(
                "conditioner."
            )
        }

        supplied_keys = set(
            parent_state_dict.keys()
        )

        if (
            supplied_keys
            != parent_expected_keys
        ):
            missing = sorted(
                parent_expected_keys
                -
                supplied_keys
            )

            extra = sorted(
                supplied_keys
                -
                parent_expected_keys
            )

            raise RuntimeError(
                "R3-1b parent state_dict does not "
                "match the R3-2 inherited parent state.\n"
                f"Missing: {missing}\n"
                f"Extra:   {extra}"
            )

        merged_state = {
            key:
                value
            for (
                key,
                value,
            )
            in current_state.items()
        }

        for (
            key,
            value,
        ) in parent_state_dict.items():
            merged_state[
                key
            ] = value

        self.load_state_dict(
            merged_state,
            strict=True,
        )

        # Reassert the frozen contract after loading.
        self.freeze_m6()
        self.freeze_parent_raw_alpha()

        self._validate_trainable_contract()

    # ============================================================
    # Safety: R3-2 must never unfreeze the M6 parent.
    # ============================================================

    def unfreeze_m6(
        self,
    ) -> None:

        raise RuntimeError(
            "R3-2 contract forbids unfreezing M6."
        )

    # ============================================================
    # Feature helpers
    # ============================================================

    @staticmethod
    def _spatial_mean(
        x: torch.Tensor,
    ) -> torch.Tensor:

        return torch.mean(
            x,
            dim=(-2, -1),
        )

    @classmethod
    def _spatial_rms(
        cls,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return torch.sqrt(
            torch.mean(
                x * x,
                dim=(-2, -1),
            )
            +
            cls.RMS_EPS
        )

    def _build_full_conditioner_features(
        self,
        *,
        x_norm: torch.Tensor,
        params: torch.Tensor,
        compiled_terms: Mapping[
            str,
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        """
        Build the 10-D REAL inference-visible feature vector.

        ParamOnly masking is deliberately NOT performed here.

        This lets both experimental arms construct the same
        underlying feature vector first; the control then masks
        features 0..7 to exact zero immediately before the shared
        conditioner.
        """

        if x_norm.ndim != 4:
            raise ValueError(
                "R3-2 expects x_norm [B,C,X,Y], "
                f"got {tuple(x_norm.shape)}"
            )

        if (
            params.ndim != 2
            or
            params.shape[-1] != 2
        ):
            raise ValueError(
                "R3-2 requires params "
                "[B,2]=[logRa,logPr], got "
                f"{tuple(params.shape)}"
            )

        if (
            params.shape[0]
            !=
            x_norm.shape[0]
        ):
            raise ValueError(
                "R3-2 x_norm / params batch "
                "size mismatch."
            )

        num_fields = len(
            FIELD_ORDER
        )

        if (
            x_norm.shape[1]
            <
            num_fields
        ):
            raise ValueError(
                "R3-2 history does not contain "
                "a complete latest state."
            )

        latest = (
            x_norm[
                :,
                -num_fields:,
                :,
                :,
            ]
        )

        b = latest[
            :,
            B_IDX,
            :,
            :,
        ]

        ux = latest[
            :,
            UX_IDX,
            :,
            :,
        ]

        uy = latest[
            :,
            UY_IDX,
            :,
            :,
        ]

        state_features = torch.stack(
            [
                self._spatial_mean(
                    b
                ),
                self._spatial_rms(
                    b
                ),

                self._spatial_mean(
                    ux
                ),
                self._spatial_rms(
                    ux
                ),

                self._spatial_mean(
                    uy
                ),
                self._spatial_rms(
                    uy
                ),
            ],
            dim=-1,
        )

        signal_feature_list = []

        for term_name in (
            self.ACTIVE_TERMS
        ):

            if (
                term_name
                not in
                compiled_terms
            ):
                raise RuntimeError(
                    "Missing compiled R3-2 term "
                    f"{term_name!r} while building "
                    "conditioner features."
                )

            signal = (
                compiled_terms[
                    term_name
                ]
            )

            if signal.ndim != 3:
                raise RuntimeError(
                    "R3-2 canonical signal must "
                    "have shape [B,X,Y], got "
                    f"{tuple(signal.shape)} "
                    f"for term {term_name!r}."
                )

            signal_rms = (
                self._spatial_rms(
                    signal
                )
            )

            signal_feature_list.append(
                torch.log1p(
                    signal_rms
                )
            )

        signal_features = torch.stack(
            signal_feature_list,
            dim=-1,
        )

        param_features = params.to(
            device=x_norm.device,
            dtype=x_norm.dtype,
        )

        features = torch.cat(
            [
                state_features,
                signal_features,
                param_features,
            ],
            dim=-1,
        )

        if (
            features.shape[-1]
            !=
            R3StateAdaptiveConditioner.INPUT_DIM
        ):
            raise RuntimeError(
                "R3-2 feature-construction contract "
                "was violated: expected 10 features, "
                f"got {features.shape[-1]}."
            )

        return features

    def _apply_arm_mask(
        self,
        full_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Capacity-matched causal control.

        StateParam:
            use all 10 real features.

        ParamOnly:
            copy the SAME 10-D feature tensor,
            then set indices 0..7 to exact zero.

            indices 8..9 remain real logRa/logPr.
        """

        conditioner_input = (
            full_features.clone()
        )

        if (
            self.adaptation_mode
            ==
            "paramonly"
        ):

            conditioner_input[
                :,
                :
                self.STATE_FEATURE_COUNT,
            ] = 0.0

        elif (
            self.adaptation_mode
            ==
            "stateparam"
        ):

            pass

        else:
            raise RuntimeError(
                "Invalid R3-2 adaptation mode "
                f"{self.adaptation_mode!r}."
            )

        return conditioner_input

    # ============================================================
    # Gate helpers
    # ============================================================

    def parent_gate_value(
        self,
        term_name: str,
    ) -> torch.Tensor:
        """
        Frozen scalar R3-1b gate.
        """

        return super().gate_value(
            term_name
        )

    def effective_gate_value(
        self,
        term_name: str,
        delta_raw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Per-sample R3-2 gate.

        delta_raw:
            [B]

        Returns:
            [B]
        """

        if (
            term_name
            not in
            self.raw_alpha
        ):
            raise KeyError(
                "Inactive / unknown R3-2 term "
                f"{term_name!r}"
            )

        if delta_raw.ndim != 1:
            raise ValueError(
                "delta_raw for one R3-2 term "
                "must have shape [B], got "
                f"{tuple(delta_raw.shape)}"
            )

        alpha_max = (
            self.alpha_max_by_term[
                term_name
            ]
        )

        parent_raw = (
            self.raw_alpha[
                term_name
            ]
        )

        return (
            alpha_max
            *
            torch.tanh(
                parent_raw
                +
                delta_raw
            )
        )

    # ============================================================
    # Forward
    # ============================================================

    def forward(
        self,
        x_norm: torch.Tensor,
        params: torch.Tensor,
        return_components: bool = False,
    ):
        """
        Parameters
        ----------
        x_norm:
            [B,16,X,Y] normalized autoregressive history.

        params:
            [B,2] = [log10(Ra), log10(Pr)].

        Returns
        -------
        normalized finite-step delta:
            [B,4,X,Y]
        """

        if params is None:
            raise ValueError(
                "R3-2 requires params=[logRa,logPr] "
                "because both matched arms include "
                "parameter conditioning."
            )

        # --------------------------------------------------------
        # 1. Frozen M6 base delta
        # --------------------------------------------------------

        base_delta_norm = (
            self.m6(
                x_norm
            )
        )

        # --------------------------------------------------------
        # 2. Compile BOTH canonical PDE signals first.
        #
        # Important:
        # conditioner features use the UNWEIGHTED signal RMS,
        # before any adaptive alpha is applied.
        # --------------------------------------------------------

        compiled_terms: Dict[
            str,
            torch.Tensor,
        ] = {}

        for term_name in (
            self.ACTIVE_TERMS
        ):

            spec = get_rbc_term_spec(
                term_name
            )

            term_param = (
                params
                if
                spec.requires_param_coefficients
                else
                None
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
                    f"Active R3-2 term "
                    f"{term_name!r} "
                    "did not produce a "
                    "target delta."
                )

            signal = (
                compiled.target_delta_norm
            )

            if signal.ndim != 3:
                raise RuntimeError(
                    f"R3-2 active term "
                    f"{term_name!r} "
                    "must produce [B,X,Y], "
                    f"got {tuple(signal.shape)}."
                )

            compiled_terms[
                term_name
            ] = signal

        # --------------------------------------------------------
        # 3. Build SAME real 10-D feature tensor for both arms.
        # --------------------------------------------------------

        full_conditioner_features = (
            self._build_full_conditioner_features(
                x_norm=x_norm,
                params=params,
                compiled_terms=(
                    compiled_terms
                ),
            )
        )

        # --------------------------------------------------------
        # 4. Apply ONLY experimental arm difference.
        # --------------------------------------------------------

        conditioner_input = (
            self._apply_arm_mask(
                full_conditioner_features
            )
        )

        # --------------------------------------------------------
        # 5. State/parameter-dependent raw-gate residual.
        #
        # At initialization:
        #
        #     delta_raw == exact zero
        #
        # because the final Linear is zero-initialized.
        # --------------------------------------------------------

        delta_raw = (
            self.conditioner(
                conditioner_input
            )
        )

        if (
            delta_raw.ndim != 2
            or
            delta_raw.shape[-1]
            != len(
                self.ACTIVE_TERMS
            )
        ):
            raise RuntimeError(
                "R3-2 conditioner output contract "
                "must be [B,2], got "
                f"{tuple(delta_raw.shape)}."
            )

        # --------------------------------------------------------
        # 6. Apply adaptive canonical coupling.
        # --------------------------------------------------------

        physics_residual_norm = (
            torch.zeros_like(
                base_delta_norm
            )
        )

        parent_gate_values: Dict[
            str,
            torch.Tensor,
        ] = {}

        effective_gate_values: Dict[
            str,
            torch.Tensor,
        ] = {}

        term_delta_raw: Dict[
            str,
            torch.Tensor,
        ] = {}

        term_corrections: Dict[
            str,
            torch.Tensor,
        ] = {}

        for term_name in (
            self.ACTIVE_TERMS
        ):

            conditioner_index = (
                self.TERM_TO_CONDITIONER_INDEX[
                    term_name
                ]
            )

            this_delta_raw = (
                delta_raw[
                    :,
                    conditioner_index,
                ]
            )

            parent_alpha = (
                self.parent_gate_value(
                    term_name
                )
            )

            effective_alpha = (
                self.effective_gate_value(
                    term_name,
                    this_delta_raw,
                )
            )

            signal = (
                compiled_terms[
                    term_name
                ]
            )

            correction = (
                effective_alpha[
                    :,
                    None,
                    None,
                ]
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

            parent_gate_values[
                term_name
            ] = parent_alpha

            effective_gate_values[
                term_name
            ] = effective_alpha

            term_delta_raw[
                term_name
            ] = this_delta_raw

            term_corrections[
                term_name
            ] = correction

        # --------------------------------------------------------
        # 7. Same output-space merge as R3-1b.
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
                "canonical",

            "adaptation_mode":
                self.adaptation_mode,

            "active_terms":
                self.ACTIVE_TERMS,

            "alpha_max_by_term":
                dict(
                    self.alpha_max_by_term
                ),

            "feature_names":
                self.FEATURE_NAMES,

            "base_delta_norm":
                base_delta_norm,

            "physics_residual_norm":
                physics_residual_norm,

            "compiled_terms":
                compiled_terms,

            # Frozen R3-1b scalar gates.
            "parent_gate_values":
                parent_gate_values,

            # Backward-friendly semantic:
            # gate_values means the ACTUAL R3-2 gates used.
            "gate_values":
                effective_gate_values,

            "effective_gate_values":
                effective_gate_values,

            "delta_raw":
                delta_raw,

            "term_delta_raw":
                term_delta_raw,

            "full_conditioner_features":
                full_conditioner_features,

            "conditioner_input":
                conditioner_input,

            "term_corrections":
                term_corrections,
        }

    # ============================================================
    # Contract audits
    # ============================================================

    def conditioner_parameter_names(
        self,
    ):

        return [
            name
            for (
                name,
                _,
            )
            in self.named_parameters()
            if name.startswith(
                "conditioner."
            )
        ]

    def conditioner_parameter_count(
        self,
    ) -> int:

        return sum(
            parameter.numel()
            for (
                name,
                parameter,
            )
            in self.named_parameters()
            if name.startswith(
                "conditioner."
            )
        )

    def frozen_parent_raw_alpha_names(
        self,
    ):

        return [
            name
            for (
                name,
                parameter,
            )
            in self.named_parameters()
            if (
                name.startswith(
                    "raw_alpha."
                )
                and
                not parameter.requires_grad
            )
        ]

    def _validate_trainable_contract(
        self,
    ) -> None:

        trainable_names = (
            self.trainable_parameter_names()
        )

        bad = [
            name
            for name
            in trainable_names
            if not name.startswith(
                "conditioner."
            )
        ]

        if bad:
            raise RuntimeError(
                "R3-2 contract violation: "
                "non-conditioner parameters are trainable:\n"
                f"{bad}"
            )

        expected_conditioner_names = set(
            self.conditioner_parameter_names()
        )

        if set(
            trainable_names
        ) != expected_conditioner_names:
            raise RuntimeError(
                "R3-2 contract violation: "
                "not every conditioner parameter "
                "is trainable."
            )

        if self.conditioner_parameter_count() != 914:
            raise RuntimeError(
                "R3-2 conditioner parameter count "
                "must be exactly 914, got "
                f"{self.conditioner_parameter_count()}."
            )

        expected_parent_raw = {
            (
                "raw_alpha."
                "buoyancy_advection"
            ),
            (
                "raw_alpha."
                "buoyancy_forcing"
            ),
        }

        actual_parent_raw = set(
            self.frozen_parent_raw_alpha_names()
        )

        if (
            actual_parent_raw
            !=
            expected_parent_raw
        ):
            raise RuntimeError(
                "R3-2 frozen parent raw-alpha "
                "contract mismatch.\n"
                f"Expected: {sorted(expected_parent_raw)}\n"
                f"Actual:   {sorted(actual_parent_raw)}"
            )

        if any(
            parameter.requires_grad
            for parameter
            in self.m6.parameters()
        ):
            raise RuntimeError(
                "R3-2 contract violation: "
                "M6 must remain frozen."
            )


# Explicit aliases for experiment naming.

R32ParamOnlyStateParamSharedFNO2d = (
    R3StateAdaptivePDECouplingFNO2d
)

R3StateAdaptiveDimensionallyValidCouplingFNO2d = (
    R3StateAdaptivePDECouplingFNO2d
)
