import torch
import torch.nn as nn

from constants import B_IDX

from models.operators.fno2d_m10_1_rmscap import (
    M10RMSCapTwoPathFNO2d,
)


class M10BOnlyRMSCapFNO2d(
    M10RMSCapTwoPathFNO2d
):
    """
    M10-2 formal-candidate rebase:
    B-only StateParam RMSCap coupling.

    ------------------------------------------------------------
    Goal
    ------------------------------------------------------------

    Build a fair Path-B-only candidate from the frozen audited M6
    backbone, removing Path-A OUTPUT injection during training.

    Only physics residual injected into the model output:

        Path B:
            P_B = -u · grad(b)
            -> buoyancy normalized delta

    Path A:
        NO output injection.

    The horizontal buoyancy-anomaly quantity
        b - mean_x(b)
    is still allowed inside the compact state summary because
    Legacy10 uses it only as an inference-visible state descriptor.

    ------------------------------------------------------------
    Prediction
    ------------------------------------------------------------

        delta_b =
            delta_b_M6
            + alpha_B * P_B_safe

    where

        P_B_safe = RMSCap(P_B)

    All other output channels remain exactly the M6 delta.

    M6 stays frozen.
    """

    def __init__(
        self,
        field_mean,
        field_std,
        mode,
        *,
        dx,
        dy,
        path_b_rms_cap,
        path_b_rms_eps=1.0e-12,
        in_channels=16,
        out_channels=4,
        modes1=16,
        modes2=16,
        width=32,
        alpha_max=0.25,
        conditioner_hidden=32,
        freeze_m6=True,
    ):

        super().__init__(
            field_mean=field_mean,
            field_std=field_std,
            mode=mode,
            dx=dx,
            dy=dy,
            path_b_rms_cap=path_b_rms_cap,
            path_b_rms_eps=path_b_rms_eps,
            in_channels=in_channels,
            out_channels=out_channels,
            modes1=modes1,
            modes2=modes2,
            width=width,
            alpha_max=alpha_max,
            conditioner_hidden=conditioner_hidden,
            freeze_m6=freeze_m6,
        )

        # ==========================================================
        # Path A is permanently disabled as an OUTPUT gate.
        #
        # Keep inherited parameter only for checkpoint / diagnostic
        # compatibility, but it is fixed at exactly zero.
        # ==========================================================

        with torch.no_grad():
            self.base_raw_alpha_a.zero_()

        self.base_raw_alpha_a.requires_grad_(
            False
        )

        # ==========================================================
        # Replace inherited 2-output conditioner by a B-only
        # conditioner.
        #
        # StateParam:
        #   8 state summary features + 2 parameter features
        #   -> hidden
        #   -> one dynamic Path-B correction
        # ==========================================================

        condition_dim = (
            self._condition_dim_for_mode(
                mode
            )
        )

        if condition_dim == 0:

            self.conditioner = None

        else:

            self.conditioner = nn.Sequential(
                nn.Linear(
                    condition_dim,
                    self.conditioner_hidden,
                ),
                nn.GELU(),
                nn.Linear(
                    self.conditioner_hidden,
                    1,
                ),
            )

            # Zero dynamic residual at initialization.
            #
            # Together with base_raw_alpha_b = 0 inherited from
            # M10, this guarantees epoch-0 output == pure M6.
            final_layer = self.conditioner[-1]

            nn.init.zeros_(
                final_layer.weight
            )

            nn.init.zeros_(
                final_layer.bias
            )

    # ==============================================================
    # B-only gate
    # ==============================================================

    def _gate_b_values(
        self,
        batch_size,
        condition,
        device,
        dtype,
    ):

        base_raw_b = (
            self.base_raw_alpha_b
            .view(1)
            .expand(batch_size)
        )

        if self.conditioner is None:

            dynamic_raw_b = torch.zeros(
                batch_size,
                device=device,
                dtype=dtype,
            )

        else:

            if condition is None:
                raise RuntimeError(
                    "Dynamic B-only mode received "
                    "no condition."
                )

            dynamic_raw_b = torch.tanh(
                self.conditioner(
                    condition
                ).squeeze(-1)
            )

        combined_raw_b = (
            base_raw_b
            +
            dynamic_raw_b
        )

        alpha_b = (
            self.alpha_max
            *
            torch.tanh(
                combined_raw_b
            )
        )

        return (
            alpha_b,
            base_raw_b,
            dynamic_raw_b,
            combined_raw_b,
        )

    # ==============================================================
    # Forward
    # ==============================================================

    def forward(
        self,
        x_norm,
        params=None,
        return_components=False,
    ):

        # ----------------------------------------------------------
        # 1. Frozen audited M6 prediction.
        # ----------------------------------------------------------

        base_delta_norm = self.m6(
            x_norm
        )

        # ----------------------------------------------------------
        # 2. Construct physics signals.
        #
        # Path A is computed only because its RMS remains part of
        # the compact Legacy state summary.
        # It is NEVER injected into u_y.
        # ----------------------------------------------------------

        latest_phys = (
            self._latest_state_phys(
                x_norm
            )
        )

        (
            path_a_norm,
            path_b_norm_raw,
        ) = self._physics_signals(
            latest_phys
        )

        # ----------------------------------------------------------
        # 3. Same compact StateParam condition.
        # ----------------------------------------------------------

        (
            condition,
            state_summary,
            param_features,
        ) = self._build_condition(
            x_norm=x_norm,
            path_a_norm=path_a_norm,
            path_b_norm=path_b_norm_raw,
            params=params,
        )

        # ----------------------------------------------------------
        # 4. Single Path-B strength.
        # ----------------------------------------------------------

        (
            alpha_b,
            base_raw_b,
            dynamic_raw_b,
            combined_raw_b,
        ) = self._gate_b_values(
            batch_size=x_norm.shape[0],
            condition=condition,
            device=x_norm.device,
            dtype=x_norm.dtype,
        )

        # ----------------------------------------------------------
        # 5. RMSCap Path B.
        # ----------------------------------------------------------

        (
            path_b_norm_safe,
            path_b_rms_raw,
            path_b_rms_safe,
            path_b_scale,
        ) = self._stabilize_path_b(
            path_b_norm_raw
        )

        # ----------------------------------------------------------
        # 6. B-only sparse output residual.
        # ----------------------------------------------------------

        residual = torch.zeros_like(
            base_delta_norm
        )

        residual[
            :,
            B_IDX,
            :,
            :,
        ] = (
            alpha_b[
                :,
                None,
                None,
            ]
            *
            path_b_norm_safe
        )

        output = (
            base_delta_norm
            +
            residual
        )

        if return_components:

            # Path A output strength is exactly zero.
            alpha_a = torch.zeros_like(
                alpha_b
            )

            # Keep compatibility with diagnostics expecting [B,2].
            base_raw = torch.stack(
                [
                    torch.zeros_like(
                        base_raw_b
                    ),
                    base_raw_b,
                ],
                dim=-1,
            )

            dynamic_raw = torch.stack(
                [
                    torch.zeros_like(
                        dynamic_raw_b
                    ),
                    dynamic_raw_b,
                ],
                dim=-1,
            )

            combined_raw = torch.stack(
                [
                    torch.zeros_like(
                        combined_raw_b
                    ),
                    combined_raw_b,
                ],
                dim=-1,
            )

            return output, {
                "mode":
                    self.mode,

                "base_delta_norm":
                    base_delta_norm,

                "physics_residual_norm":
                    residual,

                # State-summary diagnostic only.
                "path_a_norm":
                    path_a_norm,

                "path_b_norm":
                    path_b_norm_safe,

                "path_b_norm_raw":
                    path_b_norm_raw,

                "path_b_norm_safe":
                    path_b_norm_safe,

                "path_b_rms_raw":
                    path_b_rms_raw,

                "path_b_rms_safe":
                    path_b_rms_safe,

                "path_b_scale":
                    path_b_scale,

                "path_b_cap_active":
                    (
                        path_b_scale
                        <
                        (1.0 - 1.0e-7)
                    ),

                "alpha_a":
                    alpha_a,

                "alpha_b":
                    alpha_b,

                "base_raw":
                    base_raw,

                "dynamic_raw":
                    dynamic_raw,

                "combined_raw":
                    combined_raw,

                "condition":
                    condition,

                "state_summary":
                    state_summary,

                "param_features":
                    param_features,
            }

        return output


M10UtilityCandidateBOnlyFNO2d = (
    M10BOnlyRMSCapFNO2d
)
