import torch

from constants import B_IDX, UY_IDX
from models.operators.fno2d_m10_2path import M10TwoPathFNO2d


class M10RMSCapTwoPathFNO2d(M10TwoPathFNO2d):
    """
    M10-1a PathB-RMSCap-2Path.

    Lightweight residual-stabilization control built directly on
    M10-0 M10TwoPathFNO2d.

    ------------------------------------------------------------
    What stays EXACTLY the same as M10-0
    ------------------------------------------------------------

    - frozen audited M6 backbone
    - Path A definition:
          b' = b - <b>_x
          b' -> u_y delta
    - Path B raw definition:
          -u · grad(b)
          -> buoyancy delta
    - Static / Param / State / StateParam modes
    - state-summary definition
    - parameter features [log10(Ra), log10(Pr)]
    - conditioner architecture
    - bounded gate parameterization
    - zero-gate initialization
    - sparse residual topology

    ------------------------------------------------------------
    The ONLY structural change
    ------------------------------------------------------------

    Before Path B is injected into the buoyancy residual,
    its per-sample spatial RMS is constrained to the maximum
    support observed from TRAIN-ONLY ground-truth states.

        raw Path B:
            P_B

        per-sample RMS:
            r_B = RMS(P_B)

        scale:
            gamma_B = min(
                1,
                cap_B / (r_B + eps)
            )

        stabilized signal:
            P_B_safe = gamma_B * P_B

    Therefore:

        if RMS(P_B) <= cap_B:
            P_B_safe == P_B
            approximately exactly apart from floating-point epsilon
            handling.

        if RMS(P_B) > cap_B:
            only the global magnitude is reduced;
            the spatial pattern and sign structure are preserved.

    ------------------------------------------------------------
    IMPORTANT control decision
    ------------------------------------------------------------

    State / StateParam conditioning continues to receive the
    ORIGINAL raw Path-B signal when constructing the state summary.

    The cap is applied ONLY to the Path-B residual injection.

    This isolates the experimental variable:

        M10-0:
            alpha_B * P_B_raw

        M10-1a:
            alpha_B * P_B_safe

    without simultaneously changing the state conditioner inputs.
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
        if path_b_rms_cap <= 0.0:
            raise ValueError(
                "path_b_rms_cap must be positive."
            )

        if path_b_rms_eps <= 0.0:
            raise ValueError(
                "path_b_rms_eps must be positive."
            )

        super().__init__(
            field_mean=field_mean,
            field_std=field_std,
            mode=mode,
            dx=dx,
            dy=dy,
            in_channels=in_channels,
            out_channels=out_channels,
            modes1=modes1,
            modes2=modes2,
            width=width,
            alpha_max=alpha_max,
            conditioner_hidden=conditioner_hidden,
            freeze_m6=freeze_m6,
        )

        # Fixed, non-learnable stabilization configuration.
        #
        # Store as buffers so that:
        #   - they move with .to(device)
        #   - they are recorded in checkpoints
        #   - they are never optimized
        self.register_buffer(
            "path_b_rms_cap",
            torch.tensor(
                float(path_b_rms_cap),
                dtype=torch.float32,
            ),
        )

        self.register_buffer(
            "path_b_rms_eps",
            torch.tensor(
                float(path_b_rms_eps),
                dtype=torch.float32,
            ),
        )

    # ==============================================================
    # Path-B stabilization
    # ==============================================================

    def _stabilize_path_b(
        self,
        path_b_norm_raw,
    ):
        """
        Apply a per-sample spatial RMS cap.

        Parameters
        ----------
        path_b_norm_raw:
            [B, X, Y]
            original normalized Path-B signal.

        Returns
        -------
        path_b_norm_safe:
            [B, X, Y]

        path_b_rms_raw:
            [B]

        path_b_rms_safe:
            [B]

        path_b_scale:
            [B], in (0, 1]
        """

        if path_b_norm_raw.ndim != 3:
            raise ValueError(
                "Expected Path-B signal [B,X,Y], "
                f"got shape={tuple(path_b_norm_raw.shape)}"
            )

        # Per-sample spatial RMS.
        path_b_rms_raw = torch.sqrt(
            torch.mean(
                path_b_norm_raw
                * path_b_norm_raw,
                dim=(-2, -1),
            )
            + self.path_b_rms_eps.to(
                dtype=path_b_norm_raw.dtype
            )
        )

        cap = self.path_b_rms_cap.to(
            dtype=path_b_norm_raw.dtype
        )

        eps = self.path_b_rms_eps.to(
            dtype=path_b_norm_raw.dtype
        )

        # No amplification is ever allowed.
        #
        # Normal in-support signal:
        #     scale = 1
        #
        # Out-of-support signal:
        #     scale < 1
        path_b_scale = torch.clamp(
            cap / (
                path_b_rms_raw + eps
            ),
            max=1.0,
        )

        path_b_norm_safe = (
            path_b_norm_raw
            * path_b_scale[
                :,
                None,
                None,
            ]
        )

        path_b_rms_safe = torch.sqrt(
            torch.mean(
                path_b_norm_safe
                * path_b_norm_safe,
                dim=(-2, -1),
            )
            + eps
        )

        return (
            path_b_norm_safe,
            path_b_rms_raw,
            path_b_rms_safe,
            path_b_scale,
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
        """
        Returns normalized delta [B,4,X,Y].

        Compared with M10-0, only Path-B residual injection is
        stabilized by the fixed train-support RMS cap.
        """

        # ----------------------------------------------------------
        # 1. Frozen audited M6 prediction.
        # ----------------------------------------------------------

        base_delta_norm = self.m6(
            x_norm
        )

        # ----------------------------------------------------------
        # 2. Construct ORIGINAL raw physics signals.
        # ----------------------------------------------------------

        latest_phys = self._latest_state_phys(
            x_norm
        )

        (
            path_a_norm,
            path_b_norm_raw,
        ) = self._physics_signals(
            latest_phys
        )

        # ----------------------------------------------------------
        # 3. Build condition using RAW signals.
        #
        # This deliberately preserves the exact M10-0
        # State / StateParam conditioner semantics.
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
        # 4. Same M10-0 bounded gates.
        # ----------------------------------------------------------

        (
            alpha_a,
            alpha_b,
            base_raw,
            dynamic_raw,
            combined_raw,
        ) = self._gate_values(
            batch_size=x_norm.shape[0],
            condition=condition,
            device=x_norm.device,
            dtype=x_norm.dtype,
        )

        # ----------------------------------------------------------
        # 5. ONLY new operation:
        #    stabilize Path B before residual injection.
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
        # 6. Same sparse residual topology.
        #
        # Path A -> only u_y
        # Path B -> only buoyancy
        # ----------------------------------------------------------

        residual = torch.zeros_like(
            base_delta_norm
        )

        residual[
            :,
            UY_IDX,
            :,
            :,
        ] = (
            alpha_a[
                :,
                None,
                None,
            ]
            * path_a_norm
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
            * path_b_norm_safe
        )

        output = (
            base_delta_norm
            + residual
        )

        if return_components:
            return output, {
                "mode": self.mode,

                "base_delta_norm":
                    base_delta_norm,

                "physics_residual_norm":
                    residual,

                "path_a_norm":
                    path_a_norm,

                # Compatibility name:
                # actual Path B used in residual.
                "path_b_norm":
                    path_b_norm_safe,

                # Explicit M10-1a diagnostics.
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
                        < (1.0 - 1.0e-7)
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


# Explicit aliases for experiment naming.
M10PathBRMSCapFNO2d = M10RMSCapTwoPathFNO2d
M10_1a_PathBRMSCapFNO2d = M10RMSCapTwoPathFNO2d
