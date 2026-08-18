"""
R1-2: Canonical State Transform
===============================

Transform between the neural-network normalized representation and
the canonical nondimensional RBC field representation.

Current field layout:
    [..., 4, X, Y]

Current flattened history layout:
    [..., T*4, X, Y]

Normalization:
    y_norm = (y - mean_y) / (std_y + eps)

Inverse:
    y = y_norm * (std_y + eps) + mean_y
"""

from __future__ import annotations

from typing import Tuple

import torch

from constants import CONTEXT_LENGTH, NUM_FIELDS
from physics.canonical_metadata import CanonicalMetadata


class Canonicalizer:
    def __init__(
        self,
        metadata: CanonicalMetadata,
    ) -> None:
        self.metadata = metadata

    # ============================================================
    # Validation
    # ============================================================

    def _validate_state_shape(
        self,
        state: torch.Tensor,
    ) -> None:
        if not torch.is_tensor(state):
            raise TypeError(
                f"state must be torch.Tensor, got {type(state)}"
            )

        if state.ndim < 3:
            raise ValueError(
                "state must have layout [...,4,X,Y], "
                f"got shape={tuple(state.shape)}"
            )

        if state.shape[-3] != NUM_FIELDS:
            raise ValueError(
                f"Expected {NUM_FIELDS} fields at axis -3, "
                f"got shape={tuple(state.shape)}"
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

    def _validate_history_shape(
        self,
        history_norm: torch.Tensor,
        context_length: int,
    ) -> None:
        if not torch.is_tensor(history_norm):
            raise TypeError(
                "history_norm must be torch.Tensor, "
                f"got {type(history_norm)}"
            )

        if history_norm.ndim < 3:
            raise ValueError(
                "history_norm must have layout [...,T*4,X,Y], "
                f"got shape={tuple(history_norm.shape)}"
            )

        expected_channels = (
            int(context_length)
            *
            NUM_FIELDS
        )

        if history_norm.shape[-3] != expected_channels:
            raise ValueError(
                "Flattened history channel mismatch. "
                f"Expected {expected_channels}, "
                f"got {history_norm.shape[-3]}"
            )

        if history_norm.shape[-2] != self.metadata.grid.nx:
            raise ValueError(
                "History X resolution mismatch. "
                f"Expected {self.metadata.grid.nx}, "
                f"got {history_norm.shape[-2]}"
            )

        if history_norm.shape[-1] != self.metadata.grid.ny:
            raise ValueError(
                "History Y resolution mismatch. "
                f"Expected {self.metadata.grid.ny}, "
                f"got {history_norm.shape[-1]}"
            )

    # ============================================================
    # Affine normalization tensors
    # ============================================================

    def _affine_tensors(
        self,
        reference: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        means = [
            self.metadata.normalization.mean_for(field)
            for field in self.metadata.field_order
        ]

        scales = [
            (
                self.metadata.normalization.std_for(field)
                +
                self.metadata.normalization.eps
            )
            for field in self.metadata.field_order
        ]

        shape = [1] * reference.ndim
        shape[-3] = NUM_FIELDS

        mean_tensor = torch.as_tensor(
            means,
            dtype=reference.dtype,
            device=reference.device,
        ).reshape(shape)

        scale_tensor = torch.as_tensor(
            scales,
            dtype=reference.dtype,
            device=reference.device,
        ).reshape(shape)

        return mean_tensor, scale_tensor

    # ============================================================
    # State transforms
    # ============================================================

    def denormalize_state(
        self,
        state_norm: torch.Tensor,
    ) -> torch.Tensor:

        self._validate_state_shape(
            state_norm
        )

        mean, scale = self._affine_tensors(
            state_norm
        )

        return (
            state_norm * scale
            +
            mean
        )

    def normalize_state(
        self,
        state_canonical: torch.Tensor,
    ) -> torch.Tensor:

        self._validate_state_shape(
            state_canonical
        )

        mean, scale = self._affine_tensors(
            state_canonical
        )

        return (
            state_canonical - mean
        ) / scale

    # ============================================================
    # History
    # ============================================================

    def latest_state_norm(
        self,
        history_norm: torch.Tensor,
        *,
        context_length: int = CONTEXT_LENGTH,
    ) -> torch.Tensor:

        if context_length <= 0:
            raise ValueError(
                "context_length must be positive, "
                f"got {context_length}"
            )

        self._validate_history_shape(
            history_norm,
            context_length=context_length,
        )

        return history_norm[
            ...,
            -NUM_FIELDS:,
            :,
            :,
        ]

    def latest_state_canonical(
        self,
        history_norm: torch.Tensor,
        *,
        context_length: int = CONTEXT_LENGTH,
    ) -> torch.Tensor:

        latest_norm = self.latest_state_norm(
            history_norm,
            context_length=context_length,
        )

        return self.denormalize_state(
            latest_norm
        )
