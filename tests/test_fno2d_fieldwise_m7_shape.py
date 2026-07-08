import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.operators.fno2d_fieldwise import FieldWiseFNO2d
from models.operators.fno2d_fieldwise_m7 import M7FieldCouplingFNO2d


def main():
    torch.manual_seed(123)

    B = 2
    H = 32
    W = 64

    model = M7FieldCouplingFNO2d(
        in_channels=16,
        out_channels=4,
        modes1=8,
        modes2=8,
        width=32,
        context_length=4,
        num_fields=4,
    )

    x = torch.randn(B, 16, H, W, requires_grad=True)
    y = model(x)

    print("========== M7FieldCouplingFNO2d Shape Test ==========")
    print(f"Input shape : {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}")

    assert y.shape == (B, 4, H, W), f"Wrong output shape: {y.shape}"
    assert torch.isfinite(y).all(), "Output contains NaN or Inf."

    loss = y.square().mean()
    loss.backward()

    assert x.grad is not None, "Input gradient is None."
    assert torch.isfinite(x.grad).all(), "Input gradient contains NaN or Inf."
    assert model.field_coupling.coupling_matrix.grad is not None, (
        "field_coupling.coupling_matrix grad is None."
    )
    assert model.field_coupling.residual_gate.grad is not None, (
        "field_coupling.residual_gate grad is None."
    )

    info = model(x.detach(), return_features=True)

    assert info["out"].shape == (B, 4, H, W)
    assert len(info["field_features"]) == 4
    assert len(info["coupled_field_features"]) == 4
    assert info["field_features"][0].shape == (B, 8, H, W)
    assert info["coupled_field_features"][0].shape == (B, 8, H, W)

    gate = info["coupling_gate"].detach().cpu()
    coupling = info["coupling_matrix"].detach().cpu()

    print(f"Loss: {loss.item():.6f}")
    print(f"Gate values: {gate.tolist()}")
    print("Effective coupling matrix:")
    print(coupling)

    # Check state_dict compatibility direction:
    # M7 should be able to load M6 FieldWiseEncoder weights with strict=False.
    m6 = FieldWiseFNO2d(
        in_channels=16,
        out_channels=4,
        modes1=8,
        modes2=8,
        width=32,
        context_length=4,
        num_fields=4,
    )

    m7_from_m6 = M7FieldCouplingFNO2d(
        in_channels=16,
        out_channels=4,
        modes1=8,
        modes2=8,
        width=32,
        context_length=4,
        num_fields=4,
    )

    missing_keys, unexpected_keys = m7_from_m6.load_state_dict(
        m6.state_dict(),
        strict=False,
    )

    print("Missing keys when loading M6 -> M7:")
    for k in missing_keys:
        print("  ", k)

    print("Unexpected keys when loading M6 -> M7:")
    for k in unexpected_keys:
        print("  ", k)

    assert len(unexpected_keys) == 0, "Unexpected keys should be empty."
    assert all(k.startswith("field_coupling.") for k in missing_keys), (
        "Only field_coupling parameters should be missing when loading M6 weights."
    )

    print("✅ Forward shape test passed.")
    print("✅ Backward test passed.")
    print("✅ return_features test passed.")
    print("✅ M6 -> M7 strict=False loading compatibility passed.")
    print("======================================================")


if __name__ == "__main__":
    main()
