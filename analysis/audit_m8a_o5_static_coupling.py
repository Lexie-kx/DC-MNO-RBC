from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "checkpoints/tuning/m8_a_o5"
OUT_DIR = ROOT / "outputs/tables/m8_direction1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIELDS = [
    "buoyancy",
    "u_x",
    "u_y",
    "pressure",
]


def find_checkpoint(split_name: str) -> Path:
    matches = sorted(
        path
        for path in CKPT_DIR.rglob("*best.pth")
        if split_name in path.name
        and "m8_a_o5" in path.name
    )

    if len(matches) != 1:
        print(
            f"\n❌ {split_name} checkpoint数量不是1："
            f"{len(matches)}"
        )
        for path in matches:
            print(f"   {path}")
        raise RuntimeError(
            "请先确认M8-A-O5 best checkpoint。"
        )

    return matches[0]


def load_coupling(path: Path):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }

    matrix_key = (
        "field_coupling.coupling_matrix"
    )
    gate_key = (
        "field_coupling.residual_gate"
    )

    if matrix_key not in state_dict:
        raise KeyError(
            f"checkpoint中缺少：{matrix_key}"
        )

    if gate_key not in state_dict:
        raise KeyError(
            f"checkpoint中缺少：{gate_key}"
        )

    matrix = (
        state_dict[matrix_key]
        .detach()
        .float()
        .clone()
    )

    matrix.fill_diagonal_(0.0)

    gate_logit = (
        state_dict[gate_key]
        .detach()
        .float()
        .clone()
    )
    gate = torch.sigmoid(gate_logit)

    # 实际残差路径中的标量部分：
    # target gate × matrix[target, source]
    gated_matrix = gate[:, None] * matrix

    abs_row_sum = (
        gated_matrix.abs().sum(dim=1, keepdim=True)
        .clamp_min(1e-12)
    )
    row_normalized_abs = (
        gated_matrix.abs() / abs_row_sum
    )

    return {
        "matrix": matrix,
        "gate_logit": gate_logit,
        "gate": gate,
        "gated_matrix": gated_matrix,
        "row_normalized_abs": (
            row_normalized_abs
        ),
    }


def dataframe(matrix: torch.Tensor) -> pd.DataFrame:
    return pd.DataFrame(
        matrix.numpy(),
        index=[
            f"target:{field}"
            for field in FIELDS
        ],
        columns=[
            f"source:{field}"
            for field in FIELDS
        ],
    )


def print_matrix(
    title: str,
    matrix: torch.Tensor,
) -> None:
    print("\n" + title)
    print(
        dataframe(matrix).to_string(
            float_format=lambda value: (
                f"{value:+.6f}"
            )
        )
    )


records = []
loaded = {}

for split_name in ["unseen_pr", "unseen_ra"]:
    checkpoint_path = find_checkpoint(
        split_name
    )
    info = load_coupling(checkpoint_path)
    loaded[split_name] = info

    print("\n" + "=" * 96)
    print(f"{split_name}")
    print("=" * 96)
    print(f"checkpoint: {checkpoint_path}")

    gate_table = pd.DataFrame(
        {
            "target_field": FIELDS,
            "gate_logit": (
                info["gate_logit"].numpy()
            ),
            "gate_value": (
                info["gate"].numpy()
            ),
        }
    )

    print("\nPer-target residual gate")
    print(
        gate_table.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:+.6f}"
            ),
        )
    )

    print_matrix(
        "Base static matrix "
        "[target, source]",
        info["matrix"],
    )

    print_matrix(
        "Gate × base matrix",
        info["gated_matrix"],
    )

    print_matrix(
        "Row-normalized absolute contribution",
        info["row_normalized_abs"],
    )

    for target_index, target in enumerate(
        FIELDS
    ):
        for source_index, source in enumerate(
            FIELDS
        ):
            if target_index == source_index:
                continue

            records.append(
                {
                    "split": split_name,
                    "checkpoint": str(
                        checkpoint_path
                    ),
                    "target": target,
                    "source": source,
                    "base_weight": float(
                        info["matrix"][
                            target_index,
                            source_index,
                        ]
                    ),
                    "target_gate": float(
                        info["gate"][target_index]
                    ),
                    "gated_weight": float(
                        info["gated_matrix"][
                            target_index,
                            source_index,
                        ]
                    ),
                    "abs_gated_weight": float(
                        info["gated_matrix"][
                            target_index,
                            source_index,
                        ].abs()
                    ),
                    "row_normalized_abs": float(
                        info[
                            "row_normalized_abs"
                        ][
                            target_index,
                            source_index,
                        ]
                    ),
                }
            )


pr_vector = loaded[
    "unseen_pr"
]["gated_matrix"].flatten()

ra_vector = loaded[
    "unseen_ra"
]["gated_matrix"].flatten()

mask = torch.ones(
    4,
    4,
    dtype=torch.bool,
)
mask.fill_diagonal_(False)
mask = mask.flatten()

pr_vector = pr_vector[mask]
ra_vector = ra_vector[mask]

cosine = torch.nn.functional.cosine_similarity(
    pr_vector.unsqueeze(0),
    ra_vector.unsqueeze(0),
).item()

absolute_cosine = (
    torch.nn.functional.cosine_similarity(
        pr_vector.abs().unsqueeze(0),
        ra_vector.abs().unsqueeze(0),
    ).item()
)

sign_agreement = (
    torch.sign(pr_vector)
    == torch.sign(ra_vector)
).float().mean().item()

difference = (
    loaded["unseen_ra"]["gated_matrix"]
    - loaded["unseen_pr"]["gated_matrix"]
)

print("\n" + "=" * 96)
print("Unseen Pr 与 Unseen Ra 静态耦合对比")
print("=" * 96)
print(f"signed cosine similarity: {cosine:+.6f}")
print(
    "absolute-pattern cosine similarity: "
    f"{absolute_cosine:+.6f}"
)
print(
    "off-diagonal sign agreement: "
    f"{sign_agreement * 12:.0f}/12"
)

print_matrix(
    "Gated matrix difference "
    "(unseen_ra - unseen_pr)",
    difference,
)

output_path = (
    OUT_DIR
    / "m8a_o5_static_coupling_audit.csv"
)

pd.DataFrame(records).to_csv(
    output_path,
    index=False,
)

print("\n✅ 审计结果已保存：")
print(output_path)
