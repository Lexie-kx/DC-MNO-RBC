"""
Fair O5 rollout evaluation for:

- M7-FieldCoupling-O5-H4
- M7-ParamTokenOnly-O5-H4
- M8-A-O5-FullStatic-H4
- M8-Gate-O5-GateOnly-H4

The dataset, normalization, rollout and metric accumulation functions are
reused from evaluate_cross_param_rollout_m8_o5.py so that the comparison
protocol remains unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from constants import FIELD_ORDER
from training.normalization import FieldWiseNormalizer

from models.operators.fno2d_fieldwise_m7 import (
    M7FieldCouplingFNO2d,
)
from models.operators.fno2d_fieldwise_paramtoken import (
    M7FieldWiseParamTokenFNO2d,
)
from models.operators.fno2d_m8_module_gate import (
    M8ModuleGateFNO2d,
)

from evaluate_cross_param_rollout_m8_o5 import (
    RolloutDataset,
    resolve_path,
    load_model_state,
    build_m8_model_from_checkpoint,
    rollout_without_param,
    rollout_with_param,
    update_error_stats,
    print_and_save_difference,
)


MODEL_FC = "M7-FieldCoupling-O5-H4"
MODEL_PT = "M7-ParamTokenOnly-O5-H4"
MODEL_A = "M8-A-O5-FullStatic-H4"
MODEL_GATE = "M8-Gate-O5-GateOnly-H4"

MODEL_ORDER = [
    MODEL_FC,
    MODEL_PT,
    MODEL_A,
    MODEL_GATE,
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fair rollout evaluation for M7-O5, "
            "M8-A-O5 and M8-Gate-O5."
        )
    )

    parser.add_argument(
        "--split",
        required=True,
    )
    parser.add_argument(
        "--stats",
        required=True,
    )
    parser.add_argument(
        "--m7_fieldcoupling_checkpoint",
        required=True,
    )
    parser.add_argument(
        "--m7_paramtoken_checkpoint",
        required=True,
    )
    parser.add_argument(
        "--m8_full_static_checkpoint",
        required=True,
    )
    parser.add_argument(
        "--m8_module_gate_checkpoint",
        required=True,
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--horizons",
        default="1,4,8,16",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Debug only. Omit for full evaluation.",
    )

    return parser.parse_args()


def build_gate_model_from_checkpoint(
    checkpoint_path,
    device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "M8-Gate checkpoint必须是包含配置的字典格式。"
        )

    if "model_state_dict" not in checkpoint:
        raise RuntimeError(
            "M8-Gate checkpoint缺少model_state_dict。"
        )

    if "model_config" not in checkpoint:
        raise RuntimeError(
            "M8-Gate checkpoint缺少model_config。"
        )

    state_dict = checkpoint["model_state_dict"]
    model_config = dict(checkpoint["model_config"])

    # Gate模型内部固定使用static FieldCoupling。
    model_config.pop("coupling_mode", None)
    model_config.setdefault("gate_scale", 0.25)

    model = M8ModuleGateFNO2d(
        **model_config,
    ).to(device)

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    gate_weight = (
        model.module_gate.weight
        .detach()
        .cpu()
        .reshape(-1)
        .tolist()
    )

    gate_bias = float(
        model.module_gate.bias
        .detach()
        .cpu()
        .item()
    )

    print(
        "✅ Restored M8-Gate checkpoint | "
        f"gate_scale={model.gate_scale} | "
        f"gate_weight={gate_weight} | "
        f"gate_bias={gate_bias:.8f}"
    )

    return model


def build_rows(
    stats_dict,
    horizons,
):
    rows = []

    for model_name in MODEL_ORDER:
        for horizon in horizons:
            key = (model_name, horizon)

            if key not in stats_dict:
                raise RuntimeError(
                    f"缺少误差统计：{key}"
                )

            bucket = stats_dict[key]

            for channel, field in enumerate(
                FIELD_ORDER
            ):
                relative_l2 = torch.sqrt(
                    bucket["field_sse"][channel]
                    / (
                        bucket[
                            "field_target_sq"
                        ][channel]
                        + 1e-12
                    )
                ) * 100.0

                mse = (
                    bucket["field_sse"][channel]
                    / bucket["field_numel"][channel]
                )

                rows.append(
                    {
                        "model": model_name,
                        "horizon": horizon,
                        "field": field,
                        "rel_l2_percent": (
                            relative_l2.item()
                        ),
                        "mse": mse.item(),
                    }
                )

            global_relative_l2 = torch.sqrt(
                bucket["global_sse"]
                / (
                    bucket["global_target_sq"]
                    + 1e-12
                )
            ) * 100.0

            global_mse = (
                bucket["global_sse"]
                / bucket["global_numel"]
            )

            rows.append(
                {
                    "model": model_name,
                    "horizon": horizon,
                    "field": "global",
                    "rel_l2_percent": (
                        global_relative_l2.item()
                    ),
                    "mse": global_mse.item(),
                }
            )

    return rows


def main():
    args = parse_args()

    project_root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
        )
    )

    split_path = resolve_path(
        project_root,
        args.split,
    )
    stats_path = resolve_path(
        project_root,
        args.stats,
    )

    fc_checkpoint = resolve_path(
        project_root,
        args.m7_fieldcoupling_checkpoint,
    )
    pt_checkpoint = resolve_path(
        project_root,
        args.m7_paramtoken_checkpoint,
    )
    m8_a_checkpoint = resolve_path(
        project_root,
        args.m8_full_static_checkpoint,
    )
    gate_checkpoint = resolve_path(
        project_root,
        args.m8_module_gate_checkpoint,
    )
    output_path = resolve_path(
        project_root,
        args.output,
    )

    paths_to_check = [
        split_path,
        stats_path,
        fc_checkpoint,
        pt_checkpoint,
        m8_a_checkpoint,
        gate_checkpoint,
    ]

    for path in paths_to_check:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"❌ 文件不存在：{path}"
            )

    horizons = [
        int(value.strip())
        for value in args.horizons.split(",")
        if value.strip()
    ]

    if not horizons:
        raise ValueError("horizons不能为空")

    if any(horizon <= 0 for horizon in horizons):
        raise ValueError(
            f"horizons必须为正数：{horizons}"
        )

    max_horizon = max(horizons)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "🚀 [M8-Gate O5 Rollout Evaluation]"
    )
    print(f"📌 Device: {device}")
    print(f"📌 Split: {split_path}")
    print(f"📌 Stats: {stats_path}")
    print(f"📌 M7-FieldCoupling: {fc_checkpoint}")
    print(f"📌 M7-ParamToken: {pt_checkpoint}")
    print(f"📌 M8-A: {m8_a_checkpoint}")
    print(f"📌 M8-Gate: {gate_checkpoint}")
    print(f"📌 Horizons: {horizons}")
    print(f"📌 Max samples: {args.max_samples}")
    print(f"📌 Output: {output_path}")

    with open(
        split_path,
        "r",
        encoding="utf-8",
    ) as file:
        split_config = json.load(file)

    dataset = RolloutDataset(
        split_config=split_config["test"],
        max_horizon=max_horizon,
        max_samples=args.max_samples,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    normalizer = FieldWiseNormalizer(
        stats_path
    ).to(device)

    fieldcoupling_model = (
        M7FieldCouplingFNO2d(
            in_channels=16,
            out_channels=4,
            modes1=16,
            modes2=16,
            width=32,
            context_length=4,
            num_fields=4,
            coupling_hidden_channels=8,
            coupling_dropout=0.0,
            coupling_init_gate=-4.0,
            coupling_use_norm=True,
        ).to(device)
    )

    fieldcoupling_model = load_model_state(
        fieldcoupling_model,
        fc_checkpoint,
        device,
    )
    fieldcoupling_model.eval()

    paramtoken_model = (
        M7FieldWiseParamTokenFNO2d(
            in_channels=16,
            out_channels=4,
            modes1=16,
            modes2=16,
            width=32,
            context_length=4,
            num_fields=4,
            token_hidden_dim=64,
        ).to(device)
    )

    paramtoken_model = load_model_state(
        paramtoken_model,
        pt_checkpoint,
        device,
    )
    paramtoken_model.eval()

    m8_a_model = build_m8_model_from_checkpoint(
        checkpoint_path=m8_a_checkpoint,
        device=device,
        expected_coupling_mode="static",
    )

    gate_model = build_gate_model_from_checkpoint(
        checkpoint_path=gate_checkpoint,
        device=device,
    )

    stats_dict = {}

    print()
    print("🔥 开始四模型公平 rollout evaluation...")

    with torch.no_grad():
        for batch_index, batch in enumerate(
            loader,
            start=1,
        ):
            (
                x0_phys,
                future_phys,
                param,
            ) = batch

            x0_phys = x0_phys.to(device)
            future_phys = future_phys.to(device)
            param = param.to(device)

            predictions = {
                MODEL_FC: rollout_without_param(
                    model=fieldcoupling_model,
                    x0_phys=x0_phys,
                    normalizer=normalizer,
                    max_horizon=max_horizon,
                ),
                MODEL_PT: rollout_with_param(
                    model=paramtoken_model,
                    x0_phys=x0_phys,
                    param=param,
                    normalizer=normalizer,
                    max_horizon=max_horizon,
                ),
                MODEL_A: rollout_with_param(
                    model=m8_a_model,
                    x0_phys=x0_phys,
                    param=param,
                    normalizer=normalizer,
                    max_horizon=max_horizon,
                    coupling_mode="static",
                ),
                MODEL_GATE: rollout_with_param(
                    model=gate_model,
                    x0_phys=x0_phys,
                    param=param,
                    normalizer=normalizer,
                    max_horizon=max_horizon,
                ),
            }

            for horizon in horizons:
                horizon_index = horizon - 1

                true_horizon = future_phys[
                    :,
                    horizon_index,
                    :,
                    :,
                    :,
                ]

                for (
                    model_name,
                    model_predictions,
                ) in predictions.items():
                    update_error_stats(
                        pred_phys=(
                            model_predictions[
                                horizon_index
                            ]
                        ),
                        true_phys=true_horizon,
                        stats_dict=stats_dict,
                        horizon=horizon,
                        model_name=model_name,
                    )

            if batch_index % 10 == 0:
                print(
                    f"  已处理 batch "
                    f"{batch_index}/{len(loader)}"
                )

    rows = build_rows(
        stats_dict=stats_dict,
        horizons=horizons,
    )

    dataframe = pd.DataFrame(rows)

    output_directory = os.path.dirname(
        output_path
    )

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True,
        )

    dataframe.to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        "✅ Rollout evaluation saved to: "
        f"{output_path}"
    )

    wide = dataframe.pivot_table(
        index=["model", "horizon"],
        columns="field",
        values="rel_l2_percent",
    ).reset_index()

    metric_order = [
        "buoyancy",
        "u_x",
        "u_y",
        "pressure",
        "global",
    ]

    wide = wide[
        ["model", "horizon"]
        + metric_order
    ]

    wide["model"] = pd.Categorical(
        wide["model"],
        categories=MODEL_ORDER,
        ordered=True,
    )

    wide = wide.sort_values(
        ["model", "horizon"]
    )

    print()
    print(
        "================ "
        "M8-Gate O5 Rollout Summary "
        "================"
    )
    print(
        wide.to_string(index=False)
    )

    global_table = wide.pivot_table(
        index="model",
        columns="horizon",
        values="global",
        observed=False,
    ).reindex(MODEL_ORDER).reset_index()

    global_table.columns = [
        (
            "model"
            if column == "model"
            else f"h={column}"
        )
        for column in global_table.columns
    ]

    print()
    print(
        "================ "
        "Global Rel-L2 (%) "
        "================"
    )
    print(
        global_table.to_string(index=False)
    )

    print_and_save_difference(
        wide=wide,
        model_a=MODEL_GATE,
        model_b=MODEL_A,
        title=(
            "M8-Gate-O5-GateOnly-H4 - "
            "M8-A-O5-FullStatic-H4"
        ),
        output_path=output_path,
    )

    print_and_save_difference(
        wide=wide,
        model_a=MODEL_GATE,
        model_b=MODEL_PT,
        title=(
            "M8-Gate-O5-GateOnly-H4 - "
            "M7-ParamTokenOnly-O5-H4"
        ),
        output_path=output_path,
    )

    print_and_save_difference(
        wide=wide,
        model_a=MODEL_GATE,
        model_b=MODEL_FC,
        title=(
            "M8-Gate-O5-GateOnly-H4 - "
            "M7-FieldCoupling-O5-H4"
        ),
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
