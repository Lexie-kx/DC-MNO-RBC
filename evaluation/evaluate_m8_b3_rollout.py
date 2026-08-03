import os
import sys
import json
import argparse

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

from models.operators.fno2d_m8_b3_direct_residual import (
    M8B3DirectResidualAdapterFNO2d,
)

from evaluation.evaluate_cross_param_rollout_m8 import (
    RolloutDataset,
    build_m8_model_from_checkpoint,
    rollout_with_param,
    update_error_stats,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fair rollout comparison for "
            "M8-A, M8-A3 and M8-B3."
        )
    )

    parser.add_argument("--split", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument(
        "--m8_a_checkpoint",
        required=True,
    )
    parser.add_argument(
        "--m8_a3_checkpoint",
        required=True,
    )
    parser.add_argument(
        "--m8_b3_checkpoint",
        required=True,
    )
    parser.add_argument("--output", required=True)

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default="1,4,8,16",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
    )

    return parser.parse_args()


def resolve_path(project_root, path):
    if os.path.isabs(path):
        return path

    return os.path.abspath(
        os.path.join(project_root, path)
    )


def load_b3_model(
    checkpoint_path,
    device,
    expected_adapter_mode,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if (
        not isinstance(checkpoint, dict)
        or "model_state_dict" not in checkpoint
        or "model_config" not in checkpoint
    ):
        raise RuntimeError(
            "❌ B3 checkpoint 缺少 "
            "model_state_dict 或 model_config"
        )

    model_config = dict(
        checkpoint["model_config"]
    )

    actual_mode = model_config.get(
        "adapter_mode"
    )

    if actual_mode != expected_adapter_mode:
        raise RuntimeError(
            "❌ adapter_mode 不符合预期："
            f"expected={expected_adapter_mode}, "
            f"actual={actual_mode}, "
            f"path={checkpoint_path}"
        )

    model = M8B3DirectResidualAdapterFNO2d(
        **model_config
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    print(
        "✅ Restored B3 checkpoint | "
        f"adapter_mode={actual_mode} | "
        f"epoch={checkpoint.get('epoch')} | "
        f"val={checkpoint.get('val_loss')}"
    )

    return model


def build_rows(
    stats_dict,
    model_order,
    horizons,
):
    rows = []

    for model_name in model_order:
        for horizon in horizons:
            bucket = stats_dict[
                (model_name, horizon)
            ]

            for channel, field in enumerate(
                FIELD_ORDER
            ):
                rel_l2 = torch.sqrt(
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
                            rel_l2.item()
                        ),
                        "mse": mse.item(),
                    }
                )

            global_rel_l2 = torch.sqrt(
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
                        global_rel_l2.item()
                    ),
                    "mse": global_mse.item(),
                }
            )

    return rows


def print_difference(
    wide,
    model_a,
    model_b,
    output_path,
    suffix,
):
    metrics = [
        "buoyancy",
        "u_x",
        "u_y",
        "pressure",
        "global",
    ]

    a = (
        wide[wide["model"] == model_a]
        .set_index("horizon")
    )

    b = (
        wide[wide["model"] == model_b]
        .set_index("horizon")
    )

    difference = (
        a[metrics] - b[metrics]
    ).reset_index()

    print(
        f"\n================ "
        f"{model_a} - {model_b} "
        f"================"
    )
    print(
        "说明：负数表示前者更好；"
        "正数表示前者更差。"
    )
    print(
        difference.to_string(index=False)
    )

    all_values = (
        difference[metrics]
        .to_numpy()
        .reshape(-1)
    )

    field_pass = int(
        (all_values < 0).sum()
    )
    total_count = int(
        all_values.size
    )
    global_pass = int(
        (difference["global"] < 0).sum()
    )

    print(
        "\n📌 Strict field-level pass: "
        f"{field_pass}/{total_count}"
    )
    print(
        "📌 Strict global pass: "
        f"{global_pass}/{len(difference)}"
    )

    if (
        field_pass == total_count
        and global_pass == len(difference)
    ):
        print(
            "✅ 当前 split 达到全部指标全面领先"
        )
    else:
        print(
            "❌ 当前 split 尚未达到全面领先标准"
        )

    difference_path = output_path.replace(
        ".csv",
        f"_{suffix}.csv",
    )

    difference.to_csv(
        difference_path,
        index=False,
    )

    print(
        "✅ Difference table saved to: "
        f"{difference_path}"
    )


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
    m8_a_checkpoint = resolve_path(
        project_root,
        args.m8_a_checkpoint,
    )
    m8_a3_checkpoint = resolve_path(
        project_root,
        args.m8_a3_checkpoint,
    )
    m8_b3_checkpoint = resolve_path(
        project_root,
        args.m8_b3_checkpoint,
    )
    output_path = resolve_path(
        project_root,
        args.output,
    )

    for path in [
        split_path,
        stats_path,
        m8_a_checkpoint,
        m8_a3_checkpoint,
        m8_b3_checkpoint,
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"❌ 文件不存在: {path}"
            )

    horizons = [
        int(value)
        for value in args.horizons.split(",")
    ]
    max_horizon = max(horizons)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "🚀 [M8-B3 Fair Rollout Evaluation]"
    )
    print(f"📌 Device: {device}")
    print(f"📌 Split: {split_path}")
    print(f"📌 Horizons: {horizons}")
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

    m8_a = build_m8_model_from_checkpoint(
        checkpoint_path=m8_a_checkpoint,
        device=device,
        expected_coupling_mode="static",
    )

    m8_a3 = load_b3_model(
        checkpoint_path=m8_a3_checkpoint,
        device=device,
        expected_adapter_mode="static_adapter",
    )

    m8_b3 = load_b3_model(
        checkpoint_path=m8_b3_checkpoint,
        device=device,
        expected_adapter_mode="parameter_adapter",
    )

    model_order = [
        "M8-A-O5",
        "M8-A3-StaticDirect",
        "M8-B3-ParameterDirect",
    ]

    stats_dict = {}

    print(
        "\n🔥 开始 A / A3 / B3 rollout..."
    )

    with torch.no_grad():
        for batch_index, (
            x0_phys,
            future_phys,
            param,
        ) in enumerate(loader):
            x0_phys = x0_phys.to(device)
            future_phys = future_phys.to(device)
            param = param.to(device)

            predictions = {
                "M8-A-O5": rollout_with_param(
                    model=m8_a,
                    x0_phys=x0_phys,
                    param=param,
                    normalizer=normalizer,
                    max_horizon=max_horizon,
                    coupling_mode="static",
                ),
                "M8-A3-StaticDirect": (
                    rollout_with_param(
                        model=m8_a3,
                        x0_phys=x0_phys,
                        param=param,
                        normalizer=normalizer,
                        max_horizon=max_horizon,
                    )
                ),
                "M8-B3-ParameterDirect": (
                    rollout_with_param(
                        model=m8_b3,
                        x0_phys=x0_phys,
                        param=param,
                        normalizer=normalizer,
                        max_horizon=max_horizon,
                    )
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

            if (batch_index + 1) % 10 == 0:
                print(
                    f"  已处理 batch "
                    f"{batch_index + 1}/"
                    f"{len(loader)}"
                )

    rows = build_rows(
        stats_dict=stats_dict,
        model_order=model_order,
        horizons=horizons,
    )

    dataframe = pd.DataFrame(rows)

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True,
    )

    dataframe.to_csv(
        output_path,
        index=False,
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
        categories=model_order,
        ordered=True,
    )

    wide = wide.sort_values(
        ["model", "horizon"]
    )

    print(
        "\n================ "
        "Compact Rollout Summary "
        "================"
    )
    print(
        wide.to_string(index=False)
    )

    print_difference(
        wide=wide,
        model_a="M8-B3-ParameterDirect",
        model_b="M8-A3-StaticDirect",
        output_path=output_path,
        suffix="m8_b3_minus_m8_a3",
    )

    print_difference(
        wide=wide,
        model_a="M8-B3-ParameterDirect",
        model_b="M8-A-O5",
        output_path=output_path,
        suffix="m8_b3_minus_m8_a_o5",
    )

    print(
        f"\n✅ Full result saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
