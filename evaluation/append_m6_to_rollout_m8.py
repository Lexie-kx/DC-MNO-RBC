import os
import sys
import json
import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.append(PROJECT_ROOT)

from constants import FIELD_ORDER
from training.normalization import FieldWiseNormalizer
from models.operators.fno2d_fieldwise import FieldWiseFNO2d

from evaluation.evaluate_cross_param_rollout_m8 import (
    RolloutDataset,
    load_model_state,
    rollout_without_param,
    update_error_stats,
)


M6_NAME = "M6-FieldWiseEncoder-H4-continue"

MODEL_ORDER = [
    M6_NAME,
    "M7-FieldCoupling-H4",
    "M7-ParamTokenOnly-H4",
    "M8-A-FullStatic-H4",
    "M8-B-ParamConditionedCoupling-H4",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate only the M6 control model, append it to an "
            "existing M8 rollout CSV, and generate complete comparisons."
        )
    )

    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--stats", type=str, required=True)

    parser.add_argument(
        "--m6_checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--existing_csv",
        type=str,
        required=True,
        help="Existing full M8 rollout CSV containing M7 and M8 models.",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV containing M6, M7 and M8 models.",
    )

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

    return parser.parse_args()


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(
        os.path.join(PROJECT_ROOT, path)
    )


def build_m6_rows(
    stats_dict,
    horizons,
):
    rows = []

    for horizon in horizons:
        bucket = stats_dict[
            (M6_NAME, horizon)
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
                    "model": M6_NAME,
                    "horizon": horizon,
                    "field": field,
                    "rel_l2_percent": rel_l2.item(),
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
                "model": M6_NAME,
                "horizon": horizon,
                "field": "global",
                "rel_l2_percent": (
                    global_rel_l2.item()
                ),
                "mse": global_mse.item(),
            }
        )

    return rows


def calculate_difference(
    wide,
    model_a,
    model_b,
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

    if a.empty or b.empty:
        raise RuntimeError(
            f"Missing comparison model: "
            f"{model_a} or {model_b}"
        )

    difference = (
        a[metrics] - b[metrics]
    ).reset_index()

    return difference[
        ["horizon"] + metrics
    ]


def safe_name(text):
    return (
        text.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(":", "")
    )


def main():
    args = parse_args()

    split_path = resolve_path(args.split)
    stats_path = resolve_path(args.stats)
    m6_checkpoint = resolve_path(
        args.m6_checkpoint
    )
    existing_csv = resolve_path(
        args.existing_csv
    )
    output_path = resolve_path(
        args.output
    )

    for path in [
        split_path,
        stats_path,
        m6_checkpoint,
        existing_csv,
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
        "🚀 [Append M6 Control to M8 Rollout]"
    )
    print(f"📌 Device: {device}")
    print(f"📌 Split: {split_path}")
    print(f"📌 Stats: {stats_path}")
    print(f"📌 M6 checkpoint: {m6_checkpoint}")
    print(f"📌 Existing CSV: {existing_csv}")
    print(f"📌 Output: {output_path}")
    print(f"📌 Horizons: {horizons}")
    print(
        "👉 只补算 M6，不重新计算 "
        "M7 / M8 模型。"
    )

    with open(
        split_path,
        "r",
        encoding="utf-8",
    ) as file:
        split_config = json.load(file)

    dataset = RolloutDataset(
        split_config=split_config["test"],
        max_horizon=max_horizon,
        max_samples=None,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    normalizer = FieldWiseNormalizer(
        stats_path
    ).to(device)

    m6_model = FieldWiseFNO2d(
        in_channels=16,
        out_channels=4,
        modes1=16,
        modes2=16,
        width=32,
    ).to(device)

    m6_model = load_model_state(
        m6_model,
        m6_checkpoint,
        device,
    )
    m6_model.eval()

    stats_dict = {}

    print("\n🔥 开始补算 M6 rollout...")

    with torch.no_grad():
        for batch_index, (
            x0_phys,
            future_phys,
            _param,
        ) in enumerate(loader):
            x0_phys = x0_phys.to(device)
            future_phys = future_phys.to(device)

            m6_predictions = rollout_without_param(
                model=m6_model,
                x0_phys=x0_phys,
                normalizer=normalizer,
                max_horizon=max_horizon,
            )

            for horizon in horizons:
                horizon_index = horizon - 1

                true_horizon = future_phys[
                    :,
                    horizon_index,
                    :,
                    :,
                    :,
                ]

                update_error_stats(
                    pred_phys=(
                        m6_predictions[
                            horizon_index
                        ]
                    ),
                    true_phys=true_horizon,
                    stats_dict=stats_dict,
                    horizon=horizon,
                    model_name=M6_NAME,
                )

            if (batch_index + 1) % 10 == 0:
                print(
                    f"  已处理 batch "
                    f"{batch_index + 1}/"
                    f"{len(loader)}"
                )

    m6_dataframe = pd.DataFrame(
        build_m6_rows(
            stats_dict=stats_dict,
            horizons=horizons,
        )
    )

    existing_dataframe = pd.read_csv(
        existing_csv
    )

    existing_dataframe = existing_dataframe[
        existing_dataframe["model"] != M6_NAME
    ].copy()

    combined = pd.concat(
        [
            m6_dataframe,
            existing_dataframe,
        ],
        ignore_index=True,
    )

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True,
    )

    combined.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\n✅ 五模型完整结果保存到: "
        f"{output_path}"
    )

    wide = combined.pivot_table(
        index=["model", "horizon"],
        columns="field",
        values="rel_l2_percent",
    ).reset_index()

    metrics = [
        "buoyancy",
        "u_x",
        "u_y",
        "pressure",
        "global",
    ]

    wide = wide[
        ["model", "horizon"] + metrics
    ]

    wide["model"] = pd.Categorical(
        wide["model"],
        categories=MODEL_ORDER,
        ordered=True,
    )

    wide = wide.sort_values(
        ["model", "horizon"]
    )

    print(
        "\n================ "
        "Complete Rollout Summary: M6–M8 "
        "================"
    )
    print(
        "控制组、两个单模块消融、"
        "静态 Full 与条件耦合候选。"
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

    print(
        "\n================ "
        "Complete Global Rel-L2 (%) "
        "================"
    )
    print(
        global_table.to_string(index=False)
    )

    comparison_pairs = [
        (
            "M7-FieldCoupling-H4",
            M6_NAME,
        ),
        (
            "M7-ParamTokenOnly-H4",
            M6_NAME,
        ),
        (
            "M8-A-FullStatic-H4",
            M6_NAME,
        ),
        (
            "M8-B-ParamConditionedCoupling-H4",
            M6_NAME,
        ),
        (
            "M8-A-FullStatic-H4",
            "M7-FieldCoupling-H4",
        ),
        (
            "M8-A-FullStatic-H4",
            "M7-ParamTokenOnly-H4",
        ),
        (
            "M8-B-ParamConditionedCoupling-H4",
            "M7-FieldCoupling-H4",
        ),
        (
            "M8-B-ParamConditionedCoupling-H4",
            "M7-ParamTokenOnly-H4",
        ),
        (
            "M8-B-ParamConditionedCoupling-H4",
            "M8-A-FullStatic-H4",
        ),
    ]

    global_difference_rows = []
    detailed_differences = {}

    output_object = Path(output_path)

    for model_a, model_b in comparison_pairs:
        title = f"{model_a} - {model_b}"

        difference = calculate_difference(
            wide=wide,
            model_a=model_a,
            model_b=model_b,
        )

        detailed_differences[title] = (
            difference
        )

        detail_path = (
            output_object.parent
            / (
                output_object.stem
                + "_"
                + safe_name(title)
                + ".csv"
            )
        )

        difference.to_csv(
            detail_path,
            index=False,
        )

        row = {
            "comparison": title,
        }

        for horizon in horizons:
            value = difference.loc[
                difference["horizon"]
                == horizon,
                "global",
            ].iloc[0]

            row[f"h={horizon}"] = value

        global_difference_rows.append(row)

    global_differences = pd.DataFrame(
        global_difference_rows
    )

    print(
        "\n================ "
        "Complete Global Difference Table "
        "================"
    )
    print(
        "说明：前者减后者；"
        "负数表示前者更好。"
    )
    print(
        global_differences.to_string(
            index=False
        )
    )

    global_difference_path = (
        output_object.parent
        / (
            output_object.stem
            + "_complete_global_differences.csv"
        )
    )

    global_differences.to_csv(
        global_difference_path,
        index=False,
    )

    print(
        f"✅ Global difference table saved to: "
        f"{global_difference_path}"
    )

    # 终端重点打印三个最关键的逐场差值。
    key_titles = [
        (
            "M8-A-FullStatic-H4 - "
            f"{M6_NAME}"
        ),
        (
            "M8-B-ParamConditionedCoupling-H4 "
            f"- {M6_NAME}"
        ),
        (
            "M8-B-ParamConditionedCoupling-H4 "
            "- M8-A-FullStatic-H4"
        ),
    ]

    for title in key_titles:
        print(
            f"\n================ "
            f"{title}: Field-wise Difference "
            f"================"
        )
        print(
            "负数表示前者更好；"
            "正数表示前者更差。"
        )
        print(
            detailed_differences[
                title
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
