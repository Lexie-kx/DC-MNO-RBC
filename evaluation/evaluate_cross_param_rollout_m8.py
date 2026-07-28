import os
import sys
import json
import math
import argparse

import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from constants import (
    DATA_PATH,
    FIELD_ORDER,
    CONTEXT_LENGTH,
    DTYPE,
)
from training.normalization import FieldWiseNormalizer
from models.operators.fno2d_fieldwise_m7 import (
    M7FieldCouplingFNO2d,
)
from models.operators.fno2d_fieldwise_paramtoken import (
    M7FieldWiseParamTokenFNO2d,
)
from models.operators.fno2d_m8_full_conditioned import (
    M8FullConditionedFNO2d,
)


class RolloutDataset(Dataset):
    """
    Cross-parameter rollout dataset.

    Returns:
        x0_phys:
            [16, H, W]

        future_phys:
            [max_horizon, 4, H, W]

        param:
            [2] = [log10(Ra), log10(Pr)]
    """

    def __init__(
        self,
        split_config,
        max_horizon=16,
        max_samples=None,
    ):
        self.split_config = split_config
        self.max_horizon = max_horizon
        self.max_samples = max_samples

        self.data_store = {}
        self.index = []

        self._load_data()

    @staticmethod
    def _parse_ra_pr_from_group(group_name):
        import re

        ra_match = re.search(
            r"[Rr]a_?([0-9.eE+-]+)",
            group_name,
        )
        pr_match = re.search(
            r"[Pp]r_?([0-9.eE+-]+)",
            group_name,
        )

        if ra_match is None or pr_match is None:
            raise ValueError(
                f"无法从 group_name 解析 Ra/Pr: {group_name}"
            )

        return (
            float(ra_match.group(1)),
            float(pr_match.group(1)),
        )

    def _make_param(self, group_name):
        ra, pr = self._parse_ra_pr_from_group(
            group_name
        )

        return torch.tensor(
            [
                math.log10(ra),
                math.log10(pr),
            ],
            dtype=DTYPE,
        )

    def _load_data(self):
        print(
            f"📦 正在加载 rollout 数据，"
            f"共 {len(self.split_config)} 个 group 批次"
        )

        with h5py.File(DATA_PATH, "r") as file:
            for item in self.split_config:
                group_name = item["group"]
                trajectory_indices = item[
                    "trajectories"
                ]

                if group_name not in file:
                    print(
                        f"⚠️ group 不存在，跳过: "
                        f"{group_name}"
                    )
                    continue

                group = file[group_name]

                fields_data = [
                    group[field][:]
                    for field in FIELD_ORDER
                ]

                numpy_module = __import__("numpy")

                stacked_data = torch.tensor(
                    numpy_module.stack(
                        fields_data,
                        axis=0,
                    ),
                    dtype=DTYPE,
                )
                # [4, trajectory, time, H, W]

                selected_data = stacked_data[
                    :,
                    trajectory_indices,
                ]

                data = (
                    selected_data
                    .permute(1, 2, 0, 3, 4)
                    .contiguous()
                )
                # [trajectory, time, 4, H, W]

                self.data_store[group_name] = data

                param = self._make_param(
                    group_name
                )

                (
                    num_trajectories,
                    num_steps,
                    _,
                    _,
                    _,
                ) = data.shape

                max_start = (
                    num_steps
                    - CONTEXT_LENGTH
                    - self.max_horizon
                    + 1
                )

                if max_start <= 0:
                    print(
                        f"⚠️ group {group_name} "
                        "时间长度不足，跳过"
                    )
                    continue

                for trajectory_index in range(
                    num_trajectories
                ):
                    for t0 in range(max_start):
                        self.index.append(
                            {
                                "group": group_name,
                                "trajectory": (
                                    trajectory_index
                                ),
                                "t0": t0,
                                "param": param,
                            }
                        )

        if self.max_samples is not None:
            self.index = self.index[
                : self.max_samples
            ]

        print(
            f"✅ Rollout 样本数: "
            f"{len(self.index)}"
        )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        item = self.index[index]

        data = self.data_store[
            item["group"]
        ]

        trajectory = item["trajectory"]
        t0 = item["t0"]

        history = data[
            trajectory,
            t0 : t0 + CONTEXT_LENGTH,
        ]

        future = data[
            trajectory,
            (
                t0 + CONTEXT_LENGTH
            ) : (
                t0
                + CONTEXT_LENGTH
                + self.max_horizon
            ),
        ]

        _, _, height, width = history.shape

        x0_phys = history.reshape(
            CONTEXT_LENGTH * 4,
            height,
            width,
        )

        return (
            x0_phys,
            future,
            item["param"],
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Current-stage rollout evaluation for "
            "M7 ablations and M8-A/M8-B."
        )
    )

    parser.add_argument(
        "--split",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--stats",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--m7_fieldcoupling_checkpoint",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--m7_paramtoken_checkpoint",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--m8_full_static_checkpoint",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--m8_param_conditioned_checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
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
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help=(
            "Debug limit. Omit for full evaluation."
        ),
    )

    return parser.parse_args()


def resolve_path(project_root, path):
    if os.path.isabs(path):
        return path

    return os.path.abspath(
        os.path.join(project_root, path)
    )


def load_model_state(
    model,
    checkpoint_path,
    device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
        else checkpoint
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    return model



def build_m8_model_from_checkpoint(
    checkpoint_path,
    device,
    expected_coupling_mode,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]
        model_config = checkpoint.get("model_config")
    else:
        state_dict = checkpoint
        model_config = None

    if model_config is None:
        print(
            "⚠️ Legacy M8 checkpoint：未发现 model_config，"
            "使用原始 M8 默认结构恢复。"
        )

        model_config = {
            "in_channels": 16,
            "out_channels": 4,
            "modes1": 16,
            "modes2": 16,
            "width": 32,
            "context_length": 4,
            "num_fields": 4,
            "field_width": None,
            "coupling_mode": expected_coupling_mode,
            "coupling_hidden_channels": 8,
            "coupling_dropout": 0.0,
            "coupling_init_gate": -4.0,
            "coupling_use_norm": True,
            "coupling_param_hidden_dim": 64,
            "coupling_condition_scale": 0.10,
            "token_hidden_dim": 64,
            "alpha_token": 1.0,
        }

        config_source = "legacy defaults"

    else:
        model_config = dict(model_config)

        # 兼容早期 config。
        model_config.setdefault("alpha_token", 1.0)
        model_config.setdefault(
            "coupling_mode",
            expected_coupling_mode,
        )

        config_source = "checkpoint model_config"

    actual_mode = model_config["coupling_mode"]

    if actual_mode != expected_coupling_mode:
        raise ValueError(
            "❌ M8 checkpoint 的 coupling_mode 与参数位置不一致："
            f"expected={expected_coupling_mode}, "
            f"checkpoint={actual_mode}, "
            f"path={checkpoint_path}"
        )

    model = M8FullConditionedFNO2d(
        **model_config
    ).to(device)

    model.load_state_dict(
        state_dict,
        strict=True,
    )
    model.eval()

    print(
        "✅ Restored M8 checkpoint | "
        f"source={config_source} | "
        f"mode={model_config['coupling_mode']} | "
        f"alpha_token={model_config['alpha_token']} | "
        "condition_scale="
        f"{model_config['coupling_condition_scale']}"
    )

    return model


def rollout_without_param(
    model,
    x0_phys,
    normalizer,
    max_horizon,
):
    x_norm = normalizer.normalize_x(
        x0_phys
    )

    predictions = []

    for _ in range(max_horizon):
        current_state_norm = x_norm[
            :,
            -4:,
            :,
            :,
        ]

        pred_delta_norm = model(
            x_norm
        )

        pred_next_norm = (
            current_state_norm
            + pred_delta_norm
        )

        pred_next_phys = (
            normalizer.denormalize_y(
                pred_next_norm
            )
        )

        predictions.append(
            pred_next_phys
        )

        x_norm = torch.cat(
            [
                x_norm[:, 4:, :, :],
                pred_next_norm,
            ],
            dim=1,
        )

    return torch.stack(
        predictions,
        dim=0,
    )


def rollout_with_param(
    model,
    x0_phys,
    param,
    normalizer,
    max_horizon,
    coupling_mode=None,
):
    x_norm = normalizer.normalize_x(
        x0_phys
    )

    predictions = []

    for _ in range(max_horizon):
        current_state_norm = x_norm[
            :,
            -4:,
            :,
            :,
        ]

        if coupling_mode is None:
            pred_delta_norm = model(
                x_norm,
                param,
            )
        else:
            pred_delta_norm = model(
                x_norm,
                param,
                coupling_mode=coupling_mode,
            )

        pred_next_norm = (
            current_state_norm
            + pred_delta_norm
        )

        pred_next_phys = (
            normalizer.denormalize_y(
                pred_next_norm
            )
        )

        predictions.append(
            pred_next_phys
        )

        x_norm = torch.cat(
            [
                x_norm[:, 4:, :, :],
                pred_next_norm,
            ],
            dim=1,
        )

    return torch.stack(
        predictions,
        dim=0,
    )


def update_error_stats(
    pred_phys,
    true_phys,
    stats_dict,
    horizon,
    model_name,
):
    difference = (
        pred_phys - true_phys
    )

    key = (
        model_name,
        horizon,
    )

    if key not in stats_dict:
        stats_dict[key] = {
            "field_sse": torch.zeros(
                4,
                dtype=torch.float64,
                device=pred_phys.device,
            ),
            "field_target_sq": torch.zeros(
                4,
                dtype=torch.float64,
                device=pred_phys.device,
            ),
            "field_numel": torch.zeros(
                4,
                dtype=torch.float64,
                device=pred_phys.device,
            ),
            "global_sse": torch.tensor(
                0.0,
                dtype=torch.float64,
                device=pred_phys.device,
            ),
            "global_target_sq": torch.tensor(
                0.0,
                dtype=torch.float64,
                device=pred_phys.device,
            ),
            "global_numel": torch.tensor(
                0.0,
                dtype=torch.float64,
                device=pred_phys.device,
            ),
        }

    bucket = stats_dict[key]

    for channel in range(4):
        difference_channel = difference[
            :,
            channel,
            :,
            :,
        ].double()

        target_channel = true_phys[
            :,
            channel,
            :,
            :,
        ].double()

        bucket["field_sse"][channel] += (
            difference_channel.square().sum()
        )
        bucket[
            "field_target_sq"
        ][channel] += (
            target_channel.square().sum()
        )
        bucket[
            "field_numel"
        ][channel] += (
            difference_channel.numel()
        )

    bucket["global_sse"] += (
        difference.double().square().sum()
    )
    bucket["global_target_sq"] += (
        true_phys.double().square().sum()
    )
    bucket["global_numel"] += (
        difference.numel()
    )


def print_and_save_difference(
    wide,
    model_a,
    model_b,
    title,
    output_path,
):
    a = (
        wide[wide["model"] == model_a]
        .set_index("horizon")
    )
    b = (
        wide[wide["model"] == model_b]
        .set_index("horizon")
    )

    metrics = [
        "buoyancy",
        "u_x",
        "u_y",
        "pressure",
        "global",
    ]

    difference = (
        a[metrics] - b[metrics]
    ).reset_index()

    print(
        f"\n================ "
        f"{title}: Rel-L2 (%) "
        f"================"
    )
    print(
        "说明：负数表示前者更好；"
        "正数表示前者更差。"
    )
    print(
        difference.to_string(
            index=False
        )
    )

    safe_name = (
        title.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(":", "")
    )

    difference_path = output_path.replace(
        ".csv",
        f"_{safe_name}.csv",
    )

    difference.to_csv(
        difference_path,
        index=False,
    )

    print(
        f"✅ Difference table saved to: "
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

    m7_fieldcoupling_checkpoint = (
        resolve_path(
            project_root,
            args.m7_fieldcoupling_checkpoint,
        )
    )
    m7_paramtoken_checkpoint = (
        resolve_path(
            project_root,
            args.m7_paramtoken_checkpoint,
        )
    )
    m8_full_static_checkpoint = (
        resolve_path(
            project_root,
            args.m8_full_static_checkpoint,
        )
    )
    m8_param_conditioned_checkpoint = (
        resolve_path(
            project_root,
            args.m8_param_conditioned_checkpoint,
        )
    )

    output_path = resolve_path(
        project_root,
        args.output,
    )

    paths_to_check = [
        split_path,
        stats_path,
        m7_fieldcoupling_checkpoint,
        m7_paramtoken_checkpoint,
        m8_full_static_checkpoint,
        m8_param_conditioned_checkpoint,
    ]

    for path in paths_to_check:
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
        "🚀 [M8 Current-Stage "
        "Rollout Evaluation]"
    )
    print(f"📌 Device: {device}")
    print(f"📌 Split: {split_path}")
    print(f"📌 Stats: {stats_path}")
    print(
        "📌 M7-FieldCoupling-H4: "
        f"{m7_fieldcoupling_checkpoint}"
    )
    print(
        "📌 M7-ParamTokenOnly-H4: "
        f"{m7_paramtoken_checkpoint}"
    )
    print(
        "📌 M8-A-FullStatic-H4: "
        f"{m8_full_static_checkpoint}"
    )
    print(
        "📌 M8-B-ParamConditionedCoupling-H4: "
        f"{m8_param_conditioned_checkpoint}"
    )
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

    m7_fieldcoupling = M7FieldCouplingFNO2d(
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

    m7_fieldcoupling = load_model_state(
        m7_fieldcoupling,
        m7_fieldcoupling_checkpoint,
        device,
    )
    m7_fieldcoupling.eval()

    m7_paramtoken = M7FieldWiseParamTokenFNO2d(
        in_channels=16,
        out_channels=4,
        modes1=16,
        modes2=16,
        width=32,
        context_length=4,
        num_fields=4,
        token_hidden_dim=64,
    ).to(device)

    m7_paramtoken = load_model_state(
        m7_paramtoken,
        m7_paramtoken_checkpoint,
        device,
    )
    m7_paramtoken.eval()

    m8_full_static = build_m8_model_from_checkpoint(
        checkpoint_path=m8_full_static_checkpoint,
        device=device,
        expected_coupling_mode="static",
    )

    m8_param_conditioned = build_m8_model_from_checkpoint(
        checkpoint_path=m8_param_conditioned_checkpoint,
        device=device,
        expected_coupling_mode="parameter_conditioned",
    )

    stats_dict = {}

    print(
        "\n🔥 开始 M8 current-stage "
        "rollout evaluation..."
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
                "M7-FieldCoupling-H4": (
                    rollout_without_param(
                        model=m7_fieldcoupling,
                        x0_phys=x0_phys,
                        normalizer=normalizer,
                        max_horizon=max_horizon,
                    )
                ),
                "M7-ParamTokenOnly-H4": (
                    rollout_with_param(
                        model=m7_paramtoken,
                        x0_phys=x0_phys,
                        param=param,
                        normalizer=normalizer,
                        max_horizon=max_horizon,
                    )
                ),
                "M8-A-FullStatic-H4": (
                    rollout_with_param(
                        model=m8_full_static,
                        x0_phys=x0_phys,
                        param=param,
                        normalizer=normalizer,
                        max_horizon=max_horizon,
                        coupling_mode="static",
                    )
                ),
                (
                    "M8-B-ParamConditioned"
                    "Coupling-H4"
                ): (
                    rollout_with_param(
                        model=m8_param_conditioned,
                        x0_phys=x0_phys,
                        param=param,
                        normalizer=normalizer,
                        max_horizon=max_horizon,
                        coupling_mode=(
                            "parameter_conditioned"
                        ),
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

    model_order = [
        "M7-FieldCoupling-H4",
        "M7-ParamTokenOnly-H4",
        "M8-A-FullStatic-H4",
        "M8-B-ParamConditionedCoupling-H4",
    ]

    rows = []

    for model_name in model_order:
        for horizon in horizons:
            bucket = stats_dict[
                (model_name, horizon)
            ]

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
                    bucket[
                        "field_sse"
                    ][channel]
                    / bucket[
                        "field_numel"
                    ][channel]
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
                    bucket[
                        "global_target_sq"
                    ]
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

    dataframe = pd.DataFrame(rows)

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True,
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\n✅ Rollout evaluation saved to: "
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
        categories=model_order,
        ordered=True,
    )

    wide = wide.sort_values(
        ["model", "horizon"]
    )

    print(
        "\n================ "
        "Compact Rollout Summary: M8 Stage "
        "================"
    )
    print(
        "只打印 M7 单模块消融与 "
        "M8-A/M8-B。"
    )
    print(
        wide.to_string(index=False)
    )

    global_table = wide.pivot_table(
        index="model",
        columns="horizon",
        values="global",
        observed=False,
    ).reindex(model_order).reset_index()

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
        "Compact Global Rel-L2 (%) "
        "================"
    )
    print(
        global_table.to_string(
            index=False
        )
    )

    print_and_save_difference(
        wide=wide,
        model_a="M8-A-FullStatic-H4",
        model_b="M7-ParamTokenOnly-H4",
        title=(
            "M8-A-FullStatic-H4 - "
            "M7-ParamTokenOnly-H4"
        ),
        output_path=output_path,
    )

    print_and_save_difference(
        wide=wide,
        model_a="M8-A-FullStatic-H4",
        model_b="M7-FieldCoupling-H4",
        title=(
            "M8-A-FullStatic-H4 - "
            "M7-FieldCoupling-H4"
        ),
        output_path=output_path,
    )

    print_and_save_difference(
        wide=wide,
        model_a=(
            "M8-B-ParamConditionedCoupling-H4"
        ),
        model_b="M8-A-FullStatic-H4",
        title=(
            "M8-B-ParamConditionedCoupling-H4 "
            "- M8-A-FullStatic-H4"
        ),
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
