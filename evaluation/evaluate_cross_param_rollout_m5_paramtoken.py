import os
import sys
import json
import math
import argparse
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from constants import DATA_PATH, FIELD_ORDER, CONTEXT_LENGTH, DTYPE
from training.normalization import FieldWiseNormalizer
from models.operators.fno2d import PlainFNO2d
from models.operators.fno2d_film import FiLMFNO2d
from models.operators.fno2d_paramtoken import ParamTokenFNO2d


class RolloutDataset(Dataset):
    """
    用于 cross-parameter rollout evaluation。

    每个样本返回：
        x0_phys:      [16, H, W] 初始 4 帧历史，物理空间
        future_phys:  [max_horizon, 4, H, W] 未来连续真值，物理空间
        param:        [2] = [log10(Ra), log10(Pr)]
    """

    def __init__(self, split_config, max_horizon=16, max_samples=None):
        self.split_config = split_config
        self.max_horizon = max_horizon
        self.max_samples = max_samples

        self.data_store = {}
        self.index = []

        self._load_data()

    def _parse_ra_pr_from_group(self, group_name):
        import re

        ra_match = re.search(r"[Rr]a_?([0-9.eE+-]+)", group_name)
        pr_match = re.search(r"[Pp]r_?([0-9.eE+-]+)", group_name)

        if ra_match is None or pr_match is None:
            raise ValueError(f"无法从 group_name 解析 Ra/Pr: {group_name}")

        ra = float(ra_match.group(1))
        pr = float(pr_match.group(1))

        return ra, pr

    def _make_param(self, group_name):
        ra, pr = self._parse_ra_pr_from_group(group_name)

        return torch.tensor(
            [math.log10(float(ra)), math.log10(float(pr))],
            dtype=DTYPE,
        )

    def _load_data(self):
        print(f"📦 正在加载 rollout 数据，共 {len(self.split_config)} 个 group 批次")

        with h5py.File(DATA_PATH, "r") as f:
            for item in self.split_config:
                group_name = item["group"]
                traj_indices = item["trajectories"]

                if group_name not in f:
                    print(f"⚠️ group 不存在，跳过: {group_name}")
                    continue

                group = f[group_name]

                fields_data = [group[field][:] for field in FIELD_ORDER]
                stacked_data = torch.tensor(
                    __import__("numpy").stack(fields_data, axis=0),
                    dtype=DTYPE,
                )
                # stacked_data: [4, num_traj, T, H, W]

                selected_data = stacked_data[:, traj_indices]
                # [4, selected_traj, T, H, W]

                data = selected_data.permute(1, 2, 0, 3, 4).contiguous()
                # [selected_traj, T, 4, H, W]

                key = group_name
                self.data_store[key] = data

                param = self._make_param(group_name)

                num_traj, num_steps, _, _, _ = data.shape

                max_start = num_steps - CONTEXT_LENGTH - self.max_horizon + 1

                if max_start <= 0:
                    print(f"⚠️ group {group_name} 时间长度不足，跳过")
                    continue

                for local_traj_idx in range(num_traj):
                    for t0 in range(max_start):
                        self.index.append(
                            {
                                "group": key,
                                "traj": local_traj_idx,
                                "t0": t0,
                                "param": param,
                            }
                        )

        if self.max_samples is not None:
            self.index = self.index[: self.max_samples]

        print(f"✅ Rollout 样本数: {len(self.index)}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        item = self.index[idx]

        group_name = item["group"]
        traj_idx = item["traj"]
        t0 = item["t0"]
        param = item["param"]

        data = self.data_store[group_name]
        # [traj, T, 4, H, W]

        history = data[
            traj_idx,
            t0 : t0 + CONTEXT_LENGTH,
        ]
        # [4, 4, H, W]

        future = data[
            traj_idx,
            t0 + CONTEXT_LENGTH : t0 + CONTEXT_LENGTH + self.max_horizon,
        ]
        # [max_horizon, 4, H, W]

        _, _, h, w = history.shape
        x0_phys = history.reshape(CONTEXT_LENGTH * 4, h, w)
        # [16, H, W]

        return x0_phys, future, param


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-parameter rollout evaluation for M0 / M3-Delta / M3-Delta-FiLM / M5-Delta-H4 / M5-Delta-FiLM-H4 / M5-Delta-ParameterToken-H4."
    )

    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--stats", type=str, required=True)

    parser.add_argument(
        "--m3_delta_checkpoint",
        type=str,
        required=True,
        help="Checkpoint for M3-Delta."
    )

    parser.add_argument(
        "--film_checkpoint",
        type=str,
        required=True,
        help="Checkpoint for M3-Delta-FiLM."
    )

    parser.add_argument(
        "--m5_delta_h4_checkpoint",
        type=str,
        required=True,
        help="Checkpoint for M5-Delta-H4."
    )

    parser.add_argument(
        "--m5_delta_film_h4_checkpoint",
        type=str,
        required=True,
        help="Checkpoint for M5-Delta-FiLM-H4."
    )

    parser.add_argument(
        "--m5_delta_paramtoken_h4_checkpoint",
        type=str,
        required=True,
        help="Checkpoint for M5-Delta-ParameterToken-H4."
    )

    parser.add_argument("--output", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument(
        "--horizons",
        type=str,
        default="1,4,8,16",
        help="Comma-separated horizons, e.g. 1,4,8,16"
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional debug limit. Use None for full evaluation."
    )

    return parser.parse_args()


def resolve_path(project_root, path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(project_root, path))


def load_model_state(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return model


def update_error_stats(pred_phys, true_phys, stats_dict, horizon, model_name):
    """
    pred_phys: [B, 4, H, W]
    true_phys: [B, 4, H, W]
    """

    diff = pred_phys - true_phys

    key = (model_name, horizon)

    if key not in stats_dict:
        stats_dict[key] = {
            "field_sse": torch.zeros(4, dtype=torch.float64, device=pred_phys.device),
            "field_target_sq": torch.zeros(4, dtype=torch.float64, device=pred_phys.device),
            "field_numel": torch.zeros(4, dtype=torch.float64, device=pred_phys.device),
            "global_sse": torch.tensor(0.0, dtype=torch.float64, device=pred_phys.device),
            "global_target_sq": torch.tensor(0.0, dtype=torch.float64, device=pred_phys.device),
            "global_numel": torch.tensor(0.0, dtype=torch.float64, device=pred_phys.device),
        }

    bucket = stats_dict[key]

    for c in range(4):
        diff_c = diff[:, c, :, :].double()
        true_c = true_phys[:, c, :, :].double()

        bucket["field_sse"][c] += torch.sum(diff_c ** 2)
        bucket["field_target_sq"][c] += torch.sum(true_c ** 2)
        bucket["field_numel"][c] += diff_c.numel()

    bucket["global_sse"] += torch.sum(diff.double() ** 2)
    bucket["global_target_sq"] += torch.sum(true_phys.double() ** 2)
    bucket["global_numel"] += diff.numel()


def rollout_m0(x0_phys, max_horizon):
    """
    Persistence rollout。

    x0_phys: [B, 16, H, W]
    返回:
        preds: [max_horizon, B, 4, H, W]
    """

    current = x0_phys[:, -4:, :, :]
    preds = []

    for _ in range(max_horizon):
        preds.append(current)
        # persistence feed-back 后仍然是 current，不变

    return torch.stack(preds, dim=0)


def rollout_m3_delta(model, x0_phys, normalizer, max_horizon):
    """
    M3-Delta autoregressive rollout。

    使用 normalized state 进行滚动：
        x_norm -> pred_delta_norm
        pred_y_norm = x_last_norm + pred_delta_norm
        append pred_y_norm to normalized history
    """

    x_norm = normalizer.normalize_x(x0_phys)

    preds_phys = []

    for _ in range(max_horizon):
        x_last_norm = x_norm[:, -4:, :, :]

        pred_delta_norm = model(x_norm)
        pred_y_norm = x_last_norm + pred_delta_norm

        pred_y_phys = normalizer.denormalize_y(pred_y_norm)
        preds_phys.append(pred_y_phys)

        x_norm = torch.cat(
            [
                x_norm[:, 4:, :, :],
                pred_y_norm,
            ],
            dim=1,
        )

    return torch.stack(preds_phys, dim=0)


def rollout_film(model, x0_phys, param, normalizer, max_horizon):
    """
    M3-Delta-FiLM autoregressive rollout。
    """

    x_norm = normalizer.normalize_x(x0_phys)

    preds_phys = []

    for _ in range(max_horizon):
        x_last_norm = x_norm[:, -4:, :, :]

        pred_delta_norm = model(x_norm, param)
        pred_y_norm = x_last_norm + pred_delta_norm

        pred_y_phys = normalizer.denormalize_y(pred_y_norm)
        preds_phys.append(pred_y_phys)

        x_norm = torch.cat(
            [
                x_norm[:, 4:, :, :],
                pred_y_norm,
            ],
            dim=1,
        )

    return torch.stack(preds_phys, dim=0)


def main():
    args = parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    split_path = resolve_path(project_root, args.split)
    stats_path = resolve_path(project_root, args.stats)
    m3_delta_ckpt = resolve_path(project_root, args.m3_delta_checkpoint)
    film_ckpt = resolve_path(project_root, args.film_checkpoint)
    m5_delta_h4_ckpt = resolve_path(project_root, args.m5_delta_h4_checkpoint)
    m5_delta_film_h4_ckpt = resolve_path(project_root, args.m5_delta_film_h4_checkpoint)
    m5_delta_paramtoken_h4_ckpt = resolve_path(project_root, args.m5_delta_paramtoken_h4_checkpoint)
    output_path = resolve_path(project_root, args.output)

    horizons = [int(x) for x in args.horizons.split(",")]
    max_horizon = max(horizons)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("🚀 [Cross-Parameter Rollout Evaluation]")
    print(f"📌 Device: {device}")
    print(f"📌 Split: {split_path}")
    print(f"📌 Stats: {stats_path}")
    print(f"📌 M3-Delta checkpoint: {m3_delta_ckpt}")
    print(f"📌 FiLM checkpoint: {film_ckpt}")
    print(f"📌 M5-Delta-H4 checkpoint: {m5_delta_h4_ckpt}")
    print(f"📌 M5-Delta-FiLM-H4 checkpoint: {m5_delta_film_h4_ckpt}")
    print(f"📌 M5-Delta-ParameterToken-H4 checkpoint: {m5_delta_paramtoken_h4_ckpt}")
    print(f"📌 Horizons: {horizons}")
    print(f"📌 Output: {output_path}")

    with open(split_path, "r", encoding="utf-8") as f:
        split_config = json.load(f)

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

    normalizer = FieldWiseNormalizer(stats_path).to(device)

    m3_delta = PlainFNO2d(
        in_channels=16,
        out_channels=4,
        modes1=16,
        modes2=16,
        width=32,
    ).to(device)

    m3_delta = load_model_state(m3_delta, m3_delta_ckpt, device)
    m3_delta.eval()

    m5_delta_h4 = PlainFNO2d(
        in_channels=16,
        out_channels=4,
        modes1=16,
        modes2=16,
        width=32,
    ).to(device)

    m5_delta_h4 = load_model_state(m5_delta_h4, m5_delta_h4_ckpt, device)
    m5_delta_h4.eval()

    film = FiLMFNO2d(
        in_channels=16,
        out_channels=4,
        modes1=16,
        modes2=16,
        width=32,
        film_hidden_dim=64,
    ).to(device)

    film = load_model_state(film, film_ckpt, device)
    film.eval()

    m5_delta_film_h4 = FiLMFNO2d(
        in_channels=16,
        out_channels=4,
        modes1=16,
        modes2=16,
        width=32,
        film_hidden_dim=64,
    ).to(device)

    m5_delta_film_h4 = load_model_state(m5_delta_film_h4, m5_delta_film_h4_ckpt, device)
    m5_delta_film_h4.eval()

    m5_delta_paramtoken_h4 = ParamTokenFNO2d(
        in_channels=16,
        out_channels=4,
        modes1=16,
        modes2=16,
        width=32,
        token_hidden_dim=64,
    ).to(device)

    m5_delta_paramtoken_h4 = load_model_state(
        m5_delta_paramtoken_h4,
        m5_delta_paramtoken_h4_ckpt,
        device,
    )
    m5_delta_paramtoken_h4.eval()

    stats_dict = {}

    print("\n🔥 开始 rollout evaluation...")

    with torch.no_grad():
        for batch_idx, (x0_phys, future_phys, param) in enumerate(loader):
            x0_phys = x0_phys.to(device)
            future_phys = future_phys.to(device)
            param = param.to(device)

            # M0
            m0_preds = rollout_m0(x0_phys, max_horizon)

            # M3-Delta
            m3_preds = rollout_m3_delta(
                model=m3_delta,
                x0_phys=x0_phys,
                normalizer=normalizer,
                max_horizon=max_horizon,
            )

            # M5-Delta-H4
            # 结构与 M3-Delta 相同，区别是 checkpoint 经过 4-step autoregressive fine-tuning
            m5_preds = rollout_m3_delta(
                model=m5_delta_h4,
                x0_phys=x0_phys,
                normalizer=normalizer,
                max_horizon=max_horizon,
            )

            # M3-Delta-FiLM
            film_preds = rollout_film(
                model=film,
                x0_phys=x0_phys,
                param=param,
                normalizer=normalizer,
                max_horizon=max_horizon,
            )

            # M5-Delta-FiLM-H4
            # 结构与 M3-Delta-FiLM 相同，区别是 checkpoint 经过 4-step autoregressive fine-tuning
            m5_film_preds = rollout_film(
                model=m5_delta_film_h4,
                x0_phys=x0_phys,
                param=param,
                normalizer=normalizer,
                max_horizon=max_horizon,
            )

            # M5-Delta-ParameterToken-H4
            # 和 FiLM 一样需要 param 输入，但参数注入方式改为 additive parameter token
            m5_paramtoken_preds = rollout_film(
                model=m5_delta_paramtoken_h4,
                x0_phys=x0_phys,
                param=param,
                normalizer=normalizer,
                max_horizon=max_horizon,
            )

            for horizon in horizons:
                idx = horizon - 1

                true_h = future_phys[:, idx, :, :, :]

                update_error_stats(
                    pred_phys=m0_preds[idx],
                    true_phys=true_h,
                    stats_dict=stats_dict,
                    horizon=horizon,
                    model_name="M0",
                )

                update_error_stats(
                    pred_phys=m3_preds[idx],
                    true_phys=true_h,
                    stats_dict=stats_dict,
                    horizon=horizon,
                    model_name="M3-Delta",
                )

                update_error_stats(
                    pred_phys=m5_preds[idx],
                    true_phys=true_h,
                    stats_dict=stats_dict,
                    horizon=horizon,
                    model_name="M5-Delta-H4",
                )

                update_error_stats(
                    pred_phys=film_preds[idx],
                    true_phys=true_h,
                    stats_dict=stats_dict,
                    horizon=horizon,
                    model_name="M3-Delta-FiLM",
                )

                update_error_stats(
                    pred_phys=m5_film_preds[idx],
                    true_phys=true_h,
                    stats_dict=stats_dict,
                    horizon=horizon,
                    model_name="M5-Delta-FiLM-H4",
                )

                update_error_stats(
                    pred_phys=m5_paramtoken_preds[idx],
                    true_phys=true_h,
                    stats_dict=stats_dict,
                    horizon=horizon,
                    model_name="M5-Delta-ParameterToken-H4",
                )

            if (batch_idx + 1) % 10 == 0:
                print(f"  已处理 batch {batch_idx + 1}/{len(loader)}")

    rows = []

    model_order = ["M0", "M3-Delta", "M3-Delta-FiLM", "M5-Delta-H4", "M5-Delta-FiLM-H4", "M5-Delta-ParameterToken-H4"]

    for model_name in model_order:
        for horizon in horizons:
            bucket = stats_dict[(model_name, horizon)]

            for c, field in enumerate(FIELD_ORDER):
                rel_l2 = torch.sqrt(
                    bucket["field_sse"][c] / (bucket["field_target_sq"][c] + 1e-12)
                ) * 100.0

                mse = bucket["field_sse"][c] / bucket["field_numel"][c]

                rows.append(
                    {
                        "model": model_name,
                        "horizon": horizon,
                        "field": field,
                        "rel_l2_percent": rel_l2.item(),
                        "mse": mse.item(),
                    }
                )

            global_rel_l2 = torch.sqrt(
                bucket["global_sse"] / (bucket["global_target_sq"] + 1e-12)
            ) * 100.0

            global_mse = bucket["global_sse"] / bucket["global_numel"]

            rows.append(
                {
                    "model": model_name,
                    "horizon": horizon,
                    "field": "global",
                    "rel_l2_percent": global_rel_l2.item(),
                    "mse": global_mse.item(),
                }
            )

    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n✅ Rollout evaluation saved to: {output_path}")

    wide = df.pivot_table(
        index=["model", "horizon"],
        columns="field",
        values="rel_l2_percent",
    ).reset_index()

    field_order = ["buoyancy", "u_x", "u_y", "pressure", "global"]
    wide = wide[["model", "horizon"] + field_order]

    print("\n================ Rollout Summary: Rel-L2 (%) ================")
    print(wide.to_string(index=False))

    # 更直观的分块打印：按 horizon 分开
    print("\n================ By Horizon: Field-wise Rel-L2 (%) ================")
    for h in horizons:
        block = wide[wide["horizon"] == h].copy()
        block = block[["model", "buoyancy", "u_x", "u_y", "pressure", "global"]]
        print(f"\n---------- Horizon h={h} ----------")
        print(block.to_string(index=False))

    # 单独打印 global，方便看整体 rollout drift
    global_table = wide.pivot_table(
        index="model",
        columns="horizon",
        values="global",
    ).reset_index()

    global_table.columns = [
        "model" if col == "model" else f"h={col}" for col in global_table.columns
    ]

    print("\n================ Global Rel-L2 (%) by Horizon ================")
    print(global_table.to_string(index=False))

    # 单独打印速度场，方便看 u_x / u_y 是否改善
    ux_table = wide.pivot_table(
        index="model",
        columns="horizon",
        values="u_x",
    ).reset_index()
    ux_table.columns = [
        "model" if col == "model" else f"h={col}" for col in ux_table.columns
    ]

    uy_table = wide.pivot_table(
        index="model",
        columns="horizon",
        values="u_y",
    ).reset_index()
    uy_table.columns = [
        "model" if col == "model" else f"h={col}" for col in uy_table.columns
    ]

    print("\n================ u_x Rel-L2 (%) by Horizon ================")
    print(ux_table.to_string(index=False))

    print("\n================ u_y Rel-L2 (%) by Horizon ================")
    print(uy_table.to_string(index=False))

    # 单独打印 M5 相对 M3-Delta 的变化，负数表示 M5 更好，正数表示 M5 更差
    if "M5-Delta-H4" in set(wide["model"]) and "M3-Delta" in set(wide["model"]):
        m3 = wide[wide["model"] == "M3-Delta"].set_index("horizon")
        m5 = wide[wide["model"] == "M5-Delta-H4"].set_index("horizon")

        delta_rows = []
        for h in horizons:
            row = {"horizon": h}
            for field in field_order:
                row[field] = float(m5.loc[h, field] - m3.loc[h, field])
            delta_rows.append(row)

        delta_df = pd.DataFrame(delta_rows)

        print("\n================ M5 - M3-Delta Difference: Rel-L2 (%) ================")
        print("说明：负数表示 M5 更好；正数表示 M5 更差。")
        print(delta_df.to_string(index=False))


    def print_model_difference(wide_df, model_a, model_b, title):
        """
        Print field-wise difference table: model_a - model_b.
        Negative means model_a is better.
        """
        a = wide_df[wide_df["model"] == model_a].copy()
        b = wide_df[wide_df["model"] == model_b].copy()

        if a.empty or b.empty:
            print(f"\n⚠️ 跳过差值表：找不到 {model_a} 或 {model_b}")
            return

        a = a.set_index("horizon")
        b = b.set_index("horizon")

        diff = a[["buoyancy", "u_x", "u_y", "pressure", "global"]] - b[["buoyancy", "u_x", "u_y", "pressure", "global"]]
        diff = diff.reset_index()
        diff = diff[["horizon", "buoyancy", "u_x", "u_y", "pressure", "global"]]

        print(f"\n================ {title}: Rel-L2 (%) ================")
        print("说明：负数表示前者更好；正数表示前者更差。")
        print(diff.to_string(index=False))

        # Save difference table next to rollout output
        diff_name = title.lower()
        diff_name = diff_name.replace(" ", "_").replace("-", "_").replace(":", "")
        diff_path = output_path.replace(".csv", f"_{diff_name}.csv")
        diff.to_csv(diff_path, index=False)
        print(f"✅ Difference table saved to: {diff_path}")

    print_model_difference(
        wide_df=wide,
        model_a="M5-Delta-FiLM-H4",
        model_b="M5-Delta-H4",
        title="M5-Delta-FiLM-H4 - M5-Delta-H4"
    )

    print_model_difference(
        wide_df=wide,
        model_a="M5-Delta-FiLM-H4",
        model_b="M3-Delta",
        title="M5-Delta-FiLM-H4 - M3-Delta"
    )


if __name__ == "__main__":
    main()
