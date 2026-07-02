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
from models.operators.fno2d_fieldwise import FieldWiseFNO2d


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
        description="Cross-parameter rollout evaluation for M0 / M3-Delta / FiLM / M5-H4 / ParamToken / M6 CouplingToken."
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

    parser.add_argument(
        "--m6_fieldwise_encoder_h4_checkpoint",
        type=str,
        required=True,
        help="Checkpoint for M6-FieldWiseEncoder-H4."
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



def make_probe_inputs(x0_phys):
    """
    Create evaluation-time coupling probes.

    x0_phys: [B, 16, H, W]
    channel layout: context_length=4, num_fields=4
    fields: 0=buoyancy, 1=u_x, 2=u_y, 3=pressure

    Probe meaning:
    - full: normal input
    - mask_*: remove one field history by setting it to zero
    - shuffle_*: shuffle one field history across batch samples
    """
    bsz, channels, h, w = x0_phys.shape
    assert channels == CONTEXT_LENGTH * 4, f"Expected 16 channels, got {channels}"

    def reshape(x):
        return x.reshape(bsz, CONTEXT_LENGTH, 4, h, w)

    def flatten(x):
        return x.reshape(bsz, CONTEXT_LENGTH * 4, h, w)

    probes = {}

    probes["full"] = x0_phys.clone()

    # mask probes
    x = reshape(x0_phys.clone())
    x[:, :, 0, :, :] = 0.0
    probes["mask_buoyancy"] = flatten(x)

    x = reshape(x0_phys.clone())
    x[:, :, 1, :, :] = 0.0
    x[:, :, 2, :, :] = 0.0
    probes["mask_velocity"] = flatten(x)

    x = reshape(x0_phys.clone())
    x[:, :, 3, :, :] = 0.0
    probes["mask_pressure"] = flatten(x)

    # shuffle probes: if last batch has bsz=1, shuffle is no-op
    if bsz > 1:
        perm = torch.randperm(bsz, device=x0_phys.device)

        x = reshape(x0_phys.clone())
        x[:, :, 0, :, :] = x[perm, :, 0, :, :]
        probes["shuffle_buoyancy"] = flatten(x)

        perm = torch.randperm(bsz, device=x0_phys.device)
        x = reshape(x0_phys.clone())
        x[:, :, 1, :, :] = x[perm, :, 1, :, :]
        x[:, :, 2, :, :] = x[perm, :, 2, :, :]
        probes["shuffle_velocity"] = flatten(x)
    else:
        probes["shuffle_buoyancy"] = x0_phys.clone()
        probes["shuffle_velocity"] = x0_phys.clone()

    return probes


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
    m6_fieldwise_encoder_h4_ckpt = resolve_path(project_root, args.m6_fieldwise_encoder_h4_checkpoint)
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
    print(f"📌 M6-FieldWiseEncoder-H4 checkpoint: {m6_fieldwise_encoder_h4_ckpt}")
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
).to(device)

    m5_delta_paramtoken_h4 = load_model_state(
        m5_delta_paramtoken_h4,
        m5_delta_paramtoken_h4_ckpt,
        device,
    )
    m5_delta_paramtoken_h4.eval()

    m6_fieldwise_encoder_h4 = FieldWiseFNO2d(
        in_channels=16,
        out_channels=4,
        modes1=16,
        modes2=16,
        width=32,
).to(device)

    m6_fieldwise_encoder_h4 = load_model_state(
        m6_fieldwise_encoder_h4,
        m6_fieldwise_encoder_h4_ckpt,
        device,
    )
    m6_fieldwise_encoder_h4.eval()

    stats_dict = {}

    print("\n🔥 开始 Coupling Probe evaluation...")

    with torch.no_grad():
        for batch_idx, (x0_phys, future_phys, param) in enumerate(loader):
            x0_phys = x0_phys.to(device)
            future_phys = future_phys.to(device)
            param = param.to(device)

            # Coupling Probe for M6-FieldWiseEncoder-H4
            probe_inputs = make_probe_inputs(x0_phys)
            probe_preds = {}

            for probe_name, probe_x0 in probe_inputs.items():
                probe_preds[probe_name] = rollout_m3_delta(
                    model=m6_fieldwise_encoder_h4,
                    x0_phys=probe_x0,
                    normalizer=normalizer,
                    max_horizon=max_horizon,
                )

            for horizon in horizons:
                idx = horizon - 1

                true_h = future_phys[:, idx, :, :, :]

                for probe_name, seq_preds in probe_preds.items():
                    update_error_stats(
                        pred_phys=seq_preds[idx],
                        true_phys=true_h,
                        stats_dict=stats_dict,
                        horizon=horizon,
                        model_name=probe_name,
                    )

            if (batch_idx + 1) % 10 == 0:
                print(f"  已处理 batch {batch_idx + 1}/{len(loader)}")

    rows = []

    model_order = [
        "full",
        "mask_buoyancy",
        "mask_velocity",
        "mask_pressure",
        "shuffle_buoyancy",
        "shuffle_velocity",
    ]

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

    print(f"\n✅ Coupling Probe saved to: {output_path}")

    wide = df.pivot_table(
        index=["model", "horizon"],
        columns="field",
        values="rel_l2_percent",
    ).reset_index()

    field_order = ["buoyancy", "u_x", "u_y", "pressure", "global"]
    wide = wide[["model", "horizon"] + field_order]

    print("\n================ Coupling Probe Summary: Rel-L2 (%) ================")
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

    def print_probe_difference(wide_df, probe_name, base_name="full"):
        """
        Print probe difference: probe - full.
        Positive means the probe makes error larger, indicating the removed/shuffled field is useful.
        """
        probe = wide_df[wide_df["model"] == probe_name].copy()
        base = wide_df[wide_df["model"] == base_name].copy()

        if probe.empty or base.empty:
            print(f"\n⚠️ 跳过差值表：找不到 {probe_name} 或 {base_name}")
            return None

        probe = probe.set_index("horizon")
        base = base.set_index("horizon")

        diff = probe[["buoyancy", "u_x", "u_y", "pressure", "global"]] - base[["buoyancy", "u_x", "u_y", "pressure", "global"]]
        diff = diff.reset_index()
        diff.insert(0, "probe", probe_name)

        print(f"\n================ {probe_name} - full: Rel-L2 (%) ================")
        print("说明：正数表示扰动后误差变大，即该物理场信息对预测有贡献。")
        print(diff.to_string(index=False))
        return diff

    diff_rows = []
    for probe_name in [
        "mask_buoyancy",
        "mask_velocity",
        "mask_pressure",
        "shuffle_buoyancy",
        "shuffle_velocity",
    ]:
        d = print_probe_difference(wide, probe_name)
        if d is not None:
            diff_rows.append(d)

    if diff_rows:
        diff_df = pd.concat(diff_rows, ignore_index=True)
        diff_path = output_path.replace(".csv", "_probe_minus_full.csv")
        diff_df.to_csv(diff_path, index=False)
        print(f"\n✅ Probe difference table saved to: {diff_path}")


if __name__ == "__main__":
    main()
