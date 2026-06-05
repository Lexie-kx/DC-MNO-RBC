import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
import json
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import DATA_PATH, FIELD_ORDER, DTYPE
from constants import CONTEXT_LENGTH

from training.normalization import IdentityNormalizer, FieldWiseNormalizer


class RBCDataset(Dataset):
    def __init__(
        self,
        split_config,
        normalize=False,
        stats_path=None,
        return_params=False,
    ):
        self.data_path = DATA_PATH
        self.split_config = split_config
        self.normalize = normalize
        self.return_params = return_params

        if self.normalize:
            assert stats_path is not None, "❌ 开启归一化必须提供 stats_path"
            self.normalizer = FieldWiseNormalizer(stats_path)
        else:
            self.normalizer = IdentityNormalizer()

        self.inputs = []
        self.targets = []
        self.params = []

        self._load_data()

    def _parse_ra_pr_from_group(self, group_name):
        """
        从 group_name 中解析 Ra / Pr。

        这里先写成兼容版本：
        只要 group_name 里包含类似 Ra1e6 / Pr1 / Ra_1e6 / Pr_0.5，
        就可以解析。

        如果你的 group_name 格式不一样，把这里发我，我再帮你精确改。
        """
        import re

        # 常见格式：
        # Ra1e6_Pr1
        # Ra_1e6_Pr_1
        # ra1e6_pr0.5
        ra_match = re.search(r"[Rr]a_?([0-9.eE+-]+)", group_name)
        pr_match = re.search(r"[Pp]r_?([0-9.eE+-]+)", group_name)

        if ra_match is None or pr_match is None:
            raise ValueError(
                f"❌ 无法从 group_name 解析 Ra/Pr: {group_name}\n"
                f"请检查 HDF5 group 名字格式，或者把 group_name 发给我。"
            )

        ra = float(ra_match.group(1))
        pr = float(pr_match.group(1))
        return ra, pr

    def _make_param_vector(self, group_name):
        """
        返回 [log10(Ra), log10(Pr)]
        shape: [2]
        """
        ra, pr = self._parse_ra_pr_from_group(group_name)

        param = np.array(
            [math.log10(float(ra)), math.log10(float(pr))],
            dtype=np.float32,
        )
        return param

    def _load_data(self):
        print(f"正在加载数据... 包含 {len(self.split_config)} 个工况组批次。")

        with h5py.File(self.data_path, 'r') as f:
            for item in self.split_config:
                group_name = item["group"]
                traj_indices = item["trajectories"]

                if group_name not in f:
                    print(f"⚠️ group 不存在，跳过: {group_name}")
                    continue

                group = f[group_name]

                fields_data = [group[field][:] for field in FIELD_ORDER]
                stacked_data = np.stack(fields_data, axis=0)
                selected_data = stacked_data[:, traj_indices]
                data = np.transpose(selected_data, (1, 2, 0, 3, 4))

                num_trajs, num_steps, c, h, w = data.shape

                # 每个 group 对应一个固定的 Ra/Pr
                if self.return_params:
                    param_vector = self._make_param_vector(group_name)

                for traj_idx in range(num_trajs):
                    for t in range(num_steps - CONTEXT_LENGTH):
                        # x_history: [T, C, H, W]
                        x_history = data[traj_idx, t: t + CONTEXT_LENGTH]

                        # [T, C, H, W] -> [T*C, H, W]
                        x_history_flat = x_history.reshape(CONTEXT_LENGTH * c, h, w)

                        # y_target: [C, H, W]
                        y_target = data[traj_idx, t + CONTEXT_LENGTH]

                        self.inputs.append(x_history_flat)
                        self.targets.append(y_target)

                        if self.return_params:
                            self.params.append(param_vector)

        self.inputs = torch.tensor(np.array(self.inputs), dtype=DTYPE)
        self.targets = torch.tensor(np.array(self.targets), dtype=DTYPE)

        if self.return_params:
            self.params = torch.tensor(np.array(self.params), dtype=DTYPE)

        print(f"✅ 加载完毕！共生成 {len(self.inputs)} 个样本对。")
        print(f"👉 输入 X 形状: {self.inputs[0].shape} (应为 [{CONTEXT_LENGTH * 4}, 256, 64])")
        print(f"👉 目标 Y 形状: {self.targets[0].shape} (应为 [4, 256, 64])")

        if self.return_params:
            print(f"👉 参数 param 形状: {self.params[0].shape} (应为 [2])")
            print(f"👉 示例 param = [log10(Ra), log10(Pr)] = {self.params[0].tolist()}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx]
        y = self.targets[idx]

        if self.normalize:
            x = self.normalizer.normalize_x(x)
            y = self.normalizer.normalize_y(y)

        if self.return_params:
            param = self.params[idx]
            return x, y, param

        return x, y


# ================= 连通性测试 =================
if __name__ == "__main__":
    split_file = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'data', 'splits', 'iid_split.json')
    )

    with open(split_file, 'r', encoding='utf-8') as f:
        full_split_config = json.load(f)

    print("\n--- 开始实例化 Dataset：测试 return_params=True ---")

    stats_file = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'data', 'stats', 'rbc_field_stats.json')
    )

    try:
        val_dataset = RBCDataset(
            split_config=full_split_config["val"],
            normalize=True,
            stats_path=stats_file,
            return_params=True,
        )
        print("✅ 成功开启归一化模式 + 参数返回模式！")
    except Exception as e:
        print(f"⚠️ 开启归一化失败或参数解析失败: {e}")
        print("退回普通模式，但仍测试 return_params=True ...")
        val_dataset = RBCDataset(
            split_config=full_split_config["val"],
            normalize=False,
            return_params=True,
        )

    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    for batch_x, batch_y, batch_param in val_loader:
        print(f"\n🎉 成功取出携带历史帧和物理参数的 Batch!")
        print(f"Batch X shape: {batch_x.shape}")
        print(f"Batch Y shape: {batch_y.shape}")
        print(f"Batch param shape: {batch_param.shape}")
        print(f"Batch param 示例: {batch_param[0].tolist()}")

        if val_dataset.normalize:
            print(f"X 均值 (期望接近 0): {batch_x.mean().item():.4f}")
            print(f"Y 均值 (期望接近 0): {batch_y.mean().item():.4f}")

        break