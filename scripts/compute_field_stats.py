import os
import json
import torch
import h5py
import numpy as np

# 将上级目录加入环境变量
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import DATA_PATH, FIELD_ORDER

def compute_stats_low_ram():
    split_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'splits', 'iid_split.json'))
    with open(split_file, 'r', encoding='utf-8') as f:
        full_split_config = json.load(f)
        train_config = full_split_config["train"]

    sums = torch.zeros(4, dtype=torch.float64)
    sum_sqs = torch.zeros(4, dtype=torch.float64)
    total_pixels = 0

    print("🚀 启动低内存流式扫描 (Low-RAM Streaming)...")
    
    with h5py.File(DATA_PATH, 'r') as f:
        for item in train_config:
            group_name = item["group"]
            traj_indices = item["trajectories"]
            
            if group_name not in f:
                continue
                
            group = f[group_name]
            print(f"正在处理工况组: {group_name} ...")
            
            # 核心优化：每次只取 1 条轨迹进入内存
            for traj_idx in traj_indices:
                traj_data = []
                for field in FIELD_ORDER:
                    # group[field] 形状通常是 [T, Num_trajs, H, W]
                    # 我们只切片当前 traj_idx 的数据
                    field_array = group[field][traj_idx, :, :, :] 
                    traj_data.append(field_array)
                
                # stack 后的形状: [4, T, H, W]
                traj_tensor = torch.tensor(np.stack(traj_data, axis=0), dtype=torch.float64)
                
                # 累加统计量
                sums += traj_tensor.sum(dim=(1, 2, 3))
                sum_sqs += (traj_tensor ** 2).sum(dim=(1, 2, 3))
                
                T, H, W = traj_tensor.shape[1], traj_tensor.shape[2], traj_tensor.shape[3]
                total_pixels += T * H * W

    mean = sums / total_pixels
    variance = (sum_sqs / total_pixels) - mean ** 2
    std = torch.sqrt(torch.clamp(variance, min=1e-12))

    fields = ["buoyancy", "u_x", "u_y", "pressure"]
    stats_dict = {}

    print("\n✅ 统计完成！全局场统计量如下：")
    for i, field in enumerate(fields):
        stats_dict[field] = {
            "mean": mean[i].item(),
            "std": std[i].item(),
        }
        print(f"[{field}] mean={mean[i].item():.6e}, std={std[i].item():.6e}")

    os.makedirs(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'stats')), exist_ok=True)
    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'stats', 'rbc_field_stats.json'))
    
    with open(out_path, "w") as f:
        json.dump(stats_dict, f, indent=4)

    print(f"📁 统计量已成功保存至: {out_path}")

if __name__ == "__main__":
    compute_stats_low_ram()