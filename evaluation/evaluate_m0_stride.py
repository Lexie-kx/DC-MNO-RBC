import os
import sys
import json
import h5py
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from constants import DATA_PATH, FIELD_ORDER, DTYPE
from training.metrics import (
    compute_field_wise_rel_l2,
    compute_total_rel_l2,
    compute_field_wise_mse,
    compute_total_mse
)


def init_metrics():
    return {
        "rel": {
            "buoyancy": 0.0,
            "u_x": 0.0,
            "u_y": 0.0,
            "pressure": 0.0,
            "global": 0.0,
        },
        "mse": {
            "buoyancy": 0.0,
            "u_x": 0.0,
            "u_y": 0.0,
            "pressure": 0.0,
            "global": 0.0,
        },
        "num_batches": 0,
    }


def update_metrics(metrics, pred, target):
    field_rel = compute_field_wise_rel_l2(pred, target)
    total_rel = compute_total_rel_l2(pred, target)

    field_mse = compute_field_wise_mse(pred, target)
    total_mse = compute_total_mse(pred, target)

    for k in ["buoyancy", "u_x", "u_y", "pressure"]:
        metrics["rel"][k] += field_rel[k]
        metrics["mse"][k] += field_mse[k]

    metrics["rel"]["global"] += total_rel
    metrics["mse"]["global"] += total_mse
    metrics["num_batches"] += 1


def average_metrics(metrics):
    n = max(metrics["num_batches"], 1)

    for metric_type in ["rel", "mse"]:
        for k in metrics[metric_type]:
            metrics[metric_type][k] /= n

    return metrics


def evaluate_m0_for_stride(split_config, stride, batch_size=16):
    """
    M0 persistence stride 诊断。

    对于每个样本：
        输入当前帧 X_t
        M0 预测 X_t
        目标 X_{t+stride}

    注意：
        这里不是使用 history window，而是直接从 HDF5 里取单帧。
        这样可以干净评估 persistence 随 stride 增大的误差变化。
    """

    metrics = init_metrics()

    pred_buffer = []
    target_buffer = []

    with h5py.File(DATA_PATH, "r") as f:
        for item in split_config:
            group_name = item["group"]
            traj_indices = item["trajectories"]

            if group_name not in f:
                print(f"⚠️ group 不存在，跳过: {group_name}")
                continue

            group = f[group_name]

            # fields_data: list of [num_traj, num_steps, H, W]
            fields_data = [group[field][:] for field in FIELD_ORDER]

            # stacked: [C, num_traj, num_steps, H, W]
            stacked = np.stack(fields_data, axis=0)

            # selected: [C, selected_traj, num_steps, H, W]
            selected = stacked[:, traj_indices]

            # data: [selected_traj, num_steps, C, H, W]
            data = np.transpose(selected, (1, 2, 0, 3, 4))

            num_trajs, num_steps, c, h, w = data.shape

            for traj_idx in range(num_trajs):
                # t + stride 不能越界
                for t in range(num_steps - stride):
                    x_t = data[traj_idx, t]
                    y_target = data[traj_idx, t + stride]

                    pred_buffer.append(x_t)
                    target_buffer.append(y_target)

                    if len(pred_buffer) == batch_size:
                        pred = torch.tensor(np.array(pred_buffer), dtype=DTYPE)
                        target = torch.tensor(np.array(target_buffer), dtype=DTYPE)

                        update_metrics(metrics, pred, target)

                        pred_buffer = []
                        target_buffer = []

    # 处理最后不足一个 batch 的样本
    if len(pred_buffer) > 0:
        pred = torch.tensor(np.array(pred_buffer), dtype=DTYPE)
        target = torch.tensor(np.array(target_buffer), dtype=DTYPE)

        update_metrics(metrics, pred, target)

    metrics = average_metrics(metrics)
    return metrics


def print_stride_results(results):
    fields = ["buoyancy", "u_x", "u_y", "pressure", "global"]

    print("=" * 105)
    print("📊 M0 Persistence Stride Diagnostic")
    print("=" * 105)
    print(
        f"{'Stride':<10} | {'Field':<14} | {'Rel-L2 (%)':>12} | {'MSE':>14}"
    )
    print("-" * 105)

    for stride, metrics in results.items():
        for field in fields:
            rel = metrics["rel"][field] * 100.0
            mse = metrics["mse"][field]

            print(
                f"{stride:<10} | {field:<14} | {rel:>11.2f} % | {mse:>14.6f}"
            )

        print("-" * 105)

    print("=" * 105)


def save_stride_csv(results, save_path):
    import pandas as pd

    rows = []

    for stride, metrics in results.items():
        for field in ["buoyancy", "u_x", "u_y", "pressure", "global"]:
            rows.append({
                "stride": stride,
                "field": field,
                "rel_l2_percent": metrics["rel"][field] * 100.0,
                "mse": metrics["mse"][field],
            })

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"✅ stride 诊断 CSV 已保存至: {save_path}")


def plot_stride_global(results, save_dir):
    import matplotlib.pyplot as plt

    strides = list(results.keys())

    global_rel = [
        results[s]["rel"]["global"] * 100.0
        for s in strides
    ]

    ux_rel = [
        results[s]["rel"]["u_x"] * 100.0
        for s in strides
    ]

    uy_rel = [
        results[s]["rel"]["u_y"] * 100.0
        for s in strides
    ]

    buoyancy_rel = [
        results[s]["rel"]["buoyancy"] * 100.0
        for s in strides
    ]

    pressure_rel = [
        results[s]["rel"]["pressure"] * 100.0
        for s in strides
    ]

    # 图 1：global
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(strides, global_rel, marker="o")
    ax.set_xlabel("Stride")
    ax.set_ylabel("M0 Global Rel-L2 (%)")
    ax.set_title("M0 Persistence Error Increases with Prediction Stride", fontweight="bold")
    ax.set_xticks(strides)
    ax.grid(True, alpha=0.3)

    global_path = os.path.join(save_dir, "m0_stride_global_rel_l2.png")
    plt.savefig(global_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"✅ global stride 图已保存至: {global_path}")

    # 图 2：field-wise
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(strides, buoyancy_rel, marker="o", label="buoyancy")
    ax.plot(strides, ux_rel, marker="o", label="u_x")
    ax.plot(strides, uy_rel, marker="o", label="u_y")
    ax.plot(strides, pressure_rel, marker="o", label="pressure")

    ax.set_xlabel("Stride")
    ax.set_ylabel("M0 Field-wise Rel-L2 (%)")
    ax.set_title("Field-wise M0 Persistence Error under Different Strides", fontweight="bold")
    ax.set_xticks(strides)
    ax.grid(True, alpha=0.3)
    ax.legend()

    field_path = os.path.join(save_dir, "m0_stride_fieldwise_rel_l2.png")
    plt.savefig(field_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"✅ field-wise stride 图已保存至: {field_path}")


def main():
    print("🚀 启动 M0 Persistence stride 诊断")

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    split_file = os.path.join(root_dir, "data", "splits", "iid_split.json")

    with open(split_file, "r", encoding="utf-8") as f:
        split_config_all = json.load(f)

    # 用 val split 做诊断，和你前面的评估保持一致
    val_split = split_config_all["val"]

    strides = [1, 2, 4, 8]
    batch_size = 16

    results = {}

    for stride in strides:
        print(f"\n⏳ 正在评估 M0 Persistence | stride = {stride}")
        metrics = evaluate_m0_for_stride(
            split_config=val_split,
            stride=stride,
            batch_size=batch_size
        )
        results[stride] = metrics

    print_stride_results(results)

    save_dir = os.path.join(root_dir, "outputs", "figures", "stride_diagnostic")
    os.makedirs(save_dir, exist_ok=True)

    csv_path = os.path.join(save_dir, "m0_stride_diagnostic.csv")
    save_stride_csv(results, csv_path)

    plot_stride_global(results, save_dir)

    print(f"\n✅ M0 stride 诊断完成，结果保存到: {save_dir}")


if __name__ == "__main__":
    main()