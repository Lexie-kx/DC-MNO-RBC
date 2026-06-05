import sys
import os
import torch
from torch.utils.data import DataLoader

# 1. 路径配置：project_root 指向 DC-MNO_PROJECT 根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

from DC_MNO.datasets.rbc_dataset import RBCDataset
from DC_MNO.training.metrics import evaluate_physics_fields
from DC_MNO.constants import DEVICE, B_IDX, UY_IDX
from DC_MNO.visualization.m0_visualizer import plot_m0_comparison

def run_m0_evaluation():
    print("正在运行 M0 基线评估（内部路径模式）...")
    
    dataset = RBCDataset(split="train", window_size=4)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    total_metrics = None
    num_batches = 0
    last_batch_x, last_batch_y = None, None
    
    for batch in dataloader:
        x, y = batch["x"].to(DEVICE), batch["y"].to(DEVICE)
        pred = x[:, -1]
        batch_metrics = evaluate_physics_fields(pred, y)
        
        if total_metrics is None:
            total_metrics = {k: 0.0 for k in batch_metrics.keys()}
        for k, v in batch_metrics.items():
            total_metrics[k] += v
        num_batches += 1
        last_batch_x, last_batch_y = x, y
        
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
    print("\n--- M0 Baseline 结果 ---")
    for k, v in avg_metrics.items():
        print(f"{k}: {v:.6f}")

    # 2. 【核心修改】：精确定位到工程包内部的 outputs 文件夹
    # 路径：DC_MNO/scripts/ -> .. (DC_MNO/) -> outputs/
    internal_output_dir = os.path.abspath(os.path.join(current_dir, "..", "outputs"))
    fig_dir = os.path.join(internal_output_dir, "figures", "m0")
    
    # 确保内部目录存在
    os.makedirs(fig_dir, exist_ok=True)

    sample_gt = last_batch_y[0]
    sample_pred = last_batch_x[0, -1]

    # 执行绘图
    plot_m0_comparison(
        sample_gt[B_IDX], 
        sample_pred[B_IDX], 
        "Buoyancy", 
        os.path.join(fig_dir, "m0_buoyancy.png")
    )

    plot_m0_comparison(
        sample_gt[UY_IDX], 
        sample_pred[UY_IDX], 
        "u_y", 
        os.path.join(fig_dir, "m0_uy.png")
    )
    
    print(f"\n可视化结果已保存至工程包内部：\n{fig_dir}")

if __name__ == "__main__":
    run_m0_evaluation()