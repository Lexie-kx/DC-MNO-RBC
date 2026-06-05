import matplotlib.pyplot as plt
import numpy as np
import os

def plot_m0_comparison(gt, pred, field_name, save_path):
    """
    绘制 GT (t+1), Prediction (M0) 和 Absolute Error 的对比云图
    """
    # 转换为 numpy 数组
    gt_np = gt.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()
    error_np = np.abs(gt_np - pred_np)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    vmin, vmax = gt_np.min(), gt_np.max()

    # 1. Ground Truth
    im0 = axes[0].imshow(gt_np, cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')
    axes[0].set_title(f"Ground Truth (Next Frame)")
    plt.colorbar(im0, ax=axes[0])

    # 2. Persistence Prediction
    im1 = axes[1].imshow(pred_np, cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')
    axes[1].set_title(f"M0 Prediction (Current Frame)")
    plt.colorbar(im1, ax=axes[1])

    # 3. Absolute Error
    im2 = axes[2].imshow(error_np, cmap='viridis', origin='lower')
    axes[2].set_title(f"Absolute Error (Evolution)")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    
    # 强制获取绝对路径并创建目录
    abs_save_path = os.path.abspath(save_path)
    save_dir = os.path.dirname(abs_save_path)
    os.makedirs(save_dir, exist_ok=True)
    
    plt.savefig(abs_save_path, dpi=200)
    plt.close()
    print(f"可视化结果已保存至绝对路径: {abs_save_path}")