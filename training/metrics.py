import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 训练专用 Loss 类
# ==========================================
class FieldWiseRelativeL2Loss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        # pred/target: [B, C, H, W]
        # 展平空间维度: [B, C, H*W]
        pred_flat = pred.reshape(pred.shape[0], pred.shape[1], -1)
        target_flat = target.reshape(target.shape[0], target.shape[1], -1)

        # 在空间维度上计算 L2 范数
        diff = torch.norm(pred_flat - target_flat, p=2, dim=-1)
        denom = torch.norm(target_flat, p=2, dim=-1)

        # 计算相对误差
        rel = diff / (denom + self.eps)

        # 对 batch 和 field 都求平均
        return rel.mean()

# ==========================================
# 评估诊断函数库
# ==========================================
def compute_field_wise_rel_l2(pred, target, field_names=['buoyancy', 'u_x', 'u_y', 'pressure']):
    """计算每个物理场独立的 Relative L2 Error (%)"""
    assert pred.shape == target.shape, "Pred and Target shapes must match!"
    B, C, H, W = pred.shape
    
    field_metrics = {}
    for i, field in enumerate(field_names):
        p = pred[:, i, :, :].reshape(B, -1)
        t = target[:, i, :, :].reshape(B, -1)
        
        diff_norm = torch.norm(p - t, p=2, dim=1)
        target_norm = torch.norm(t, p=2, dim=1)
        
        rel_l2 = diff_norm / (target_norm + 1e-6)
        field_metrics[field] = torch.mean(rel_l2).item()
        
    return field_metrics

def compute_total_rel_l2(pred, target):
    """计算全局的 Relative L2 Error"""
    p = pred.reshape(pred.shape[0], -1)
    t = target.reshape(target.shape[0], -1)
    
    diff_norm = torch.norm(p - t, p=2, dim=1)
    target_norm = torch.norm(t, p=2, dim=1)
    
    return torch.mean(diff_norm / (target_norm + 1e-6)).item()

def compute_field_wise_mse(pred, target, field_names=['buoyancy', 'u_x', 'u_y', 'pressure']):
    """计算每个物理场独立的绝对 MSE Error (用于排查分母问题)"""
    assert pred.shape == target.shape, "Pred and Target shapes must match!"
    
    field_metrics = {}
    for i, field in enumerate(field_names):
        mse = F.mse_loss(pred[:, i, :, :], target[:, i, :, :])
        field_metrics[field] = mse.item()
        
    return field_metrics

def compute_total_mse(pred, target):
    """计算全局的 MSE Error"""
    return F.mse_loss(pred, target).item()