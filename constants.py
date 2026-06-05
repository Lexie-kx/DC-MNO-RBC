import os
import torch

# ==========================================
# 1. 全局环境与种子配置
# ==========================================
# 全局随机种子，保证 Baseline 和 M5 实验的严格可复现性
RANDOM_SEED = 42

# ==========================================
# 2. 全局路径与设备配置
# ==========================================
# 绝对路径：彻底解决相对路径执行崩溃的问题
DATA_PATH = "./data/raw/rbc_uniform_dataset_90.h5"

# 自动检测 GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 统一浮点精度，物理仿真建议至少使用 float32
DTYPE = torch.float32 

# ==========================================
# 3. 物理场通道字典与索引 (Physics Dictionary)
# ==========================================
# 使用 tuple 防止运行时被意外篡改
FIELD_ORDER = ('buoyancy', 'u_x', 'u_y', 'pressure')

B_IDX = 0
UX_IDX = 1
UY_IDX = 2
P_IDX = 3

NUM_FIELDS = len(FIELD_ORDER)

# 物理场双向映射字典，方便解包调试
FIELD_TO_INDEX = {
    'buoyancy': B_IDX,
    'u_x': UX_IDX,
    'u_y': UY_IDX,
    'pressure': P_IDX
}

INDEX_TO_FIELD = {
    B_IDX: 'buoyancy',
    UX_IDX: 'u_x',
    UY_IDX: 'u_y',
    P_IDX: 'pressure'
}

# ==========================================
# 4. 空间与时间维度规格 (Spatiotemporal Specs)
# ==========================================
SPATIAL_RESOLUTION = (256, 64)  # (X, Y)

# 时序演化窗口核心参数
CONTEXT_LENGTH = 4         # 模型观察的历史帧数 (输入通道数将是 4 * NUM_FIELDS)
PREDICT_STEPS = 1          # 模型一次预测未来的帧数
DEFAULT_ROLLOUT_STEPS = 16 # 默认评估多步滚动误差的步长

# ==========================================
# 5. 物理参数域 (Physical Parameter Space)
# ==========================================
# 以后在 unseen Ra/Pr 测试或者 DataLoader 划分时直接调用
RA_VALUES = [1e6, 1e7, 1e8]
PR_VALUES = [0.5, 1.0, 2.0]

# ==========================================
# 6. 数值稳定常量
# ==========================================
# 全局数值稳定极小值
EPS = 1e-8