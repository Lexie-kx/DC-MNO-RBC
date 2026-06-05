import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
    """2D 傅里叶层：执行 FFT，在频域进行线性变换，然后进行逆 FFT。"""
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        
        # 1. 转换到频域
        x_ft = torch.fft.rfft2(x)

        # 2. 乘以相关的傅里叶模式 (低频部分)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
            
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # 3. 逆转换回物理空间
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class PlainFNO2d(nn.Module):
    """
    最朴素的 Plain FNO 2D 基线模型。
    输入: [B, 16, 256, 64] (4 history frames * 4 fields)
    输出: [B, 4, 256, 64] (next step: buoyancy, ux, uy, pressure)
    """
    def __init__(self, in_channels=16, out_channels=4, modes1=16, modes2=16, width=32):
        super(PlainFNO2d, self).__init__()
        
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        # 投影层：将输入的 16 个通道映射到 FNO 的隐层特征维度
        self.p = nn.Linear(in_channels, self.width) 
        
        # 4 层傅里叶卷积块
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        
        # 旁路 1x1 卷积 (Local linear transform)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        # 解码层：将隐层特征映射回 4 个物理场
        self.mlp0 = nn.Linear(self.width, 128)
        self.mlp1 = nn.Linear(128, out_channels)

    def forward(self, x):
        # (b, c, x, y) -> (b, x, y, c)
        x = x.permute(0, 2, 3, 1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2 # 最后一层不加激活

        # 解码回物理空间维度
        x = x.permute(0, 2, 3, 1)
        x = self.mlp0(x)
        x = F.gelu(x)
        x = self.mlp1(x)
        x = x.permute(0, 3, 1, 2)
        
        return x

if __name__ == '__main__':
    # ==========================================
    # Step 1 验收标准测试：验证张量形状流转
    # ==========================================
    B = 16
    C_in = 16  # 4 frames * 4 fields
    H = 256
    W = 64
    
    # 模拟 DataLoader 吐出的 Batch X
    dummy_input = torch.randn(B, C_in, H, W)
    print(f"📥 输入 Tensor 形状: {dummy_input.shape} (预期: [16, 16, 256, 64])")
    
    # 实例化模型
    model = PlainFNO2d(in_channels=16, out_channels=4, modes1=16, modes2=16, width=32)
    
    # 前向传播
    pred = model(dummy_input)
    print(f"📤 输出 Tensor 形状: {pred.shape} (预期: [16, 4, 256, 64])")
    
    assert pred.shape == (B, 4, H, W), "🚨 形状不匹配！"
    print("\n✅ Step 1 验收通过！模型前向传播形状完全正确，没有发生维度崩塌。")