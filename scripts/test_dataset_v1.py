import os
import sys
import json
import torch
from torch.utils.data import DataLoader

# 确保能找到项目根目录的包
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.rbc_dataset import RBCDataset

def main():
    print(" 正在初始化 DC-MNO 数据管道测试...")

    # 1. 读取最新的 IID Split 协议
    split_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'splits', 'iid_split.json'))
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"找不到划分文件: {split_file}")
        
    with open(split_file, 'r', encoding='utf-8') as f:
        split_config = json.load(f)

    # 2. 使用最新的 split_config 接口初始化 Dataset
    print("加载 Train Dataset...")
    dataset = RBCDataset(split_config=split_config["train"])
    
    # 3. 构造 DataLoader
    BATCH_SIZE = 16
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 4. 抽取第一个 Batch 用于 Shape 校验
    print("抽取 Batch 进行维度校验...\n")
    batch_x, batch_y = next(iter(dataloader))

    # 5. 打印用于截图的完美格式
    print("=" * 60)
    print("  数据输入格式校验 (DataLoader 输出)")
    print("=" * 60)
    print(f"{'Item':<10} | {'Shape':<25} | {'Meaning'}")
    print("-" * 60)
    print(f"{'X':<10} | {str(list(batch_x.shape)):<25} | 4 帧历史场展开为 16 通道")
    print(f"{'Y':<10} | {str(list(batch_y.shape)):<25} | 下一时刻真实物理场")
    print("=" * 60)
    print("\n 提示：你可以直接截图上面的表格，放入你的实验设计文档中，证明数据工程闭环已完成。")

if __name__ == "__main__":
    main()