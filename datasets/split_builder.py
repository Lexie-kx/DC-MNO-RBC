# Script to generate json splits
import json
import os
import h5py

# 引入常量，保证路径一致
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import DATA_PATH

def build_iid_split(output_dir="../data/splits"):
    """
    生成增强版的 IID 数据划分方案 (结合了自动探测和 Metadata)。
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 修正 3：加入 Metadata
    split_config = {
        "split_type": "iid",
        "description": "Trajectory-level IID split. Train(0-7), Val(8), Test(9) for all available groups.",
        "train": [],
        "val": [],
        "test": []
    }
    
    # 修正 1：自动探测真实的 Group Names
    print(f"正在从 {DATA_PATH} 探测真实的工况组...")
    try:
        with h5py.File(DATA_PATH, 'r') as f:
            valid_group_names = list(f.keys())
    except Exception as e:
        print(f"❌ 读取 HDF5 失败: {e}")
        return

    print(f"✅ 探测到 {len(valid_group_names)} 个工况组。开始分配轨迹...")

    for group_name in valid_group_names:
        # Train: 前 8 条轨迹 (0-7)
        split_config["train"].append({
            "group": group_name,
            "trajectories": list(range(0, 8))
        })
        
        # Val: 第 9 条轨迹 (8)
        split_config["val"].append({
            "group": group_name,
            "trajectories": [8]
        })
        
        # Test: 第 10 条轨迹 (9)
        split_config["test"].append({
            "group": group_name,
            "trajectories": [9]
        })
            
    # 修正 2：Sanity Print (打印前两组看看长什么样)
    print("\n--- Split Config 预览 (部分) ---")
    preview = {"train": split_config["train"][:2]} # 只打印两组看看格式
    print(json.dumps(preview, indent=4))
    print("--------------------------------\n")
    
    output_path = os.path.join(output_dir, "iid_split.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(split_config, f, indent=4)
        
    print(f"🎯 IID 划分配置已完美生成至: {output_path}")

if __name__ == "__main__":
    build_iid_split(output_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'splits')))