#!/usr/bin/env python3
"""
检查数据集结构和形状
"""

import zarr
import numpy as np

def check_dataset():
    try:
        # 打开数据集
        root = zarr.open('./data/six-tasks.zarr', mode='r')
        print("=== 数据集结构 ===")
        print("根目录键:", list(root.keys()))
        
        if 'data' in root:
            print("\n=== 数据子目录 ===")
            data_keys = list(root['data'].keys())
            print("数据键:", data_keys)
            
            print("\n=== 数据形状 ===")
            for key in data_keys:
                if key in root['data']:
                    shape = root['data'][key].shape
                    dtype = root['data'][key].dtype
                    print(f"{key}: shape={shape}, dtype={dtype}")
                    
                    # 如果是图像数据，显示更多信息
                    if len(shape) >= 3 and key.endswith('_cam'):
                        print(f"  - 图像数据: {shape[1]}x{shape[2]} 通道")
                        
        else:
            print("❌ 未找到 'data' 目录")
            
    except Exception as e:
        print(f"❌ 检查数据集时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_dataset()
