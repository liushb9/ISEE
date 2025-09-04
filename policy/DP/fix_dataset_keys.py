#!/usr/bin/env python3
"""
Fix dataset key mapping issues by ensuring consistent naming between dataset and config
"""

import os
import shutil
import zarr
import numpy as np
from diffusion_policy.common.replay_buffer import ReplayBuffer

def fix_dataset_keys():
    """
    Fix the key mapping issues in the dataset by ensuring consistent naming
    """
    
    # 源数据集路径
    source_path = "data/six_tasks-demo_clean-300.zarr"
    # 修复后的数据集路径
    fixed_path = "data/six_tasks-demo_clean-300-fixed.zarr"
    
    if not os.path.exists(source_path):
        print(f"Error: Source dataset {source_path} not found!")
        return False
    
    # 如果目标已存在，先删除
    if os.path.exists(fixed_path):
        print(f"Removing existing fixed dataset: {fixed_path}")
        shutil.rmtree(fixed_path)
    
    print(f"Creating fixed dataset from {source_path}...")
    
    # 复制源数据集
    shutil.copytree(source_path, fixed_path)
    
    # 打开数据集进行修复
    root = zarr.open(fixed_path, mode='a')
    
    # 检查数据集结构
    print("Original dataset structure:")
    if 'data' in root:
        for key in root['data'].keys():
            if key in root['data']:
                data = root['data'][key]
                if hasattr(data, 'shape'):
                    print(f"  {key}: shape={data.shape}, dtype={data.dtype}")
                else:
                    print(f"  {key}: no shape info")
    
    # 检查是否需要重命名键
    if 'head_camera' in root['data'] and 'head_cam' not in root['data']:
        print("\nRenaming 'head_camera' to 'head_cam'...")
        
        # 获取head_camera数据
        head_camera_data = root['data']['head_camera'][:]
        
        # 删除原有的head_camera
        del root['data']['head_camera']
        
        # 计算合适的数据块大小，确保不超过2GB
        total_size = head_camera_data.nbytes
        max_chunk_size = 2 * 1024 * 1024 * 1024  # 2GB
        
        # 计算合适的chunk大小
        if total_size > max_chunk_size:
            # 如果总大小超过2GB，使用更小的chunk
            chunk_size = max(1, max_chunk_size // (head_camera_data.shape[1] * head_camera_data.shape[2] * head_camera_data.shape[3]))
            chunks = (chunk_size, head_camera_data.shape[1], head_camera_data.shape[2], head_camera_data.shape[3])
            print(f"  Using chunk size: {chunks} to handle large dataset")
        else:
            chunks = head_camera_data.shape
        
        # 创建新的head_cam数据集，使用适当的压缩设置
        compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
        
        root['data'].create_dataset(
            'head_cam',
            data=head_camera_data,
            chunks=chunks,
            overwrite=True,
            compressor=compressor
        )
        
        print("Successfully renamed 'head_camera' to 'head_cam'")
    
    # 验证修复后的数据集
    print("\nVerifying fixed dataset...")
    try:
        test_buffer = ReplayBuffer.copy_from_path(fixed_path)
        print(f"Fixed dataset contains {test_buffer.n_episodes} episodes")
        print(f"Available keys: {list(test_buffer.keys())}")
        for key in test_buffer.keys():
            if key in test_buffer:
                print(f"  {key}: shape={test_buffer[key].shape}, dtype={test_buffer[key].dtype}")
    except Exception as e:
        print(f"Error verifying fixed dataset: {e}")
        return False
    
    print(f"\nDataset fixed successfully! Fixed dataset saved to: {fixed_path}")
    return True

def check_image_shapes():
    """
    Check the actual image shapes in the dataset
    """
    print("\nChecking image shapes in datasets...")
    
    datasets = [
        "data/six_tasks-demo_clean-300.zarr",
        "data/six_tasks.zarr"
    ]
    
    for dataset_path in datasets:
        if os.path.exists(dataset_path):
            print(f"\nDataset: {dataset_path}")
            try:
                root = zarr.open(dataset_path, 'r')
                if 'data' in root:
                    for key in root['data'].keys():
                        if key in root['data']:
                            data = root['data'][key]
                            if hasattr(data, 'shape'):
                                print(f"  {key}: shape={data.shape}, dtype={data.dtype}")
                            else:
                                print(f"  {key}: no shape info")
            except Exception as e:
                print(f"  Error reading dataset: {e}")
        else:
            print(f"Dataset not found: {dataset_path}")

if __name__ == "__main__":
    print("Dataset Key Mapping Fix Tool")
    print("=" * 40)
    
    # 检查图像形状
    check_image_shapes()
    
    # 修复数据集
    print("\n" + "=" * 40)
    success = fix_dataset_keys()
    
    if success:
        print("\nDataset fixing completed successfully!")
        print("You can now use the fixed dataset for training.")
    else:
        print("\nDataset fixing failed!")
        exit(1)
