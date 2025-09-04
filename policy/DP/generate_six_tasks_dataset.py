#!/usr/bin/env python3
"""
Generate six_tasks-demo_clean-300.zarr dataset by merging individual task datasets
"""

import os
import shutil
from diffusion_policy.common.replay_buffer import ReplayBuffer
import zarr
import numpy as np

def generate_six_tasks_dataset():
    # 需要合并的zarr数据集路径
    zarr_paths = [
        "data/stack_blocks_three-demo_clean-50.zarr",
        "data/stack_bowls_three-demo_clean-50.zarr",
        "data/blocks_ranking_size-demo_clean-50.zarr",
        "data/blocks_ranking_rgb-demo_clean-50.zarr",
        "data/hanging_mug-demo_clean-50.zarr",
        "data/place_cans_plasticbox-demo_clean-50.zarr",
    ]
    
    # 目标路径
    target_path = "data/six_tasks-demo_clean-300.zarr"
    
    # 检查源数据集是否存在
    for zarr_path in zarr_paths:
        if not os.path.exists(zarr_path):
            print(f"Error: Source dataset {zarr_path} not found!")
            return False
    
    # 如果目标已存在，先删除
    if os.path.exists(target_path):
        print(f"Removing existing dataset: {target_path}")
        shutil.rmtree(target_path)
    
    print("Creating merged dataset...")
    
    # 创建空buffer
    merged_buffer = ReplayBuffer.create_empty_numpy()
    
    # 合并所有数据集
    for zarr_path in zarr_paths:
        print(f"Loading {zarr_path}")
        try:
            buffer = ReplayBuffer.copy_from_path(zarr_path)
            for i in range(buffer.n_episodes):
                episode = buffer.get_episode(i, copy=True)
                # 扩展 text_feat
                ep_len = episode['action'].shape[0]
                text_feat = buffer['text_feat'][i]  # shape: (feat_dim,)
                episode['text_feat'] = np.tile(text_feat, (ep_len, 1))  # shape: (ep_len, feat_dim)
                merged_buffer.add_episode(episode)
            print(f"Added {buffer.n_episodes} episodes from {zarr_path}")
        except Exception as e:
            print(f"Error loading {zarr_path}: {e}")
            return False
    
    # 保存到zarr
    print(f"Saving merged dataset to {target_path}")
    merged_buffer.save_to_path(target_path)
    print(f"Successfully created {target_path}")
    
    # 验证数据集
    print("Verifying dataset...")
    try:
        test_buffer = ReplayBuffer.copy_from_path(target_path)
        print(f"Dataset contains {test_buffer.n_episodes} episodes")
        print(f"Available keys: {list(test_buffer.keys())}")
        for key in test_buffer.keys():
            if key in test_buffer:
                print(f"  {key}: shape={test_buffer[key].shape}, dtype={test_buffer[key].dtype}")
    except Exception as e:
        print(f"Error verifying dataset: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = generate_six_tasks_dataset()
    if success:
        print("Dataset generation completed successfully!")
    else:
        print("Dataset generation failed!")
        exit(1)