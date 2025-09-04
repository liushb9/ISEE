#!/usr/bin/env python3
"""
测试优化后的归一化函数
验证内存使用和归一化效果
"""

import numpy as np
import sys
import os

# 添加路径
sys.path.append('/home/shengbang/RoboTwin/policy/DP')

from diffusion_policy.dataset.robot_image_dataset import RobotImageDataset

def test_normalizer_memory_usage():
    """测试归一化函数的内存使用情况"""
    print("=== 测试归一化函数内存使用 ===")
    
    # 模拟数据集路径（需要替换为实际路径）
    zarr_path = "data/six-tasks.zarr"  # 根据实际情况调整
    
    if not os.path.exists(zarr_path):
        print(f"❌ 数据集不存在: {zarr_path}")
        print("请先运行数据处理脚本创建数据集")
        return
    
    try:
        # 创建数据集实例
        print("📊 创建数据集实例...")
        dataset = RobotImageDataset(
            zarr_path=zarr_path,
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            batch_size=128,
            max_train_episodes=None,
        )
        
        print(f"✅ 数据集创建成功")
        print(f"   - 总episodes: {dataset.replay_buffer.n_episodes}")
        print(f"   - Action维度: {dataset.replay_buffer['action'].shape[1]}")
        print(f"   - State维度: {dataset.replay_buffer['state'].shape[1]}")
        
        # 测试归一化函数
        print("\n🔧 测试归一化函数...")
        import psutil
        import gc
        
        # 记录内存使用
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        print(f"   内存使用前: {mem_before:.2f} MB")
        
        # 运行归一化
        normalizer = dataset.get_normalizer()
        
        # 清理内存
        gc.collect()
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        print(f"   内存使用后: {mem_after:.2f} MB")
        print(f"   内存增长: {mem_after - mem_before:.2f} MB")
        
        # 检查归一化器
        print(f"\n📋 归一化器信息:")
        print(f"   - Action归一化器: {type(normalizer['action'])}")
        
        if hasattr(normalizer['action'], 'params_dict'):
            params = normalizer['action'].params_dict
            print(f"   - Scale shape: {params['scale'].shape}")
            print(f"   - Offset shape: {params['offset'].shape}")
            
            # 检查前3维（位置）是否被归一化
            pos_scale = params['scale'][:3]
            pos_offset = params['offset'][:3]
            print(f"   - 位置归一化参数:")
            print(f"     Scale: {pos_scale}")
            print(f"     Offset: {pos_offset}")
            
            # 检查中间6维（旋转）是否保持不变
            rot_scale = params['scale'][3:9]
            rot_offset = params['offset'][3:9]
            print(f"   - 旋转归一化参数:")
            print(f"     Scale: {rot_scale}")
            print(f"     Offset: {rot_offset}")
            
            # 检查最后1维（夹爪）是否保持不变
            gripper_scale = params['scale'][9:]
            gripper_offset = params['offset'][9:]
            print(f"   - 夹爪归一化参数:")
            print(f"     Scale: {gripper_scale}")
            print(f"     Offset: {gripper_offset}")
        
        print("\n✅ 归一化测试完成!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_normalizer_memory_usage()
