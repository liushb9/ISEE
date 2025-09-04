#!/usr/bin/env python3
"""
400条数据训练系统内存(RAM)需求估算
"""

import os
import sys
import subprocess
import psutil

def get_system_memory():
    """获取系统内存信息"""
    try:
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percent': memory.percent
        }
    except Exception as e:
        print(f"⚠️  无法获取系统内存信息: {e}")
        return None

def estimate_400_data_ram():
    """估算400条数据训练的系统内存需求"""
    
    print("=== 400条数据训练系统内存(RAM)需求估算 ===")
    
    # 训练参数
    expert_data_num = 400
    batch_size = 128  # 从配置文件读取
    sequence_length = 8  # horizon from config
    n_obs_steps = 3
    n_action_steps = 6
    action_dim = 10
    obs_shape = [3, 256, 256]  # RGB图像
    
    print(f"📋 训练配置:")
    print(f"   数据条数: {expert_data_num}")
    print(f"   Batch Size: {batch_size}")
    print(f"   序列长度: {sequence_length}")
    print(f"   观测步数: {n_obs_steps}")
    print(f"   动作步数: {n_action_steps}")
    print(f"   动作维度: {action_dim}")
    print(f"   图像尺寸: {obs_shape}")
    
    # 计算数据大小
    print(f"\n📊 数据大小计算:")
    
    # 单条数据大小
    obs_size_per_step = obs_shape[0] * obs_shape[1] * obs_shape[2] * 4  # float32, bytes
    action_size_per_step = action_dim * 4  # float32, bytes
    
    single_episode_obs_size = obs_size_per_step * n_obs_steps  # bytes
    single_episode_action_size = action_size_per_step * n_action_steps  # bytes
    single_episode_size = single_episode_obs_size + single_episode_action_size  # bytes
    
    print(f"   单步观测大小: {obs_size_per_step / (1024**2):.2f} MB")
    print(f"   单步动作大小: {action_size_per_step / (1024**2):.2f} MB")
    print(f"   单条数据大小: {single_episode_size / (1024**2):.2f} MB")
    
    # 400条数据总大小
    total_data_size_gb = (single_episode_size * expert_data_num) / (1024**3)
    print(f"   400条数据总大小: {total_data_size_gb:.2f} GB")
    
    # 系统内存需求估算
    print(f"\n🧮 系统内存需求估算:")
    
    # 基础内存需求（GB）
    base_memory = {
        'python_process': 1.0,        # Python进程基础内存
        'pytorch_framework': 2.0,     # PyTorch框架内存
        'data_loading': 1.0,          # 数据加载器内存
        'file_io_buffer': 0.5,        # 文件I/O缓冲区
        'system_overhead': 1.0        # 系统开销
    }
    
    # 数据相关内存
    data_memory = {
        'dataset_cache': total_data_size_gb * 0.3,  # 数据集缓存(30%)
        'batch_preprocessing': 0.5,   # Batch预处理内存
        'augmentation_buffer': 0.3,   # 数据增强缓冲区
    }
    
    # 训练相关内存
    training_memory = {
        'model_cpu_copy': 0.5,        # 模型CPU副本
        'optimizer_cpu_state': 1.0,   # 优化器CPU状态
        'gradient_cpu_buffer': 0.5,   # 梯度CPU缓冲区
        'checkpoint_buffer': 0.5,     # 检查点缓冲区
    }
    
    # 计算总内存
    total_base = sum(base_memory.values())
    total_data = sum(data_memory.values())
    total_training = sum(training_memory.values())
    total_memory = total_base + total_data + total_training
    
    print(f"📊 内存需求详情:")
    print(f"   基础内存:")
    for key, value in base_memory.items():
        print(f"     {key}: {value:.1f} GB")
    
    print(f"   数据内存:")
    for key, value in data_memory.items():
        print(f"     {key}: {value:.1f} GB")
    
    print(f"   训练内存:")
    for key, value in training_memory.items():
        print(f"     {key}: {value:.1f} GB")
    
    print(f"\n💾 总系统内存需求:")
    print(f"   总内存: {total_memory:.1f} GB")
    print(f"   推荐内存: {total_memory * 1.3:.1f} GB (包含30%缓冲)")
    
    # 检查系统内存状态
    print(f"\n🔍 检查系统内存状态...")
    memory_info = get_system_memory()
    
    if memory_info:
        print(f"系统内存状态:")
        print(f"   总内存: {memory_info['total_gb']:.1f} GB")
        print(f"   已用内存: {memory_info['used_gb']:.1f} GB ({memory_info['percent']:.1f}%)")
        print(f"   可用内存: {memory_info['available_gb']:.1f} GB")
        
        if memory_info['available_gb'] >= total_memory:
            print(f"   ✅ 内存充足，可以运行400条数据训练")
        else:
            print(f"   ❌ 内存不足，建议:")
            print(f"      - 关闭其他程序释放内存")
            print(f"      - 减小batch_size")
            print(f"      - 减少数据缓存比例")
    else:
        print(f"⚠️  无法检查系统内存状态")
    
    # 优化建议
    print(f"\n💡 优化建议:")
    print(f"   1. 如果内存不足，可以:")
    print(f"      - 减小batch_size到64: 内存需求约 {total_memory * 0.8:.1f} GB")
    print(f"      - 减小batch_size到32: 内存需求约 {total_memory * 0.6:.1f} GB")
    print(f"      - 减少数据缓存比例到10%: 内存需求约 {total_memory * 0.9:.1f} GB")
    print(f"   2. 监控内存使用: watch -n 1 'free -h'")
    print(f"   3. 400条数据相对较少，训练时间不会太长")
    print(f"   4. 建议在训练前关闭不必要的程序")

if __name__ == "__main__":
    estimate_400_data_ram()
