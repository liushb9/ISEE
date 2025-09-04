#!/usr/bin/env python3
"""
400条数据训练内存需求估算
"""

import os
import sys
import subprocess

def get_gpu_info():
    """获取GPU信息"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            if line:
                total, used, free = map(int, line.split(', '))
                gpu_info.append({
                    'total': total,
                    'used': used,
                    'free': free
                })
        return gpu_info
    except Exception as e:
        print(f"⚠️  无法获取GPU信息: {e}")
        return None

def estimate_400_data_memory():
    """估算400条数据训练的内存需求"""
    
    print("=== 400条数据训练内存需求估算 ===")
    
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
    
    # 内存需求估算
    print(f"\n🧮 内存需求估算:")
    
    # 基础内存需求（GB）
    base_memory = {
        'model_weights': 2.0,        # ResNet18 + UNet
        'optimizer_states': 4.0,     # AdamW优化器状态
        'gradients': 2.0,            # 梯度
        'activations': 1.0,          # 激活值
        'system_overhead': 0.5       # 系统开销
    }
    
    # 数据相关内存
    batch_obs_size = batch_size * n_obs_steps * obs_size_per_step / (1024**3)  # GB
    batch_action_size = batch_size * n_action_steps * action_size_per_step / (1024**3)  # GB
    
    data_memory = {
        'batch_obs_data': batch_obs_size,
        'batch_action_data': batch_action_size,
        'cached_data': total_data_size_gb * 0.1,  # 假设缓存10%的数据
    }
    
    # 计算总内存
    total_base = sum(base_memory.values())
    total_data = sum(data_memory.values())
    total_memory = total_base + total_data
    
    print(f"📊 内存需求详情:")
    print(f"   基础内存:")
    for key, value in base_memory.items():
        print(f"     {key}: {value:.1f} GB")
    
    print(f"   数据内存:")
    for key, value in data_memory.items():
        print(f"     {key}: {value:.1f} GB")
    
    print(f"\n💾 总内存需求:")
    print(f"   总内存: {total_memory:.1f} GB")
    print(f"   推荐显存: {total_memory * 1.2:.1f} GB (包含20%缓冲)")
    
    # 检查GPU状态
    print(f"\n🔍 检查GPU状态...")
    gpu_info = get_gpu_info()
    
    if gpu_info:
        print(f"GPU状态:")
        for i, gpu in enumerate(gpu_info):
            free_gb = gpu['free'] / 1024
            total_gb = gpu['total'] / 1024
            used_gb = gpu['used'] / 1024
            
            status = "✅ 可用" if free_gb >= total_memory else f"❌ 显存不足 (需要{total_memory:.1f}GB)"
            
            print(f"   GPU {i}: {status}")
            print(f"     总显存: {total_gb:.1f} GB")
            print(f"     已用显存: {used_gb:.1f} GB")
            print(f"     可用显存: {free_gb:.1f} GB")
            
            if free_gb >= total_memory:
                print(f"     ✅ 可以运行400条数据训练")
            else:
                print(f"     ❌ 显存不足，建议:")
                print(f"        - 减小batch_size到64或32")
                print(f"        - 使用gradient checkpointing")
                print(f"        - 等待其他进程释放GPU")
    else:
        print(f"⚠️  无法检查GPU状态")
    
    # 优化建议
    print(f"\n💡 优化建议:")
    print(f"   1. 如果显存不足，可以减小batch_size:")
    print(f"      - batch_size=64: 内存需求约 {total_memory * 0.7:.1f} GB")
    print(f"      - batch_size=32: 内存需求约 {total_memory * 0.5:.1f} GB")
    print(f"   2. 使用gradient checkpointing可以节省约30%显存")
    print(f"   3. 400条数据相对较少，训练时间应该不会太长")
    print(f"   4. 建议监控训练过程中的显存使用情况")

if __name__ == "__main__":
    estimate_400_data_memory()
