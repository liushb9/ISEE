#!/usr/bin/env python3
"""
多卡训练内存使用估算脚本
基于你的服务器配置和训练参数进行估算
"""

import os
import sys
import json
import subprocess
from pathlib import Path

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

def estimate_dataset_size(zarr_path):
    """估算数据集大小"""
    if not os.path.exists(zarr_path):
        print(f"❌ 数据集不存在: {zarr_path}")
        return None
    
    try:
        # 使用zarr获取数据集信息
        import zarr
        store = zarr.open(zarr_path, mode='r')
        
        total_size = 0
        dataset_info = {}
        
        for key, arr in store.items():
            if hasattr(arr, 'shape') and hasattr(arr, 'dtype'):
                # 计算数组大小
                array_size = arr.nbytes / (1024**3)  # GB
                total_size += array_size
                dataset_info[key] = {
                    'shape': arr.shape,
                    'dtype': str(arr.dtype),
                    'size_gb': array_size
                }
        
        return {
            'total_size_gb': total_size,
            'details': dataset_info
        }
    except Exception as e:
        print(f"⚠️  无法读取数据集信息: {e}")
        return None

def estimate_training_memory(dataset_size_gb, num_gpus, batch_size, sequence_length, action_dim=10):
    """估算训练内存需求"""
    
    # 基础内存需求（GB）
    base_memory = {
        'model_weights': 2.0,        # 模型权重
        'optimizer_states': 4.0,     # 优化器状态
        'gradients': 2.0,            # 梯度
        'activations': 1.0,          # 激活值
        'system_overhead': 0.5       # 系统开销
    }
    
    # 数据相关内存
    data_memory = {
        'batch_data': (batch_size * sequence_length * action_dim * 4) / (1024**3),  # float32
        'cached_data': dataset_size_gb * 0.1,  # 假设缓存10%的数据
    }
    
    # 多卡训练额外开销
    ddp_overhead = {
        'gradient_sync': 0.5,        # 梯度同步
        'communication': 0.3,        # 进程间通信
        'redundant_storage': 0.2     # 重复存储
    }
    
    # 计算总内存
    total_base = sum(base_memory.values())
    total_data = sum(data_memory.values())
    total_ddp = sum(ddp_overhead.values()) * num_gpus
    
    total_memory = total_base + total_data + total_ddp
    
    return {
        'base_memory': base_memory,
        'data_memory': data_memory,
        'ddp_overhead': ddp_overhead,
        'total_memory': total_memory,
        'per_gpu_memory': total_memory / num_gpus
    }

def check_available_resources(gpu_ids, required_memory_per_gpu):
    """检查可用资源"""
    gpu_info = get_gpu_info()
    if not gpu_info:
        return None
    
    available_gpus = []
    total_free_memory = 0
    
    for gpu_id in gpu_ids:
        if gpu_id < len(gpu_info):
            gpu = gpu_info[gpu_id]
            free_gb = gpu['free'] / 1024  # 转换为GB
            
            if free_gb >= required_memory_per_gpu:
                available_gpus.append({
                    'id': gpu_id,
                    'free_gb': free_gb,
                    'status': '✅ 可用'
                })
                total_free_memory += free_gb
            else:
                available_gpus.append({
                    'id': gpu_id,
                    'free_gb': free_gb,
                    'status': f'❌ 显存不足 (需要{required_memory_per_gpu:.1f}GB, 可用{free_gb:.1f}GB)'
                })
    
    return {
        'gpus': available_gpus,
        'total_free_memory': total_free_memory,
        'can_run': all(gpu['status'].startswith('✅') for gpu in available_gpus)
    }

def main():
    """主函数"""
    print("=== 多卡训练内存需求估算 ===")
    
    # 训练参数
    task_name = "six_tasks"
    config_name = "demo_clean"
    num_episodes = 1200
    seed = 0
    num_gpus = 3
    gpu_ids = [0, 1, 2]
    batch_size = 128  # 默认batch size
    sequence_length = 1  # 默认序列长度
    
    print(f"📋 训练配置:")
    print(f"   任务名称: {task_name}")
    print(f"   配置名称: {config_name}")
    print(f"   Episode数量: {num_episodes}")
    print(f"   GPU数量: {num_gpus}")
    print(f"   GPU ID: {gpu_ids}")
    print(f"   Batch Size: {batch_size}")
    print(f"   序列长度: {sequence_length}")
    
    # 检查数据集
    print(f"\n📊 检查数据集...")
    dataset_paths = [
        "data/six-tasks.zarr",
        "data/six_tasks-demo_clean-1200.zarr"
    ]
    
    dataset_info = None
    for path in dataset_paths:
        if os.path.exists(path):
            print(f"✅ 找到数据集: {path}")
            dataset_info = estimate_dataset_size(path)
            break
    
    if not dataset_info:
        print("❌ 未找到数据集，使用默认估算")
        dataset_info = {'total_size_gb': 50.0, 'details': {}}  # 默认50GB
    
    print(f"   数据集大小: {dataset_info['total_size_gb']:.2f} GB")
    
    # 估算内存需求
    print(f"\n🧮 估算内存需求...")
    memory_estimate = estimate_training_memory(
        dataset_size_gb=dataset_info['total_size_gb'],
        num_gpus=num_gpus,
        batch_size=batch_size,
        sequence_length=sequence_length
    )
    
    print(f"📊 内存需求详情:")
    print(f"   基础内存:")
    for key, value in memory_estimate['base_memory'].items():
        print(f"     {key}: {value:.1f} GB")
    
    print(f"   数据内存:")
    for key, value in memory_estimate['data_memory'].items():
        print(f"     {key}: {value:.1f} GB")
    
    print(f"   多卡开销:")
    for key, value in memory_estimate['ddp_overhead'].items():
        print(f"     {key}: {value:.1f} GB × {num_gpus} = {value * num_gpus:.1f} GB")
    
    print(f"\n💾 总内存需求:")
    print(f"   总内存: {memory_estimate['total_memory']:.1f} GB")
    print(f"   每GPU内存: {memory_estimate['per_gpu_memory']:.1f} GB")
    
    # 检查可用资源
    print(f"\n🔍 检查可用资源...")
    resource_check = check_available_resources(gpu_ids, memory_estimate['per_gpu_memory'])
    
    if resource_check:
        print(f"GPU状态:")
        for gpu in resource_check['gpus']:
            print(f"   GPU {gpu['id']}: {gpu['status']} (可用显存: {gpu['free_gb']:.1f} GB)")
        
        print(f"\n总可用显存: {resource_check['total_free_memory']:.1f} GB")
        
        if resource_check['can_run']:
            print(f"\n✅ 资源充足，可以开始训练!")
            print(f"   建议:")
            print(f"   - 监控GPU显存使用: watch -n 1 nvidia-smi")
            print(f"   - 如果显存不足，可以减小batch_size")
            print(f"   - 使用gradient checkpointing节省显存")
        else:
            print(f"\n❌ 资源不足，无法开始训练!")
            print(f"   建议:")
            print(f"   - 减少GPU数量")
            print(f"   - 减小batch_size")
            print(f"   - 等待其他进程释放GPU")
    else:
        print(f"⚠️  无法检查GPU状态")
    
    # 保存估算结果
    result = {
        'training_config': {
            'task_name': task_name,
            'num_gpus': num_gpus,
            'gpu_ids': gpu_ids,
            'batch_size': batch_size
        },
        'memory_estimate': memory_estimate,
        'dataset_info': dataset_info,
        'resource_check': resource_check
    }
    
    with open('memory_estimate_result.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"\n💾 估算结果已保存到: memory_estimate_result.json")

if __name__ == "__main__":
    main()
