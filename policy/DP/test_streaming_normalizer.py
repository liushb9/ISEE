#!/usr/bin/env python3
"""
测试流式归一化器
验证修改后的normalizer.fit()是否真正避免了内存爆炸
"""

import numpy as np
import sys
import os
import psutil
import gc
import time

# 添加路径
sys.path.append('/home/shengbang/RoboTwin/policy/DP')

def create_large_mock_dataset():
    """创建大型模拟数据集"""
    print("📊 创建大型模拟数据集...")
    
    # 模拟100万个时间步，10维action
    n_timesteps = 1_000_000
    action_dim = 10
    
    # 创建随机数据
    action_data = np.random.randn(n_timesteps, action_dim).astype(np.float32)
    
    print(f"✅ 数据集创建完成")
    print(f"   - 时间步数: {n_timesteps:,}")
    print(f"   - Action维度: {action_dim}")
    print(f"   - 数据大小: {action_data.nbytes / 1024 / 1024:.2f} MB")
    
    return action_data

def test_traditional_normalizer(data):
    """测试传统归一化方法"""
    print("\n🔄 测试传统归一化方法...")
    
    from diffusion_policy.model.common.normalizer import LinearNormalizer
    
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"   内存使用前: {mem_before:.2f} MB")
    
    # 创建归一化器
    normalizer = LinearNormalizer()
    
    # 测试传统方法（use_streaming=False）
    start_time = time.time()
    try:
        normalizer.fit(
            data={"action": data},
            last_n_dims=1,
            mode="limits",
            use_streaming=False  # 强制使用传统方法
        )
        end_time = time.time()
        
        mem_after = process.memory_info().rss / 1024 / 1024
        print(f"   内存使用后: {mem_after:.2f} MB (+{mem_after - mem_before:.2f} MB)")
        print(f"   计算时间: {end_time - start_time:.4f} 秒")
        
        # 检查归一化器参数
        if "action" in normalizer.params_dict:
            params = normalizer.params_dict["action"]
            print(f"   ✅ 归一化器创建成功")
            print(f"      Scale shape: {params['scale'].shape}")
            print(f"      Offset shape: {params['offset'].shape}")
        else:
            print(f"   ❌ 归一化器创建失败")
            
    except Exception as e:
        print(f"   ❌ 传统方法失败: {e}")
        return None
    
    return normalizer

def test_streaming_normalizer(data):
    """测试流式归一化方法"""
    print("\n🚀 测试流式归一化方法...")
    
    from diffusion_policy.model.common.normalizer import LinearNormalizer
    
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"   内存使用前: {mem_before:.2f} MB")
    
    # 创建归一化器
    normalizer = LinearNormalizer()
    
    # 测试流式方法（use_streaming=True，默认值）
    start_time = time.time()
    try:
        normalizer.fit(
            data={"action": data},
            last_n_dims=1,
            mode="limits",
            use_streaming=True  # 使用流式方法
        )
        end_time = time.time()
        
        mem_after = process.memory_info().rss / 1024 / 1024
        print(f"   内存使用后: {mem_after:.2f} MB (+{mem_after - mem_before:.2f} MB)")
        print(f"   计算时间: {end_time - start_time:.4f} 秒")
        
        # 检查归一化器参数
        if "action" in normalizer.params_dict:
            params = normalizer.params_dict["action"]
            print(f"   ✅ 归一化器创建成功")
            print(f"      Scale shape: {params['scale'].shape}")
            print(f"      Offset shape: {params['offset'].shape}")
        else:
            print(f"   ❌ 归一化器创建失败")
            
    except Exception as e:
        print(f"   ❌ 流式方法失败: {e}")
        return None
    
    return normalizer

def compare_normalizers(traditional_norm, streaming_norm):
    """比较两种归一化器的结果"""
    if traditional_norm is None or streaming_norm is None:
        print("❌ 无法比较，归一化器创建失败")
        return
    
    print("\n📊 归一化器结果对比:")
    print("=" * 50)
    
    try:
        # 比较参数
        trad_params = traditional_norm.params_dict["action"]
        stream_params = streaming_norm.params_dict["action"]
        
        # 比较scale
        scale_diff = torch.max(torch.abs(trad_params["scale"] - stream_params["scale"]))
        print(f"Scale差异: {scale_diff:.2e}")
        
        # 比较offset
        offset_diff = torch.max(torch.abs(trad_params["offset"] - stream_params["offset"]))
        print(f"Offset差异: {offset_diff:.2e}")
        
        # 比较统计信息
        trad_stats = trad_params["input_stats"]
        stream_stats = stream_params["input_stats"]
        
        min_diff = torch.max(torch.abs(trad_stats["min"] - stream_stats["min"]))
        max_diff = torch.max(torch.abs(trad_stats["max"] - stream_stats["max"]))
        mean_diff = torch.max(torch.abs(trad_stats["mean"] - stream_stats["mean"]))
        std_diff = torch.max(torch.abs(trad_stats["std"] - stream_stats["std"]))
        
        print(f"Min差异: {min_diff:.2e}")
        print(f"Max差异: {max_diff:.2e}")
        print(f"Mean差异: {mean_diff:.2e}")
        print(f"Std差异: {std_diff:.2e}")
        
        if max_diff < 1e-10:
            print("✅ 结果完全一致！")
        else:
            print("⚠️  结果存在微小差异")
            
    except Exception as e:
        print(f"❌ 比较失败: {e}")

def main():
    """主函数"""
    print("=== 流式归一化器测试 ===")
    
    # 创建大型数据集
    action_data = create_large_mock_dataset()
    
    # 测试传统方法
    traditional_norm = test_traditional_normalizer(action_data)
    
    # 清理内存
    del traditional_norm
    gc.collect()
    
    # 测试流式方法
    streaming_norm = test_streaming_normalizer(action_data)
    
    # 比较结果
    compare_normalizers(traditional_norm, streaming_norm)
    
    print("\n🎉 测试完成!")

if __name__ == "__main__":
    main()
