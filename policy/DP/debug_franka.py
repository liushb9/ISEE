#!/usr/bin/env python3
import zarr
import numpy as np
from diffusion_policy.common.replay_buffer import ReplayBuffer

def debug_zarr_file(zarr_path):
    """调试zarr文件的问题"""
    print(f"=== 调试文件: {zarr_path} ===")
    
    try:
        # 尝试直接打开zarr
        print("1. 尝试直接打开zarr...")
        root = zarr.open(zarr_path, "r")
        print("   ✅ 成功打开zarr")
        
        # 检查meta部分
        print("\n2. 检查meta部分...")
        if "meta" in root:
            meta = root["meta"]
            print(f"   meta keys: {list(meta.keys())}")
            
            if "episode_ends" in meta:
                episode_ends = meta["episode_ends"][:]
                print(f"   episode_ends shape: {episode_ends.shape}")
                print(f"   episode_ends: {episode_ends}")
            
            if "embodiment_type" in meta:
                try:
                    embodiment_type = meta["embodiment_type"]
                    # 检查是否是标量还是数组
                    if hasattr(embodiment_type, 'shape') and len(embodiment_type.shape) == 0:
                        # 标量值
                        print(f"   embodiment_type (scalar): {embodiment_type[()]}")
                    else:
                        # 数组值
                        print(f"   embodiment_type (array): {embodiment_type[:]}")
                except Exception as e:
                    print(f"   ❌ 读取embodiment_type失败: {e}")
        else:
            print("   ❌ 没有meta部分")
        
        # 检查data部分
        print("\n3. 检查data部分...")
        if "data" in root:
            data = root["data"]
            print(f"   data keys: {list(data.keys())}")
            
            for key in data.keys():
                try:
                    dataset = data[key]
                    print(f"   {key}: shape={dataset.shape}, dtype={dataset.dtype}")
                    
                    # 尝试读取一小部分数据
                    if len(dataset.shape) > 0:
                        sample = dataset[:min(5, dataset.shape[0])]
                        print(f"     sample: {sample}")
                except Exception as e:
                    print(f"     ❌ 读取 {key} 失败: {e}")
        else:
            print("   ❌ 没有data部分")
            
    except Exception as e:
        print(f"   ❌ 打开zarr失败: {e}")
        return False
    
    print("\n4. 尝试使用ReplayBuffer加载...")
    try:
        buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            keys=["head_cam", "state", "action", "text_feat"],
        )
        print(f"   ✅ ReplayBuffer加载成功")
        print(f"   episodes数量: {buffer.n_episodes}")
        print(f"   数据形状:")
        for key in ["head_cam", "state", "action", "text_feat"]:
            if key in buffer:
                print(f"     {key}: {buffer[key].shape}")
        
        # 尝试获取第一个episode
        print(f"\n5. 尝试获取第一个episode...")
        episode = buffer.get_episode(0, copy=True)
        print(f"   ✅ 成功获取episode")
        print(f"   episode keys: {list(episode.keys())}")
        for key, value in episode.items():
            if hasattr(value, 'shape'):
                print(f"     {key}: {value.shape}")
        
    except Exception as e:
        print(f"   ❌ ReplayBuffer加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    zarr_path = "data/stack_blocks_three-demo_clean-franka-panda-50.zarr"
    success = debug_zarr_file(zarr_path)
    
    if success:
        print("\n✅ 文件检查完成，没有发现问题")
    else:
        print("\n❌ 文件检查发现问题")
