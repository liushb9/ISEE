#!/usr/bin/env python3
"""
测试normalizer修复
验证agent_pos归一化参数是否正确设置
"""

import sys
import os

# 设置正确的Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_normalizer_initialization():
    """测试normalizer初始化"""
    print("=== 测试normalizer初始化 ===")
    
    try:
        # 测试导入
        from diffusion_policy.dataset.robot_image_dataset import RobotImageDataset
        print("✅ RobotImageDataset导入成功")
        
        # 检查数据集路径
        zarr_path = "data/six-tasks.zarr"
        if not os.path.exists(zarr_path):
            print(f"❌ 数据集不存在: {zarr_path}")
            print("请先运行数据处理脚本创建数据集")
            return False
        
        # 创建数据集实例
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
        print("✅ 数据集创建成功")
        
        # 检查数据维度
        action_dim = dataset.replay_buffer["action"].shape[1]
        state_dim = dataset.replay_buffer["state"].shape[1]
        print(f"   Action维度: {action_dim}")
        print(f"   State维度: {state_dim}")
        
        # 获取normalizer
        print("\n🔧 获取normalizer...")
        normalizer = dataset.get_normalizer()
        print("✅ normalizer获取成功")
        
        # 检查normalizer参数
        print("\n📊 检查normalizer参数...")
        
        # 检查action参数
        if "action" in normalizer.params_dict:
            print("✅ action参数存在")
            action_params = normalizer.params_dict["action"]
            if "scale" in action_params and "offset" in action_params:
                print(f"   Action scale shape: {action_params['scale'].shape}")
                print(f"   Action offset shape: {action_params['offset'].shape}")
            else:
                print("❌ action参数不完整")
                return False
        else:
            print("❌ action参数不存在")
            return False
        
        # 检查agent_pos参数
        if "agent_pos" in normalizer.params_dict:
            print("✅ agent_pos参数存在")
            agent_pos_params = normalizer.params_dict["agent_pos"]
            if "scale" in agent_pos_params and "offset" in agent_pos_params:
                print(f"   Agent_pos scale shape: {agent_pos_params['scale'].shape}")
                print(f"   Agent_pos offset shape: {agent_pos_params['offset'].shape}")
            else:
                print("❌ agent_pos参数不完整")
                return False
        else:
            print("❌ agent_pos参数不存在")
            return False
        
        # 检查图像参数
        image_keys = ["head_cam", "front_cam", "left_cam", "right_cam"]
        for key in image_keys:
            if key in normalizer.params_dict:
                print(f"✅ {key}参数存在")
            else:
                print(f"⚠️  {key}参数不存在（可能正常）")
        
        print("\n✅ normalizer初始化测试通过")
        return True
        
    except Exception as e:
        print(f"❌ normalizer初始化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_normalizer_usage():
    """测试normalizer使用"""
    print("\n=== 测试normalizer使用 ===")
    
    try:
        # 测试导入
        from diffusion_policy.dataset.robot_image_dataset import RobotImageDataset
        
        # 创建数据集实例
        zarr_path = "data/six-tasks.zarr"
        if not os.path.exists(zarr_path):
            print(f"❌ 数据集不存在: {zarr_path}")
            return False
        
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
        
        # 获取normalizer
        normalizer = dataset.get_normalizer()
        
        # 测试归一化
        print("🔧 测试归一化...")
        
        # 创建测试数据
        import torch
        test_batch = {
            "obs": {
                "head_cam": torch.randn(2, 3, 256, 256),
                "agent_pos": torch.randn(2, 10),
                "text_feat": torch.randn(2, 512)
            },
            "action": torch.randn(2, 10)
        }
        
        # 测试归一化
        try:
            normalized_batch = normalizer.normalize(test_batch)
            print("✅ 归一化成功")
            print(f"   归一化后action shape: {normalized_batch['action'].shape}")
            print(f"   归一化后agent_pos shape: {normalized_batch['obs']['agent_pos'].shape}")
        except Exception as e:
            print(f"❌ 归一化失败: {e}")
            return False
        
        print("✅ normalizer使用测试通过")
        return True
        
    except Exception as e:
        print(f"❌ normalizer使用测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("=== Normalizer修复测试 ===")
    
    # 测试1: normalizer初始化
    init_ok = test_normalizer_initialization()
    
    # 测试2: normalizer使用
    usage_ok = test_normalizer_usage()
    
    # 总结
    print(f"\n=== 测试总结 ===")
    print(f"Normalizer初始化: {'✅' if init_ok else '❌'}")
    print(f"Normalizer使用: {'✅' if usage_ok else '❌'}")
    
    if init_ok and usage_ok:
        print("🎉 所有测试通过！Normalizer修复成功")
        print("\n现在可以重新运行训练:")
        print("bash train_multi_gpu.sh six_tasks demo_clean 1200 0 1 '0'")
    else:
        print("⚠️  部分测试失败，需要进一步检查")

if __name__ == "__main__":
    main()
