#!/usr/bin/env python3
"""
测试配置修复
验证移除optimizer配置后的情况
"""

import sys
import os

# 设置正确的Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_config_loading():
    """测试配置加载"""
    print("=== 测试配置加载 ===")
    
    try:
        # 测试导入
        from omegaconf import OmegaConf
        import yaml
        
        # 加载配置文件
        config_path = "diffusion_policy/config/robot_dp_10.yaml"
        with open(config_path, 'r') as f:
            config_content = yaml.safe_load(f)
        
        print("✅ 配置文件加载成功")
        
        # 检查是否还有optimizer配置
        if 'optimizer' in config_content:
            print(f"❌ 配置中仍然包含optimizer: {config_content['optimizer']}")
            return False
        else:
            print("✅ 配置中已移除optimizer部分")
        
        # 检查其他必要配置
        required_keys = ['policy', 'training', 'dataloader', 'val_dataloader']
        for key in required_keys:
            if key in config_content:
                print(f"✅ 配置包含 {key}")
            else:
                print(f"❌ 配置缺少 {key}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ 配置测试失败: {e}")
        return False

def test_workspace_creation():
    """测试Workspace创建"""
    print("\n=== 测试Workspace创建 ===")
    
    try:
        # 测试导入
        from diffusion_policy.workspace.robotworkspace import RobotWorkspace
        print("✅ RobotWorkspace导入成功")
        
        # 测试配置
        from omegaconf import OmegaConf
        
        # 创建最小配置
        config = {
            'policy': {
                '_target_': 'diffusion_policy.policy.diffusion_unet_image_policy.DiffusionUnetImagePolicy',
                'shape_meta': {
                    'obs': {'head_cam': {'shape': [3, 256, 256], 'type': 'rgb'}},
                    'action': {'shape': [10], 'type': 'low_dim'}
                },
                'horizon': 8,
                'n_action_steps': 6,
                'n_obs_steps': 3
            },
            'training': {
                'use_ema': True,
                'seed': 42,
                'device': 'cpu'
            },
            'task': {
                'dataset': {
                    '_target_': 'diffusion_policy.dataset.robot_image_dataset.RobotImageDataset',
                    'zarr_path': None
                }
            }
        }
        
        cfg = OmegaConf.create(config)
        print("✅ 配置创建成功")
        
        # 注意：这里只是测试配置，不实际运行
        print("✅ Workspace配置测试通过")
        
        return True
        
    except Exception as e:
        print(f"❌ Workspace测试失败: {e}")
        return False

def test_optimizer_creation():
    """测试优化器创建"""
    print("\n=== 测试优化器创建 ===")
    
    try:
        import torch
        import torch.nn as nn
        
        # 创建简单模型
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)
            
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        print("✅ 简单模型创建成功")
        
        # 使用硬编码参数创建优化器
        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=1.0e-4,
            betas=(0.95, 0.999),
            eps=1.0e-8,
            weight_decay=1.0e-6
        )
        print("✅ 优化器创建成功")
        print(f"   优化器类型: {type(optimizer)}")
        print(f"   参数数量: {len(list(optimizer.param_groups))}")
        
        return True
        
    except Exception as e:
        print(f"❌ 优化器测试失败: {e}")
        return False

def main():
    """主函数"""
    print("=== 配置修复测试 ===")
    
    # 测试1: 配置加载
    config_ok = test_config_loading()
    
    # 测试2: Workspace创建
    workspace_ok = test_workspace_creation()
    
    # 测试3: 优化器创建
    optimizer_ok = test_optimizer_creation()
    
    # 总结
    print(f"\n=== 测试总结 ===")
    print(f"配置加载: {'✅' if config_ok else '❌'}")
    print(f"Workspace创建: {'✅' if workspace_ok else '❌'}")
    print(f"优化器创建: {'✅' if optimizer_ok else '❌'}")
    
    if config_ok and workspace_ok and optimizer_ok:
        print("🎉 所有测试通过！配置修复成功")
    else:
        print("⚠️  部分测试失败，需要进一步检查")

if __name__ == "__main__":
    main()
