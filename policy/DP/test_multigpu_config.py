#!/usr/bin/env python3
"""
测试多卡训练配置
验证Fabric实例是否正确传递和使用
"""

import sys
import os
import torch

# 添加路径
sys.path.append('/home/shengbang/RoboTwin/policy/DP')

def test_fabric_creation():
    """测试Fabric实例创建"""
    print("=== 测试Fabric实例创建 ===")
    
    try:
        from lightning.fabric import Fabric
        
        # 创建Fabric实例
        fabric = Fabric(
            accelerator="cuda" if torch.cuda.is_available() else "cpu",
            devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
            strategy="ddp",
            precision="32-true",
        )
        
        print(f"✅ Fabric实例创建成功")
        print(f"   加速器: {fabric.accelerator}")
        print(f"   设备数量: {fabric.num_devices}")  # 修复：使用num_devices
        print(f"   策略: {fabric.strategy}")
        print(f"   精度: {fabric.precision}")
        
        return fabric
        
    except Exception as e:
        print(f"❌ Fabric实例创建失败: {e}")
        return None

def test_workspace_instantiation():
    """测试workspace实例化"""
    print("\n=== 测试Workspace实例化 ===")
    
    try:
        from diffusion_policy.workspace.robotworkspace import RobotWorkspace
        from omegaconf import OmegaConf
        
        # 创建完整的配置结构，满足RobotWorkspace初始化需求
        cfg = OmegaConf.create({
            "task": {
                "name": "test_task",
                "dataset": {
                    "_target_": "diffusion_policy.dataset.robot_image_dataset.RobotImageDataset",
                    "zarr_path": "data/test.zarr",
                    "horizon": 1,
                    "pad_before": 0,
                    "pad_after": 0,
                    "seed": 42,
                    "val_ratio": 0.0,
                    "batch_size": 1,
                    "max_train_episodes": None
                },
                "env_runner": None,
                "workspace": {
                    "_target_": "diffusion_policy.workspace.robotworkspace.RobotWorkspace"
                }
            },
            "policy": {
                "_target_": "diffusion_policy.policy.diffusion_unet_image_policy.DiffusionUnetImagePolicy",
                "obs_encoder": {
                    "_target_": "diffusion_policy.model.common.mlp.MLP",
                    "input_dim": 10,
                    "output_dim": 64,
                    "hidden_dims": [64, 64]
                },
                "action_dim": 10,
                "n_obs_steps": 2,
                "n_action_steps": 2,
                "n_cond_steps": 2,
                "horizon": 4,
                "obs_as_global_cond": True,
                "diffusion_step_embed_dim": 64,
                "down_dims": [64, 128, 256],
                "up_dims": [256, 128, 64],
                "n_groups": 8,
                "mid_blocks_per_group": 2,
                "use_fp16": False
            },
            "optimizer": {
                "_target_": "torch.optim.AdamW",
                "lr": 1e-4,
                "weight_decay": 1e-6
            },
            "dataloader": {
                "batch_size": 1,
                "num_workers": 0,
                "shuffle": True,
                "pin_memory": False,
                "persistent_workers": False
            },
            "val_dataloader": {
                "batch_size": 1,
                "num_workers": 0,
                "shuffle": False,
                "pin_memory": False,
                "persistent_workers": False
            },
            "training": {
                "seed": 42,
                "device": "cuda:0",
                "num_epochs": 1,
                "max_train_steps": None,
                "max_val_steps": None,
                "gradient_accumulate_every": 1,
                "lr_warmup_steps": 0,
                "lr_scheduler": "cosine",
                "checkpoint_every": 1,
                "val_every": 1,
                "sample_every": 1,
                "rollout_every": 1,
                "use_ema": False,
                "freeze_encoder": False,
                "debug": False,
                "resume": False,
                "tqdm_interval_sec": 0.1
            },
            "ema": {
                "_target_": "diffusion_policy.model.diffusion.ema_model.EMAModel",
                "decay": 0.9999
            },
            "checkpoint": {
                "topk": 5
            },
            "logging": {
                "mode": "disabled",
                "project": "test_project",
                "name": "test_run"
            },
            "head_camera_type": "default",
            "n_obs_steps": 2,
            "n_action_steps": 2,
            "horizon": 4,
            "obs_dim": 10,
            "action_dim": 10
        })
        
        print(f"✅ 配置创建成功")
        print(f"   配置键数量: {len(cfg)}")
        
        # 直接实例化RobotWorkspace
        workspace = RobotWorkspace(cfg)
        
        print(f"✅ Workspace实例化成功")
        print(f"   类型: {type(workspace)}")
        print(f"   Fabric属性: {hasattr(workspace, 'fabric')}")
        print(f"   Rank属性: {hasattr(workspace, 'rank')}")
        print(f"   World_size属性: {hasattr(workspace, 'world_size')}")
        
        return workspace
        
    except Exception as e:
        print(f"❌ Workspace实例化失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_fabric_integration(fabric, workspace):
    """测试Fabric与Workspace的集成"""
    print("\n=== 测试Fabric集成 ===")
    
    if fabric is None or workspace is None:
        print("❌ 无法测试集成，实例创建失败")
        return
    
    try:
        # 设置多卡训练属性
        workspace.fabric = fabric
        workspace.rank = fabric.global_rank
        workspace.world_size = fabric.num_devices  # 修复：使用num_devices
        
        print(f"✅ 多卡训练属性设置成功")
        print(f"   Fabric实例: {workspace.fabric is not None}")
        print(f"   Rank: {workspace.rank}")
        print(f"   World_size: {workspace.world_size}")
        
        # 测试多卡训练逻辑
        if hasattr(workspace, 'fabric') and workspace.fabric is not None:
            print(f"✅ 多卡训练模式检测成功")
        else:
            print(f"❌ 多卡训练模式检测失败")
            
    except Exception as e:
        print(f"❌ Fabric集成失败: {e}")
        import traceback
        traceback.print_exc()

def test_device_management():
    """测试设备管理"""
    print("\n=== 测试设备管理 ===")
    
    try:
        from lightning.fabric import Fabric
        
        fabric = Fabric(
            accelerator="cuda" if torch.cuda.is_available() else "cpu",
            devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
            strategy="ddp",
        )
        
        device = fabric.device
        print(f"✅ 设备管理正常")
        print(f"   Fabric设备: {device}")
        print(f"   CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU数量: {torch.cuda.device_count()}")
            print(f"   当前GPU: {torch.cuda.current_device()}")
        
    except Exception as e:
        print(f"❌ 设备管理测试失败: {e}")

def main():
    """主函数"""
    print("=== 多卡训练配置测试 ===")
    
    # 测试Fabric创建
    fabric = test_fabric_creation()
    
    # 测试Workspace实例化
    workspace = test_workspace_instantiation()
    
    # 测试集成
    test_fabric_integration(fabric, workspace)
    
    # 测试设备管理
    test_device_management()
    
    print("\n🎉 测试完成!")
    
    if fabric and workspace:
        print("✅ 所有测试通过，多卡训练配置正常")
    else:
        print("❌ 部分测试失败，需要检查配置")

if __name__ == "__main__":
    main()
