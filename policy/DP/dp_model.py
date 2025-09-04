import numpy as np
import torch
import hydra
import dill
import sys, os

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
sys.path.append(parent_dir)

from diffusion_policy.workspace.robotworkspace import RobotWorkspace
from diffusion_policy.env_runner.dp_runner import DPRunner

class DP:

    def __init__(self, ckpt_file: str, n_obs_steps, n_action_steps):
        self.policy = self.get_policy(ckpt_file, None, "cuda:0")
        self.runner = DPRunner(n_obs_steps=n_obs_steps, n_action_steps=n_action_steps)

    def update_obs(self, observation):
        self.runner.update_obs(observation)
    
    def reset_obs(self):
        self.runner.reset_obs()

    def get_action(self, observation=None):
        action = self.runner.get_action(self.policy, observation)
        return action

    def get_last_obs(self):
        return self.runner.obs[-1]

    def get_policy(self, checkpoint, output_dir, device):
        # load checkpoint
        payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
        cfg = payload["cfg"]
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=output_dir)
        workspace: RobotWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        # get policy from workspace
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        # 强制更新obs_encoder的配置以匹配实际数据形状
        if hasattr(policy, 'obs_encoder') and hasattr(policy.obs_encoder, 'key_shape_map'):
            # 将head_cam的形状改为(3, 240, 320)以匹配实际输入图像形状
            if 'head_cam' in policy.obs_encoder.key_shape_map:
                policy.obs_encoder.key_shape_map['head_cam'] = (3, 240, 320)
            
            # 同时更新shape_meta中的配置
            if hasattr(policy.obs_encoder, 'shape_meta') and 'obs' in policy.obs_encoder.shape_meta:
                if 'head_cam' in policy.obs_encoder.shape_meta['obs']:
                    policy.obs_encoder.shape_meta['obs']['head_cam']['shape'] = [3, 240, 320]
            
            # 重新构建key_transform_map以确保resize操作正确应用
            if hasattr(policy.obs_encoder, 'key_transform_map') and 'head_cam' in policy.obs_encoder.key_transform_map:
                import torchvision
                # 创建新的resize transform
                new_resizer = torchvision.transforms.Resize(size=(256, 256))
                # 获取现有的其他transforms
                existing_transform = policy.obs_encoder.key_transform_map['head_cam']
                if hasattr(existing_transform, 'modules'):
                    # 如果transform是Sequential，替换第一个模块（resizer）
                    modules = list(existing_transform.modules())
                    if len(modules) > 1:  # 跳过Sequential本身
                        new_transform = torch.nn.Sequential(
                            new_resizer,
                            *modules[2:]  # 跳过Sequential和原来的resizer
                        )
                        policy.obs_encoder.key_transform_map['head_cam'] = new_transform

        device = torch.device(device)
        policy.to(device)
        policy.eval()

        return policy
