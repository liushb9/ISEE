from typing import Dict
import numba
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler,
    get_val_mask,
    downsample_mask,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer
import pdb
from diffusion_policy.model.common.normalizer import SingleFieldLinearNormalizer


class RobotImageDataset(BaseImageDataset):

    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        batch_size=128,
        max_train_episodes=None,
    ):

        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            # keys=['head_camera', 'front_camera', 'left_camera', 'right_camera', 'state', 'action'],
            keys=["head_cam", "state", "action", "text_feat"],  # 改为head_cam
        )

        val_mask = get_val_mask(n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        self.batch_size = batch_size
        sequence_length = self.sampler.sequence_length
        self.buffers = {
            k: np.zeros((batch_size, sequence_length, *v.shape[1:]), dtype=v.dtype)
            for k, v in self.sampler.replay_buffer.items()
        }
        self.buffers_torch = {k: torch.from_numpy(v) for k, v in self.buffers.items()}
        for v in self.buffers_torch.values():
            v.pin_memory()

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        """
        优化的归一化函数：
        1. 预计算统计信息，避免内存爆炸
        2. 只对xyz位置进行归一化，保持rotation_6d和gripper不变
        """
        normalizer = LinearNormalizer()
        
        # 预计算action的统计信息（避免内存爆炸）
        action_data = self.replay_buffer["action"]
        action_stat = {
            "min": np.min(action_data, axis=0),
            "max": np.max(action_data, axis=0),
            "mean": np.mean(action_data, axis=0),
            "std": np.std(action_data, axis=0),
        }
        
        # 预计算state的统计信息（用于agent_pos）
        state_data = self.replay_buffer["state"]
        state_stat = {
            "min": np.min(state_data, axis=0),
            "max": np.max(state_data, axis=0),
            "mean": np.mean(state_data, axis=0),
            "std": np.std(state_data, axis=0),
        }
        
        # 智能归一化：只对xyz位置归一化，保持rotation_6d和gripper不变
        action_dim = action_data.shape[1]
        if action_dim == 10:  # endpose: [x, y, z, rx, ry, rz, rw, rx2, ry2, gripper]
            # 位置部分 (前3维): 归一化到[-1, 1]
            pos_scale, pos_offset = self._get_pos_normalizer_params(
                action_stat["min"][:3], action_stat["max"][:3]
            )
            
            # 旋转部分 (中间6维): 保持原值
            rot_scale = np.ones(6)
            rot_offset = np.zeros(6)
            
            # 夹爪部分 (最后1维): 保持原值
            gripper_scale = np.ones(1)
            gripper_offset = np.zeros(1)
            
            # 组合所有参数
            action_scale = np.concatenate([pos_scale, rot_scale, gripper_scale])
            action_offset = np.concatenate([pos_offset, rot_offset, gripper_offset])
            
            # 创建action归一化器
            action_normalizer = SingleFieldLinearNormalizer.create_manual(
                scale=action_scale,
                offset=action_offset,
                input_stats_dict=action_stat
            )
            normalizer["action"] = action_normalizer
            
            # 创建agent_pos归一化器（使用相同的逻辑）
            state_dim = state_data.shape[1]
            if state_dim == 10:  # 如果state也是10维
                state_pos_scale, state_pos_offset = self._get_pos_normalizer_params(
                    state_stat["min"][:3], state_stat["max"][:3]
                )
                state_rot_scale = np.ones(6)
                state_rot_offset = np.zeros(6)
                state_gripper_scale = np.ones(1)
                state_gripper_offset = np.zeros(1)
                
                state_scale = np.concatenate([state_pos_scale, state_rot_scale, state_gripper_scale])
                state_offset = np.concatenate([state_pos_offset, state_rot_offset, state_gripper_offset])
                
                state_normalizer = SingleFieldLinearNormalizer.create_manual(
                    scale=state_scale,
                    offset=state_offset,
                    input_stats_dict=state_stat
                )
                normalizer["agent_pos"] = state_normalizer
            else:
                # 对于其他维度，使用传统方法
                state_normalizer = SingleFieldLinearNormalizer.create_manual(
                    scale=np.ones(state_dim),
                    offset=np.zeros(state_dim),
                    input_stats_dict=state_stat
                )
                normalizer["agent_pos"] = state_normalizer
        else:
            # 对于其他维度，使用传统方法
            data = {
                "action": action_data,
                "agent_pos": state_data,
            }
            normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        
        # 图像归一化
        normalizer["head_cam"] = get_image_range_normalizer()
        normalizer["front_cam"] = get_image_range_normalizer()
        normalizer["left_cam"] = get_image_range_normalizer()
        normalizer["right_cam"] = get_image_range_normalizer()
        
        return normalizer
    
    def _get_pos_normalizer_params(self, min_vals, max_vals, output_max=1, output_min=-1, range_eps=1e-7):
        """
        计算位置归一化参数，将位置数据归一化到[output_min, output_max]范围
        """
        input_range = max_vals - min_vals
        ignore_dim = input_range < range_eps
        input_range[ignore_dim] = output_max - output_min
        
        scale = (output_max - output_min) / input_range
        offset = output_min - scale * min_vals
        offset[ignore_dim] = (output_max + output_min) / 2 - min_vals[ignore_dim]
        
        return scale, offset

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample["state"].astype(np.float32)  # (agent_posx2, block_posex3)
        # 数据集中的图像已经是CHW格式，不需要moveaxis
        head_cam = sample["head_cam"].astype(np.float32) / 255  # 改为head_cam，移除moveaxis
        # front_cam = np.moveaxis(sample['front_camera'],-1,1)/255
        # left_cam = np.moveaxis(sample['left_camera'],-1,1)/255
        # right_cam = np.moveaxis(sample['right_camera'],-1,1)/255
        text_feat = sample["text_feat"].astype(np.float32)

        data = {
            "obs": {
                "head_cam": head_cam,  # T, 3, H, W - 确保键名与配置文件一致
                # 'front_cam': front_cam, # T, 3, H, W
                # 'left_cam': left_cam, # T, 3, H, W
                # 'right_cam': right_cam, # T, 3, H, W
                "agent_pos": agent_pos,  # T, D
                "text_feat": text_feat,  # T, D_text
            },
            "action": sample["action"].astype(np.float32),  # T, D
        }
        return data

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if isinstance(idx, slice):
            raise NotImplementedError  # Specialized
        elif isinstance(idx, int):
            sample = self.sampler.sample_sequence(idx)
            # 确保数据类型为float32
            sample = self._sample_to_data(sample)
            # 转换为torch tensor并确保数据类型
            torch_sample = dict_apply(sample, lambda x: torch.from_numpy(x).float())
            return torch_sample
        elif isinstance(idx, np.ndarray):
            assert len(idx) == self.batch_size
            for k, v in self.sampler.replay_buffer.items():
                batch_sample_sequence(
                    self.buffers[k],
                    v,
                    self.sampler.indices,
                    idx,
                    self.sampler.sequence_length,
                )
            return self.buffers_torch
        else:
            raise ValueError(idx)

    def postprocess(self, samples, device):
        agent_pos = samples["state"].to(device, non_blocking=True)
        # 数据集中的图像已经是CHW格式，不需要额外的格式转换
        head_cam = samples["head_cam"].to(device, non_blocking=True) / 255.0  # 改为head_cam
        # front_cam = samples['front_camera'].to(device, non_blocking=True) / 255.0
        # left_cam = samples['left_camera'].to(device, non_blocking=True) / 255.0
        # right_cam = samples['right_camera'].to(device, non_blocking=True) / 255.0
        action = samples["action"].to(device, non_blocking=True)
        text_feat = samples["text_feat"].to(device, non_blocking=True)
        return {
            "obs": {
                "head_cam": head_cam,  # B, T, 3, H, W - 确保键名与配置文件一致
                # 'front_cam': front_cam, # B, T, 3, H, W
                # 'left_cam': left_cam, # B, T, 3, H, W
                # 'right_cam': right_cam, # B, T, 3, H, W
                "agent_pos": agent_pos,  # B, T, D
                "text_feat": text_feat,  # B, T, D_text
            },
            "action": action,  # B, T, D
        }


def _batch_sample_sequence(
    data: np.ndarray,
    input_arr: np.ndarray,
    indices: np.ndarray,
    idx: np.ndarray,
    sequence_length: int,
):
    for i in numba.prange(len(idx)):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = indices[idx[i]]
        data[i, sample_start_idx:sample_end_idx] = input_arr[buffer_start_idx:buffer_end_idx]
        if sample_start_idx > 0:
            data[i, :sample_start_idx] = data[i, sample_start_idx]
        if sample_end_idx < sequence_length:
            data[i, sample_end_idx:] = data[i, sample_end_idx - 1]


_batch_sample_sequence_sequential = numba.jit(_batch_sample_sequence, nopython=True, parallel=False)
_batch_sample_sequence_parallel = numba.jit(_batch_sample_sequence, nopython=True, parallel=True)


def batch_sample_sequence(
    data: np.ndarray,
    input_arr: np.ndarray,
    indices: np.ndarray,
    idx: np.ndarray,
    sequence_length: int,
):
    batch_size = len(idx)
    assert data.shape == (batch_size, sequence_length, *input_arr.shape[1:])
    if batch_size >= 16 and data.nbytes // batch_size >= 2**16:
        _batch_sample_sequence_parallel(data, input_arr, indices, idx, sequence_length)
    else:
        _batch_sample_sequence_sequential(data, input_arr, indices, idx, sequence_length)
