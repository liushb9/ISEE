import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader

import IPython
import cv2

e = IPython.embed


class EpisodicDataset(torch.utils.data.Dataset):

    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, max_action_len):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.max_action_len = max_action_len
        self.is_sim = None
        self.__getitem__(0)  # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            is_sim = None
            original_action_shape = root["/action"].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root["/observations/qpos"][start_ts]
            # 获取text_feat（每个episode的text_feat都相同）
            text_feat = root["/observations/text_feat"][()]
            
            # 验证text_feat维度
            if text_feat.ndim == 1:
                text_feat_dim = text_feat.shape[0]
            else:
                text_feat_dim = text_feat.shape[-1]
            
            if text_feat_dim != 512 and text_feat_dim != 1024:
                print(f"警告: episode_{episode_id} 的text_feat维度异常: {text_feat.shape}")
            
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f"/observations/images/{cam_name}"][start_ts]
            # get all actions after and including start_ts
            if is_sim:
                action = root["/action"][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root["/action"][max(0, start_ts - 1):]  # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1)  # hack, to make timesteps more aligned

        self.is_sim = is_sim
        # Use the maximum action dimension from norm_stats if available
        max_action_dim = self.norm_stats.get("max_action_dim", action.shape[1] if len(action.shape) > 1 else action.shape[0])
        padded_action = np.zeros((self.max_action_len, max_action_dim), dtype=np.float32)
        padded_action[:action_len, :action.shape[1]] = action
        is_pad = np.ones(self.max_action_len, dtype=bool)  # 初始化为全1（True）
        is_pad[:action_len] = 0  # 前action_len个位置设置为0（False），表示非填充部分

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            # 确保图像是256x256
            img = image_dict[cam_name]
            if img.shape[:2] != (256, 256):
                # 如果图像尺寸不是256x256，则resize
                img = cv2.resize(img, (256, 256))
            all_cam_images.append(img)
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()
        text_feat_data = torch.from_numpy(text_feat).float()

        # channel last
        image_data = torch.einsum("k h w c -> k c h w", image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0

        # Normalize action data (pad to max_action_dim if necessary)
        current_action_dim = action_data.size(-1) if action_data.dim() > 1 else action_data.size(0)
        max_action_dim = self.norm_stats.get("max_action_dim", current_action_dim)

        if current_action_dim < max_action_dim:
            # Pad action_data to max_action_dim
            if action_data.dim() == 1:
                # For 1D tensor (single timestep)
                pad_dim = torch.zeros(max_action_dim - current_action_dim, dtype=action_data.dtype, device=action_data.device)
                action_data = torch.cat([action_data, pad_dim], dim=-1)
            else:
                # For 2D tensor (multiple timesteps)
                pad_dim = torch.zeros(action_data.size(0), max_action_dim - current_action_dim, dtype=action_data.dtype, device=action_data.device)
                action_data = torch.cat([action_data, pad_dim], dim=-1)

        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]

        # Normalize qpos data (pad to max_state_dim if necessary)
        current_dim = qpos_data.size(-1) if qpos_data.dim() > 0 else qpos_data.size(0)
        max_dim = self.norm_stats.get("max_state_dim", current_dim)

        if current_dim < max_dim:
            # Pad qpos_data to max_state_dim
            if qpos_data.dim() == 1:
                # For 1D tensor (single timestep)
                pad_dim = torch.zeros(max_dim - current_dim, dtype=qpos_data.dtype, device=qpos_data.device)
                qpos_data = torch.cat([qpos_data, pad_dim], dim=-1)
            else:
                # For 2D tensor (multiple timesteps)
                pad_dim = torch.zeros(qpos_data.size(0), max_dim - current_dim, dtype=qpos_data.dtype, device=qpos_data.device)
                qpos_data = torch.cat([qpos_data, pad_dim], dim=-1)

        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        # text_feat不归一化

        return image_data, qpos_data, action_data, is_pad, text_feat_data


def get_norm_stats(dataset_dir, num_episodes):
    # Handle mixed state dimensions by processing each dimension separately
    all_qpos_data = []
    all_action_data = []
    state_dims = set()

    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            qpos = root["/observations/qpos"][()]  # Assuming this is a numpy array
            action = root["/action"][()]

            # Record state dimension
            current_dim = qpos.shape[1] if len(qpos.shape) > 1 else qpos.shape[0]
            state_dims.add(current_dim)

            all_qpos_data.append((current_dim, torch.from_numpy(qpos)))
            all_action_data.append(torch.from_numpy(action))

    print(f"Detected state dimensions in dataset: {sorted(state_dims)}")

    # Find the maximum dimension for padding
    max_state_dim = max(state_dims) if state_dims else 14
    print(f"Using maximum state dimension for padding: {max_state_dim}")

    # Also check action dimensions
    action_dims = set()
    for action in all_action_data:
        current_dim = action.size(1) if len(action.size()) > 1 else action.size(0)
        action_dims.add(current_dim)

    max_action_dim = max(action_dims) if action_dims else 14
    print(f"Action dimensions found: {sorted(action_dims)}, using max: {max_action_dim}")

    # Pad all tensors to the maximum size
    max_qpos_len = max(qpos_tensor.size(0) for _, qpos_tensor in all_qpos_data)
    max_action_len = max(a.size(0) for a in all_action_data)

    padded_qpos = []
    for current_dim, qpos in all_qpos_data:
        current_len = qpos.size(0)
        # Pad time dimension
        if current_len < max_qpos_len:
            # Pad with the last element
            pad = qpos[-1:].repeat(max_qpos_len - current_len, 1)
            qpos = torch.cat([qpos, pad], dim=0)

        # Pad state dimension to maximum
        if current_dim < max_state_dim:
            # Pad state dimension with zeros (assuming the extra dimensions are at the end)
            pad_dim = torch.zeros(qpos.size(0), max_state_dim - current_dim, dtype=qpos.dtype)
            qpos = torch.cat([qpos, pad_dim], dim=1)

        padded_qpos.append(qpos)

    padded_action = []
    for action in all_action_data:
        current_len = action.size(0)
        current_action_dim = action.size(1) if len(action.size()) > 1 else action.size(0)

        # Pad time dimension
        if current_len < max_action_len:
            pad = action[-1:].repeat(max_action_len - current_len, 1)
            action = torch.cat([action, pad], dim=0)

        # Pad action dimension to maximum
        if current_action_dim < max_action_dim:
            pad_dim = torch.zeros(action.size(0), max_action_dim - current_action_dim, dtype=action.dtype)
            action = torch.cat([action, pad_dim], dim=1)

        padded_action.append(action)

    all_qpos_data = torch.stack(padded_qpos)
    all_action_data = torch.stack(padded_action)

    # normalize action data (assuming actions have consistent dimensions)
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping

    stats = {
        "action_mean": action_mean.numpy().squeeze(),
        "action_std": action_std.numpy().squeeze(),
        "qpos_mean": qpos_mean.numpy().squeeze(),
        "qpos_std": qpos_std.numpy().squeeze(),
        "example_qpos": qpos,
        "state_dims": sorted(list(state_dims)),
        "max_state_dim": max_state_dim,
        "action_dims": sorted(list(action_dims)),
        "max_action_dim": max_action_dim,
        # 不包含text_feat的归一化统计信息
    }

    return stats, max_action_len


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f"\nData from: {dataset_dir}\n")
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats, max_action_len = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, max_action_len)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, max_action_len)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
    )

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim, norm_stats.get("max_state_dim")


### env utils


def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


### helper functions


def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
