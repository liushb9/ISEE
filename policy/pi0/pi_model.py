#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import json
import sys
import jax
import numpy as np
from openpi.models import model as _model
from openpi.policies import aloha_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

import cv2
from PIL import Image

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


class PI0:

    def __init__(self, train_config_name, model_name, checkpoint_id, pi0_step):
        self.train_config_name = train_config_name
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id

        config = _config.get_config(self.train_config_name)
        self.policy = _policy_config.create_trained_policy(
            config,
            f"policy/pi0/checkpoints/{self.train_config_name}/{self.model_name}/{self.checkpoint_id}",
            robotwin_repo_id=model_name)
        print("loading model success!")
        self.img_size = (224, 224)
        self.observation_window = None
        self.pi0_step = pi0_step
        # Store the model's action_dim for reference
        self.model_action_dim = config.model.action_dim
        print(f"Model action_dim: {self.model_action_dim}")

    # set img_size
    def set_img_size(self, img_size):
        self.img_size = img_size

    # set language randomly
    def set_language(self, instruction):
        self.instruction = instruction
        print(f"successfully set instruction:{instruction}")

    # Update the observation window buffer
    def update_observation_window(self, img_arr, state):
        img_front, img_right, img_left, puppet_arm = (
            img_arr[0],
            img_arr[1],
            img_arr[2],
            state,
        )
        img_front = np.transpose(img_front, (2, 0, 1))
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))

        self.observation_window = {
            "state": state,
            "images": {
                "cam_high": img_front,
                "cam_left_wrist": img_left,
                "cam_right_wrist": img_right,
            },
            "prompt": self.instruction,
        }

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        actions = self.policy.infer(self.observation_window)["actions"]
        
        # The model outputs actions with shape [action_horizon, model_action_dim]
        # We need to adapt this to the actual robot's DOF
        # The input state was padded to model_action_dim, but we need to extract only the relevant dimensions
        
        # Detect actual DOF by finding non-zero values in the state
        state = self.observation_window["state"]
        non_zero_mask = np.abs(state) > 1e-6  # Threshold for non-zero values
        actual_dof = np.sum(non_zero_mask)
        
        if actual_dof > 0 and actual_dof < self.model_action_dim:
            # Extract only the relevant action dimensions (first actual_dof dimensions)
            actions = actions[:, :actual_dof]
            print(f"Extracted actions for {actual_dof}DOF robot from {self.model_action_dim}DOF model output")
        elif actual_dof == self.model_action_dim:
            # All dimensions are non-zero, use full action
            print(f"Using full {actual_dof}DOF actions from model output")
        else:
            # Fallback: assume 14DOF or 16DOF based on common robot configurations
            if actual_dof == 0:  # All zeros, this shouldn't happen
                print("Warning: All state values are zero, assuming 14DOF robot")
                actual_dof = 14
                actions = actions[:, :actual_dof]
            else:
                print(f"Warning: Unexpected DOF detected: {actual_dof}, using full model output")
        
        return actions

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        print("successfully unset obs and language intruction")
