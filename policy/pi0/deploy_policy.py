import numpy as np
import torch
import dill
import os, sys

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)

from pi_model import *


# Encode observation for the model
def encode_obs(observation):
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]
    
    # Handle variable DOF: pad or truncate to match model's expected action_dim (32)
    # The model was trained with action_dim=32, so we need to ensure consistent input size
    expected_dim = 32
    actual_dim = len(input_state)
    
    # Store original state for reference
    original_state = input_state.copy()
    
    if actual_dim < expected_dim:
        # Pad with zeros for smaller DOF robots (e.g., 14DOF -> 32DOF)
        padded_state = np.zeros(expected_dim)
        padded_state[:actual_dim] = input_state
        input_state = padded_state
        print(f"Padded state from {actual_dim}DOF to {expected_dim}DOF")
    elif actual_dim > expected_dim:
        # Truncate for larger DOF robots (e.g., 16DOF -> 32DOF, but this shouldn't happen with our current setup)
        input_state = input_state[:expected_dim]
        print(f"Truncated state from {actual_dim}DOF to {expected_dim}DOF")
    else:
        print(f"State dimension matches expected: {actual_dim}DOF")

    return input_rgb_arr, input_state


def get_model(usr_args):
    train_config_name, model_name, checkpoint_id, pi0_step = (usr_args["train_config_name"], usr_args["model_name"],
                                                              usr_args["checkpoint_id"], usr_args["pi0_step"])
    return PI0(train_config_name, model_name, checkpoint_id, pi0_step)


def eval(TASK_ENV, model, observation):

    if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)

    input_rgb_arr, input_state = encode_obs(observation)
    model.update_observation_window(input_rgb_arr, input_state)

    # ======== Get Action ========

    actions = model.get_action()[:model.pi0_step]

    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        input_rgb_arr, input_state = encode_obs(observation)
        model.update_observation_window(input_rgb_arr, input_state)

    # ============================


def reset_model(model):
    model.reset_obsrvationwindows()
