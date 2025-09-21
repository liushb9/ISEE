STATE_VEC_IDX_MAPPING = {
    # [0, 10): right arm joint positions
    **{
        "arm_joint_{}_pos".format(i): i
        for i in range(10)
    },
    **{
        "right_arm_joint_{}_pos".format(i): i
        for i in range(10)
    },
    # [10, 15): right gripper joint positions
    **{
        "gripper_joint_{}_pos".format(i): i + 10
        for i in range(5)
    },
    **{
        "right_gripper_joint_{}_pos".format(i): i + 10
        for i in range(5)
    },
    "gripper_open": 10,  # alias of right_gripper_joint_0_pos
    "right_gripper_open": 10,
    # [15, 25): right arm joint velocities
    **{
        "arm_joint_{}_vel".format(i): i + 15
        for i in range(10)
    },
    **{
        "right_arm_joint_{}_vel".format(i): i + 15
        for i in range(10)
    },
    # [25, 30): right gripper joint velocities
    **{
        "gripper_joint_{}_vel".format(i): i + 25
        for i in range(5)
    },
    **{
        "right_gripper_joint_{}_vel".format(i): i + 25
        for i in range(5)
    },
    "gripper_open_vel": 25,  # alias of right_gripper_joint_0_vel
    "right_gripper_open_vel": 25,
    # [30, 33): right end effector positions
    "eef_pos_x": 30,
    "right_eef_pos_x": 30,
    "eef_pos_y": 31,
    "right_eef_pos_y": 31,
    "eef_pos_z": 32,
    "right_eef_pos_z": 32,
    # [33, 39): right end effector 6D pose
    "eef_angle_0": 33,
    "right_eef_angle_0": 33,
    "eef_angle_1": 34,
    "right_eef_angle_1": 34,
    "eef_angle_2": 35,
    "right_eef_angle_2": 35,
    "eef_angle_3": 36,
    "right_eef_angle_3": 36,
    "eef_angle_4": 37,
    "right_eef_angle_4": 37,
    "eef_angle_5": 38,
    "right_eef_angle_5": 38,
    # [39, 42): right end effector velocities
    "eef_vel_x": 39,
    "right_eef_vel_x": 39,
    "eef_vel_y": 40,
    "right_eef_vel_y": 40,
    "eef_vel_z": 41,
    "right_eef_vel_z": 41,
    # [42, 45): right end effector angular velocities
    "eef_angular_vel_roll": 42,
    "right_eef_angular_vel_roll": 42,
    "eef_angular_vel_pitch": 43,
    "right_eef_angular_vel_pitch": 43,
    "eef_angular_vel_yaw": 44,
    "right_eef_angular_vel_yaw": 44,
    # [45, 50): reserved
    # [50, 60): left arm joint positions
    **{
        "left_arm_joint_{}_pos".format(i): i + 50
        for i in range(10)
    },
    # [60, 65): left gripper joint positions
    **{
        "left_gripper_joint_{}_pos".format(i): i + 60
        for i in range(5)
    },
    "left_gripper_open": 60,  # alias of left_gripper_joint_0_pos
    # [65, 75): left arm joint velocities
    **{
        "left_arm_joint_{}_vel".format(i): i + 65
        for i in range(10)
    },
    # [75, 80): left gripper joint velocities
    **{
        "left_gripper_joint_{}_vel".format(i): i + 75
        for i in range(5)
    },
    "left_gripper_open_vel": 75,  # alias of left_gripper_joint_0_vel
    # [80, 83): left end effector positions
    "left_eef_pos_x": 80,
    "left_eef_pos_y": 81,
    "left_eef_pos_z": 82,
    # [83, 89): left end effector 6D pose
    "left_eef_angle_0": 83,
    "left_eef_angle_1": 84,
    "left_eef_angle_2": 85,
    "left_eef_angle_3": 86,
    "left_eef_angle_4": 87,
    "left_eef_angle_5": 88,
    # [89, 92): left end effector velocities
    "left_eef_vel_x": 89,
    "left_eef_vel_y": 90,
    "left_eef_vel_z": 91,
    # [92, 95): left end effector angular velocities
    "left_eef_angular_vel_roll": 92,
    "left_eef_angular_vel_pitch": 93,
    "left_eef_angular_vel_yaw": 94,
    # [95, 100): reserved
    # [100, 102): base linear velocities
    "base_vel_x": 100,
    "base_vel_y": 101,
    # [102, 103): base angular velocities
    "base_angular_vel": 102,
    # [103, 128): reserved
}
STATE_VEC_LEN = 128

def create_dynamic_arm_indices(arm_dof, side="right"):
    """
    动态创建机械臂关节索引映射

    Args:
        arm_dof (int): 机械臂自由度 (6 或 7)
        side (str): "left" 或 "right"

    Returns:
        list: 关节索引列表
    """
    if arm_dof not in [6, 7]:
        raise ValueError(f"不支持的自由度: {arm_dof}，只支持6或7自由度")

    # 关节位置索引
    arm_indices = []
    for i in range(arm_dof):
        key = f"{side}_arm_joint_{i}_pos"
        if key in STATE_VEC_IDX_MAPPING:
            arm_indices.append(STATE_VEC_IDX_MAPPING[key])

    # 夹爪索引
    gripper_key = f"{side}_gripper_open"
    if gripper_key in STATE_VEC_IDX_MAPPING:
        arm_indices.append(STATE_VEC_IDX_MAPPING[gripper_key])

    return arm_indices

def create_bimanual_indices(left_dof=6, right_dof=6):
    """
    创建双臂的索引映射

    Args:
        left_dof (int): 左臂自由度
        right_dof (int): 右臂自由度

    Returns:
        dict: 包含左臂、右臂和联合索引的字典
    """
    left_indices = create_dynamic_arm_indices(left_dof, "left")
    right_indices = create_dynamic_arm_indices(right_dof, "right")

    return {
        "left": left_indices,
        "right": right_indices,
        "combined": left_indices + right_indices
    }
