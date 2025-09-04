from ._base_task import Base_Task
from .utils import *
import sapien
from ._GLOBAL_CONFIGS import *
import math

class merged(Base_Task):
    def __init__(self, task_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = kwargs.get('sub_task_name', task_name)
        self.block_middle_pose = [0, 0.0, 0.9, 0, 1, 0, 0]  # for handover_block

    def setup_demo(self, **kwargs):
        if 'sub_task_name' in kwargs:
            self.task_name = kwargs['sub_task_name']
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        if self.task_name == "beat_block_hammer":
            self.hammer = create_actor(
                scene=self,
                pose=sapien.Pose([0, -0.06, 0.783], [0, 0, 0.995, 0.105]),
                modelname="020_hammer",
                convex=True,
                model_id=0,
            )
            block_pose = rand_pose(
                xlim=[-0.25, 0.25],
                ylim=[-0.05, 0.15],
                zlim=[0.76],
                qpos=[1, 0, 0, 0],
                rotate_rand=True,
                rotate_lim=[0, 0, 0.5],
            )
            while abs(block_pose.p[0]) < 0.05 or np.sum(pow(block_pose.p[:2], 2)) < 0.001:
                block_pose = rand_pose(
                    xlim=[-0.25, 0.25],
                    ylim=[-0.05, 0.15],
                    zlim=[0.76],
                    qpos=[1, 0, 0, 0],
                    rotate_rand=True,
                    rotate_lim=[0, 0, 0.5],
                )
            self.block = create_box(
                scene=self,
                pose=block_pose,
                half_size=(0.025, 0.025, 0.025),
                color=(1, 0, 0),
                name="box",
                is_static=True,
            )
            self.hammer.set_mass(0.001)
            self.add_prohibit_area(self.hammer, padding=0.10)
            self.prohibited_area.append([
                block_pose.p[0] - 0.05,
                block_pose.p[1] - 0.05,
                block_pose.p[0] + 0.05,
                block_pose.p[1] + 0.05,
            ])
        elif self.task_name == "handover_block":
            rand_pos = rand_pose(
                xlim=[-0.25, -0.05],
                ylim=[0, 0.25],
                zlim=[0.842],
                qpos=[0.981, 0, 0, 0.195],
                rotate_rand=True,
                rotate_lim=[0, 0, 0.2],
            )
            self.box = create_box(
                scene=self,
                pose=rand_pos,
                half_size=(0.03, 0.03, 0.1),
                color=(1, 0, 0),
                name="box",
                boxtype="long",
            )
            rand_pos = rand_pose(
                xlim=[0.1, 0.25],
                ylim=[0.15, 0.2],
            )
            self.target = create_box(
                scene=self,
                pose=rand_pos,
                half_size=(0.05, 0.05, 0.005),
                color=(0, 0, 1),
                name="target",
                is_static=True,
            )
            self.add_prohibit_area(self.box, padding=0.1)
            self.add_prohibit_area(self.target, padding=0.1)
        elif self.task_name == "turn_switch":
            self.model_name = "056_switch"
            self.model_id = np.random.randint(0, 8)
            self.switch = rand_create_sapien_urdf_obj(
                scene=self,
                modelname=self.model_name,
                modelid=self.model_id,
                xlim=[-0.25, 0.25],
                ylim=[0.0, 0.1],
                zlim=[0.81, 0.84],
                rotate_rand=True,
                rotate_lim=[0, 0, np.pi / 4],
                qpos=[0.704141, 0, 0, 0.71006],
                fix_root_link=True,
            )
            self.prohibited_area.append([-0.4, -0.2, 0.4, 0.2])

    def play_once(self):
        if self.task_name == "beat_block_hammer":
            block_pose = self.block.get_functional_point(0, "pose").p
            arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")
            self.move(self.grasp_actor(self.hammer, arm_tag=arm_tag, pre_grasp_dis=0.12, grasp_dis=0.01))
            self.move(self.move_by_displacement(arm_tag, z=0.07, move_axis="arm"))
            self.move(
                self.place_actor(
                    self.hammer,
                    target_pose=self.block.get_functional_point(1, "pose"),
                    arm_tag=arm_tag,
                    functional_point_id=0,
                    pre_dis=0.06,
                    dis=0,
                    is_open=False,
                ))
            self.info["info"] = {"{A}": "020_hammer/base0", "{a}": str(arm_tag)}
            return self.info
        elif self.task_name == "handover_block":
            grasp_arm_tag = ArmTag("left" if self.box.get_pose().p[0] < 0 else "right")
            place_arm_tag = grasp_arm_tag.opposite
            self.move(
                self.grasp_actor(
                    self.box,
                    arm_tag=grasp_arm_tag,
                    pre_grasp_dis=0.07,
                    grasp_dis=0.0,
                    contact_point_id=[0, 1, 2, 3],
                ))
            self.move(self.move_by_displacement(grasp_arm_tag, z=0.1))
            self.move(
                self.place_actor(
                    self.box,
                    target_pose=self.block_middle_pose,
                    arm_tag=grasp_arm_tag,
                    functional_point_id=0,
                    pre_dis=0,
                    dis=0,
                    is_open=False,
                    constrain="free",
                ))
            self.move(
                self.grasp_actor(
                    self.box,
                    arm_tag=place_arm_tag,
                    pre_grasp_dis=0.07,
                    grasp_dis=0.0,
                    contact_point_id=[4, 5, 6, 7],
                ))
            self.move(self.open_gripper(grasp_arm_tag))
            self.move(self.move_by_displacement(grasp_arm_tag, z=0.1, move_axis="arm"))
            self.move(
                self.back_to_origin(grasp_arm_tag),
                self.place_actor(
                    self.box,
                    target_pose=self.target.get_functional_point(1, "pose"),
                    arm_tag=place_arm_tag,
                    functional_point_id=0,
                    pre_dis=0.05,
                    dis=0.,
                    constrain="align",
                    pre_dis_axis="fp",
                ),
            )
            return self.info
        elif self.task_name == "turn_switch":
            switch_pose = self.switch.get_pose()
            face_dir = -switch_pose.to_transformation_matrix()[:3, 0]
            arm_tag = ArmTag("right" if face_dir[0] > 0 else "left")
            self.move(self.close_gripper(arm_tag=arm_tag, pos=0))
            self.move(self.grasp_actor(self.switch, arm_tag=arm_tag, pre_grasp_dis=0.04))
            self.info["info"] = {"{A}": f"056_switch/base{self.model_id}", "{a}": str(arm_tag)}
            return self.info

    def check_success(self):
        if self.task_name == "beat_block_hammer":
            hammer_target_pose = self.hammer.get_functional_point(0, "pose").p
            block_pose = self.block.get_functional_point(1, "pose").p
            eps = np.array([0.02, 0.02])
            return np.all(abs(hammer_target_pose[:2] - block_pose[:2]) < eps) and self.check_actors_contact(
                self.hammer.get_name(), self.block.get_name())
        elif self.task_name == "handover_block":
            box_pos = self.box.get_functional_point(0, "pose").p
            target_pose = self.target.get_functional_point(1, "pose").p
            eps = [0.03, 0.03]
            return (np.all(np.abs(box_pos[:2] - target_pose[:2]) < eps) and abs(box_pos[2] - target_pose[2]) < 0.01
                    and self.is_right_gripper_open())
        elif self.task_name == "turn_switch":
            limit = self.switch.get_qlimits()[0]
            return self.switch.get_qpos()[0] >= limit[1] - 0.05
