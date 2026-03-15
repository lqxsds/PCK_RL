from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.robots.panda import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Actor, Pose
from mani_skill.utils.structs.types import SimConfig
from scipy.spatial.transform import Rotation as R

def _build_box_with_hole(
    scene: ManiSkillScene, inner_radius, outer_radius, depth, center=(0, 0)
):
    builder = scene.create_actor_builder()
    thickness = (outer_radius - inner_radius) * 0.5
    # x-axis is hole direction
    half_center = [x * 0.5 for x in center]
    half_sizes = [
        [depth, thickness - half_center[0], outer_radius],
        [depth, thickness + half_center[0], outer_radius],
        [depth, outer_radius, thickness - half_center[1]],
        [depth, outer_radius, thickness + half_center[1]],
    ]
    offset = thickness + inner_radius
    poses = [
        sapien.Pose([0, offset + half_center[0], 0]),
        sapien.Pose([0, -offset + half_center[0], 0]),
        sapien.Pose([0, 0, offset + half_center[1]]),
        sapien.Pose([0, 0, -offset + half_center[1]]),
    ]

    mat = sapien.render.RenderMaterial(
        base_color=sapien_utils.hex2rgba("#FFD289"), roughness=0.5, specular=0.5
    )

    for half_size, pose in zip(half_sizes, poses):
        builder.add_box_collision(pose, half_size)
        builder.add_box_visual(pose, half_size, material=mat)
    return builder


@register_env("PegInsertionSide-v1", max_episode_steps=100)
class PegInsertionSideEnv(BaseEnv):
    """
    **Task Description:**
    Pick up a orange-white peg and insert the orange end into the box with a hole in it.

    **Randomizations:**
    - Peg half length is randomized between 0.085 and 0.125 meters. Box half length is the same value. (during reconfiguration)
    - Peg radius/half-width is randomized between 0.015 and 0.025 meters. Box hole's radius is same value + 0.003m of clearance. (during reconfiguration)
    - Peg is laid flat on table and has it's xy position and z-axis rotation randomized
    - Box is laid flat on table and has it's xy position and z-axis rotation randomized

    **Success Conditions:**
    - The white end of the peg is within 0.015m of the center of the box (inserted mid way).
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PegInsertionSide-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda_wristcam"]
    agent: Union[PandaWristCam]
    #_clearance = 0.003
    _clearance = 0.005
    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        num_envs=1,
        reconfiguration_freq=None,
        **kwargs,
    ):
        if reconfiguration_freq is None:
            if num_envs == 1:
                reconfiguration_freq = 1
            else:
                reconfiguration_freq = 0
        super().__init__(
            *args,
            robot_uids=robot_uids,
            num_envs=num_envs,
            reconfiguration_freq=reconfiguration_freq,
            **kwargs,
        )

    @property
    def _default_sim_config(self):
        return SimConfig()

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([0, -0.3, 0.2], [0, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.5, -0.5, 0.8], [0.05, -0.1, 0.4])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        with torch.device(self.device):
            self.table_scene = TableSceneBuilder(self)
            self.table_scene.build()

            lengths = self._batched_episode_rng.uniform(0.1, 0.1)
            radii = self._batched_episode_rng.uniform(0.02, 0.02)
            centers = (
                0.5
                * (lengths - radii)[:, None]
                * self._batched_episode_rng.uniform(-1, 1, size=(2,))
            )
            centers = np.zeros((self.num_envs, 2))

            # save some useful values for use later
            self.peg_half_sizes = common.to_tensor(np.vstack([lengths, radii, radii])).T
            peg_head_offsets = torch.zeros((self.num_envs, 3))
            peg_head_offsets[:, 0] = self.peg_half_sizes[:, 0]
            self.peg_head_offsets = Pose.create_from_pq(p=peg_head_offsets)

            box_hole_offsets = torch.zeros((self.num_envs, 3))
            box_hole_offsets[:, 1:] = common.to_tensor(centers)
            self.box_hole_offsets = Pose.create_from_pq(p=box_hole_offsets)
            self.box_hole_radii = common.to_tensor(radii + self._clearance)

            # in each parallel env we build a different box with a hole and peg (the task is meant to be quite difficult)
            pegs = []
            boxes = []

            for i in range(self.num_envs):
                scene_idxs = [i]
                length = lengths[i]
                radius = radii[i]
                builder = self.scene.create_actor_builder()
                builder.add_box_collision(sapien.Pose([length, 0, 0]),
                    half_size=[length, radius, radius])
                builder.add_box_collision( sapien.Pose([-length / 6, 0, 0]),
                    half_size=[length / 6, radius*2, radius])
                # peg head
                mat = sapien.render.RenderMaterial(
                    base_color=sapien_utils.hex2rgba("#EC7357"),
                    roughness=0.5,
                    specular=0.5,
                )
                builder.add_box_visual(
                    sapien.Pose([length, 0, 0]),
                    half_size=[length, radius, radius],
                    material=mat,
                )
                # peg tail
                mat = sapien.render.RenderMaterial(
                    base_color=sapien_utils.hex2rgba("#EDF6F9"),
                    roughness=0.5,
                    specular=0.5,
                )
                builder.add_box_visual(
                    sapien.Pose([-length / 6, 0, 0]),
                    half_size=[length / 6, radius*2, radius],
                    material=mat,
                )
                builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
                builder.set_scene_idxs(scene_idxs)
                peg = builder.build(f"peg_{i}")
                self.remove_from_state_dict_registry(peg)
                # box with hole

                inner_radius, outer_radius, depth = (
                    radius + self._clearance,
                    length,
                    length,
                )
                builder = _build_box_with_hole(
                    self.scene, inner_radius, outer_radius, depth, center=centers[i]
                )
                builder.initial_pose = sapien.Pose(p=[0, 1, 0.1])
                builder.set_scene_idxs(scene_idxs)
                box = builder.build_kinematic(f"box_with_hole_{i}")
                self.remove_from_state_dict_registry(box)
                pegs.append(peg)
                boxes.append(box)
            self.peg = Actor.merge(pegs, "peg")
            self.box = Actor.merge(boxes, "box_with_hole")

            # to support heterogeneous simulation state dictionaries we register merged versions
            # of the parallel actors
            self.add_to_state_dict_registry(self.peg)
            self.add_to_state_dict_registry(self.box)
###############################################################################################
            global q
            q = [1, 0, 0, 0]  # [w,x,y,z] 格式

            #################
            # 创建旋转对象（注意转换成[x,y,z,w]格式）
            rot_original = R.from_quat([q[1], q[2], q[3], q[0]])

            # 创建 z 轴旋转 90 度
            rot_z = R.from_euler('x', 0, degrees=True)

            # 应用旋转
            rot_final = rot_z * rot_original

            # 获取结果四元数（转换回[w,x,y,z]格式）
            quat_result = rot_final.as_quat()  # 返回[x,y,z,w]格式
            result = [quat_result[3], quat_result[0], quat_result[1], quat_result[2]]  # 转回[w,x,y,z]格式
            q = result
            #################
            # 创建旋转对象（注意转换成[x,y,z,w]格式）
            rot_original = R.from_quat([q[1], q[2], q[3], q[0]])

            # 创建 z 轴旋转 90 度
            rot_z = R.from_euler('z', 180, degrees=True)

            # 应用旋转
            rot_final = rot_z * rot_original

            # 获取结果四元数（转换回[w,x,y,z]格式）
            quat_result = rot_final.as_quat()  # 返回[x,y,z,w]格式
            result = [quat_result[3], quat_result[0], quat_result[1], quat_result[2]]  # 转回[w,x,y,z]格式
            q = result
            #################
            # self.peg.set_pose(sapien.Pose(p=[0,0,0.05],q=q))

            ################# #################
            ################# #################
            ################# #################
            ################# #################
            ################# #################
            ################# #################
            ################# #################


            builder = self.scene.create_actor_builder()
            builder.add_nonconvex_collision_from_file(filename='hammer.obj')
            builder.add_visual_from_file(filename='hammer.obj')
            builder.set_mass_and_inertia(mass=1, inertia=[0, 0, 0],
                                         cmass_local_pose=(sapien.Pose(p=[0, -0.5, 0])))
            # builder.physx_body_type = "static"
            mesh2 = builder.build(name='mesh2')
            self.hammer = mesh2
            self.hammer.set_collision_group_bit(group=2, bit_idx=30, bit=1)
            ###############################################################################################
            global q_hammer
            q = [1, 0, 0, 0]  # [w,x,y,z] 格式

            #################
            # 创建旋转对象（注意转换成[x,y,z,w]格式）
            rot_original = R.from_quat([q[1], q[2], q[3], q[0]])

            # 创建 z 轴旋转 90 度
            rot_z = R.from_euler('x', -90, degrees=True)

            # 应用旋转
            rot_final = rot_z * rot_original

            # 获取结果四元数（转换回[w,x,y,z]格式）
            quat_result = rot_final.as_quat()  # 返回[x,y,z,w]格式
            result = [quat_result[3], quat_result[0], quat_result[1], quat_result[2]]  # 转回[w,x,y,z]格式
            q = result
            #################
            # 创建旋转对象（注意转换成[x,y,z,w]格式）
            rot_original = R.from_quat([q[1], q[2], q[3], q[0]])

            # 创建 z 轴旋转 90 度
            rot_z = R.from_euler('z', 180, degrees=True)

            # 应用旋转
            rot_final = rot_z * rot_original

            # 获取结果四元数（转换回[w,x,y,z]格式）
            quat_result = rot_final.as_quat()  # 返回[x,y,z,w]格式
            result = [quat_result[3], quat_result[0], quat_result[1], quat_result[2]]  # 转回[w,x,y,z]格式
            q_hammer = result
            #################


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # initialize the box
            xy=torch.tensor([[0,0.3]])

            pos = torch.zeros((b, 3))
            pos[:, :2] = xy
            pos[:, 2] = self.peg_half_sizes[env_idx, 0]
            quat = randomization.random_quaternions(
                b,
                self.device,
                lock_x=True,
                lock_y=True,
                bounds=(np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8),
            )
            quat=[0.7071,0,0,0.7071]
            self.box.set_pose(Pose.create_from_pq(pos, quat))

            # # Initialize the robot
            # qpos = np.array(
            #     [
            #         0.0,
            #         np.pi / 8,
            #         0,
            #         -np.pi * 5 / 8,
            #         0,
            #         np.pi * 3 / 4,
            #         -np.pi / 4,
            #         0.04,
            #         0.04,
            #     ]
            # )
            # qpos = self._episode_rng.normal(0, 0.02, (b, len(qpos))) + qpos
            # qpos[:, -2:] = 0.04
            # self.agent.robot.set_qpos(qpos)
            self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))

            self.hammer.set_pose(sapien.Pose(p=[0, -0.3, 0], q=q_hammer))
            ############################################
            self.grasp_peg = False
            self.grasp_hammer = False

            self.agent.robot.set_qpos([ 0.0999, 0.3733, -0.0913, -1.9119, 0.0421, 2.2790, -0.7897, 0.0179,0.0179])
            self.peg.set_pose(sapien.Pose(p=[ 8.3195e-03, -1.7439e-04, 2.0174e-01], q=[ 0.7098, -0.0081, 0.0118, 0.7042]))

    # save some commonly used attributes
    @property
    def peg_head_pos(self):
        return self.peg.pose.p + self.peg_head_offsets.p

    @property
    def peg_head_pose(self):
        return self.peg.pose * self.peg_head_offsets

    @property
    def box_hole_pose(self):
        return self.box.pose * self.box_hole_offsets

    @property
    def goal_pose(self):
        # NOTE (stao): this is fixed after each _initialize_episode call. You can cache this value
        # and simply store it after _initialize_episode or set_state_dict calls.
        return self.box.pose * self.box_hole_offsets * self.peg_head_offsets.inv()

    def has_peg_inserted(self):
        # Only head position is used in fact
        peg_head_pos_at_hole = (self.box_hole_pose.inv() * self.peg_head_pose).p
        # x-axis is hole direction
        #x_flag = -0.015 <= peg_head_pos_at_hole[:, 0]
        x_flag = -0.1 <= peg_head_pos_at_hole[:, 0]
        y_flag = (-self.box_hole_radii <= peg_head_pos_at_hole[:, 1]) & (
            peg_head_pos_at_hole[:, 1] <= self.box_hole_radii
        )
        z_flag = (-self.box_hole_radii <= peg_head_pos_at_hole[:, 2]) & (
            peg_head_pos_at_hole[:, 2] <= self.box_hole_radii
        )


        return (
            x_flag & y_flag & z_flag,
            peg_head_pos_at_hole,
        )

    def evaluate(self):
        success, peg_head_pos_at_hole = self.has_peg_inserted()
        #print(dict(success=success, peg_head_pos_at_hole=peg_head_pos_at_hole))
        return dict(success=success, peg_head_pos_at_hole=peg_head_pos_at_hole)

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                peg_pose=self.peg.pose.raw_pose,
                peg_half_size=self.peg_half_sizes,
                box_hole_pose=self.box_hole_pose.raw_pose,
                box_hole_radius=self.box_hole_radii,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # Stage 1: Encourage gripper to be rotated to be lined up with the peg

        # Stage 2: Encourage gripper to move close to peg tail and grasp it
        gripper_pos = self.agent.tcp.pose.p
        tgt_gripper_pose = self.peg.pose  # account for panda gripper width with a bit more leeway
        gripper_to_peg_dist = torch.linalg.norm(
            gripper_pos - tgt_gripper_pose.p, axis=1
        )

        reaching_reward = 1 - torch.tanh(4.0 * gripper_to_peg_dist)

        # check with max_angle=20 to ensure gripper isn't grasping peg at an awkward pose
        is_grasped = self.agent.is_grasping(self.peg, max_angle=20)
        reward = reaching_reward + is_grasped

        # Stage 3: Orient the grasped peg properly towards the hole

        # pre-insertion award, encouraging both the peg center and the peg head to match the yz coordinates of goal_pose
        peg_head_wrt_goal = self.goal_pose.inv() * self.peg_head_pose
        peg_head_wrt_goal_yz_dist = torch.linalg.norm(
            peg_head_wrt_goal.p[:, 1:], axis=1
        )
        peg_wrt_goal = self.goal_pose.inv() * self.peg.pose
        peg_wrt_goal_yz_dist = torch.linalg.norm(peg_wrt_goal.p[:, 1:], axis=1)

        pre_insertion_reward = 3 * (
            1
            - torch.tanh(
                0.5 * (peg_head_wrt_goal_yz_dist + peg_wrt_goal_yz_dist)
                + 4.5 * torch.maximum(peg_head_wrt_goal_yz_dist, peg_wrt_goal_yz_dist)
            )
        )
        reward += pre_insertion_reward * is_grasped
        # stage 3 passes if peg is correctly oriented in order to insert into hole easily
        pre_inserted = (peg_head_wrt_goal_yz_dist < 0.01) & (
            peg_wrt_goal_yz_dist < 0.01
        )

        # Stage 4: Insert the peg into the hole once it is grasped and lined up
        peg_head_wrt_goal_inside_hole = self.box_hole_pose.inv() * self.peg_head_pose
        insertion_reward = 5 * (
            1
            - torch.tanh(
                5.0 * torch.linalg.norm(peg_head_wrt_goal_inside_hole.p, axis=1)
            )
        )
        reward += insertion_reward * (is_grasped & pre_inserted)

        reward[info["success"]] = 10
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs, action, info) / 10

    def step(self, action: Union[None, np.ndarray, torch.Tensor, Dict]):
        return super().step(action)