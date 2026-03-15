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
from mani_skill.utils.building import actors

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


            # in each parallel env we build a different box with a hole and peg (the task is meant to be quite difficult)
            pegs = []
            for i in range(self.num_envs):
                scene_idxs=[i]
                builder = self.scene.create_actor_builder()
                builder.set_scene_idxs(scene_idxs)
                builder.add_nonconvex_collision_from_file(filename='/home/lqx/ManiSkill-3.0.0b15/mani_skill/envs/tasks/tabletop/hammer.obj')
                builder.add_visual_from_file(filename='/home/lqx/ManiSkill-3.0.0b15/mani_skill/envs/tasks/tabletop/hammer.obj')
                builder.set_mass_and_inertia(mass=1, inertia=[0, 0, 0],cmass_local_pose=sapien.Pose())

                #builder.physx_body_type = "static"
                mesh2 = builder.build(name=f"hammer{i}")
                peg = mesh2
                peg.set_collision_group_bit(group=2, bit_idx=30, bit=1)
                ###############################################################################################
                pegs.append(peg)
                self.remove_from_state_dict_registry(peg)


            self.peg = Actor.merge(pegs, "peg")


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

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
            qpos=np.array([ 0.1416,  0.5398, -0.2288, -1.9562,  0.1879,  2.4768,  0.5820,  0.0400,
          0.0400])
            self.agent.robot.set_qpos(qpos)
            self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))

            xyz = torch.zeros((b, 3))
            xyz[..., 2] = 0.00
            q = [1, 0, 0, 0]
            obj_pose = Pose.create_from_pq(p=xyz, q=q)
            self.peg.set_pose(obj_pose)


    # save some commonly used attributes
    @property
    def peg_head_pos(self):
        return self.peg.pose.p + self.peg_head_offsets.p

    @property
    def peg_head_pose(self):
        return self.peg.pose * self.peg_head_offsets




    def has_peg_lifted(self):
        # Only head position is used in fact
        peg_position = self.peg.pose.p

        z_flag = self.peg.pose.p[:, 2] >= 0.05

        return (
            z_flag,
            peg_position,
        )

    def evaluate(self):
        success, peg_position = self.has_peg_lifted()
        return dict(success=success, peg_position=peg_position)

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                peg_pose=self.peg.pose.raw_pose,
                # peg_half_size=self.peg_half_sizes,
                # box_hole_pose=self.box_hole_pose.raw_pose,
                # box_hole_radius=self.box_hole_radii,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # Stage 1: Encourage gripper to be rotated to be lined up with the peg

        # Stage 2: Encourage gripper to move close to peg tail and grasp it
        gripper_pos = self.agent.tcp.pose.p
        tgt_gripper_pose = self.peg.pose.p

        gripper_to_peg_dist = torch.linalg.norm(
            gripper_pos - tgt_gripper_pose, axis=1
        )

        reaching_reward = 1 - torch.tanh(4.0 * gripper_to_peg_dist)
        # check with max_angle=20 to ensure gripper isn't grasping peg at an awkward pose
        is_grasped = self.agent.is_grasping(self.peg, max_angle=20)
        is_lift = self.peg.pose.p[:, 2] >= 0
        reward =  reaching_reward*2+is_lift*5+is_grasped*5
        reward=reaching_reward*0
        reward[info["success"]] = 10
        return reward

    def compute_normalized_dense_reward(
            self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs, action, info) / 10
