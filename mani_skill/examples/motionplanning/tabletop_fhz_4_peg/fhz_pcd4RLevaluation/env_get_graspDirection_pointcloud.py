from typing import Dict

import numpy as np
import sapien
import torch

from mani_skill.agents.robots.fetch.fetch import Fetch
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.utils.building import actors

from mani_skill.utils.building import urdf_loader
from scipy.spatial.transform import Rotation as R
import json
from mani_skill.utils.structs import Actor, Pose
from mani_skill.utils import common, sapien_utils


name=("peg")


with open(f'{name}_grasp.json','r') as f:
    grasp_point = json.load(f)
with open(f'{name}_PCA.json','r') as f:
    data_read = json.load(f)
    center_point=data_read['center_point']
    principal_axes=data_read['principal_axes']
    axe=[0,0,0]
    axe[0]=principal_axes[0]['component']
    axe[1]=principal_axes[1]['component']
    axe[2]=np.cross(axe[0],axe[1])
direction=input("Enter direction 1/2/3")
if direction=="1":
        axes_selected=axe[0]
elif direction=="2":
        axes_selected=axe[1]
elif direction=="3":
        axes_selected=axe[2]
grasp_point=np.array(grasp_point)
axes_selected=np.array(axes_selected)
grasp_point_camera=grasp_point-axes_selected*0.3

@register_env("CustomEnv-v1", max_episode_steps=200000)
class CustomEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = ["dense"]
    """
    This is just a dummy environment for showcasing robots in a empty scene
    """

    def __init__(self, *args, robot_uids="panda", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        # 提取 pca_center_top 变量并转换回 NumPy 数组
        pose = sapien_utils.look_at(grasp_point_camera, grasp_point)
        ###############################################################################################
        q = pose.q.tolist()[0]  # [w,x,y,z] 格式

        #################
        # 创建旋转对象（注意转换成[x,y,z,w]格式）
        rot_original = R.from_quat([q[1], q[2], q[3], q[0]])

        # 创建 z 轴旋转 90 度
        rot_z = R.from_euler('z', 0, degrees=True)

        # 应用旋转
        rot_final = rot_z * rot_original

        # 获取结果四元数（转换回[w,x,y,z]格式）
        quat_result = rot_final.as_quat()  # 返回[x,y,z,w]格式
        result = [quat_result[3], quat_result[0], quat_result[1], quat_result[2]]  # 转回[w,x,y,z]格式
        q = result
        #################
        pose.q = q
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]
    @property
    def _default_human_render_camera_configs(self):

        pose = sapien_utils.look_at([1.25, -1.25, 1.5], [0.0, 0.0, 0.2])
        return CameraConfig("render_camera", pose, 2048, 2048, 1, 0.01, 100)

    def _load_scene(self, options: dict):
            with torch.device(self.device):


                lengths = self._batched_episode_rng.uniform(0.085, 0.125)
                lengths = np.array([0.1])
                radii = self._batched_episode_rng.uniform(0.015, 0.025)
                radii = np.array([0.02])
                centers = (
                        0.5
                        * (lengths - radii)[:, None]
                        * self._batched_episode_rng.uniform(-1, 1, size=(2,))
                )
                centers = np.array([[0.0, 0.0]])

                # save some useful values for use later
                self.peg_half_sizes = common.to_tensor(np.vstack([lengths, radii, radii])).T
                peg_head_offsets = torch.zeros((self.num_envs, 3))
                peg_head_offsets[:, 0] = self.peg_half_sizes[:, 0]
                self.peg_head_offsets = Pose.create_from_pq(p=peg_head_offsets)

                # box_hole_offsets = torch.zeros((self.num_envs, 3))
                # box_hole_offsets[:, 1:] = common.to_tensor(centers)
                # self.box_hole_offsets = Pose.create_from_pq(p=box_hole_offsets)
                # self.box_hole_radii = common.to_tensor(radii + self._clearance)

                # in each parallel env we build a different box with a hole and peg (the task is meant to be quite difficult)
                pegs = []
                boxes = []

                for i in range(self.num_envs):
                    scene_idxs = [i]
                    length = lengths[i]
                    radius = radii[i]
                    builder = self.scene.create_actor_builder()
                    builder.physx_body_type = "static"
                    builder.add_box_collision(sapien.Pose([length, 0, 0]),
                                              half_size=[length, radius, radius])
                    builder.add_box_collision(sapien.Pose([-length / 6, 0, 0]),
                                              half_size=[length / 6, radius * 2, radius])
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
                        half_size=[length / 6, radius * 2, radius],
                        material=mat,
                    )
                    builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
                    builder.set_scene_idxs(scene_idxs)
                    peg = builder.build(f"peg_{i}")
                    self.remove_from_state_dict_registry(peg)
                    # box with hole

                    # inner_radius, outer_radius, depth = (
                    #     radius + self._clearance,
                    #     length,
                    #     length,
                    # )
                    # builder = _build_box_with_hole(
                    #     self.scene, inner_radius, outer_radius, depth, center=centers[i]
                    # )
                    # builder.initial_pose = sapien.Pose(p=[0, 1, 0.1])
                    # builder.set_scene_idxs(scene_idxs)
                    # box = builder.build_kinematic(f"box_with_hole_{i}")
                    # self.remove_from_state_dict_registry(box)
                    pegs.append(peg)
                    #boxes.append(box)
                self.peg = Actor.merge(pegs, "peg")
                #self.box = Actor.merge(boxes, "box_with_hole")

                # to support heterogeneous simulation state dictionaries we register merged versions
                # of the parallel actors
                self.add_to_state_dict_registry(self.peg)
                #self.add_to_state_dict_registry(self.box)
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
                self.peg.set_pose(sapien.Pose(p=[0, 0, 0], q=q))


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)

            # Initialize the robot
            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    -np.pi / 4,
                    0.04,
                    0.04,
                ]
            )
            qpos = self._episode_rng.normal(0, 0.02, (b, len(qpos))) + qpos
            qpos[:, -2:] = 0.04
            self.agent.robot.set_qpos(qpos)
            self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))

            initial_position = torch.tensor([0, 0, 10])  # Example coordinates
            initial_orientation = torch.tensor([0, 0, 0, 1])  # Example quaternion

            self.agent.robot.set_pose(sapien.Pose(p=initial_position, q=initial_orientation))

            rotation = [0.7071, 0, 0, 0.7071]
            #
            # self.center = actors.build_sphere(
            #     self.scene,
            #     radius=0.02,
            #     color=[1, 1, 1, 1],
            #     name="center",
            #     body_type="kinematic",
            #     add_collision=False,
            #     initial_pose=sapien.Pose(p=grasp_point_camera, q=[1, 0, 0, 0])
            # )
            # self.center2 = actors.build_sphere(
            #     self.scene,
            #     radius=0.02,
            #     color=[1, 1, 1, 1],
            #     name="center2",
            #     body_type="kinematic",
            #     add_collision=False,
            #     initial_pose=sapien.Pose(p=grasp_point, q=[1, 0, 0, 0])
            # )

        pass

    def evaluate(self):
        return {}

    def _get_obs_extra(self, info: Dict):
        return dict()

    def compute_dense_reward(self, obs: any, action: torch.Tensor, info: Dict):

        # Get the device from the action tensor
        reward=0

        return reward
    def get_chair_name_fhz(self):
        return name
    def get_direction_fhz(self):
        return direction