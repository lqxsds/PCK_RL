import argparse

import gymnasium as gym
import numpy as np

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.template import CustomEnv
from mani_skill.sensors.camera import CameraConfig
import trimesh
import trimesh.scene
import matplotlib.pyplot as plt
from mani_skill.utils import common
from sklearn.decomposition import PCA

import json
import string



def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="CustomEnv-v1", help="The environment ID of the task you want to simulate")
    parser.add_argument("--cam-width", type=int, help="Override the width of every camera in the environment")
    parser.add_argument("--cam-height", type=int, help="Override the height of every camera in the environment")
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="Seed the random actions and environment. Default is no seed",
    )
    args = parser.parse_args()
    return args


def main(args):
    # if args.seed is not None:
    #     np.random.seed(args.seed)
    # sensor_configs = dict()
    # if args.cam_width:
    #     sensor_configs["width"] = args.cam_width
    # if args.cam_height:
    #     sensor_configs["height"] = args.cam_height
    env: CustomEnv = gym.make(
        args.env_id,
        obs_mode="pointcloud",
        reward_mode="dense",
    )

    obs, _ = env.reset(seed=args.seed)
    #while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    xyz = common.to_numpy(obs["pointcloud"]["xyzw"][0, ..., :3])
    colors = common.to_numpy(obs["pointcloud"]["rgb"][0])
    segment = common.to_numpy(obs["pointcloud"]["segmentation"][0])
    xyz_selected=[]
    colors_selected=[]

    for element in range(len(segment)):
            if segment[element]==16:
                xyz_selected.append(xyz[element])
                colors_selected.append(colors[element])




    env.close()

    # pcd = trimesh.points.PointCloud(xyz_selected, colors_selected)
    # for uid, config in env.unwrapped._sensor_configs.items():
    #     test = uid
    #     print(test)
    #     if isinstance(config, CameraConfig) and uid == 'base_camera':
    #         cam2world = obs["sensor_param"][uid]["cam2world_gl"][0]
    #         camera = trimesh.scene.Camera(uid, (1024, 1024), fov=(np.rad2deg(config.fov), np.rad2deg(config.fov)))
    #
    # axis = trimesh.creation.axis()
    # print("-------------------------------------------------")
    # print("-------------------------------------------------")
    # print("-------------------------------------------------")
    # print("this is pointcloud from all cameras.close the window to write as numpy")
    # print("-------------------------------------------------")
    # print("-------------------------------------------------")
    # print("-------------------------------------------------")
    # trimesh.Scene([pcd, axis], camera=camera, camera_transform=cam2world).show()



    xyz_selected_np = np.array(xyz_selected)
    np.save(f"peg_pointcloud_full.npy", xyz_selected_np)
    print("write successful")



    # data=xyz_selected_np
    # # 提取 X, Y, Z 坐标
    # x, y, z = data[:, 0], data[:, 1], data[:, 2]
    #
    # # 创建 3D 图形
    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(111, projection='3d')
    #
    # # 绘制点云
    # ax.scatter(x, y, z, s=1, c=z, cmap='viridis', alpha=0.8)
    #
    # # 设置坐标轴标签
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    # ax.set_title("3D Point Cloud Visualization")
    #
    # # 显示图像
    # plt.show()


if __name__ == "__main__":
    main(parse_args())
