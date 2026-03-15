import argparse

import gymnasium as gym
import numpy as np

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
import trimesh
import trimesh.scene

from mani_skill.utils import common
from sklearn.decomposition import PCA

import json
import string
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



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
    if args.seed is not None:
        np.random.seed(args.seed)
    sensor_configs = dict()
    if args.cam_width:
        sensor_configs["width"] = args.cam_width
    if args.cam_height:
        sensor_configs["height"] = args.cam_height
    env: BaseEnv = gym.make(
        args.env_id,
        obs_mode="pointcloud",
        reward_mode="dense",
        sensor_configs=sensor_configs,
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
            if segment[element]==17:
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

    data =np.array(xyz_selected)
    x, y, z = data[:, 0], data[:, 1], data[:, 2]  # 假设数据是 (N, 3)

    # 3D 散点图
    fig = plt.figure(figsize=(10, 5))

    # 3D 子图
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(x, y, z, s=0.1)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    max_range=x_range if x_range>y_range else y_range
    max_range=max_range/2

    ax1.set_title("3D Scatter Plot")
    ax1.set_xlim([np.mean(x)-max_range, np.mean(x)+max_range])
    ax1.set_ylim([np.mean(y)-max_range, np.mean(y)+max_range])

    # 2D 子图（只画 xy 轴）
    ax2 = fig.add_subplot(122)
    ax2.scatter(x, y, s=0.1)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title("XY Plane Scatter Plot")
    ax2.set_xlim([np.mean(x)-max_range, np.mean(x)+max_range])
    ax2.set_ylim([np.mean(y)-max_range, np.mean(y)+max_range])

    plt.show()


    # 计算 x, y, z 的取值范围
    print(f"x 取值范围: [{x.min()}, {x.max()}]")
    print(f"y 取值范围: [{y.min()}, {y.max()}]")
    print(f"z 取值范围: [{z.min()}, {z.max()}]")

if __name__ == "__main__":
    main(parse_args())
