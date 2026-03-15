import argparse

import gymnasium as gym
import numpy as np
from sklearn.metrics import d2_pinball_score

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
import trimesh
import trimesh.scene

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
    with open('hammer_grasp.json', 'r', encoding='utf-8') as file:
        grasppoint = json.load(file)

    for element in range(len(segment)):
            if segment[element]==16:
                if np.linalg.norm(xyz[element]-grasppoint)<0.1:
                    xyz_selected.append(xyz[element])
                    colors_selected.append(colors[element])


    env.close()

    pcd = trimesh.points.PointCloud(xyz_selected, colors_selected)
    for uid, config in env.unwrapped._sensor_configs.items():
        test = uid
        print(test)
        if isinstance(config, CameraConfig) and uid == 'base_camera':
            cam2world = obs["sensor_param"][uid]["cam2world_gl"][0]
            camera = trimesh.scene.Camera(uid, (1024, 1024), fov=(np.rad2deg(config.fov), np.rad2deg(config.fov)))

    #axis = trimesh.creation.axis()
    print("-------------------------------------------------")
    print("-------------------------------------------------")
    print("-------------------------------------------------")
    print("close the window to write as numpy")
    print("-------------------------------------------------")
    print("-------------------------------------------------")
    print("-------------------------------------------------")
    #trimesh.Scene([pcd, axis], camera=camera, camera_transform=cam2world).show()
    trimesh.Scene([pcd], camera=camera, camera_transform=cam2world).show()

    name=env.get_chair_name_fhz()
    direction=env.get_direction_fhz()
    xyz_selected_np = np.array(xyz_selected)
    np.save(f"{name}_pointcloud_{direction}.npy", xyz_selected_np)
    print("write successful")



if __name__ == "__main__":
    main(parse_args())
