import gymnasium as gym
import numpy as np
import sapien

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers import RecordEpisode


import tyro
from dataclasses import dataclass
from typing import List, Optional, Annotated, Union
import cv2
import numpy as np
@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "PickCube-v1"
    """The environment ID of the task you want to simulate"""

    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "rgbd"
    """Observation mode"""

    robot_uids: Annotated[Optional[str], tyro.conf.arg(aliases=["-r"])] = None
    """Robot UID(s) to use. Can be a comma separated list of UIDs or empty string to have no agents. If not given then defaults to the environments default robot"""

    sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "auto"
    """Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'"""

    reward_mode: Optional[str] = None
    """Reward mode"""

    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    """Number of environments to run."""

    control_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-c"])] = None
    """Control mode"""

    render_mode: str = "human"
    """Render mode"""

    shader: str = "default"
    """Change shader used for all cameras in the environment for rendering. Default is 'minimal' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer"""

    record_dir: Optional[str] = None
    """Directory to save recordings"""

    pause: Annotated[bool, tyro.conf.arg(aliases=["-p"])] = False
    """If using human render mode, auto pauses the simulation upon loading"""

    quiet: bool = False
    """Disable verbose output."""

    seed: Annotated[Optional[Union[int, List[int]]], tyro.conf.arg(aliases=["-s"])] = None
    """Seed(s) for random actions and simulator. Can be a single integer or a list of integers. Default is None (no seeds)"""

def main(args: Args):
    np.set_printoptions(suppress=True, precision=3)
    verbose = not args.quiet
    if isinstance(args.seed, int):
        args.seed = [args.seed]
    if args.seed is not None:
        np.random.seed(args.seed[0])
    parallel_in_single_scene = args.render_mode == "human"
    if args.render_mode == "human" and args.obs_mode in ["sensor_data", "rgb", "rgbd", "depth", "point_cloud"]:
        print("Disabling parallel single scene/GUI render as observation mode is a visual one. Change observation mode to state or state_dict to see a parallel env render")
        parallel_in_single_scene = False
    if args.render_mode == "human" and args.num_envs == 1:
        parallel_in_single_scene = False
    env_kwargs = dict(
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        num_envs=args.num_envs,
        sim_backend=args.sim_backend,
        enable_shadow=True,
        parallel_in_single_scene=parallel_in_single_scene,
    )
    if args.robot_uids is not None:
        env_kwargs["robot_uids"] = tuple(args.robot_uids.split(","))
    env: BaseEnv = gym.make(
        args.env_id,
        **env_kwargs
    )
    record_dir = args.record_dir
    if record_dir:
        record_dir = record_dir.format(env_id=args.env_id)
        env = RecordEpisode(env, record_dir, info_on_video=False, save_trajectory=False, max_steps_per_video=gym_utils.find_max_episode_steps_value(env))

    if verbose:
        print("Observation space", env.observation_space)
        print("Action space", env.action_space)
        if env.unwrapped.agent is not None:
            print("Control mode", env.unwrapped.control_mode)
        print("Reward mode", env.unwrapped.reward_mode)

    obs, _ = env.reset(seed=args.seed, options=dict(reconfigure=True))
    if args.seed is not None and env.action_space is not None:
            env.action_space.seed(args.seed[0])
    if args.render_mode is not None:
        viewer = env.render()
        if isinstance(viewer, sapien.utils.Viewer):
            viewer.paused = args.pause
        env.render()
    step_count=0
    while True:
        action = env.action_space.sample() if env.action_space is not None else None
        obs, reward, terminated, truncated, info = env.step(action)
        # ----- 处理 hand_camera 的 RGB 和深度图 -----
        if isinstance(obs, dict) and 'sensor_data' in obs:
            sensor_data = obs['sensor_data']
            if 'hand_camera' in sensor_data:
                cam_data = sensor_data['hand_camera']
                rgb_tensor = cam_data['rgb']
                depth_tensor = cam_data['depth']

                # 转换为 numpy 数组（兼容 PyTorch / TensorFlow / numpy）
                if hasattr(rgb_tensor, 'detach'):  # PyTorch
                    rgb_img = rgb_tensor.detach().cpu().numpy()
                    depth_img = depth_tensor.detach().cpu().numpy()
                else:
                    rgb_img = np.array(rgb_tensor)
                    depth_img = np.array(depth_tensor)

                # 移除批次维度（如果存在且为1）
                if rgb_img.ndim == 4 and rgb_img.shape[0] == 1:
                    rgb_img = rgb_img[0]  # 变成 (128, 128, 3)
                if depth_img.ndim == 4 and depth_img.shape[0] == 1:
                    depth_img = depth_img[0]  # 变成 (128, 128, 1)

                # 处理 RGB 图像
                if rgb_img.ndim == 3 and rgb_img.shape[-1] == 3:
                    # 值范围处理
                    if rgb_img.max() <= 1.0:
                        rgb_img = (rgb_img * 255).astype(np.uint8)
                    else:
                        rgb_img = rgb_img.astype(np.uint8)
                    rgb_img_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

                    # 显示
                    cv2.imshow("Hand Camera RGB", rgb_img_bgr)
                else:
                    print(f"Unexpected RGB shape after processing: {rgb_img.shape}")
                    rgb_img_bgr = None

                # 处理深度图像
                if depth_img.ndim == 3 and depth_img.shape[-1] == 1:
                    # 压缩最后一维，变成 (128, 128)
                    depth_img = np.squeeze(depth_img, axis=-1)
                if depth_img.ndim == 2:
                    # 归一化到 [0,255] 用于显示
                    depth_display = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    cv2.imshow("Hand Camera Depth", depth_display)

                    # 保存深度图为 16 位 PNG（假设单位为米，转换为毫米）
                    depth_mm = (depth_img * 1000).astype(np.uint16)
                    cv2.imwrite(f"hand_depth_{step_count:04d}.png", depth_mm)
                    # 可选：保存原始 numpy 数组
                    # np.save(f"hand_depth_raw_{step_count:04d}.npy", depth_img)
                else:
                    print(f"Unexpected depth shape after processing: {depth_img.shape}")

                # 保存 RGB 图像
                if rgb_img_bgr is not None:
                    cv2.imwrite(f"hand_rgb_{step_count:04d}.png", rgb_img_bgr)

                cv2.waitKey(1)  # 更新窗口
                step_count += 1
        # ----- 处理结束 -----
        if verbose:
            print("reward", reward)
            print("terminated", terminated)
            print("truncated", truncated)
            print("info", info)
        if args.render_mode is not None:
            env.render()
            # viewer.paused=True
        if args.render_mode is None or args.render_mode != "human":
            if (terminated | truncated).any():
                break
    env.close()

    if record_dir:
        print(f"Saving video to {record_dir}")


if __name__ == "__main__":
    parsed_args = tyro.cli(Args)
    main(parsed_args)
