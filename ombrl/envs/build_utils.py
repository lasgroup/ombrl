from typing import Callable, Optional
import gymnasium as gym
import gymnasium.wrappers
from gymnasium.wrappers import RescaleAction
from gymnasium.wrappers.pixel_observation import PixelObservationWrapper
from jaxrl import wrappers

def make_humanoid_bench_env(
        env_name: str,
        seed: int,
        save_folder: Optional[str] = None,
        add_episode_monitor: bool = True,
        action_repeat: int = 1,
        action_cost: float = 0.0,
        frame_stack: int = 1,
        from_pixels: bool = False,
        pixels_only: bool = True,
        image_size: int = 84,
        sticky: bool = False,
        gray_scale: bool = False,
        flatten: bool = True,
        recording_image_size: Optional[int] = None,
        episode_trigger: Callable[[int], bool] = None,
):
    import humanoid_bench
    downscale_image = False
    if from_pixels:
        camera_id = 0
        if recording_image_size is not None and save_folder is not None:
            size = recording_image_size
            downscale_image = True
        else:
            size = image_size
        render_kwargs = {
            'height': size,
            'width': size,
            'camera_id': camera_id,
            'render_mode': 'rgb_array'
        }
    else:
        if recording_image_size is not None and save_folder:
            render_kwargs = {
                'width': recording_image_size,
                'height': recording_image_size,
                'render_mode': 'rgb_array'
            }
        else:
            render_kwargs = {'render_mode': 'rgb_array'}
    env = gym.make(env_name, **render_kwargs)

    if flatten and isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)

    if add_episode_monitor:
        env = wrappers.EpisodeMonitor(env)

    if action_repeat > 1:
        env = wrappers.RepeatAction(env, action_repeat)

    env = wrappers.ActionCost(env, action_cost=action_cost)
    env = RescaleAction(env, -1.0, 1.0)

    if save_folder is not None:
        env = gymnasium.wrappers.RecordVideo(env, save_folder, episode_trigger=episode_trigger)

    if from_pixels:
        env = PixelObservationWrapper(env,
                                      pixels_only=pixels_only)
        env = wrappers.TakeKey(env, take_key='pixels')
        if downscale_image:
            env = gymnasium.wrappers.ResizeObservation(env, shape=image_size)
        if gray_scale:
            env = wrappers.RGB2Gray(env)
    else:
        env = wrappers.SinglePrecision(env)

    if frame_stack > 1:
        env = wrappers.FrameStack(env, num_stack=frame_stack)

    if sticky:
        env = wrappers.StickyActionEnv(env)

    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env


def make_metaworld_env(
        env_name: str,
        seed: int,
        save_folder: Optional[str] = None,
        add_episode_monitor: bool = True,
        action_repeat: int = 1,
        action_cost: float = 0.0,
        frame_stack: int = 1,
        from_pixels: bool = False,
        pixels_only: bool = True,
        image_size: int = 84,
        sticky: bool = False,
        gray_scale: bool = False,
        flatten: bool = True,
        time_limit: int = 200,
        recording_image_size: int = 1024,
):
    from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE)
    assert not from_pixels, "currently only works for state based tasks."
    render_kwargs = {}
    constructor = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name]
    env = constructor(seed=seed)
    env = gymnasium.wrappers.TimeLimit(env, max_episode_steps=time_limit)

    if flatten and isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)

    if add_episode_monitor:
        env = wrappers.EpisodeMonitor(env)

    if action_repeat > 1:
        env = wrappers.RepeatAction(env, action_repeat)

    env = wrappers.ActionCost(env, action_cost=action_cost)
    env = RescaleAction(env, -1.0, 1.0)

    if save_folder is not None:
        env = gym.wrappers.RecordVideo(env, save_folder)

    if from_pixels:
        env = PixelObservationWrapper(env,
                                      pixels_only=pixels_only)
        env = wrappers.TakeKey(env, take_key='pixels')
        if gray_scale:
            env = wrappers.RGB2Gray(env)
    else:
        env = wrappers.SinglePrecision(env)

    if frame_stack > 1:
        env = wrappers.FrameStack(env, num_stack=frame_stack)

    if sticky:
        env = wrappers.StickyActionEnv(env)

    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env


def make_hopper_env(env_name: str,
             seed: int,
             save_folder: Optional[str] = None,
             add_episode_monitor: bool = True,
             action_repeat: int = 1,
             action_cost: float = 0.0,
             frame_stack: int = 1,
             from_pixels: bool = False,
             pixels_only: bool = True,
             image_size: int = 84,
             sticky: bool = False,
             gray_scale: bool = False,
             flatten: bool = True,
             recording_image_size: Optional[int] = None,
             episode_trigger: Optional[Callable[[int], bool]] = None,
             ) -> gym.Env:
    # Check if the env is in gym.
    all_envs = gym.envs.registry.values()
    env_ids = [env_spec.id for env_spec in all_envs]
    downscale_image = False
    if from_pixels:
        if env_name in env_ids:
            camera_id = 0
        else:
            domain_name, task_name = env_name.split('-')
            camera_id = 2 if domain_name == 'quadruped' else 0

        if recording_image_size is not None and save_folder is not None:
            size = recording_image_size
            downscale_image = True
        else:
            size = image_size
        render_kwargs = {
            'height': size,
            'width': size,
            'camera_id': camera_id
        }
    else:
        if env_name in env_ids:
            render_kwargs = {'render_mode': 'rgb_array'}
        else:
            render_kwargs = {}

        if recording_image_size is not None and save_folder is not None:
            render_kwargs['height'] = recording_image_size
            render_kwargs['width'] = recording_image_size

    if env_name in env_ids:
        env = gym.make(env_name, terminate_when_unhealthy=False, **render_kwargs)
    else:
        domain_name, task_name = env_name.split('-')
        env = wrappers.DMCEnv(domain_name=domain_name,
                              task_name=task_name,
                              task_kwargs={'random': seed},
                              **render_kwargs)

    if flatten and isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)

    env = wrappers.ActionCost(env, action_cost=action_cost)

    if add_episode_monitor:
        env = wrappers.EpisodeMonitor(env)

    if action_repeat > 1:
        env = wrappers.RepeatAction(env, action_repeat)

    env = RescaleAction(env, -1.0, 1.0)

    if save_folder is not None:
        env = gymnasium.wrappers.RecordVideo(env, save_folder, episode_trigger=episode_trigger)

    if from_pixels:
        env = PixelObservationWrapper(env,
                                      pixels_only=pixels_only)
        env = wrappers.TakeKey(env, take_key='pixels')
        if downscale_image:
            env = gymnasium.wrappers.ResizeObservation(env, shape=image_size)
        if gray_scale:
            env = wrappers.RGB2Gray(env)
    else:
        env = wrappers.SinglePrecision(env)

    if frame_stack > 1:
        env = wrappers.FrameStack(env, num_stack=frame_stack)

    if sticky:
        env = wrappers.StickyActionEnv(env)

    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env
