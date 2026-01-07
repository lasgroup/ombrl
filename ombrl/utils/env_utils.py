import gymnasium.wrappers

from typing import Optional, Callable

import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.wrappers import RescaleAction
from gymnasium.wrappers.pixel_observation import PixelObservationWrapper

from jaxrl import wrappers
import numpy as np
import collections


class ActionScalar(gym.Wrapper):
    def __init__(self, env, scale_factor: float):
        super().__init__(env)
        assert scale_factor > 0
        self._scale_factor = scale_factor

    def step(self, action):
        act = action * self._scale_factor
        return super().step(act)


class ActionStacker(gym.Wrapper):

    def __init__(self,
                 env,
                 buffer_size: int,
                 step_delays: int | None = None,
                 store_action_buffer_in_obs: bool = True,
                 stack_axis=-1):
        super().__init__(env)
        if step_delays is None:
            step_delays = buffer_size
        assert step_delays <= buffer_size
        self._step_delays = step_delays
        self._buffer_size = buffer_size
        self._store_action_buffer_in_obs = store_action_buffer_in_obs
        self._stack_axis = stack_axis
        self._action_buffer = collections.deque([], maxlen=buffer_size)
        if self._store_action_buffer_in_obs:
            low = np.repeat(self.action_space.low, buffer_size, axis=stack_axis)
            high = np.repeat(self.action_space.high,
                             buffer_size,
                             axis=stack_axis)
            low = np.concatenate([self.observation_space.low, low], axis=self._stack_axis)
            high = np.concatenate([self.observation_space.high, high], axis=self._stack_axis)
            self.observation_space = Box(low=low, high=high)

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset()
        dummy_action = np.zeros_like(self.action_space.sample())
        for _ in range(self._buffer_size):
            self._action_buffer.append(dummy_action)
        return self._get_obs(obs), info

    def step(self, action):
        self._action_buffer.append(action)
        action_idx = 1 + self._step_delays
        applied_action = self._action_buffer[-action_idx]
        obs, reward, done, truncate, info = self.env.step(applied_action)
        return self._get_obs(obs), reward, done, truncate, info

    def _get_obs(self, obs: np.ndarray):
        assert len(self._action_buffer) == self._buffer_size
        if self._store_action_buffer_in_obs:
            act_obs = np.concatenate(list(self._action_buffer), axis=self._stack_axis)
            obs = np.concatenate([obs, act_obs], axis=self._stack_axis)
        return obs


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
