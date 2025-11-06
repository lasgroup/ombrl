import os
import random
import time

import numpy as np
from typing import List
from ombrl.utils.multiple_reward_wrapper import RewardFunction
from ombrl.utils.rewards import PendulumKeepDown, MountainCarGoLeft, CheetahRunBackwards, HopperHopBackwards, \
    WalkerWalkBackwards, ReacherKeepAway, PusherPushAway, PusherKeepAway, PusherGoAway # , DmCheetahRunBackwards


def get_dt(
        env_name: str,
) -> float:
    env_timesteps = {
        "quadruped": 0.005,
        "fish": 0.004,
        "cartpole": 0.01,
        "dog_v2": 0.005,
        "swimmer": 0.002,
        "cartpole_suite": 0.01,
        "test_model": 0.01,
        "arena": 0.01,
        "lqr": 0.03,
        "acrobot": 0.01,
        "composer_arena": 0.002,
        "hopper": 0.005,
        "stacker": 0.001,
        "humanoid": 0.005,
        "walker": 0.0025,
        "drosophila": 0.0001,
        "dog_base": 0.005,
        "pendulum": 0.02,
        "cheetah": 0.01,
        "dog": 0.005,
        "reacher": 0.02,
        "finger": 0.01,
        "fruitfly_v2": 0.0001,
        "cartpole_no_names": 0.01,
        "manipulator": 0.001,
        "humanoid_suite": 0.005,
        "point_mass": 0.02,
        "mountaincar": 1.0,
        "MountainCarContinuous": 1.0,
        "Pendulum": 0.05,
        "HalfCheetah": 0.05,
        "Hopper": 0.008,
        "Walker2d": 0.008,
        "Reacher": 0.02,
        "Swimmer": 0.04,
        "Pusher": 0.05,
        "Humanoid": 0.015,
    }
    if env_name in env_timesteps.keys():
        return env_timesteps.get(env_name)
    else:
        domain_name, task_name = env_name.split('-')
        if domain_name in env_timesteps.keys():
            return env_timesteps.get(domain_name)
        else:
            raise ValueError(f"Invalid environment: {domain_name}")


def get_rewards(
        env_name: str,
) -> List[RewardFunction]:
    env_rewards = {
        "MountainCarContinuous": [
            RewardFunction(0), 
            MountainCarGoLeft(1)
        ],
        "Pendulum": [
            PendulumKeepDown(-1), 
            RewardFunction(1, lambda obs, act, next_obs, r: -r),
            RewardFunction(2, lambda obs, act, next_obs, r: r)
        ],
        "HalfCheetah": [
            RewardFunction(0), 
            RewardFunction(1, lambda obs, act, next_obs, r: -r),
            CheetahRunBackwards(2)
        ],
        "Hopper": [
            RewardFunction(0), 
            RewardFunction(1, lambda obs, act, next_obs, r: -r),
            HopperHopBackwards(2)
        ],
        "Walker2d": [
            RewardFunction(0), 
            RewardFunction(1, lambda obs, act, next_obs, r: -r),
            WalkerWalkBackwards(2)
        ],
        "Pusher": [
            RewardFunction(0), 
            PusherPushAway(1),
            PusherKeepAway(2),
            PusherGoAway(3)
        ],
        "Reacher": [
            RewardFunction(0), 
            RewardFunction(1, lambda obs, act, next_obs, r: -r),
            ReacherKeepAway(2)
        ],
        "cartpole": [
            RewardFunction(0), 
            RewardFunction(1, lambda obs, act, next_obs, r: np.ones_like(r)),
            RewardFunction(1, lambda obs, act, next_obs, r: -r)
        ],
        # "cheetah": [
        #     RewardFunction(0), 
        #     RewardFunction(1, lambda obs, act, next_obs, r: -r),
        #     DmCheetahRunBackwards(2)
        # ],
    }
    if env_name in env_rewards.keys():
        return env_rewards.get(env_name)
    else:
        domain_name, task_name = env_name.split('-')
        if domain_name in env_rewards.keys():
            return env_rewards.get(domain_name)
        else:
            raise ValueError(f"Invalid environment: {domain_name}")
