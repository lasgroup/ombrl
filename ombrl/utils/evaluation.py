from typing import Dict, List

import gymnasium as gym
import numpy as np
import copy

from ombrl.agents import CombrlExplorerLearner
from ombrl.utils.multiple_reward_wrapper import RewardFunction
from ombrl.utils.rewards import MountainCarGoLeft
from jaxrl.wrappers import dmc_env
"""
Adapted from: https://github.com/ikostrikov/jaxrl/blob/main/jaxrl/evaluation.py
"""

def evaluate(agent: CombrlExplorerLearner, env: gym.Env, num_episodes: int, reward_list: List[RewardFunction] = None) -> Dict[str, float]:
    stats = {'return': [], 'length': []}

    if not reward_list:
        reward_list = [RewardFunction(0)] # Reward is identity

    eval_stats = {key: copy.deepcopy(stats) for key in range(len(reward_list))}

    for reward_index, reward_function in enumerate(reward_list):
        successes = None
        eval_stats[reward_index]['eval_return'] = []
        for _ in range(num_episodes):
            observation, _ = env.reset()
            reached_goal = False # HACK for MountainCar
            finish = False
            episode_return = 0.0
            while not finish:
                action = agent.sample_actions(observation, temperature=0.0, reward_index=reward_index)
                next_observation, reward, done, truncate, info = env.step(action)
                reward = reward_function(observation, action, next_observation, reward)
                if not reached_goal:
                    episode_return += reward * agent.action_repeat
                    if isinstance(reward_function, MountainCarGoLeft):
                        # HACK: Hack for MountainCar
                        if episode_return > 50.:
                            reached_goal = True
                observation = next_observation
                finish = done or truncate
            if 'episode' in info:
                for k in stats.keys():
                    eval_stats[reward_index][k].append(info['episode'][k])
                eval_stats[reward_index]['eval_return'].append(episode_return)

            if 'is_success' in info:
                if successes is None:
                    successes = 0.0
                successes += info['is_success']

        for k, v in eval_stats[reward_index].items():
            eval_stats[reward_index][k] = np.mean(v)

        if successes is not None:
            eval_stats[reward_index]['success'] = successes / num_episodes
    return eval_stats
