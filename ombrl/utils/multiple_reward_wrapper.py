import numpy as np
from typing import List, Callable
from jaxrl.wrappers.dmc_env import DMCEnv

class RewardFunction:
    def __init__(self, reward_index=0, custom_reward_fn: Callable = None):
        self.reward_index = reward_index
        if custom_reward_fn:
            self._get_reward = custom_reward_fn
    
    def __call__(self, observation, action, next_observation, reward):
        if self.reward_index == 0:
            return reward
        else:
            return self._get_reward(observation, action, next_observation, reward)

    def _get_reward(self, observation, action, next_observation, reward):
        raise NotImplementedError(f'Reward function not set for reward_index: {self.reward_index}')
    
RewardLike = Callable | RewardFunction

class DmRewardFunction(RewardFunction):    
    def __call__(self, observation, action, next_observation, reward, env: DMCEnv):
        if self.reward_index == 0:
            return reward
        else:
            return self._get_reward(observation, action, next_observation, reward, env)
        
    def _get_reward(self, observation, action, next_observation, reward, env: DMCEnv = None):
        assert env is not None
        raise NotImplementedError(f'Reward function not set for reward_index: {self.reward_index}')


if __name__ == '__main__':
    observation = np.array(1)
    action = np.array(0.1)
    next_observation = np.array(2)
    reward = np.array(3)

    custom_reward_1 = lambda obs, act, next_obs, r: -r

    class CustomReward2(RewardFunction):
        def _get_reward(self, observation, action, next_observation, reward):
            return reward * 2

    reward_list: List[RewardFunction] = [
        RewardFunction(0),
        RewardFunction(1, custom_reward_1),
        CustomReward2(2)
    ]
    num_rewards = 3

    for reward_fn in reward_list:
        print(reward_fn(observation, action, next_observation, reward))

    print("Done")
