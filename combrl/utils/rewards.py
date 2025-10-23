import numpy as np
import jax.numpy as jnp

from typing import List, Callable
from dm_control.utils import rewards
from combrl.utils.multiple_reward_wrapper import RewardFunction, DmRewardFunction
from jaxrl.wrappers import DMCEnv


def angle_normalize(x):
    # From: https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py#L270
    return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi


class PendulumKeepDown(RewardFunction):
    def _get_reward(self, observation, action, next_observation, reward):
        reward_array = jnp.atleast_1d(reward)
        
        theta = jnp.arctan2(observation[..., 1], observation[..., 0])
        theta_dot = observation[..., 2]
        
        cost_angle = jnp.square(angle_normalize(theta - jnp.pi))
        cost_ang_vel = 0.1 * jnp.square(theta_dot)
        cost_ctrl = 0.001 * jnp.sum(jnp.square(action), axis=-1)
        cost = cost_angle + cost_ang_vel + cost_ctrl
        
        new_reward = jnp.reshape(-cost, reward_array.shape)
        if reward_array.shape == (1,):
            return new_reward[0]
        return new_reward


class MountainCarGoLeft(RewardFunction):
    def _get_reward(self, observation, action, next_observation, reward):
        pos = jnp.atleast_1d(next_observation[..., 0])
        velocity = jnp.atleast_1d(next_observation[..., 1])
        action = jnp.atleast_1d(action)
        reward_array = jnp.atleast_1d(reward)
        terminate = jnp.logical_and(pos <= -1.65, velocity <= 0)
        
        new_reward = - (action[..., 0] ** 2) * 0.1 + 100 * terminate
        new_reward = jnp.reshape(new_reward, reward_array.shape)
        if reward_array.shape == (1,):
            return new_reward[0]
        return new_reward


class CheetahRunBackwards(RewardFunction):
    def _get_reward(self, observation, action, next_observation, reward):
        # Ensure reward is an array.
        reward_array = jnp.atleast_1d(reward)
        
        ctrl_cost_weight = 0.1
        ctrl_cost = ctrl_cost_weight * jnp.sum(jnp.square(action), axis=-1)
        
        # default reward is:
        #   reward = (x_after - x_before)/dt - ctrl_cost
        # so that (x_after - x_before)/dt = reward + ctrl_cost.
        # For backward running we want a forward reward of:
        #   new_forward_reward = - (x_after - x_before)/dt = - (reward + ctrl_cost)
        new_forward_reward = - (reward + ctrl_cost) - ctrl_cost

        new_reward = jnp.reshape(new_forward_reward, reward_array.shape)
        if reward_array.shape == (1,):
            return new_reward[0]
        return new_reward


class HopperHopBackwards(RewardFunction):
    def _get_reward(self, observation, action, next_observation, reward):
        # Ensure reward is an array.
        reward_array = jnp.atleast_1d(reward)
        
        ctrl_cost_weight = 0.001
        ctrl_cost = ctrl_cost_weight * jnp.sum(jnp.square(action), axis=-1)
        
        # Compute healthy_reward based on next_observation.
        # - Healthy z: next_observation[..., 0] >= 0.7
        # - Healthy angle: next_observation[..., 1] ∈ [-0.2, 0.2]
        # - Healthy state: all(next_observation[..., 1:] ∈ [-100, 100])
        is_healthy_state = jnp.all((next_observation[..., 1:] >= -100.0) & (next_observation[..., 1:] <= 100.0), axis=-1)
        is_healthy_z = next_observation[..., 0] >= 0.7
        is_healthy_angle = (next_observation[..., 1] >= -0.2) & (next_observation[..., 1] <= 0.2)
        healthy = is_healthy_state & is_healthy_z & is_healthy_angle
        healthy_reward = jnp.where(healthy, 1.0, 0.0)
        
        # default reward is:
        #   reward = healthy_reward + forward_reward - ctrl_cost,
        # where forward_reward = (x_after - x_before)/dt.
        # 
        # Some math to compute backward reward:
        # new_forward_reward = - (x_after - x_before)/dt = - (reward - healthy_reward + ctrl_cost).
        #
        #   new_reward = healthy_reward + new_forward_reward - ctrl_cost
        #              = 2 * healthy_reward - reward - 2 * ctrl_cost.
        new_reward = 2.0 * healthy_reward - reward - 2.0 * ctrl_cost

        new_reward = jnp.reshape(new_reward, reward_array.shape)
        if reward_array.shape == (1,):
            return new_reward[0]
        return new_reward
    

class WalkerWalkBackwards(RewardFunction):
    def _get_reward(self, observation, action, next_observation, reward):
        reward_array = jnp.atleast_1d(reward)
        
        ctrl_cost_weight = 1e-3
        ctrl_cost = ctrl_cost_weight * jnp.sum(jnp.square(action), axis=-1)
        
        # Compute healthy_reward based on next_observation.
        #   index 0: torso z (height)
        #   index 1: torso angle
        #
        # Healthy if:
        #   z-coordinate is in (0.8, 2.0) and
        #   |torso angle| < 1.0.
        
        is_healthy_state = jnp.all(jnp.isfinite(next_observation), axis=-1)
        is_healthy_z = (next_observation[..., 0] > 0.8) & (next_observation[..., 0] < 2.0)
        is_healthy_angle = (next_observation[..., 1] >= -1.0) & (next_observation[..., 1] <= 1.0)
        healthy = is_healthy_state & is_healthy_z & is_healthy_angle
        healthy_reward = jnp.where(healthy, 1.0, 0.0)
        
        # default reward is:
        #   reward = healthy_reward + forward_reward - ctrl_cost,
        # where forward_reward = (x_after - x_before)/dt.
        # 
        # Some math to compute backward reward:
        # new_forward_reward = - (x_after - x_before)/dt = - (reward - healthy_reward + ctrl_cost).
        #
        #   new_reward = healthy_reward + new_forward_reward - ctrl_cost
        #              = healthy_reward - (reward - healthy_reward + ctrl_cost) - ctrl_cost
        #              = 2 * healthy_reward - reward - 2 * ctrl_cost
        new_reward = 2.0 * healthy_reward - reward - 2.0 * ctrl_cost

        new_reward = jnp.reshape(new_reward, reward_array.shape)
        if reward_array.shape == (1,):
            return new_reward[0]
        return new_reward
    

class ReacherKeepAway(RewardFunction):
    def _get_reward(self, observation, action, next_observation, reward):
        reward_array = jnp.atleast_1d(reward)
        
        ctrl_cost = jnp.sum(jnp.square(action), axis=-1)
        
        # Original reward:
        #    reward = -d - ctrl_cost
        # where d = ||fingertip - target||.
        # New reward:
        #    new_reward = d - ctrl_cost.
        #    new_reward = (-reward - ctrl_cost) - ctrl_cost = -reward - 2 * ctrl_cost.
        new_reward = -reward - 2.0 * ctrl_cost
        
        new_reward = jnp.reshape(new_reward, reward_array.shape)
        if reward_array.shape == (1,):
            return new_reward[0]
        return new_reward
    

class PusherKeepAway(RewardFunction):
    """This reward encourages pushing the object away, and keeping the fingertip away from the object"""
    def _get_reward(self, observation, action, next_observation, reward):
        reward_array = jnp.atleast_1d(reward)
        
        ctrl_cost = jnp.sum(jnp.square(action), axis=-1)
        
        # Original reward:
        #   reward = - d_goal - 0.1 * ctrl_cost - 0.5 * d_ft
        # where d_goal = ||object - goal|| and d_ft = ||fingertip - object||.
        #
        # New reward:
        #   new_reward = d_goal + 0.5 * d_ft - 0.1 * ctrl_cost.
        new_reward = -reward - 0.2 * ctrl_cost
        
        new_reward = jnp.reshape(new_reward, reward_array.shape)
        if reward_array.shape == (1,):
            return new_reward[0]
        return new_reward
    

class PusherPushAway(RewardFunction):
    """This reward encourages pushing the object away, keeping the fingertip near the object"""
    def _get_reward(self, observation, action, next_observation, reward):
        reward_array = jnp.atleast_1d(reward)
        
        
        # d = ||object - goal||.
        # using observation, indices:
        #   17-19: object position, and 20-22: goal position.
        d_goal = jnp.linalg.norm(next_observation[..., 17:20] - next_observation[..., 20:23], axis=-1)
        
        # Original reward:
        #   reward = - d_goal - 0.1 * ctrl_cost - 0.5 * d_ft
        # where d_goal = ||object - goal|| and d_ft = ||fingertip - object||.
        #
        # New reward: 
        #   new_reward = d_goal - 0.5 * d_ft - 0.1 * ctrl_cost.
        new_reward = reward + 2.0 * d_goal
        
        new_reward = jnp.reshape(new_reward, reward_array.shape)
        if reward_array.shape == (1,):
            return new_reward[0]
        return new_reward


class PusherGoAway(RewardFunction):
    """This reward encourages keeping the fingertip AWAY from the object"""
    def _get_reward(self, observation, action, next_observation, reward):
        reward_array = jnp.atleast_1d(reward)
        
        
        # d = ||object - goal||.
        # using observation, indices:
        #   17-19: object position, and 14-16: fingertip position.
        d = jnp.linalg.norm(next_observation[..., 14:17] - next_observation[..., 17:20], axis=-1)        
        ctrl_cost = jnp.sum(jnp.square(action), axis=-1)

        new_reward = 1.0 * d - 0.1 * ctrl_cost
        
        new_reward = jnp.reshape(new_reward, reward_array.shape)
        if reward_array.shape == (1,):
            return new_reward[0]
        return new_reward
    

class DmCheetahRunBackwards(DmRewardFunction):
    def _get_reward(self, observation, action, next_observation, reward, env: DMCEnv):
        # Ensure reward is an array.
        reward_array = jnp.atleast_1d(reward)
        
        
        # default reward is:
        #   reward = (x_after - x_before)/dt - ctrl_cost
        # so that (x_after - x_before)/dt = reward + ctrl_cost.
        # For backward running we want a forward reward of:
        #   new_forward_reward = - (x_after - x_before)/dt = - (reward + ctrl_cost)
        if reward_array.shape == (1,):
            return rewards.tolerance(env.unwrapped._env._physics.speed(),
                             bounds=(-float('inf'), -10),
                             margin=10,
                             value_at_margin=0,
                             sigmoid='linear')
        raise NotImplementedError() # return new_reward


if __name__ == '__main__':
    observation = np.array([-1.0, 0.0])
    action = np.array(0.1)
    next_observation = np.array([-1.15, -0.05])    
    reward = np.array(0)

    custom_reward_1 = lambda obs, act, next_obs, r: -r

    class CustomReward2(RewardFunction):
        def _get_reward(self, observation, action, next_observation, reward):
            return reward * 2

    reward_list: List[RewardFunction] = [
        RewardFunction(0),
        MountainCarGoLeft(1)
    ]
    num_rewards = len(reward_list)

    for reward_fn in reward_list:
        print(reward_fn(observation, action, next_observation, reward))

    batch_observations = np.array([
        [-1.0, 0.0],
        [-1.0, 0.0],
        [-1.0, 0.0]
    ])
    batch_actions = np.array([
        [0.1],
        [0.2],
        [0.1]
    ])
    batch_next_observations = np.array([
        [-1.19, -0.05],
        [-1.20, -0.04],
        [-1.18, 0.06]
    ])
    batch_rewards = np.array([0.0, 0.0, 0.0])

    for reward_fn in reward_list:
        print(reward_fn(batch_observations, batch_actions, batch_next_observations, batch_rewards))

    print("Done")
