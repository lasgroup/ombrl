from gymnasium import Wrapper
from typing import Optional
import numpy as np
import gymnasium as gym

class PendulumInitWrapper(Wrapper):
    def __init__(self, env, init_angle: float = np.pi, init_vel: float = 0.0):
        """
        Wraps the pendulum environment to override its reset.
        
        Args:
            env: The original pendulum environment.
            init_angle: The desired initial angle (in radians).
            init_vel: The desired initial angular velocity.
        """
        super().__init__(env)
        self.init_angle = init_angle
        self.init_vel = init_vel

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)

        self.env.unwrapped.state = np.array([self.init_angle, self.init_vel])

        new_obs = self.env.unwrapped._get_obs()
        return new_obs, info


# Example usage:
if __name__ == "__main__":
    env = gym.make("Pendulum-v1")
    env = PendulumInitWrapper(env, init_angle=0.0, init_vel=0.0)
    
    obs, info = env.reset(seed=42)
    print("Initial observation:", obs)
    
    n_steps = 5
    for step in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\nStep {step + 1}:")
        print(" Action:", action)
        print(" Observation:", obs)
        print(" Reward:", reward)
        print(" Terminated:", terminated)
        print(" Truncated:", truncated)