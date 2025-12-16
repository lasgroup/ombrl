"""
pendulum.py

Contains:
  - A general episodic parameter wrapper (EpisodicParamWrapper)
  - A custom PendulumEnv whose physical parameters can be changed
  - A linear Optax scheduler for episodic decay
  - A factory for a CRL Pendulum with decaying max_torque

Author: Klemens Iten (2025)
"""

import numpy as np
import gym
from gym import spaces
import optax
import jax.numpy as jnp
from jaxrl import wrappers


class EpisodicParamWrapper(gym.Wrapper):
    """
    A generic wrapper that updates environment parameters at the start of
    each episode using a user-provided scheduler + apply function.

    Arguments:
        env: the base Gym environment
        scheduler_fn: function mapping (episode_idx) -> dict of parameters
        apply_fn: function (env, param_dict) -> None, applies params to env
    """

    def __init__(self, env, scheduler_fn, apply_fn):
        super().__init__(env)
        self.scheduler_fn = scheduler_fn
        self.apply_fn = apply_fn
        self.episode_idx = -1
        self.current_params = {}

    def reset(self, **kwargs):
        self.episode_idx += 1

        # Compute episodic parameters
        self.current_params = self.scheduler_fn(self.episode_idx)

        # Apply them to the underlying environment
        self.apply_fn(self.env, self.current_params)

        obs, info = self.env.reset(**kwargs)
        info = dict(info)
        info.update(self.current_params)
        info["episode_idx"] = self.episode_idx
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info.update(self.current_params)
        info["episode_idx"] = self.episode_idx
        return obs, reward, terminated, truncated, info


def make_linear_decay_scheduler(p0: float, p_min: float, n_decay: int):
    """
    Creates an Optax linear schedule mapping episode index n -> decayed parameter.
    Returns a function scheduler(n) -> float.
    """
    linear_sched = optax.linear_schedule(
        init_value=1.0,
        end_value=0.0,
        transition_steps=n_decay,
    )

    def scheduler(n: int):
        frac = float(linear_sched(n))  # ∈ [0, 1]
        return p_min + (p0 - p_min) * frac

    return scheduler


class PendulumEnv(gym.Env):
    """
    Simple pendulum with modifiable physical parameters.  
    State: [cos(theta), sin(theta), omega]  
    Action: scalar torque in [-max_torque, max_torque].
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        mass=1.0,
        length=1.0,
        damping=0.0,
        gravity=9.81,
        max_torque=2.0,
        dt=0.05,
    ):
        super().__init__()

        self.mass = mass
        self.length = length
        self.damping = damping
        self.gravity = gravity
        self.max_torque = max_torque
        self.dt = dt

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        self.state = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        high = np.array([np.pi, 1.0])
        theta, omega = self.np_random.uniform(low=-high, high=high)
        self.state = np.array([theta, omega], dtype=np.float32)
        return self._get_obs(), {"theta": theta, "omega": omega}

    def step(self, action):
        u = np.clip(action[0], -1.0, 1.0) * self.max_torque

        theta, omega = self.state
        m = self.mass
        l = self.length
        g = self.gravity
        c = self.damping

        # dynamics
        theta_dot = omega
        omega_dot = (g / l) * np.sin(theta) + (u / (m * l * l)) - c * omega

        # integrate
        theta = theta + theta_dot * self.dt
        omega = omega + omega_dot * self.dt
        self.state = np.array([theta, omega], dtype=np.float32)

        obs = self._get_obs()
        reward = -(theta**2 + 0.1 * omega**2 + 0.001 * u**2)

        return obs, reward, False, False, {}

    def _get_obs(self):
        theta, omega = self.state
        return np.array([np.cos(theta), np.sin(theta), omega], dtype=np.float32)


def make_pendulum_crl_env(
    u0=2.0, u_min=0.5, n_decay=50, **pendulum_kwargs
):
    """
    Constructs a PendulumEnv with an episodic decay of max_torque using Optax.
    """
    sched = make_linear_decay_scheduler(u0, u_min, n_decay)

    def scheduler_fn(ep_idx: int):
        return {"max_torque": float(sched(ep_idx))}

    def apply_fn(env, params):
        env.max_torque = params["max_torque"]

    base_env = PendulumEnv(**pendulum_kwargs)
    return EpisodicParamWrapper(base_env, scheduler_fn, apply_fn)


if __name__ == "__main__":
    env = make_pendulum_crl_env()

    for ep in range(5):
        obs, info = env.reset()
        print(f"[Episode {ep}] max_torque = {info['max_torque']:.3f}")
        for t in range(3):
            env.step(env.action_space.sample())
