from gymnasium import Wrapper
from typing import Callable, Any, Optional
import numpy as np
import gymnasium as gym

from optax.schedules import Schedule, linear_schedule

class InitWrapper(Wrapper):
    def __init__(self, env: gym.Env, init_state: np.ndarray):
        """
        Wraps the environment to override its reset.
        
        Args:
            env: The original pendulum environment.
            init_params: The desired initial parameters as an array.
        """
        super().__init__(env)

        self._validated = False
        self._init_state = np.asarray(init_state, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        base = self.env.unwrapped

        if not self._validated:
            if not hasattr(base, "state"):
                raise AttributeError(
                    "InitWrapper requires env.unwrapped.state to exist."
                )
            base_state = np.asarray(base.state)

            if self._init_state.shape != base_state.shape:
                raise ValueError(
                    f"init_state shape {self._init_state.shape} does not match "
                    f"env.state shape {base_state.shape}"
                )

            original_state = base_state.copy()
            base.state = self._init_state.copy()
            test_obs = self._compute_obs(base)
            base.state = original_state

            if not self.observation_space.contains(test_obs):
                raise ValueError(
                    "init_state produces an observation outside observation_space."
                )

            self._validated = True

        base.state = self._init_state.copy()
        new_obs = self._compute_obs(base)

        info = dict(info)
        info["fixed_init_state"] = self._init_state.copy()
        return new_obs, info
        
    @staticmethod
    def _compute_obs(base_env: gym.Env):
        """
        Recompute observation in the same way the env does.
        """
        if hasattr(base_env, "_get_obs"):
            return base_env._get_obs()
        return np.asarray(base_env.state, dtype=np.float32)


class EpisodicParamWrapper(gym.Wrapper):
    """
    A generic wrapper that updates environment parameters at the start of
    each episode using a user-provided scheduler.

    Args:
        env: Base Gym environment
        scheduler_fn: Callable mapping episode_idx -> dict of parameters
        apply_fn: Callable (env, params) that applies parameters to env
        apply_before_reset: Whether parameters affect the reset distribution
    """

    def __init__(
        self,
        env: gym.Env,
        scheduler_fn: Callable[[int], dict[str, Any]],
        apply_fn: Callable[[gym.Env, dict[str, Any]], None],
        *,
        apply_before_reset: bool = True,
    ):
        super().__init__(env)
        self.scheduler_fn = scheduler_fn
        self.apply_fn = apply_fn
        self.apply_before_reset = apply_before_reset

        self.episode_idx = -1
        self.current_params: dict[str, Any] = {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None, ):
        self.episode_idx += 1

        # Sample parameters for this episode
        self.current_params = dict(self.scheduler_fn(self.episode_idx))
        base_env = self.env.unwrapped

        if self.apply_before_reset:
            self.apply_fn(base_env, self.current_params)

        obs, info = self.env.reset(seed=seed, options=options)

        if not self.apply_before_reset:
            self.apply_fn(base_env, self.current_params)

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
    
    def get_current_params(self) -> dict:
        if not self.current_params:
            raise RuntimeError(
                "Parameters not initialized yet. "
                "Call reset() at least once before evaluation."
            )
        return dict(self.current_params)


class EvalEnvFactory:
    """
    Factory that creates frozen evaluation environments
    from a training environment snapshot.
    """

    def __init__(
        self,
        make_env_fn: Callable[[], gym.Env],
        apply_fn: Callable[[gym.Env, dict], None],
        *,
        init_state: np.ndarray | None = None,
    ):
        """
        Args:
            make_env_fn: function that returns a fresh base env (gym.make)
            apply_fn: function(env, params) that applies frozen parameters
            init_state: optional fixed initial state for evaluation environment (see InitWrapper)
        """
        self.make_env_fn = make_env_fn
        self.apply_fn = apply_fn
        self.init_state = init_state

    def make(self, frozen_params: dict, *args) -> gym.Env:
        env = self.make_env_fn(*args)

        if self.init_state is not None:
            env = InitWrapper(env, self.init_state)

        self.apply_fn(env.unwrapped, frozen_params)
        return env


# Example usage:
if __name__ == "__main__":

    init_angle=0.0
    init_vel=0.0
    init_state = np.array([init_angle, init_vel])
    env = gym.make("Pendulum-v1")
    env = InitWrapper(env, init_state=init_state)

    torque_schedule = linear_schedule(
        init_value=2.0,
        end_value=0.5,
        transition_steps=10,   # reach min torque after 10 episodes
    )

    def scheduler_fn(ep_idx: int):
        return {"max_torque": float(torque_schedule(ep_idx))}

    def apply_fn(base_env: gym.Env, params: dict):
        base_env.max_torque = params["max_torque"]
        base_env.action_space.low[:] = -params["max_torque"]
        base_env.action_space.high[:] = params["max_torque"]

    env = EpisodicParamWrapper(env, scheduler_fn, apply_fn)

    class DummyAgent:
        def sample_actions(self, obs, temperature: float = 1.0):
            # ignore obs, just sample
            return env.action_space.sample()    

    def evaluate(agent, env: gym.Env, num_episodes: int):
        returns = []
        lengths = []

        for _ in range(num_episodes):
            obs, _ = env.reset()
            done = False
            ep_return = 0.0
            ep_len = 0

            while not done:
                action = agent.sample_actions(obs, temperature=0.0)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_return += reward
                ep_len += 1

            returns.append(ep_return)
            lengths.append(ep_len)

        return {
            "return": float(np.mean(returns)),
            "length": float(np.mean(lengths)),
        }

    eval_env_factory = EvalEnvFactory(
        make_env_fn=lambda: gym.make("Pendulum-v1"),
        apply_fn=apply_fn,
        init_state=init_state,
    )

    agent = DummyAgent()

    max_steps = 500
    eval_interval = 50
    eval_episodes = 3

    obs, info = env.reset(seed=42)
    print("Initial observation:", obs)
    
    episode_return = 0.0
    episode_len = 0

    for step in range(1, max_steps + 1):
        action = agent.sample_actions(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        episode_return += reward
        episode_len += 1

        if terminated or truncated:
            print(
                f"[train] episode {info['episode_idx']} | "
                f"return={episode_return:.1f}, "
                f"len={episode_len}, "
                f"max_torque={info['max_torque']:.3f}"
            )

            obs, _ = env.reset()
            episode_return = 0.0
            episode_len = 0

        if step % eval_interval == 0:
            frozen_params = env.get_current_params()
            eval_env = eval_env_factory.make(frozen_params)

            eval_stats = evaluate(agent, eval_env, eval_episodes)

            print(
                f"[eval @ step {step}] "
                f"return={eval_stats['return']:.1f}, "
                f"len={eval_stats['length']:.1f}, "
                f"max_torque={frozen_params['max_torque']:.3f}"
            )
