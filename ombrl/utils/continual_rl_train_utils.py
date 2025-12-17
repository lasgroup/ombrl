import os
import random
import time

import gymnasium.wrappers
import numpy as np
import tqdm
import jax.numpy as jnp
from typing import Optional, Dict, Callable
from tensorboardX import SummaryWriter

# from jaxrl.agents import DDPGLearner, REDQLearner, SACLearner, DrQLearner
from maxinforl_jax.agents import MaxInfoSacLearner
from ombrl.agents import MaxInfoOmbrlLearner, ContinualMaxInfoLearner
from jaxrl.datasets import ReplayBuffer
from maxinforl_jax.datasets import NstepReplayBuffer
from ombrl.envs.wrappers import InitWrapper, EpisodicParamWrapper, EvalEnvFactory
from jaxrl.evaluation import evaluate
from jaxrl.utils import make_env
import wandb
import gymnasium as gym
from gymnasium.wrappers import RescaleAction
from gymnasium.wrappers.pixel_observation import PixelObservationWrapper

from jaxrl import wrappers


def train(
        project_name: str,
        entity_name: str,
        alg_name: str,
        env_name: str,
        alg_kwargs: Dict,
        env_kwargs: Dict,
        seed: int = 0,
        wandb_log: bool = True,
        log_config: Optional[Dict] = None,
        logs_dir: str = './logs/',
        save_video: bool = False,
        replay_buffer_size: int = 1_000_000,
        max_steps: int = 1_000_000,
        use_tqdm: bool = True,
        training_start: int = 0,
        updates_per_step: int = 1,
        batch_size: int = 256,
        log_interval: int = 1_000,
        eval_interval: int = 5_000,
        eval_episodes: int = 5,
        exp_hash: str = '',
        n_steps_returns: int = -1,
        recording_image_size: Optional[int] = None,
        eval_episode_trigger: Optional[Callable[[int], bool]] = None,
        episodic_param_scheduler: Optional[Callable[[int], dict]] = None,
        episodic_param_apply_fn: Optional[Callable[[gym.Env, dict], None]] = None,
        init_state: Optional[np.ndarray] = None,
):
    run_name = f"{env_name}__{alg_name}__{seed}__{int(time.time())}__{exp_hash}"

    if save_video:
        raise NotImplementedError("Video recording not implemented in this version.")
    else:
        video_train_folder = None
        video_eval_folder = None

    if env_name in [env_spec.id for env_spec in gym.envs.registry.values()]:
        env = make_env(env_name=env_name, seed=seed,
                       save_folder=video_train_folder,
                       recording_image_size=recording_image_size,
                       **env_kwargs)
        
        if init_state is not None:
            raise NotImplementedError("InitWrapper not tested in this version.") # TODO: Test
            env = InitWrapper(env, init_state=init_state)

        if episodic_param_scheduler is not None:
            assert episodic_param_apply_fn is not None
            env = EpisodicParamWrapper(
                env,
                scheduler_fn=episodic_param_scheduler,
                apply_fn=episodic_param_apply_fn,
                apply_before_reset=True,
            )
        else:
            env.episode_idx = -1  # for logging purposes
        
        eval_env_factory = EvalEnvFactory(
            make_env_fn=lambda: make_env(
            env_name=env_name,
            seed=seed + 42,
            save_folder=video_eval_folder,
            episode_trigger=eval_episode_trigger,
            recording_image_size=recording_image_size,
            **env_kwargs,
        ),
        apply_fn=episodic_param_apply_fn,
        init_state=init_state,
    )
        
    else:
        raise NotImplementedError(f'Env {env_name} not implemented in this version.')


    np.random.seed(seed)
    random.seed(seed)

    if alg_kwargs.get('pseudo_ct', False) == True:
        raise NotImplementedError(f'continuous-time RL not implemented in this version, got pseudo_ct=True.')

    if wandb_log:
        if log_config is None:
            log_config = {'alg': alg_name}
        else:
            log_config.update({'alg': alg_name})
        wandb.init(
            dir=logs_dir,
            project=project_name,
            sync_tensorboard=True,
            config=log_config,
            name=run_name,
            group=exp_hash,
            monitor_gym=True,
            save_code=True)

    summary_writer = SummaryWriter(
        os.path.join(logs_dir, run_name))

    if alg_name == 'maxinfosac':
        agent = MaxInfoSacLearner(seed,
                                  env.observation_space.sample(),
                                  env.action_space.sample(), **alg_kwargs)
    elif alg_name == 'maxinfombsac':
        agent = MaxInfoOmbrlLearner(seed,
                                    env.observation_space.sample(),
                                    env.action_space.sample(), **alg_kwargs)
    elif alg_name == 'continualmaxinfo':
        agent = ContinualMaxInfoLearner(seed,
                                    env.observation_space.sample(),
                                    env.action_space.sample(), **alg_kwargs)
    else:
        raise NotImplementedError()
    if n_steps_returns < 0:
        replay_buffer = ReplayBuffer(observation_space=env.observation_space,
                                     action_space=env.action_space,
                                     capacity=replay_buffer_size or max_steps)
    else:
        if 'discount' in alg_kwargs.keys():
            discount = alg_kwargs['discount']
        else:
            discount = 0.99
        replay_buffer = NstepReplayBuffer(observation_space=env.observation_space, action_space=env.action_space,
                                          discount=discount,
                                          n_steps=n_steps_returns,
                                          capacity=replay_buffer_size or max_steps)

    eval_returns = []
    observation, _ = env.reset()
    for i in tqdm.tqdm(range(1, max_steps + 1),
                       smoothing=0.1,
                       disable=not use_tqdm):

        if i < training_start:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation)
        next_observation, reward, terminate, truncate, info = env.step(action)

        if terminate:
            mask = 0.0
        else:
            mask = 1.0

        replay_buffer.insert(observation, action, reward, mask, float(terminate or truncate),
                             next_observation)
        observation = next_observation

        if i >= training_start:
            for _ in range(updates_per_step):
                batch = replay_buffer.sample(batch_size)
                update_info = agent.update(batch, episode_idx=env.episode_idx)

            if i % log_interval == 0:
                for k, v in update_info.items():
                    summary_writer.add_scalar(f'training/{k}', v, i)
                summary_writer.flush()

        if i % eval_interval == 0:
            if episodic_param_scheduler is not None:
                frozen_params = env.get_current_params()
                for k, v in frozen_params.items():
                    summary_writer.add_scalar(
                        f"env_params/{k}",
                        float(v),
                        info["total"]["timesteps"],
                    )
                eval_env = eval_env_factory.make(frozen_params)
            else:
                # fallback: stationary env
                eval_env = make_env(
                    env_name=env_name,
                    seed=seed + 42,
                    save_folder=video_eval_folder,
                    episode_trigger=eval_episode_trigger,
                    recording_image_size=recording_image_size,
                    **env_kwargs,
                )

            eval_stats = evaluate(agent, eval_env, eval_episodes)

            for k, v in eval_stats.items():
                summary_writer.add_scalar(f'evaluation/average_{k}s', v,
                                          info['total']['timesteps'])
            summary_writer.flush()

            eval_returns.append(
                (info['total']['timesteps'], eval_stats['return']))
            np.savetxt(os.path.join(logs_dir, f'{seed}.txt'),
                       eval_returns,
                       fmt=['%d', '%.1f'])
        
        if terminate or truncate:
            observation, _ = env.reset()
            terminate = False
            truncate = False
            for k, v in info['episode'].items():
                summary_writer.add_scalar(f'training/{k}', v,
                                          info['total']['timesteps'])

            if 'is_success' in info:
                summary_writer.add_scalar(f'training/success',
                                          info['is_success'],
                                          info['total']['timesteps'])
            if 'success' in info:
                summary_writer.add_scalar(f'training/success',
                                          info['success'],
                                          info['total']['timesteps'])
