import os
import random
import time
import pickle

import numpy as np
import tqdm
from typing import Optional, Dict, Callable, List
from tensorboardX import SummaryWriter

from ombrl.agents import COMBRLExplorerLearner, SOMBRLExplorerLearner
from jaxrl.datasets import ReplayBuffer
from maxinforl_jax.datasets import NstepReplayBuffer
from ombrl.utils.evaluation import evaluate
from ombrl.utils.multiple_reward_wrapper import RewardFunction

from ombrl.utils.wrappers import PendulumInitWrapper
from ombrl.utils.env_utils import make_metaworld_env, make_humanoid_bench_env
from jaxrl.utils import make_env
import wandb
from jaxrl.evaluation import evaluate as jaxrl_evaluate
import collections
import jax.numpy as jnp
import jax


TrainState = collections.namedtuple(
    'TrainState',
    ['replay_buffer', 'agent_state'])


def create_env(
        env_name: str,
        env_kwargs: Dict,
        seed: int = 0,
        video_train_folder: str | None = None,
        video_eval_folder: str | None = None,
        recording_image_size: Optional[int] = None,
        eval_episode_trigger: Optional[Callable[[int], bool]] = None,
):
    if 'humanoid_bench' in env_name:
        _, task_name = env_name.split('/')
        env = make_humanoid_bench_env(env_name=task_name, seed=seed,
                                      save_folder=video_train_folder,
                                      recording_image_size=recording_image_size,
                                      **env_kwargs)
        eval_env = make_humanoid_bench_env(env_name=task_name, seed=seed + 42,
                                           save_folder=video_eval_folder,
                                           recording_image_size=recording_image_size,
                                           episode_trigger=eval_episode_trigger,
                                           **env_kwargs)
    elif 'metaworld' in env_name:
        _, task_name = env_name.split('_')
        env = make_metaworld_env(env_name=task_name, seed=seed, save_folder=video_train_folder, **env_kwargs)
        eval_env = make_metaworld_env(env_name=task_name, seed=seed + 42,
                                      save_folder=video_eval_folder, **env_kwargs)
    else:
        env = make_env(env_name=env_name, seed=seed,
                       save_folder=video_train_folder,
                       recording_image_size=recording_image_size,
                       **env_kwargs)
        eval_env = make_env(env_name=env_name, seed=seed + 42,
                            save_folder=video_eval_folder,
                            episode_trigger=eval_episode_trigger,
                            recording_image_size=recording_image_size,
                            **env_kwargs)
        if env_name == 'MountainCarContinuous-v0':
            # HACK for MountainCar
            eval_env.unwrapped.min_position = -1.75
        if env_name == 'Pendulum-v1':
            env = PendulumInitWrapper(env, init_angle=np.pi, init_vel=0.0)
            eval_env = PendulumInitWrapper(eval_env, init_angle=np.pi, init_vel=0.0)
    return env, eval_env


def train(
        project_name: str,
        entity_name: str,
        alg_name: str,
        env_name: str,
        alg_kwargs: Dict,
        env_kwargs: Dict,
        seed: int = 0,
        reward_list: List[RewardFunction] | RewardFunction | None = None,
        wandb_log: bool = True,
        log_config: Optional[Dict] = None,
        logs_dir: str = './logs',
        save_video: bool = False,
        replay_buffer_size: int = 1_000_000,
        max_steps: int = 1_000_000,
        use_tqdm: bool = True,
        training_start: int = 0,
        updates_per_step: int = 1,
        batch_size: int = 256,
        log_interval: int = 1_000,
        eval_interval: int = 5_000,
        save_interval: int = 10_000,
        eval_episodes: int = 5,
        exp_hash: str = '',
        n_steps_returns: int = -1,
        recording_image_size: Optional[int] = None,
        eval_episode_trigger: Optional[Callable[[int], bool]] = None,
        prior_agent_state: TrainState | None = None,
        prior_batch_size: int = 256,
        load_prior_agent_params: bool = False,
):
    run_name = f"{env_name}__{alg_name}__{seed}__{int(time.time())}__{exp_hash}"

    if save_video:
        video_train_folder = os.path.join(logs_dir, 'video', 'train')
        video_eval_folder = os.path.join(logs_dir, 'video', 'eval')
    else:
        video_train_folder = None
        video_eval_folder = None

    env, eval_env = create_env(
        env_name=env_name,
        env_kwargs=env_kwargs,
        video_train_folder=video_train_folder,
        video_eval_folder=video_eval_folder,
        recording_image_size=recording_image_size,
        eval_episode_trigger=eval_episode_trigger,
    )

    np.random.seed(seed)
    random.seed(seed)

    if wandb_log:
        if log_config is None:
            log_config = {'alg': alg_name}
        else:
            log_config.update({'alg': alg_name})
        wandb.init(
            dir=logs_dir,
            project=project_name,
            entity=entity_name,
            sync_tensorboard=True,
            config=log_config,
            name=run_name,
            monitor_gym=True,
            save_code=True)

    summary_writer = SummaryWriter(
        os.path.join(logs_dir, run_name))

    if alg_name == 'combrl':
        agent = COMBRLExplorerLearner(seed,
                                      env.observation_space.sample(),
                                      env.action_space.sample(),
                                      reward_list, **alg_kwargs)
    elif alg_name == 'sombrl':
        if reward_list is not None:
            assert isinstance(reward_list, RewardFunction), "Only one reward function can be passed to SOMBRL"
        if prior_agent_state is not None:
            agent = SOMBRLExplorerLearner.load_from_agent_state(
                seed=seed,
                observations=env.observation_space.sample(),
                actions=env.action_space.sample(),
                agent_state=prior_agent_state.agent_state,
                load_params=load_prior_agent_params
            )
        else:
            agent = SOMBRLExplorerLearner(
                seed,
                env.observation_space.sample(),
                env.action_space.sample(),
                reward_model=reward_list,
                **alg_kwargs)
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

    # Training Loop
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

        if i >= training_start:
            for _ in range(updates_per_step):
                batch = replay_buffer.sample(batch_size)
                if prior_agent_state is not None:
                    assert isinstance(prior_agent_state.replay_buffer, ReplayBuffer)
                    prior_batch = prior_agent_state.replay_buffer.sample(prior_batch_size)
                    batch = jax.tree_util.tree_map(lambda x, y: jnp.concatenate([x, y], axis=0), batch, prior_batch)
                update_info = agent.update(batch)

            if i % log_interval == 0:
                for k, v in update_info.items():
                    summary_writer.add_scalar(f'training/{k}', v, i)
                summary_writer.flush()

        if i % save_interval == 0:

            train_state = TrainState(
                replay_buffer=replay_buffer,
                agent_state=agent.agent_state(),
            )
            with open(f'agent_state_{i}.pkl', 'wb') as f:
                pickle.dump(train_state, f)

        if i % eval_interval == 0:
            if reward_list is None:
                eval_stats = jaxrl_evaluate(agent, eval_env, eval_episodes)
                for k, v in eval_stats.items():
                    summary_writer.add_scalar(f'evaluation/average_{k}s', v,
                                              info['total']['timesteps'])
                summary_writer.flush()

                eval_returns.append(
                    (info['total']['timesteps'], eval_stats['return']))
                np.savetxt(os.path.join(logs_dir, f'{seed}.txt'),
                           eval_returns,
                           fmt=['%d', '%.1f'])
            else:
                eval_info = evaluate(agent, eval_env, eval_episodes, reward_list)
                for reward_index, eval_stats in eval_info.items():
                    for k, v in eval_stats.items():
                        summary_writer.add_scalar(f'evaluation/task_{reward_index}_average_{k}s', v,
                                                  info['total']['timesteps'])
                    summary_writer.flush()

                    eval_returns.append(
                        (info['total']['timesteps'], eval_stats['eval_return']))
                    np.savetxt(os.path.join(logs_dir, f'{seed}.txt'),
                               eval_returns,
                               fmt=['%d', '%.1f'])
    if wandb_log:
        wandb.finish()
    return TrainState(
                replay_buffer=replay_buffer,
                agent_state=agent.agent_state(),
            )


def train_sim_to_real(
        env_name_sim: str,
        env_name_real: str,
        env_kwargs_sim: Dict,
        env_kwargs_real: Dict,
        max_steps_sim: int = 1_000_000,
        max_steps_real: int = 250_000,
        logs_dir: str = './logs',
        *args,
        **kwargs
):
    logs_dir_sim = os.path.join(logs_dir, 'sim')
    agent_state = train(env_name=env_name_sim,
                        env_kwargs=env_kwargs_sim,
                        max_steps=max_steps_sim,
                        logs_dir=logs_dir_sim,
                        *args, **kwargs)
    logs_dir_real = os.path.join(logs_dir, 'real')
    return train(
        env_name=env_name_real,
        env_kwargs=env_kwargs_real,
        max_steps=max_steps_real,
        prior_agent_state=agent_state,
        logs_dir=logs_dir_real,
        *args,
        **kwargs
    )

