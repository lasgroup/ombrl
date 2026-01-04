import os
import random
import time

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
from ombrl.envs.build_utils import make_metaworld_env, make_humanoid_bench_env
from jaxrl.utils import make_env
import wandb
from jaxrl.evaluation import evaluate as jaxrl_evaluate


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
        eval_episodes: int = 5,
        exp_hash: str = '',
        n_steps_returns: int = -1,
        recording_image_size: Optional[int] = None,
        eval_episode_trigger: Optional[Callable[[int], bool]] = None,
):
    run_name = f"{env_name}__{alg_name}__{seed}__{int(time.time())}__{exp_hash}"

    if save_video:
        video_train_folder = os.path.join(logs_dir, 'video', 'train')
        video_eval_folder = os.path.join(logs_dir, 'video', 'eval')
    else:
        video_train_folder = None
        video_eval_folder = None

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
        if env_name=='MountainCarContinuous-v0':
            # HACK for MountainCar
            eval_env.unwrapped.min_position = -1.75
        if env_name=='Pendulum-v1':
            env = PendulumInitWrapper(env, init_angle=np.pi, init_vel=0.0)
            eval_env = PendulumInitWrapper(eval_env, init_angle=np.pi, init_vel=0.0)

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
                update_info = agent.update(batch)

            if i % log_interval == 0:
                for k, v in update_info.items():
                    summary_writer.add_scalar(f'training/{k}', v, i)
                summary_writer.flush()

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
