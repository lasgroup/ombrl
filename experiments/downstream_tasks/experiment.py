import numpy as np
import os
import sys
import argparse


def experiment(
        logs_dir: str,
        project_name: str,
        entity_name: str,
        alg_name: str,
        env_name: str,
        action_cost: float = 0.0,
        action_repeat: int = 2,
        lr: float = 1e-4,
        ens_lr: float = 1e-4,
        ens_wd: float = 0.0,
        tau: float = 0.01,
        seed: int = 0,
        wandb_log: bool = True,
        save_video: bool = False,
        replay_buffer_size: int = 1_000_000,
        max_steps: int = 1_500_000,
        use_tqdm: bool = True,
        training_start: int = 0,
        updates_per_step: int = 1,
        batch_size: int = 256,
        hidden_dims: int = 256,
        log_interval: int = 1_000,
        eval_interval: int = 5_000,
        eval_episodes: int = 5,
        exp_hash: str = '',
        n_steps_returns: int = 3,
        init_temperature: float = 1.0,
        sample_model: bool = False,
        critic_real_data_update_period: int = 2,
        perturb_policy: bool = True,
        perturb_model: bool = True,
        explore_until: int = 1_500_000,
        use_bronet: bool = True,
        pseudo_ct: bool = True,
        int_rew_weight_start: float = -1.0,
        int_rew_weight_end: float = 0.0,
        int_rew_weight_decrease_steps: int = -1,
):
    """ Core experiment function """
    print(f"Starting experiment {project_name} for {alg_name} on {env_name}")
    print(f"Logging to: {logs_dir}")
    print(f"Seed: {seed}, Learning rate: {lr}, Action Repeat: {action_repeat}")
    print(f"Replay Buffer Size: {replay_buffer_size}, Max Steps: {max_steps}")
    print(f"Hidden Dims: {hidden_dims}, Use TQDM: {use_tqdm}")
    print(f"Training start: {training_start}, Updates per step: {updates_per_step}")
    print(f"Critic real data update period: {critic_real_data_update_period}")
    print(f"Perturbation - Policy: {perturb_policy}, Model: {perturb_model}")
    print(f"Use Bronet: {use_bronet}, Pseudo CT: {pseudo_ct}")

    # import jax
    # jax.config.update("jax_debug_nans", True)
    from combrl.utils.train_utils import train
    if alg_name in ['maxentdrq', 'drq']:
        n_steps_returns = -1
    alg_kwargs = {
        'actor_lr': lr,
        'critic_lr': lr,
        'temp_lr': lr,
        'hidden_dims': (hidden_dims, hidden_dims),
        'discount': 0.99,
        'tau': tau,
        'target_update_period': 1,
        'target_entropy': None,
        'backup_entropy': True,
        'use_bronet': use_bronet,
        'init_temperature': init_temperature,
    }

    replay_buffer_size = min(replay_buffer_size, max_steps)
    if alg_name == 'combrl':
        alg_kwargs['ens_lr'] = ens_lr
        alg_kwargs['ens_wd'] = ens_wd
        alg_kwargs['model_hidden_dims'] = (hidden_dims, hidden_dims)
        alg_kwargs['sample_model'] = sample_model
        alg_kwargs['critic_real_data_update_period'] = critic_real_data_update_period
        alg_kwargs['perturb_policy'] = perturb_policy
        alg_kwargs['perturb_model'] = perturb_model
        alg_kwargs['explore_until'] = explore_until
        alg_kwargs['pseudo_ct'] = pseudo_ct
        alg_kwargs['int_rew_weight_start'] = int_rew_weight_start
        alg_kwargs['int_rew_weight_end'] = int_rew_weight_end
        alg_kwargs['int_rew_weight_decrease_steps'] = int_rew_weight_decrease_steps


    env_kwargs = {'action_cost': action_cost,
                  'action_repeat': action_repeat,
                  }
    
    if int_rew_weight_start >= 0.:
        exp_name = exp_hash + '__' + str(int_rew_weight_start)
    else:
        exp_name = exp_hash

    log_config = {
        'alg_name': alg_name,
        'action_cost': action_cost,
        'lr': lr,
        'ens_lr': ens_lr,
        'ens_wd': ens_wd,
        'action_repeat': action_repeat,
        'env_name': env_name,
        'exp_hash': exp_hash,
        'exp_name': exp_name,
        'hidden_dims': hidden_dims,
        'batch_size': batch_size,
        'n_steps_returns': n_steps_returns,
        'tau': tau,
        'init_temperature': init_temperature,
        'sample_model': sample_model,
        'critic_real_data_update_period': critic_real_data_update_period,
        'use_bronet': use_bronet,
        'perturb_policy': perturb_policy,
        'perturb_model': perturb_model,
        'explore_until': explore_until,
        'pseudo_ct': pseudo_ct,
        'int_rew_weight_start': int_rew_weight_start,
        'int_rew_weight_end': int_rew_weight_end,
        'int_rew_weight_decrease_steps': int_rew_weight_decrease_steps,
    }

    train(
        project_name=project_name,
        entity_name=entity_name,
        alg_name=alg_name,
        env_name=env_name,
        alg_kwargs=alg_kwargs,
        env_kwargs=env_kwargs,
        seed=seed,
        wandb_log=wandb_log,
        log_config=log_config,
        logs_dir=logs_dir,
        save_video=save_video,
        replay_buffer_size=replay_buffer_size,
        max_steps=max_steps,
        use_tqdm=use_tqdm,
        training_start=training_start,
        updates_per_step=updates_per_step,
        batch_size=batch_size,
        log_interval=log_interval,
        eval_interval=eval_interval,
        eval_episodes=eval_episodes,
        exp_hash=exp_name,
        n_steps_returns=n_steps_returns,
    )


def main(args):
    """"""
    from pprint import pprint
    print(args)

    """ Generate experiment hash and set up redirect of output streams """

    """ Experiment core """
    np.random.seed(args.seed)

    experiment(
        logs_dir=args.logs_dir,
        project_name=args.project_name,
        entity_name=args.entity_name,
        alg_name=args.alg_name,
        env_name=args.env_name,
        action_cost=args.action_cost,
        action_repeat=args.action_repeat,
        lr=args.lr,
        ens_lr=args.ens_lr,
        ens_wd=args.ens_wd,
        tau=args.tau,
        seed=args.seed,
        wandb_log=bool(args.wandb_log),
        save_video=bool(args.save_video),
        replay_buffer_size=args.replay_buffer_size,
        max_steps=args.max_steps,
        use_tqdm=bool(args.use_tqdm),
        training_start=args.training_start,
        updates_per_step=args.updates_per_step,
        batch_size=args.batch_size,
        hidden_dims=args.hidden_dims,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        exp_hash=args.exp_hash,
        n_steps_returns=args.n_steps_returns,
        init_temperature=args.init_temperature,
        sample_model=bool(args.sample_model),
        critic_real_data_update_period=args.critic_real_data_update_period,
        perturb_policy=bool(args.perturb_policy),
        perturb_model=bool(args.perturb_model),
        explore_until=args.explore_until,
        use_bronet=bool(args.use_bronet),
        pseudo_ct=bool(args.pseudo_ct),
        int_rew_weight_start=args.int_rew_weight_start,
        int_rew_weight_end=args.int_rew_weight_end,
        int_rew_weight_decrease_steps=args.int_rew_weight_decrease_steps,
    )




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MTTest')

    # general experiment args
    parser.add_argument('--logs_dir', type=str, default='./logs/')
    parser.add_argument('--project_name', type=str, default='CombrlSAC_Test')
    parser.add_argument('--entity_name', type=str, default='kiten')
    parser.add_argument('--alg_name', type=str, default='combrl')
    parser.add_argument('--env_name', type=str, default='Reacher-v4')
    # 'Walker2d-v4', 'Swimmer-v4', 'Pusher-v4', 'Reacher-v4'
    # 'HalfCheetah-v4') # 'cheetah-run') # 'Hopper-v4') #'cheetah-run') # 'MountainCarContinuous-v0') # 'cartpole-swingup_sparse') # 
    parser.add_argument('--action_cost', type=float, default=0.0)
    parser.add_argument('--action_repeat', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--ens_lr', type=float, default=1e-4)
    parser.add_argument('--ens_wd', type=float, default=0.0)
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--wandb_log', type=int, default=1)
    parser.add_argument('--save_video', type=int, default=0)
    parser.add_argument('--replay_buffer_size', type=int, default=1_000_000)
    parser.add_argument('--max_steps', type=int, default=250_000)
    parser.add_argument('--use_tqdm', type=int, default=1)
    parser.add_argument('--training_start', type=int, default=0)
    parser.add_argument('--updates_per_step', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--hidden_dims', type=int, default=256)
    parser.add_argument('--log_interval', type=int, default=1_000)
    parser.add_argument('--eval_interval', type=int, default=100)
    parser.add_argument('--eval_episodes', type=int, default=5)
    parser.add_argument('--exp_hash', type=str, default='xy')
    parser.add_argument('--n_steps_returns', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--init_temperature', type=float, default=1)
    parser.add_argument('--sample_model', type=int, default=0)
    parser.add_argument('--critic_real_data_update_period', type=int, default=2)
    parser.add_argument('--perturb_policy', type=int, default=1)
    parser.add_argument('--perturb_model', type=int, default=1)
    parser.add_argument('--explore_until', type=int, default=5)
    parser.add_argument('--use_bronet', type=int, default=1)
    parser.add_argument('--pseudo_ct', type=int, default=1)
    parser.add_argument('--int_rew_weight_start', type=float, default = 0.1)
    parser.add_argument('--int_rew_weight_end', type=float, default=1_000_000)
    parser.add_argument('--int_rew_weight_decrease_steps', type=int, default=-1)

    args = parser.parse_args()
    main(args)
