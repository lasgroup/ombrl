import numpy as np
import argparse
from experiments.utils import hash_dict


def experiment(
        project_name: str,
        entity_name: str,
        alg_name: str,
        env_name: str,
        action_cost: float = 0.0,
        action_repeat: int = 1,
        num_neurons: int = 256,
        num_hidden_layers: int = 2,
        ens_lr: float = 3e-4,
        lr: float = 3e-4,
        ens_wd: float = 0.0,
        seed: int = 0,
        wandb_log: bool = True,
        logs_dir: str = './logs/',
        save_video: bool = False,
        replay_buffer_size: int = 1_000_000,
        max_steps: int = 1_000_000,
        use_tqdm: bool = True,
        training_start: int = 0,
        batch_size: int = 256,
        log_interval: int = 1_000,
        eval_interval: int = 5_000,
        eval_episodes: int = 10,
        exp_hash: str = '',
        sample_model: bool = False,
        perturb_policy: bool = True,
        perturb_model: bool = True,
        predict_diff: bool = True,
        predict_reward: bool = True,
        actor_critic_updates_per_model_update: int = -1,
        policy_update_period: int = 1,
        target_update_period: int = 1,
        num_imagined_steps_end: int = 5,
        steps_to_imagined_steps_end: int = 10_000,
        int_rew_weight_start: float = 0.0,
        int_rew_weight_end: float = 0.0,
        int_rew_weight_decrease_steps: int = -1,
        updates_per_step: int = 1,
        reset_models: bool = False,
        reset_period: int = 500_000,
):
    from ombrl.utils.train_utils import train
    import optax

    env_kwargs = {'action_cost': action_cost,
                  'action_repeat': action_repeat,
                  }
    assert alg_name in ['sombrl', 'combrl']

    alg_kwargs = {
        'actor_lr': lr,
        'critic_lr': lr,
        'temp_lr': lr,
        'hidden_dims': (num_neurons,) * num_hidden_layers,
        'discount': 0.99,
        'tau': 0.005,
        'target_update_period': 1,
        'target_entropy': None,
        'backup_entropy': True,
    }
    max_gradient_norm = 0.5
    updates_per_step = updates_per_step
    reset_period = reset_period
    # do soft resets of the model every few steps
    reset_models = reset_models

    alg_kwargs['ens_lr'] = ens_lr
    alg_kwargs['ens_wd'] = ens_wd
    alg_kwargs['model_hidden_dims'] = (num_neurons,) * num_hidden_layers
    alg_kwargs['sample_model'] = sample_model
    alg_kwargs['max_gradient_norm'] = max_gradient_norm
    alg_kwargs['reset_models'] = reset_models
    alg_kwargs['reset_period'] = reset_period
    alg_kwargs['perturb_policy'] = perturb_policy
    alg_kwargs['perturb_model'] = perturb_model
    alg_kwargs['predict_reward'] = predict_reward
    alg_kwargs['actor_critic_updates_per_model_update'] = actor_critic_updates_per_model_update
    alg_kwargs['policy_update_period'] = policy_update_period
    alg_kwargs['target_update_period'] = target_update_period
    alg_kwargs['num_imagined_steps'] = optax.piecewise_constant_schedule(
        init_value=1, boundaries_and_scales={steps_to_imagined_steps_end: num_imagined_steps_end}

    )
    alg_kwargs['int_rew_weight_start'] = int_rew_weight_start
    alg_kwargs['int_rew_weight_end'] = int_rew_weight_end
    alg_kwargs['int_rew_weight_decrease_steps'] = int_rew_weight_decrease_steps

    log_config = {
        'alg_name': alg_name,
        'action_cost': action_cost,
        'exp_hash': exp_hash,
        'ens_lr': ens_lr,
        'ens_wd': ens_wd,
        'lr': lr,
        'num_neurons': num_neurons,
        'num_hidden_layers': num_hidden_layers,
        'action_repeat': action_repeat,
        'env_name': env_name,
        'sample_model': sample_model,
        'actor_critic_updates_per_model_update': actor_critic_updates_per_model_update,
        'max_gradient_norm': max_gradient_norm,
        'reset_models': reset_models,
        'reset_period': reset_period,
        'perturb_policy': perturb_policy,
        'perturb_model': perturb_model,
        'predict_diff': predict_diff,
        'policy_update_period': policy_update_period,
        'num_imagined_steps_end': num_imagined_steps_end,
        'steps_to_imagined_steps_end': steps_to_imagined_steps_end,
        'int_rew_weight_start': int_rew_weight_start,
        'int_rew_weight_end': int_rew_weight_end,
        'int_rew_weight_decrease_steps': int_rew_weight_decrease_steps,
        'updates_per_step': updates_per_step,
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
        exp_hash=exp_hash,
    )


def main(args):
    """"""
    from pprint import pprint
    print(args)
    pprint(args.__dict__)
    exp_hash = hash_dict(args.__dict__)
    print('\n ------------------------------------ \n')

    """ Experiment core """
    np.random.seed(args.seed)

    experiment(
        project_name=args.project_name,
        entity_name=args.entity_name,
        alg_name=args.alg_name,
        env_name=args.env_name,
        action_cost=args.action_cost,
        action_repeat=args.action_repeat,
        ens_lr=args.ens_lr,
        num_neurons=args.num_neurons,
        num_hidden_layers=args.num_hidden_layers,
        lr=args.lr,
        ens_wd=args.ens_wd,
        seed=args.seed,
        wandb_log=bool(args.wandb_log),
        logs_dir=args.logs_dir + f'{args.alg_name}/{exp_hash}',
        save_video=bool(args.save_video),
        replay_buffer_size=args.replay_buffer_size,
        max_steps=args.max_steps,
        use_tqdm=bool(args.use_tqdm),
        training_start=args.training_start,
        batch_size=args.batch_size,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        exp_hash=args.exp_hash,
        sample_model=bool(args.sample_model),
        actor_critic_updates_per_model_update=args.actor_critic_updates_per_model_update,
        perturb_policy=bool(args.perturb_policy),
        perturb_model=bool(args.perturb_model),
        policy_update_period=args.policy_update_period,
        target_update_period=args.target_update_period,
        num_imagined_steps_end=args.num_imagined_steps_end,
        steps_to_imagined_steps_end=args.steps_to_imagined_steps_end,
        predict_diff=bool(args.predict_diff),
        int_rew_weight_start=args.int_rew_weight_start,
        int_rew_weight_end=args.int_rew_weight_end,
        int_rew_weight_decrease_steps=args.int_rew_weight_decrease_steps,
        updates_per_step=args.updates_per_step,
        reset_period=args.reset_period,
        reset_models=bool(args.reset_models),

    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MTTest')

    # general experiment args
    parser.add_argument('--logs_dir', type=str, default='./logs/')
    parser.add_argument('--project_name', type=str, default='sombrl_tests')
    parser.add_argument('--entity_name', type=str, default='sukhijab')
    parser.add_argument('--alg_name', type=str, default='sombrl')
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v4')
    # 'Pendulum-v1', 'Walker2d-v4', 'Swimmer-v4', 'Pusher-v4', 'Reacher-v4', 'Humanoid-v4'
    parser.add_argument('--action_cost', type=float, default=0.0)
    parser.add_argument('--action_repeat', type=int, default=2)
    parser.add_argument('--num_neurons', type=int, default=256)
    parser.add_argument('--num_hidden_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--ens_lr', type=float, default=3e-4)
    parser.add_argument('--ens_wd', type=float, default=0.0)
    parser.add_argument('--wandb_log', type=int, default=0)
    parser.add_argument('--save_video', type=int, default=0)
    parser.add_argument('--replay_buffer_size', type=int, default=1_000_000)
    parser.add_argument('--max_steps', type=int, default=1_500_000)
    parser.add_argument('--use_tqdm', type=int, default=1)
    parser.add_argument('--training_start', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--log_interval', type=int, default=1_000)
    parser.add_argument('--eval_interval', type=int, default=100)
    parser.add_argument('--eval_episodes', type=int, default=5)
    parser.add_argument('--exp_hash', type=str, default='sombrl')
    parser.add_argument('--sample_model', type=int, default=0)
    parser.add_argument('--perturb_policy', type=int, default=1)
    parser.add_argument('--perturb_model', type=int, default=1)
    parser.add_argument('--predict_diff', type=int, default=1)
    parser.add_argument('--actor_critic_updates_per_model_update', type=int, default=-1)
    parser.add_argument('--policy_update_period', type=int, default=1)
    parser.add_argument('--target_update_period', type=int, default=1)
    parser.add_argument('--num_imagined_steps_end', type=int, default=5)
    parser.add_argument('--steps_to_imagined_steps_end', type=int, default=10_000)
    parser.add_argument('--int_rew_weight_start', type=float, default=0.0)
    parser.add_argument('--int_rew_weight_end', type=float, default=0.0)
    parser.add_argument('--int_rew_weight_decrease_steps', type=int, default=-1)
    parser.add_argument('--updates_per_step', type=int, default=1)
    parser.add_argument('--reset_period', type=int, default=500_000)
    parser.add_argument('--reset_models', type=int, default=0)

    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    main(args)
