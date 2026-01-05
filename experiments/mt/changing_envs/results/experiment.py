import numpy as np
import argparse
from experiments.utils import hash_dict, parse_string
from typing import Optional
from ombrl.envs.env_utils import get_scheduler_apply_fn


def experiment(
        project_name: str,
        entity_name: str,
        alg_name: str,
        env_name: str,
        action_cost: float = 0.0,
        action_repeat: int = 1,
        ens_lr: float = 3e-4,
        dyn_ent_lr: float = 3e-4,
        num_neurons: int = 256,
        num_hidden_layers: int = 2,
        lr: float = 3e-4,
        ens_wd: float = 0.0,
        dyn_wd: float = 0.0,
        seed: int = 0,
        wandb_log: bool = True,
        logs_dir: str = './logs/',
        save_video: bool = False,
        replay_buffer_mode: str = 'none',
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
        critic_real_data_update_period: int = 2,
        updates_per_step: int = 1,
        num_imagined_steps: int = 1,
        actor_critic_updates_per_model_update: int = 1,
        use_bronet: bool = True,
        init_temperature_dyn_entropy: float = 1.0,
        reset_models:  bool = True,
        perturb_policy: bool = True,
        perturb_critic: bool = True,
        perturb_model: bool = True,
        policy_perturb_rate: float = 0.2,
        critic_perturb_rate: float = 0.2,
        model_perturb_rate: float = 0.2,
        policy_reset_period: int = 5,
        critic_reset_period: int = 5,
        model_reset_period: int = 5,
        pseudo_ct: bool = False,
        predict_diff: bool = True,
        env_param_mode: str = 'stationary',
        init_state: Optional[np.ndarray] = None,
        parameter_decay: float = None,
):
    from ombrl.utils.continual_rl_train_utils import train

    exp_hash = env_param_mode + '_' + exp_hash # HACK
    print(f"WARNING: Experiment hash overriden to: {exp_hash}")

    env_kwargs = {'action_cost': action_cost,
                  'action_repeat': action_repeat,
                  }

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
        'use_bronet': use_bronet,
    }
    max_gradient_norm = 0.5
    updates_per_step = updates_per_step
    reset_period = 500_000
    # do soft resets of the model every few steps
    reset_models = reset_models
    if replay_buffer_mode == 'none':
        replay_buffer_size = max_steps
    elif replay_buffer_mode == 'window' or replay_buffer_mode == 'reset':
        replay_buffer_size = min(replay_buffer_size, max_steps)
    else:
        raise ValueError(f'Unknown replay buffer mode: {replay_buffer_mode}')
    
    if alg_name == 'maxinfosac' or alg_name == 'maxinfombsac'\
        or alg_name == 'continualmaxinfo':
        alg_kwargs['dyn_ent_lr'] = dyn_ent_lr
        alg_kwargs['dyn_wd'] = dyn_wd
        alg_kwargs['ens_lr'] = ens_lr
        alg_kwargs['ens_wd'] = ens_wd
        alg_kwargs['init_temperature_dyn_entropy'] = init_temperature_dyn_entropy
        alg_kwargs['model_hidden_dims'] = (num_neurons,) * num_hidden_layers
        if alg_name == 'maxinfombsac':
            alg_kwargs['sample_model'] = sample_model
            alg_kwargs['critic_real_data_update_period'] = critic_real_data_update_period
            alg_kwargs['max_gradient_norm'] = max_gradient_norm
            alg_kwargs['reset_models'] = reset_models
            alg_kwargs['reset_period'] = reset_period
            alg_kwargs['perturb_policy'] = perturb_policy
            alg_kwargs['perturb_model'] = perturb_model
            alg_kwargs['pseudo_ct'] = pseudo_ct
            alg_kwargs['predict_diff'] = predict_diff
            alg_kwargs['dt'] = None
            alg_kwargs['action_repeat'] = env_kwargs.get('action_repeat', 1)

            # set update per step such that critic is updated at least once using real data.
            updates_per_step = critic_real_data_update_period * updates_per_step
        elif alg_name == 'continualmaxinfo':

            if critic_perturb_rate < 0.0:
                critic_perturb_rate = policy_perturb_rate
                print(f'WARNING: setting critic_perturb_rate to policy_perturb_rate: {policy_perturb_rate}')

            alg_kwargs.update(dict(
                sample_model=sample_model,
                max_gradient_norm=max_gradient_norm,
                reset_models=reset_models,
                perturb_policy=perturb_policy,
                perturb_critic=perturb_critic,
                perturb_model=perturb_model,
                policy_perturb_rate = policy_perturb_rate,
                critic_perturb_rate = critic_perturb_rate,
                model_perturb_rate = model_perturb_rate,
                policy_reset_period=policy_reset_period,
                critic_reset_period=critic_reset_period,
                model_reset_period=model_reset_period, 
                pseudo_ct=pseudo_ct,
                predict_diff=predict_diff,
                dt=None,
                action_repeat=env_kwargs.get('action_repeat', 1),
                num_imagined_steps=num_imagined_steps,
                actor_critic_updates_per_model_update=actor_critic_updates_per_model_update,
            ))
            updates_per_step = updates_per_step

    model_update_delay = 1


    log_config = {
        'alg_name': alg_name,
        'action_cost': action_cost,
        'exp_hash': exp_hash,
        'ens_lr': ens_lr,
        'ens_wd': ens_wd,
        'lr': lr,
        'dyn_ent_lr': dyn_ent_lr,
        'dyn_wd': dyn_wd,
        'batch_size': batch_size,
        'seed': seed,
        'save_video': save_video,
        'replay_buffer_mode': replay_buffer_mode,
        'replay_buffer_size': replay_buffer_size,
        'max_steps': max_steps,
        'training_start': training_start,
        'num_neurons': num_neurons,
        'num_hidden_layers': num_hidden_layers,
        'action_repeat': action_repeat,
        'env_name': env_name,
        'model_update_delay': model_update_delay,
        'sample_model': sample_model,
        'critic_real_data_update_period': None if alg_name != 'maxinfombsac' else critic_real_data_update_period,
        'use_bronet': use_bronet,
        'max_gradient_norm': max_gradient_norm,
        'updates_per_step': updates_per_step,
        'init_temperature_dyn_entropy': init_temperature_dyn_entropy,
        'reset_models': reset_models,
        'reset_period': reset_period,
        'perturb_policy': perturb_policy,
        'perturb_model': perturb_model,
        'pseudo_ct': pseudo_ct,
        'predict_diff': predict_diff,
        'env_param_mode': env_param_mode,
        'init_state': None if init_state is None else np.asarray(init_state).tolist(),
        'policy_perturb_rate': policy_perturb_rate,
        'critic_perturb_rate': critic_perturb_rate,
        'model_perturb_rate': model_perturb_rate,
        'policy_reset_period': policy_reset_period,
        'critic_reset_period': critic_reset_period,
        'model_reset_period': model_reset_period,
        'num_imagined_steps': num_imagined_steps,
        'actor_critic_updates_per_model_update': actor_critic_updates_per_model_update,
    }

    scheduler_fn, apply_fn, env_log = get_scheduler_apply_fn(env_name=env_name, env_param_mode=env_param_mode, parameter_decay=parameter_decay)
    log_config = log_config | env_log


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
        replay_buffer_mode=replay_buffer_mode,
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
        episodic_param_scheduler=scheduler_fn,
        episodic_param_apply_fn=apply_fn,
        init_state=init_state,
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
        dyn_ent_lr=args.dyn_ent_lr,
        num_neurons=args.num_neurons,
        num_hidden_layers=args.num_hidden_layers,
        lr=args.lr,
        ens_wd=args.ens_wd,
        dyn_wd=args.dyn_wd,
        seed=args.seed,
        wandb_log=bool(args.wandb_log),
        logs_dir=args.logs_dir + f'{args.alg_name}/{exp_hash}',
        save_video=bool(args.save_video),
        replay_buffer_mode=args.replay_buffer_mode,
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
        critic_real_data_update_period=args.critic_real_data_update_period,
        updates_per_step=args.updates_per_step,
        num_imagined_steps=args.num_imagined_steps,
        actor_critic_updates_per_model_update=args.actor_critic_updates_per_model_update,
        init_temperature_dyn_entropy=args.init_temperature_dyn_entropy,
        reset_models=bool(args.reset_models),
        perturb_policy=bool(args.perturb_policy),
        perturb_critic=bool(args.perturb_critic),
        perturb_model=bool(args.perturb_model),
        policy_perturb_rate=args.policy_perturb_rate,
        critic_perturb_rate=args.critic_perturb_rate,
        model_perturb_rate=args.model_perturb_rate,
        policy_reset_period=args.policy_reset_period,
        critic_reset_period=args.critic_reset_period,
        model_reset_period=args.model_reset_period,
        use_bronet=bool(args.use_bronet),
        pseudo_ct=bool(args.pseudo_ct),
        predict_diff=bool(args.predict_diff),
        env_param_mode=args.env_param_mode,
        init_state=parse_string(args.init_state),
        parameter_decay=args.parameter_decay,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MTTest')

    # general experiment args
    parser.add_argument('--logs_dir', type=str, default='./logs/')
    parser.add_argument('--project_name', type=str, default='MT_Test')
    parser.add_argument('--entity_name', type=str, default='kiten')
    parser.add_argument('--alg_name', type=str, default='continualmaxinfo')
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v4')
    # 'Pendulum-v1', 'Walker2d-v4', 'Swimmer-v4', 'Pusher-v4', 'Reacher-v4', 'Humanoid-v4'
    parser.add_argument('--action_cost', type=float, default=0.0)
    parser.add_argument('--action_repeat', type=int, default=1)
    parser.add_argument('--ens_lr', type=float, default=3e-4)
    parser.add_argument('--dyn_ent_lr', type=float, default=3e-4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--ens_wd', type=float, default=0.0)
    parser.add_argument('--dyn_wd', type=float, default=0.0)
    parser.add_argument('--num_neurons', type=int, default=256)
    parser.add_argument('--num_hidden_layers', type=int, default=2)
    parser.add_argument('--wandb_log', type=int, default=1)
    parser.add_argument('--save_video', type=int, default=0)
    parser.add_argument('--replay_buffer_mode', type=str, default='reset', choices=['none', 'window', 'reset'])
    parser.add_argument('--replay_buffer_size', type=int, default=2_000)
    parser.add_argument('--max_steps', type=int, default=2001)
    parser.add_argument('--use_tqdm', type=int, default=1)
    parser.add_argument('--training_start', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--log_interval', type=int, default=1_000)
    parser.add_argument('--eval_interval', type=int, default=200)
    parser.add_argument('--eval_episodes', type=int, default=5)
    parser.add_argument('--exp_hash', type=str, default='maxinfombsac')
    parser.add_argument('--sample_model', type=int, default=0)
    parser.add_argument('--critic_real_data_update_period', type=int, default=None)
    parser.add_argument('--updates_per_step', type=int, default=1)
    parser.add_argument('--num_imagined_steps', type=int, default=1)
    parser.add_argument('--actor_critic_updates_per_model_update', type=int, default=1)
    parser.add_argument('--init_temperature_dyn_entropy', type=float, default=1.0)
    parser.add_argument('--reset_models', type=int, default=1)

    parser.add_argument('--perturb_policy', type=int, default=1)
    parser.add_argument('--perturb_critic', type=int, default=1)
    parser.add_argument('--perturb_model', type=int, default=1)

    parser.add_argument('--policy_perturb_rate', type=float, default=0.2)
    parser.add_argument('--critic_perturb_rate', type=float, default=-1)
    parser.add_argument('--model_perturb_rate', type=float, default=0.2)

    parser.add_argument('--policy_reset_period', type=int, default=999)
    parser.add_argument('--critic_reset_period', type=int, default=999)
    parser.add_argument('--model_reset_period', type=int, default=999)
    parser.add_argument('--use_bronet', type=int, default=1)
    parser.add_argument('--pseudo_ct', type=int, default=0)
    parser.add_argument('--predict_diff', type=int, default=1)
    parser.add_argument('--env_param_mode', type=str, default='stationary', choices=[
        'exponential', 'stationary', 'episodic', 'maximal', 'minimal', 'step', 'slow', 'piecewise'
        ])
    parser.add_argument('--init_state', type=str, default="None", help="Initial state for environment")
    parser.add_argument('--parameter_decay', type=float, default=0.2)

    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    main(args)
