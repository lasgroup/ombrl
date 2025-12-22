from experiments.utils import generate_run_commands, generate_base_command, dict_permutations
from experiments.combrl.downstream_tasks import experiment as exp
import argparse

PROJECT_NAME = 'COMBRL_SAC'

"""
Pendulum-v1
MountainCar continuous
Cartpole swingup sparse
Cheetah run
"""

entity = 'kiten'
_applicable_configs = {
    'batch_size': [256],
    'seed': list(range(10)),
    'wandb_log': [1],
    'project_name': [PROJECT_NAME],
    'entity_name': [entity],
    'use_tqdm': [0],
}

_applicable_configs_sac = {'alg_name': ['maxinfosac', 'sac'],
                           'exp_hash': ['sac'],
                           'ens_lr': [3e-4],
                           'dyn_ent_lr': [3e-4],
                           'lr': [3e-4],
                           'use_bronet': [0],
                           } | _applicable_configs

_applicable_configs_combrl = {'alg_name': ['combrl'],
                              'exp_hash': ['combrl'],
                             'ens_lr': [3e-4],
                             'lr': [3e-4],
                             'sample_model': [0],
                             'critic_real_data_update_period': [5],
                             'perturb_model': [1],
                             'perturb_policy': [0],
                             'use_bronet': [1],
                             'explore_until': [2_500_000],
                             'int_rew_weight_start': [0.1, 1.0, 10.0, 100., 1000., 10_000],
                             'int_rew_weight_end': [0.0],
                             'int_rew_weight_decrease_steps': [-1],
                             } | _applicable_configs

_applicable_configs_copax = {'alg_name': ['combrl'],
                             'exp_hash': ['copax'],
                             'ens_lr': [3e-4],
                             'lr': [3e-4],
                             'sample_model': [0],
                             'critic_real_data_update_period': [5],
                             'perturb_model': [1],
                             'perturb_policy': [0],
                             'use_bronet': [1],
                             'explore_until': [2_500_000],
                             'int_rew_weight_start': [-1],
                             'int_rew_weight_decrease_steps': [-1],
                             } | _applicable_configs

_applicable_configs_mean = {'alg_name': ['combrl'],
                            'exp_hash': ['mean'],
                              'ens_lr': [3e-4],
                              'lr': [3e-4],
                              'sample_model': [0],
                              'critic_real_data_update_period': [5],
                              'perturb_model': [1],
                              'perturb_policy': [0],
                              'use_bronet': [1],
                              'explore_until': [0],
                              'int_rew_weight_start': [-1],
                              'int_rew_weight_decrease_steps': [-1],
                              } | _applicable_configs

_applicable_configs_pets = {'alg_name': ['combrl'],
                            'exp_hash': ['pets'],
                              'ens_lr': [3e-4],
                              'lr': [3e-4],
                              'sample_model': [1],
                              'critic_real_data_update_period': [5],
                              'perturb_model': [1],
                              'perturb_policy': [0],
                              'use_bronet': [1],
                              'explore_until': [0],
                              'int_rew_weight_start': [-1],
                              'int_rew_weight_decrease_steps': [-1],
                              } | _applicable_configs

configs_gym = {
    'env_name': ['HalfCheetah-v4',
                 'Hopper-v4'],
    'max_steps': [1_000_000],
    'eval_interval': [10_000],
    'action_repeat': [2],
    'hidden_dims': [256],
    'int_rew_weight_start': [2.0, 10.0, 50., 100., 1000.],
    # 'num_hidden_layers': [2],
}

configs_cheetah = {
    'env_name': ['HalfCheetah-v4'],
    'max_steps': [1_000_000],
    'eval_interval': [10_000],
    'action_repeat': [2],
    'hidden_dims': [256],
    # 'int_rew_weight_start': [2.0, 10.0, 50., 100., 1000.],
    # 'num_hidden_layers': [2],
}

configs_hopper = {
    'env_name': ['Hopper-v4'],
    'max_steps': [1_000_000],
    'eval_interval': [10_000],
    'action_repeat': [2],
    'hidden_dims': [256],
    # 'int_rew_weight_start': [10.0, 100., 1000., 10_000.],
    # 'num_hidden_layers': [2],
}

all_flags_combinations = dict_permutations(configs_gym | _applicable_configs_copax) \
                         + dict_permutations(configs_gym | _applicable_configs_mean)


def main(args):
    command_list = []
    logs_dir = '../'
    if args.mode == 'euler':
        logs_dir = '/cluster/scratch/'
        logs_dir += entity + '/' + PROJECT_NAME + '/'

    for flags in all_flags_combinations:
        flags['logs_dir'] = logs_dir
        cmd = generate_base_command(exp, flags=flags)
        command_list.append(cmd)

    # submit jobs
    num_hours = 23 if args.long_run else 23
    generate_run_commands(command_list, num_cpus=args.num_cpus, num_gpus=args.num_gpus,
                          mode=args.mode, duration=f'{num_hours}:59:00', prompt=True, mem=32000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_cpus', type=int, default=1, help='number of cpus to use')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus to use')
    parser.add_argument('--mode', type=str, default='euler', help='how to launch the experiments')
    parser.add_argument('--long_run', default=False, action="store_true")

    args = parser.parse_args()
    main(args)
