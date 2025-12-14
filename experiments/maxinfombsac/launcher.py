from experiments.utils import generate_run_commands, generate_base_command, dict_permutations
from experiments.combrl.auto_tuned_lambda import experiment as exp
import argparse

PROJECT_NAME = 'MaxInfoMBSAC_Dez_14_11_30_Test_1_DT'

"""
Pendulum-v1
MountainCar continuous
Cartpole swingup sparse
Hopper hop
Cheetah run
Quadruped run
"""

entity = 'kiten'
_applicable_configs = {
    'batch_size': [256],
    'seed': list(range(1)),
    'wandb_log': [1],
    'project_name': [PROJECT_NAME],
    'entity_name': [entity],
    'use_tqdm': [0]
}

_applicable_configs_sac = {'alg_name': ['maxinfosac', 'sac'],
                           'ens_lr': [3e-4],
                           'dyn_ent_lr': [3e-4],
                           'lr': [3e-4],
                           'use_bronet': [0],
                           } | _applicable_configs

_applicable_configs_mbsac = {'alg_name': ['maxinfombsac'],
                             'ens_lr': [3e-4],
                             'dyn_ent_lr': [3e-4],
                             'lr': [3e-4],
                             'sample_model': [0],
                             'critic_real_data_update_period': [5],
                             'init_temperature_dyn_entropy': [1.0],
                             'perturb_model': [1],
                             'perturb_policy': [0,1],
                             'use_bronet': [1],
                             } | _applicable_configs

_applicable_configs_mbmean = {'alg_name': ['maxinfombsac'],
                              'ens_lr': [3e-4],
                              'dyn_ent_lr': [0.0],
                              'lr': [3e-4],
                              'sample_model': [0],
                              'critic_real_data_update_period': [5],
                              'init_temperature_dyn_entropy': [1e-8],
                              'perturb_model': [1],
                              'perturb_policy': [0,1],
                              'use_bronet': [1],
                              } | _applicable_configs

_applicable_configs_mbpets = {'alg_name': ['maxinfombsac'],
                              'ens_lr': [3e-4],
                              'dyn_ent_lr': [0.0],
                              'lr': [3e-4],
                              'sample_model': [1],
                              'critic_real_data_update_period': [5],
                              'init_temperature_dyn_entropy': [1e-8],
                              'perturb_model': [1],
                              'perturb_policy': [0,1],
                              'use_bronet': [1],
                              } | _applicable_configs

# add other high dim tasks
configs_humanoid = {
    'env_name': [  #
        # 'humanoid-walk',
        # 'humanoid-stand',
        # 'humanoid-run',
        # 'dog-walk',
        # 'dog-stand',
        # 'dog-run',
        # 'quadruped-walk',
        'quadruped-run',
    ],
    'max_steps': [2_500_000],
    'eval_interval': [10_000],
    'action_repeat': [2],
    'num_neurons': [512],
    'num_hidden_layers': [2],
}


# add other low dim tasks
configs_cartpole = {
    'env_name': ['cartpole-swingup_sparse',
                 # 'reacher-hard',
                 # 'finger-spin',
                 # 'walker-run'
                 ],
    'max_steps': [250_000],
    'eval_interval': [10_000],
    'action_repeat': [2],
    'action_cost': [0.0], #0.1, 0.25, 0.4],
    'num_neurons': [256],
    'num_hidden_layers': [2],
}

configs_others = {
    'env_name': [
        'hopper-hop',
        # 'acrobot-swingup',
        # 'finger-turn_hard',
        # 'walker-run',
        'cheetah-run',
    ],
    'max_steps': [500_000],
    'eval_interval': [10_000],
    'action_repeat': [2],
    'num_neurons': [256],
    'num_hidden_layers': [2],
}

configs_mountaincar = {
    'env_name': ['MountainCarContinuous-v0'],
    'max_steps': [50_000],
    'eval_interval': [1_000],
    'action_repeat': [1],
    'num_neurons': [256],
    'num_hidden_layers': [2],
}

configs_pendulum = {
    'env_name': ['Pendulum-v1'],
    'max_steps': [50_000],
    'eval_interval': [1_000],
    'action_repeat': [1],
    'num_neurons': [256],
    'num_hidden_layers': [2],
    'exp_hash': ['KeepUp', 'SwingUp'],
}

configs_gym = {
    'env_name': ['HalfCheetah-v4',
                 'Hopper-v4'],
    'max_steps': [500_000],
    'eval_interval': [20_000],
    'action_repeat': [2],
    'num_neurons': [256],
    'num_hidden_layers': [2],
}

configs_cheetah = {
    'env_name': ['HalfCheetah-v4'],
    'max_steps': [2_000_000],
    'eval_interval': [20_000],
    'action_repeat': [2],
    'num_neurons': [256],
    'num_hidden_layers': [2],
}

configs_walker = {
    'env_name': ['Walker2d-v4',
                'Humanoid-v4'],
    'max_steps': [2_000_000],
    'eval_interval': [10_000],
    'action_repeat': [2],
    'num_neurons': [256],
    'num_hidden_layers': [2],
}

configs_pusher = {
    'env_name': ['Pusher-v4',
                'Reacher-v4'],
    'max_steps': [250_000],
    'eval_interval': [1_250],
    'action_repeat': [2],
    'num_neurons': [256],
    'num_hidden_layers': [2],
}

"""
all_flags_combinations = dict_permutations(configs_others | _applicable_configs_mbsac) \
                         + dict_permutations(configs_cartpole | _applicable_configs_mbsac) \
                         + dict_permutations(configs_mountaincar | _applicable_configs_mbsac)\
                         + dict_permutations(configs_humanoid | _applicable_configs_mbsac)\
                         + dict_permutations(configs_others | _applicable_configs_mbmean) \
                         + dict_permutations(configs_cartpole | _applicable_configs_mbmean) \
                         + dict_permutations(configs_mountaincar | _applicable_configs_mbmean)\
                         + dict_permutations(configs_humanoid | _applicable_configs_mbmean)
"""
all_flags_combinations = dict_permutations(configs_gym | _applicable_configs_mbsac) \
                         + dict_permutations(configs_mountaincar | _applicable_configs_mbsac)\
                         + dict_permutations(configs_cartpole | _applicable_configs_mbsac)\
                         + dict_permutations(configs_pendulum | _applicable_configs_mbsac)\
                         + dict_permutations(configs_gym | _applicable_configs_mbmean) \
                         + dict_permutations(configs_mountaincar | _applicable_configs_mbmean) \
                         + dict_permutations(configs_cartpole | _applicable_configs_mbmean) \
                         + dict_permutations(configs_pendulum | _applicable_configs_mbmean) \
                         + dict_permutations(configs_gym | _applicable_configs_mbpets) \
                         + dict_permutations(configs_mountaincar | _applicable_configs_mbpets) \
                         + dict_permutations(configs_cartpole | _applicable_configs_mbpets) \    
                         + dict_permutations(configs_pendulum | _applicable_configs_mbpets)
                            

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
    num_hours = 23 if args.long_run else 3
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
