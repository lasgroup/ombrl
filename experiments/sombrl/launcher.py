from experiments.utils import generate_run_commands, generate_base_command, dict_permutations
from experiments.sombrl import experiment as exp
import argparse

PROJECT_NAME = 'SOMBRL_4_Jan2026'

entity = 'sukhijab'
_applicable_configs = {
    'batch_size': [256],
    'seed': list(range(3)),
    'wandb_log': [1],
    'project_name': [PROJECT_NAME],
    'entity_name': [entity],
    'use_tqdm': [1],
    'predict_diff': [1],
    'ens_lr': [3e-4],
    'lr': [3e-4],
    'perturb_policy': [0, 1],
    'perturb_model': [0],
    'reset_models': [1],
    'updates_per_step': [1],
    'int_rew_weight_start': [1.0, 0.0],
    'alg_name': ['sombrl'],
    'action_repeat': [2],
    'num_neurons': [512],
    'num_hidden_layers': [2],

}

cartpole_config = {
    'env_name': ['cartpole-swingup_sparse'],
    'max_steps': [500_000],
    'num_imagined_steps_end': [5],
    'steps_to_imagined_steps_end': [10_000],
    'int_rew_weight_decrease_steps': [100_000],
}

quadruped_config = {
    'env_name': ['quadruped-run'],
    'max_steps': [2_500_000],
    'num_imagined_steps_end': [4, 5],
    'steps_to_imagined_steps_end': [100_000],
    'int_rew_weight_decrease_steps': [250_000],
}


all_flags_combinations = dict_permutations(_applicable_configs | cartpole_config) \
                         + dict_permutations(_applicable_configs | quadruped_config)


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
