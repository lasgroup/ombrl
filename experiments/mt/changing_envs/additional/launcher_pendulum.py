from experiments.utils import generate_run_commands, generate_base_command, dict_permutations
from experiments.mt.changing_envs.additional import experiment as exp
import argparse

PROJECT_NAME = 'MT_Jan_21_10_45_Gym_inverted_Test_3_NoTermination'

WANDB_OFFLINE = False
LONG_EXPERIMENT = False

entity = 'kiten'
_applicable_configs = {
    'batch_size': [256],
    'seed': list(range(5)),
    'wandb_log': [1],
    'project_name': [PROJECT_NAME],
    'entity_name': [entity],
    'use_tqdm': [0],
    'pseudo_ct': [0],
    'predict_diff': [1],
    'parameter_decay': [0.0],
    'reset_models': [1],
    'save_video': [0],
    'eval_episodes': [5],
}

_applicable_configs_sac = {'alg_name': ['maxinfosac', 'sac'],
                           'ens_lr': [3e-4],
                           'dyn_ent_lr': [3e-4],
                           'lr': [3e-4],
                           'use_bronet': [0],
                           } | _applicable_configs

_applicable_configs_mbsac = {'alg_name': ['maxinfombsac'],
                             'exp_hash': ['maxinfombsac'],
                             'ens_lr': [3e-4],
                             'dyn_ent_lr': [3e-4],
                             'lr': [3e-4],
                             'sample_model': [0],
                             'critic_real_data_update_period': [5],
                             'init_temperature_dyn_entropy': [1.0],
                             'perturb_model': [0,1],
                             'perturb_policy': [0],
                             'use_bronet': [1],
                             } | _applicable_configs

_applicable_configs_continual = {'alg_name': ['continualmaxinfo'],
                             'exp_hash': ['continual'],
                             'ens_lr': [3e-4],
                             'dyn_ent_lr': [3e-4],
                             'lr': [3e-4],
                             'sample_model': [0],
                             'updates_per_step': [2],
                             'actor_critic_updates_per_model_update': [1],
                             'num_imagined_steps': [1],
                             'init_temperature_dyn_entropy': [1.0],
                             'use_bronet': [1],
                             'env_param_mode': ['stationary'],

                             # replay_buffer_size
                             'replay_buffer_mode': ['reset', 'window'],
                             'replay_buffer_size': [200_000], # TODO ablate

                              # resets / perturbations
                             'perturb_policy': [1],
                             'perturb_critic': [1],
                             'perturb_model': [1],
 
                             'policy_perturb_rate': [0.2],
                             'critic_perturb_rate': [-1],
                             'model_perturb_rate': [0.2],

                             'policy_reset_period': [200], # TODO ablate
                             'critic_reset_period': [200], # TODO ablate
                             'model_reset_period': [200], # TODO ablate
                            } | _applicable_configs

_applicable_configs_continual_mean = {'alg_name': ['continualmaxinfo'],
                             'exp_hash': ['mean'],
                             'ens_lr': [3e-4],
                             'dyn_ent_lr': [3e-4],
                             'lr': [3e-4],
                             'sample_model': [0],
                             'updates_per_step': [2],
                             'actor_critic_updates_per_model_update': [1],
                             'num_imagined_steps': [1],
                             'init_temperature_dyn_entropy': [1.0],
                             'use_bronet': [1],
                             'env_param_mode': ['stationary'],

                              # replay_buffer_size
                             'replay_buffer_mode': ['none'],
                             'replay_buffer_size': [4_000_000],

                             # resets / perturbations
                             'perturb_policy': [0],
                             'perturb_critic': [0],
                             'perturb_model': [0],
 
                             'policy_perturb_rate': [0],
                             'critic_perturb_rate': [0],
                             'model_perturb_rate': [0],
 
                             'policy_reset_period': [9999],
                             'critic_reset_period': [9999],
                             'model_reset_period': [9999],
                            } | _applicable_configs

_applicable_configs_mbmean = {'alg_name': ['maxinfombsac'],
                              'exp_hash': ['mean'],
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
                              'exp_hash': ['pets'],
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

configs_mountaincar = {
    'env_name': ['MountainCarContinuous-v0'],
    'max_steps': [25_000],
    'eval_interval': [1_000],
    'action_repeat': [1],
    'num_neurons': [256],
    'num_hidden_layers': [2],
    'init_state': ["None"],
}

configs_pendulum = {
    'env_name': ['Pendulum-v1'],
    'max_steps': [15_000],
    'eval_interval': [200],
    'action_repeat': [1],
    'num_neurons': [256],
    'num_hidden_layers': [2],
    'init_state': ["3.1415,0.0"],
}

configs_inverted_pendulum = {
    'env_name': ['InvertedPendulum-v4'],
    'max_steps': [50_000],
    'eval_interval': [500],
    'action_repeat': [1],
    'num_neurons': [256],
    'num_hidden_layers': [2],
    'init_state': ["None"],
}

configs_inverted_double_pendulum = {
    'env_name': ['InvertedDoublePendulum-v4'],
    'max_steps': [150_000],
    'eval_interval': [2_000],
    'action_repeat': [1],
    'num_neurons': [256],
    'num_hidden_layers': [2],
    'init_state': ["None"],
}


configs_gym = {
    'env_name': ['HalfCheetah-v4',
                 'Hopper-v4'],
    'max_steps': [500_000],
    'eval_interval': [20_000],
    'action_repeat': [2],
    'num_neurons': [256],
    'num_hidden_layers': [2],
    'init_state': ["None"],
}

configs_cheetah = {
    'env_name': ['HalfCheetah-v4'],
    'max_steps': [500_000],
    'eval_interval': [20_000],
    'action_repeat': [2],
    'num_neurons': [256],
    'num_hidden_layers': [2],
    'init_state': ["None"],
}

configs_swimmer = {
    'env_name': ['Swimmer-v4'],
    'max_steps': [500_000],
    'eval_interval': [20_000],
    'action_repeat': [2],
    'num_neurons': [256],
    'num_hidden_layers': [2],
    'init_state': ["None"],
}

configs_ant = {
    'env_name': ['Ant-v4'],
    'max_steps': [500_000],
    'eval_interval': [20_000],
    'action_repeat': [1],
    'num_neurons': [256],
    'num_hidden_layers': [2],
    'init_state': ["None"],
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

configs_humanoid_standup = {
    'env_name': ['HumanoidStandup-v4'],
    'max_steps': [2_000_000],
    'eval_interval': [20_000],
    'action_repeat': [1],
    'num_neurons': [256],
    'num_hidden_layers': [2],
    'init_state': ["None"],
}

"""
all_flags_combinations = dict_permutations(configs_cheetah | _applicable_configs_continual)\
    + dict_permutations(configs_cheetah | _applicable_configs_continual_mean)             

all_flags_combinations = dict_permutations(configs_gym | _applicable_configs_continual_mean)
"""

all_flags_inverted_pendulums = (
    dict_permutations(configs_inverted_pendulum | _applicable_configs_continual_mean)
  + dict_permutations(configs_inverted_double_pendulum | _applicable_configs_continual_mean)
+ dict_permutations(configs_inverted_pendulum | _applicable_configs_continual)
  + dict_permutations(configs_inverted_double_pendulum | _applicable_configs_continual)

  )

all_flags_ant_and_co = (
    dict_permutations(configs_swimmer | _applicable_configs_continual_mean)
  + dict_permutations(configs_ant | _applicable_configs_continual_mean)
)

all_flags_humanoid_and_walker = (
    dict_permutations(configs_walker | _applicable_configs_continual_mean)
  + dict_permutations(configs_humanoid_standup | _applicable_configs_continual_mean)
)

GROUP = "inverted"  # "inverted", "ant", "humanoid"

if GROUP == "inverted":
    all_flags_combinations = all_flags_inverted_pendulums
elif GROUP == "ant":
    all_flags_combinations = dict_permutations(configs_swimmer | _applicable_configs_continual_mean)


elif GROUP == "humanoid":
    all_flags_combinations = all_flags_humanoid_and_walker
    LONG_EXPERIMENT = True
else:
    raise ValueError(f"Unknown GROUP: {GROUP}")


def main(args):
    if WANDB_OFFLINE: 
        print("WARNING: wandb set to offline")
        import os
        os.system('wandb offline')
    
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
    num_hours = 23 if args.long_run or LONG_EXPERIMENT else 3
    generate_run_commands(command_list, num_cpus=args.num_cpus, num_gpus=args.num_gpus,
                          mode=args.mode, duration=f'{num_hours}:59:00', prompt=True, mem=2000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_cpus', type=int, default=1, help='number of cpus to use')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus to use')
    parser.add_argument('--mode', type=str, default='euler', help='how to launch the experiments')
    parser.add_argument('--long_run', default=False, action="store_true")

    args = parser.parse_args()
    main(args)
