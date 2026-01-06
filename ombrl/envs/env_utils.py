import gymnasium as gym
import jax.numpy as jnp
from optax.schedules import linear_schedule, constant_schedule
from typing import Callable, Optional


def get_scheduler_apply_fn(env_name: str = None, env_param_mode: str = None, **kwargs) -> tuple[
    Optional[Callable[[int], dict]], Optional[Callable[[gym.Env, dict], None]], dict
    ]:
    if env_param_mode == 'stationary':
        scheduler_fn: Optional[Callable[[int], dict]] = None
        apply_fn: Optional[Callable[[gym.Env, dict], None]] = None

        return scheduler_fn, apply_fn, {}
    
    if env_name == 'Pendulum-v1':
        def apply_fn(base_env: gym.Env, params: dict):
            base_env.max_torque = params["max_torque"]
            base_env.action_space.low[:] = -params["max_torque"]
            base_env.action_space.high[:] = params["max_torque"]

        env_logs = {}
        if env_param_mode == 'episodic':
            pendulum_max_torque_init = 5.0
            pendulum_max_torque_final = 1.0
            pendulum_max_torque_transition_steps = 10
            pendulum_max_torque_transition_begin = 5
            torque_schedule = linear_schedule(
                init_value=pendulum_max_torque_init,
                end_value=pendulum_max_torque_final,
                transition_steps=pendulum_max_torque_transition_steps,
                transition_begin=pendulum_max_torque_transition_begin # TODO see if warm up is needed
                )
                
            env_logs["pendulum_max_torque_init"] = pendulum_max_torque_init
            env_logs["pendulum_max_torque_final"] = pendulum_max_torque_final
            env_logs["pendulum_torque_transition_steps"] = pendulum_max_torque_transition_steps
            env_logs["pendulum_torque_transition_begin"] = pendulum_max_torque_transition_begin

            def scheduler_fn(ep_idx: int):
                return {"max_torque": float(torque_schedule(ep_idx))}


        elif env_param_mode == 'slow':
            pendulum_max_torque_init = 5.0
            pendulum_max_torque_final = 1.0
            pendulum_max_torque_transition_steps = 25
            pendulum_max_torque_transition_begin = 5
            torque_schedule = linear_schedule(
                init_value=pendulum_max_torque_init,
                end_value=pendulum_max_torque_final,
                transition_steps=pendulum_max_torque_transition_steps,
                transition_begin=pendulum_max_torque_transition_begin # TODO see if warm up is needed
                )
                
            env_logs["pendulum_max_torque_init"] = pendulum_max_torque_init
            env_logs["pendulum_max_torque_final"] = pendulum_max_torque_final
            env_logs["pendulum_torque_transition_steps"] = pendulum_max_torque_transition_steps
            env_logs["pendulum_torque_transition_begin"] = pendulum_max_torque_transition_begin

            def scheduler_fn(ep_idx: int):
                return {"max_torque": float(torque_schedule(ep_idx))}


        elif env_param_mode == 'maximal':
            torque_schedule = constant_schedule(5.0)
            def scheduler_fn(ep_idx: int):
                return {"max_torque": float(torque_schedule(ep_idx))}
            env_logs["pendulum_max_torque"] = 5.0

        elif env_param_mode == 'minimal':
            torque_schedule = constant_schedule(1.0)
            def scheduler_fn(ep_idx: int):
                return {"max_torque": float(torque_schedule(ep_idx))}
            env_logs["pendulum_max_torque"] = 1.0

        elif env_param_mode == 'step':
            def scheduler_fn(episode_idx: int):
                step_change_episode = 10
                if episode_idx < step_change_episode:
                    return {"max_torque": 5.0}
                else:
                    return {"max_torque": 1.0}

            env_logs["pendulum_step_change_episode"] = 10
            env_logs["pendulum_max_torque_init"] = 5.0
            env_logs["pendulum_max_torque_final"] = 1.0
            
        elif env_param_mode == 'piecewise':
            def scheduler_fn(episode_idx: int):
                if episode_idx < 5:
                    val = 5.0
                elif episode_idx < 10:
                    val = 4.0
                elif episode_idx < 15:
                    val = 3.0
                elif episode_idx < 20:
                    val = 2.0
                else:
                    val = 1.0
                return {"max_torque": val}


            env_logs["pendulum_step_change_episode"] = 5
            env_logs["pendulum_max_torque_init"] = 5.0
            env_logs["pendulum_max_torque_final"] = 1.0

        elif env_param_mode == 'exponential':
            decay_rate = kwargs.get('parameter_decay', 0.0)
            max_torque = 5.0
            min_torque = 1.0
            transition_begin = 5

            def scheduler_fn(ep_idx: int):
                t = max(0, ep_idx - transition_begin)
                val = jnp.exp(-decay_rate * t) * (max_torque - min_torque) + min_torque
                
                return {"max_torque": float(val)}

            env_logs.update({
                "pendulum_max_torque_init": max_torque,
                "pendulum_max_torque_final": min_torque,
                "pendulum_torque_decay_rate": decay_rate,
                "transition_begin": transition_begin
            })

        else:
            raise ValueError(f"env_param_mode={env_param_mode} not supported for {env_name}")
                
    elif env_name == 'MountainCarContinuous-v0':
        def apply_fn(base_env: gym.Env, params: dict):
            base_env.power = params["power"] 

        env_logs = {}
        if env_param_mode == 'episodic':
            mountaincar_power_init = 0.004
            mountaincar_power_final = 0.001
            mountaincar_power_transition_steps = 10
            power_schedule = linear_schedule(
                init_value=mountaincar_power_init,
                end_value=mountaincar_power_final,
                transition_steps=mountaincar_power_transition_steps,
                )
                
            env_logs["mountaincar_power_init"] = mountaincar_power_init
            env_logs["mountaincar_power_final"] = mountaincar_power_final
            env_logs["mountaincar_power_transition_steps"] = mountaincar_power_transition_steps

            def scheduler_fn(ep_idx: int):
                return {"power": float(power_schedule(ep_idx))}
            
        elif env_param_mode == 'maximal':
            power_schedule = constant_schedule(0.004)
            def scheduler_fn(ep_idx: int):
                return {"power": float(power_schedule(ep_idx))}
            env_logs["mountaincar_power"] = 0.004

        elif env_param_mode == 'minimal':
            power_schedule = constant_schedule(0.001)
            def scheduler_fn(ep_idx: int):
                return {"power": float(power_schedule(ep_idx))}
            env_logs["mountaincar_power"] = 0.001

        elif env_param_mode == 'exponential':
            decay_rate = kwargs.get('parameter_decay', 0.0)

            max_power = 0.004
            min_power = 0.001
            transition_begin = 50


            def scheduler_fn(ep_idx: int):
                t = max(0, ep_idx - transition_begin)
                val = jnp.exp(-decay_rate * t) * (max_power - min_power) + min_power
                return {"power": float(val)}

            env_logs.update({
                "mountaincar_power_init": max_power,
                "mountaincar_power_final": min_power,
                "mountaincar_power_decay_rate": decay_rate,
                "transition_begin": transition_begin,
            })

        else:
            raise ValueError(f"env_param_mode={env_param_mode} not supported for {env_name}")    
            
    elif env_name == 'HalfCheetah-v4':
        # Store the original gear ratios to use as a baseline for scaling
        # These are the default MuJoCo values for HalfCheetah-v4
        base_gears = jnp.array([120., 90., 60., 120., 60., 30.])

        def apply_fn(base_env: gym.Env, params: dict):
            # Multiply the entire baseline array by the gear_scale factor
            new_gears = base_gears * params["gear_scale"]
            # Convert back to numpy for the MuJoCo C-binding compatibility
            base_env.unwrapped.model.actuator_gear[:, 0] = new_gears

        env_logs = {}
        # Configuration for HalfCheetah strength
        max_scale = 1.0  # Full strength
        min_scale = 0.2  # 20% strength (very weak)

        if env_param_mode == 'episodic':
            transition_steps = 15
            transition_begin = 5
            scale_schedule = linear_schedule(
                init_value=max_scale,
                end_value=min_scale,
                transition_steps=transition_steps,
                transition_begin=transition_begin
            )
            
            env_logs.update({
                "cheetah_scale_init": max_scale,
                "cheetah_scale_final": min_scale,
                "cheetah_transition_steps": transition_steps
            })

            def scheduler_fn(ep_idx: int):
                return {"gear_scale": float(scale_schedule(ep_idx))}

        elif env_param_mode == 'maximal':
            def scheduler_fn(ep_idx: int): return {"gear_scale": max_scale}
            env_logs["cheetah_gear_scale"] = max_scale

        elif env_param_mode == 'minimal':
            def scheduler_fn(ep_idx: int): return {"gear_scale": min_scale}
            env_logs["cheetah_gear_scale"] = min_scale

        elif env_param_mode == 'step':
            def scheduler_fn(episode_idx: int):
                return {"gear_scale": max_scale if episode_idx < 10 else min_scale}
            env_logs["cheetah_step_episode"] = 10

        elif env_param_mode == 'exponential':
            decay_rate = kwargs.get('parameter_decay', 0.007)
            transition_begin = 400

            def scheduler_fn(ep_idx: int):
                t = max(0, ep_idx - transition_begin)
                val = jnp.exp(-decay_rate * t) * (max_scale - min_scale) + min_scale
                return {"gear_scale": float(val)}

            env_logs.update({
                "cheetah_scale_init": max_scale,
                "cheetah_scale_final": min_scale,
                "cheetah_decay_rate": decay_rate,
            })
        else:
            raise ValueError(f"env_param_mode={env_param_mode} not supported for {env_name}")
    
    else:
        raise ValueError(f"Unknown env_name: {env_name}")
    
    return scheduler_fn, apply_fn, env_logs

""""" 
def main():
    import matplotlib.pyplot as plt
    env_name = "MountainCarContinuous-v0"
    env_param_mode = "exponential"
    num_episodes = 10 
    alphas = [0.0]  # Testing a range of decay speeds

    plt.figure(figsize=(10, 6))

    for alpha in alphas:
        scheduler_fn, _, _ = get_scheduler_apply_fn(
            env_name=env_name,
            env_param_mode=env_param_mode,
            parameter_decay=alpha
        )

        episodes = range(num_episodes)
        torques = [scheduler_fn(ep)["power"] for ep in episodes]
        
        plt.plot([ep_idx*200 for ep_idx in episodes], torques, label=f'α = {alpha}')

    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Min Torque (1.0)')
    plt.title(f"Exponential Decay Visualization (Pendulum-v1)", fontsize=14)
    plt.xlabel("Environment steps", fontsize=12)
    plt.ylabel("Max Torque Value", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()
"""
def main():
    import matplotlib.pyplot as plt
    env_name = "MountainCarContinuous-v0"
    env_param_mode = "exponential"
    
    # Increase episodes to see the full decay curve
    num_episodes = 150 
    # Use a range of alphas to see how fast the car gets "weaker"
    alphas = [0.0, 0.05, 0.1, 0.2] 

    plt.figure(figsize=(10, 6))

    for alpha in alphas:
        scheduler_fn, _, _ = get_scheduler_apply_fn(
            env_name=env_name,
            env_param_mode=env_param_mode,
            parameter_decay=alpha
        )

        episodes = range(num_episodes)
        # Change "torques" to "powers" and access the correct dictionary key
        powers = [scheduler_fn(ep)["power"] for ep in episodes]
        
        # Plotting against episode index
        plt.plot(episodes, powers, label=f'α = {alpha}')

    plt.axhline(y=0.001, color='r', linestyle='--', alpha=0.5, label='Min Power (0.001)')
    plt.title(f"Exponential Power Decay: {env_name}", fontsize=14)
    plt.xlabel("Episode Index", fontsize=12)
    plt.ylabel("Engine Power", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

if __name__ == "__main__":
    main()
