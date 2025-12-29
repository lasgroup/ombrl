import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RescaleAction
from optax.schedules import linear_schedule
from wrappers import InitWrapper, EpisodicParamWrapper

class MountainCarAgent:
    def __init__(self, mode="direct"): self.mode = mode
    def sample_actions(self, obs):
        if self.mode == "direct": return np.array([1.0], dtype=np.float32)
        return np.array([np.sign(obs[1]) if abs(obs[1]) > 1e-4 else 1.0], dtype=np.float32)

def run_power_sweep(name, mode, start, end, steps, init_state=np.array([-0.5, 0.0])):
    print(f"\n--- Finding {name} Threshold ---")
    schedule = linear_schedule(init_value=start, end_value=end, transition_steps=steps)
    def apply_fn(env, p): env.power = p["power"]
    
    # Sweep Loop
    base_env = gym.make("MountainCarContinuous-v0")
    env = InitWrapper(base_env, init_state=init_state) if mode == "direct" else base_env
    env = RescaleAction(env, -1.0, 1.0)
    env = EpisodicParamWrapper(env, lambda ep: {"power": float(schedule(ep))}, apply_fn)
    
    agent, last_success = MountainCarAgent(mode), start
    for ep in range(steps):
        obs, info = env.reset()
        p, done, success = info["power"], False, False
        while not done:
            obs, _, term, trunc, _ = env.step(agent.sample_actions(obs))
            if mode == "direct" and obs[1] < -1e-5: break 
            if obs[0] >= 0.45: success = True; break
            done = term or trunc
        if success: last_success = p
        print(f"Ep {ep:02d} | Power: {p:.5f} | {'SUCCESS' if success else 'FAIL'}")
    return last_success

def visualize_threshold(name, mode, power, init_state=None):
    print(f"\nVisualizing {name} at Power: {power:.5f}...")
    env = gym.make("MountainCarContinuous-v0", render_mode="human")
    if init_state is not None: env = InitWrapper(env, init_state=init_state)
    env = RescaleAction(env, -1.0, 1.0)
    env.unwrapped.power = power # Manually set threshold
    
    agent = MountainCarAgent(mode)
    obs, _ = env.reset()
    for _ in range(1000):
        obs, _, term, trunc, _ = env.step(agent.sample_actions(obs))
        env.render()
        if term or trunc: break
    env.close()

if __name__ == "__main__":
    max_p = run_power_sweep("Direct-Drive Max", "direct", 0.005, 0.001, 15)
    min_p = run_power_sweep("Swing-Up Min", "swing", 0.0015, 0.0001, 20)

    # Barely gets up directly
    
    visualize_threshold("Barely Direct", "direct", 0.004, init_state=np.array([-0.5, 0.0]))
    
    # Barely swings up (Ultimate Physical Limit)
    
    visualize_threshold("Barely Swinging", "swing", 0.0004)