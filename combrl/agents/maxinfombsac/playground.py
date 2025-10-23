import gymnasium as gym

# envs = ['Pendulum-v1', 'Walker2d-v4', 'Pusher-v4', 'Reacher-v4']
envs = ['Pusher-v4']

for env in envs:
    current_env = gym.make(env)
    current_env.reset()
    action = current_env.action_space.sample()
    current_env.step(action)
