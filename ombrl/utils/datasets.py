from jaxrl.datasets import ReplayBuffer


class ResetReplayBuffer(ReplayBuffer):
    def __init__(self,
                 observation_space,
                 action_space,
                 capacity: int):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            capacity=capacity,
        )

    def reset(self):
        self.insert_index = 0
        self.size = 0


if __name__ == "__main__":
    import gymnasium as gym

    env = gym.make("Pendulum-v1")

    buffer = ResetReplayBuffer(
        observation_space=env.observation_space,
        action_space=env.action_space,
        capacity=1000,
    )

    obs, _ = env.reset()

    num_steps = 2001
    reset_count = 0

    for step in range(num_steps):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)

        mask = 0.0 if terminated else 1.0
        done_float = float(terminated or truncated)

        if buffer.size == 100:
            batch = buffer.sample(200)

        if buffer.size == buffer.capacity:
            buffer.reset()
            reset_count += 1

        buffer.insert(obs, action, reward, mask, done_float, next_obs)

        obs = next_obs

        if terminated or truncated:
            obs, _ = env.reset()

    print(f"Finished {num_steps} steps")
    print(f"Buffer capacity: {buffer.capacity}")
    print(f"Current replay buffer size: {buffer.size}")
    print(f"Number of buffer resets: {reset_count}")
