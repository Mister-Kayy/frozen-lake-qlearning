import os
import numpy as np
from environment import FrozenLakeEnv
from agent import QLearningAgent

env = FrozenLakeEnv(max_steps=200)

state_size = env.n_rows * env.n_cols
action_size = 4

agent = QLearningAgent(state_size, action_size)

q_table_path = "results/q_table.npy"
if not os.path.exists(q_table_path):
    raise FileNotFoundError(
        f"Could not find {q_table_path}. Run train.py first to generate it."
    )

agent.q_table = np.load(q_table_path)

episodes = 100
successes = 0
failures = 0
total_rewards = 0

for episode in range(episodes):
    state = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action = int(np.argmax(agent.q_table[state]))
        next_state, reward, done = env.step(action)
        state = next_state
        episode_reward += reward

    total_rewards += episode_reward

    if env.goal_reached:
        successes += 1
    else:
        failures += 1

success_rate = (successes / episodes) * 100
average_reward = total_rewards / episodes

print("Evaluation Results")
print("------------------")
print("Success Rate:", success_rate, "%")
print("Average Reward:", average_reward)
print("Successful Runs:", successes)
print("Failures:", failures)