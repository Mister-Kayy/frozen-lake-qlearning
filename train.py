import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from environment import FrozenLakeEnv
from agent import QLearningAgent

os.makedirs("results", exist_ok=True)

env = FrozenLakeEnv(max_steps=200)

state_size = env.n_rows * env.n_cols
action_size = 4

EPISODES = 5000

# --- Hyperparameters (Part C: tune these and re-run) ---
ALPHA = 0.1     # learning rate
GAMMA = 0.99    # discount factor
EPSILON_START = 1.0
EPSILON_MIN = 0.01

# Decay epsilon so it reaches EPSILON_MIN around 70% of the way through
# training, instead of in the first ~900 episodes. With this many holes,
# the agent needs a longer exploration window to actually stumble onto
# the goal before it starts exploiting.
decay_episodes = int(EPISODES * 0.7)
EPSILON_DECAY = (EPSILON_MIN / EPSILON_START) ** (1 / decay_episodes)

agent = QLearningAgent(
    state_size, action_size,
    alpha=ALPHA, gamma=GAMMA,
    epsilon=EPSILON_START, epsilon_decay=EPSILON_DECAY, epsilon_min=EPSILON_MIN
)

episode_rewards = []
epsilon_values = []
success_flags = []   # 1 if that episode reached the goal, else 0

success_count = 0

for episode in range(EPISODES):

    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    # Use the environment's own flag, not the raw terminal reward
    if env.goal_reached:
        success_count += 1
        success_flags.append(1)
    else:
        success_flags.append(0)

    episode_rewards.append(total_reward)
    epsilon_values.append(agent.epsilon)

    agent.decay_epsilon()

    if (episode + 1) % 500 == 0:
        recent_success_rate = sum(success_flags[-500:]) / 500 * 100
        print(
            f"Episode {episode+1}, "
            f"Total Successes: {success_count}, "
            f"Last 500 Success Rate: {recent_success_rate:.1f}%, "
            f"Epsilon: {agent.epsilon:.3f}"
        )

print("\nTraining Complete")
print("Successful Episodes:", success_count)
print("Success Rate (training, includes exploration phase):",
      round(success_count / EPISODES * 100, 2), "%")

# --- Required Part D output: the policy, in the exact specified symbols ---
policy_symbols = {0: "←", 1: "↓", 2: "→", 3: "↑"}

print("\nLearned Policy:\n")
for r in range(env.n_rows):
    row = ""
    for c in range(env.n_cols):
        cell = env.grid[r][c]
        if cell == "H":
            row += "H "
        elif cell == "G":
            row += "G "
        else:
            state = r * env.n_cols + c
            action = int(np.argmax(agent.q_table[state]))
            row += policy_symbols[action] + " "
    print(row)

# --- Diagnostic, not a graded deliverable on its own: how many times was
# each state actually updated? States with very few visits are off the
# optimal path and their arrows above may not be meaningful. Worth a
# sentence in the report rather than claiming every cell is "learned".
print("\nState visit counts (low numbers = undertrained / off optimal path):\n")
for r in range(env.n_rows):
    row = ""
    for c in range(env.n_cols):
        cell = env.grid[r][c]
        if cell in ("H", "G"):
            row += f"{cell:>4} "
        else:
            state = r * env.n_cols + c
            row += f"{agent.state_visits[state]:>4} "
    print(row)

UNDERTRAINED_THRESHOLD = 10
undertrained = [
    (r, c) for r in range(env.n_rows) for c in range(env.n_cols)
    if env.grid[r][c] not in ("H", "G")
    and agent.state_visits[r * env.n_cols + c] < UNDERTRAINED_THRESHOLD
]
if undertrained:
    print(f"\n{len(undertrained)} state(s) visited fewer than {UNDERTRAINED_THRESHOLD} times "
          f"(policy there is not reliable): {undertrained}")

# --- Final, exploration-free evaluation for a clean headline number ---
eval_env = FrozenLakeEnv(max_steps=200)
eval_episodes = 100
eval_successes = 0
eval_total_reward = 0.0

for _ in range(eval_episodes):
    state = eval_env.reset()
    done = False
    episode_reward = 0.0

    while not done:
        action = int(np.argmax(agent.q_table[state]))
        next_state, reward, done = eval_env.step(action)
        state = next_state
        episode_reward += reward

    eval_total_reward += episode_reward
    if eval_env.goal_reached:
        eval_successes += 1

print("\nGreedy Evaluation (100 episodes, no exploration):")
print("Success Rate:", eval_successes / eval_episodes * 100, "%")
print("Average Reward:", round(eval_total_reward / eval_episodes, 3))

# --- Save artifacts required by the assignment ---
np.save("results/q_table.npy", agent.q_table)
np.save("results/state_visits.npy", agent.state_visits)

with open("results/training_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["episode", "reward", "epsilon", "success"])
    for i in range(EPISODES):
        writer.writerow([i + 1, episode_rewards[i], epsilon_values[i], success_flags[i]])

fig, axes = plt.subplots(2, 1, figsize=(8, 8))

axes[0].plot(episode_rewards)
axes[0].set_title("Episode Rewards Over Time")
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Reward")

window = 100
rolling_success = [
    sum(success_flags[max(0, i - window):i + 1]) / min(window, i + 1) * 100
    for i in range(EPISODES)
]
axes[1].plot(rolling_success)
axes[1].set_title(f"Rolling Success Rate (window={window})")
axes[1].set_xlabel("Episode")
axes[1].set_ylabel("Success Rate (%)")

plt.tight_layout()
plt.savefig("results/training_plots.png")
plt.show()