# Frozen Lake from First Principles Using Q-Learning


## 1. Introduction

### What is Reinforcement Learning?

Reinforcement Learning (RL) is a branch of machine learning in which an agent learns to make decisions by interacting with an environment. The agent at every timestep observes a state, chooses an action and receives a reward together with a new state. There is no labeled training set like supervised learning which tells the agent the correct action. Instead, it learns from the consequences of its own actions, gradually favoring choices that lead to higher cumulative reward over time. It uses trials repeated over many episodes which allows the agent to discover a mapping from states to actions that maximizes long-term reward.

### What is Frozen Lake?

Frozen Lake is a classic grid-world RL benchmark. An agent starts at a fixed position on a frozen lake represented as a grid and  navigates to a goal tile without falling into a hole. Each grid cell compises of:

| Symbol | Meaning                                       |
|--------|-----------------------------------------------|
| `S`    | Start state                                   |
| `F`    | Frozen (safe, walkable) state                 |
| `H`    | Hole — falling in ends the episode in failure |
| `G`    | Goal — reaching it ends the episode in success|

This project uses the standard 8x8 map:

```
SFFFFFFF
FFFFFFFF
FFFHFFFF
FFFHFFFF
FFFHFFFF
FHHFFFHF
FHFFHFHF
FFFHFFFG
```

The challenge is that the agent must learn, purely from reward signals, both where the holes are and the shortest safe route to the goal.

---

## 2. Environment Design

### State Representation

States are represented as a single integer index, computed from the agent's row and column on the grid:
```
state = row * n_cols + col
```

This gives 64 possible states for the 8x8 grid (indices 0–63), which is a convenient representation for indexing directly into a 
Q-table.

### Action Representation

Four discrete actions, matching the  specification:

| Action | Value | Row/Col Delta |
|--------|-------|----------------|
| Left   | 0     | `(0, -1)`      |
| Down   | 1     | `(1, 0)`       |
| Right  | 2     | `(0, 1)`       |
| Up     | 3     | `(-1, 0)`      |



### Reward Structure

|            Outcome           | Reward |  Episode ends? |
|------------------------------|--------|----------------|
|Step onto a frozen (`F`) tile | -0.01  | No             |
|Step onto a hole (`H`)        | -1     | Yes (failure)  |
|Step onto the goal (`G`)      | +10    | Yes (success)  |

The small negative step cost discourages wandering and pushes the agent toward the shortest safe path, rather than only caring about eventually reaching the goal regardless of path length. Episodes are also capped at 200 steps as a safety limit, in case a partially trained policy gets stuck looping between non-terminal states.



## 3. Q-Learning Algorithm

### Description of Q-Learning

Q-Learning is a model-free, off-policy RL algorithm which maintains a table, `Q(s, a)`, estimates the expected total future reward of taking action `a` in state `s` and then acts optimally afterward. The agent doesn't need a model of the environment's transition probabilities but learns the Q-table purely from the observed `(state, action, reward, next_state)` transitions, updating its estimates after every single step.

### The Update Equation

The Q-value is updated using the Bellman equation:
```
Q(s,a) ← Q(s,a) + α[r + γmaxQ(s′,a′) − Q(s,a)
```
Where:
α = learning rate
γ = discount factor
r = reward
s’ = next state

### Exploration Strategy

The agent employs an epilson greedy exploration strategy which selects random actions with probability epilson and chooses the action with the highest Q-value otherwise. The exploration rate starts at 1.0 and gradually decays to 0.01 over approximately 70% of the training episodes. This slower decay provides sufficient exploration in the large 8×8 environment, allowing the agent to discover the goal state before transitioning to exploitation of the learned policy.


## 4. Training Procedure

### Hyperparameters Used

|      Hyperparameter     |  Value  |              Role                       |
|-------------------------|---------|-----------------------------------------|
| Learning rate (alpha)   | 0.1     | Step size for Q-value updates           |
| Discount factor (gamma) | 0.99    | Weight on future vs. immediate reward   |
| Epsilon (start)         | 1.0     | Initial exploration rate                |
| Epsilon (min)           | 0.01    | Floor exploration rate                  |
| Epsilon decay           |computed | Reaches the floor ~70% through training |
| Max steps per episode   | 200     | Safety cap against infinite loops       |

The selected values of α, γ, and the minimun exploration rate were determined through experimentation and not randomly. Each parameter was changed individually while keeping the others constant, and the agent's performance was compared. The results of these experiments are stored in results/hyperparameter_comparison.csv and visualized in results/hyperparameter_comparison.png.

### Number of Episodes

The agent was trained for 5,000 episodes.

## 5. Results

### Final Success Rate

- Training success rate (last 500 episodes): 99.2 — the rolling success rate once the policy has effectively converged.
- raining success rate (full 5,000 episodes): 80.0 — lower than the figure above only because it includes the early high-epsilon episodes where the agent was still mostly acting randomly.
- Greedy evaluation success rate (100 episodes, zero exploration): 100%, average reward 9.87 per episode. This number that reflects the actual learned policy's quality, with no random actions involved.

### 6. Learned Policy

```
→ ↓ ↓ ↓ ↓ ↓ → ↓
→ → → → ↓ ↓ → ↓
↑ ↑ ↑ H ↓ ↓ ↓ ↓
↑ ↑ ← H → → → ↓
↓ ↑ ↑ H → → → ↓
↓ H H → → ↑ H ↓
↓ H → ↑ H ↑ H ↓
→ → ↑ H ← ← → G
```

The arrows from the start tile (top-left) reaches the goal (bottom-right) in exactly 14 steps which is the shortest path that is geometrically possible. The agent does not just learn 'a' working route but it rahter learned the optimal one. The final policy maps each state to an optimal action that leads the agent toward the goal while avoiding holes.

### Discussion of Performance

The agent learned an effective policy that consistently reaches the goal while avoiding holes. State visitation analysis showed that a few states were visited very infrequently because they were not part of the optimal path. As exploration decreased, the agent focused on the most successful route, resulting in limited training for these states. This behavior is normal in Q-learning and does not negatively affect performance, as the agent achieved a 100% success rate during evaluation.


## 7. Execution Instructions

```bash
pip install -r requirements.txt

python3 train.py                        # trains the agent, prints the policy, saves results/
python3 evaluate.py                      # loads results/q_table.npy, runs 100 greedy episodes
python3 experiments.py                   # hyperparameter ablation (alpha / gamma / epsilon_min)
```

Each script is self-contained and only depends on `environment.py` and
`agent.py`. Run them from the project root so the relative `results/` path
resolves correctly.
