import numpy as np
import random


class QLearningAgent:
    """
    Tabular Q-Learning agent implemented from scratch.

    Update rule:
        Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
    """

    def __init__(self, state_size, action_size,
                 alpha=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):

        self.state_size = state_size
        self.action_size = action_size

        # Q-table initialized to 0
        self.q_table = np.zeros((state_size, action_size))

        # How many times each state has been updated. Used purely for
        # diagnostics: states that are rarely visited (because they're off
        # the optimal path) end up with noisy, effectively-untrained
        # Q-values, and it's worth being able to show that honestly.
        self.state_visits = np.zeros(state_size, dtype=int)

        # Hyperparameters (exposed as constructor args so Part C
        # experiments can sweep them easily)
        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def _greedy_action(self, state):
        """Argmax over Q-values, breaking ties randomly instead of always
        picking the lowest action index. Without this, every unvisited
        state (Q-values all zero) silently defaults to 'Left'."""
        q_values = self.q_table[state]
        max_q = np.max(q_values)
        best_actions = np.flatnonzero(q_values == max_q)
        return np.random.choice(best_actions)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        return self._greedy_action(state)

    def update(self, state, action, reward, next_state, done):
        self.state_visits[state] += 1

        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])

        target = reward + (0 if done else self.gamma * next_max)

        self.q_table[state, action] = old_value + self.alpha * (target - old_value)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)