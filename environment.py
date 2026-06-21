class FrozenLakeEnv:
    """
    Custom 8x8 Frozen Lake environment, implemented from scratch (no Gym).

    Actions:
        0 = Left
        1 = Down
        2 = Right
        3 = Up
    """

    def __init__(self, max_steps=200):

        self.grid = [
            "SFFFFFFF",
            "FFFFFFFF",
            "FFFHFFFF",
            "FFFHFFFF",
            "FFFHFFFF",
            "FHHFFFHF",
            "FHFFHFHF",
            "FFFHFFFG"
        ]

        self.n_rows = 8
        self.n_cols = 8

        self.start_row = 0
        self.start_col = 0

        self.agent_row = self.start_row
        self.agent_col = self.start_col

        self.done = False
        self.goal_reached = False

        # Row/col deltas for each action
        self.actions = {
            0: (0, -1),   # Left
            1: (1, 0),    # Down
            2: (0, 1),    # Right
            3: (-1, 0)    # Up
        }

        # Safety cap so training/evaluation can never loop forever if the
        # agent wanders without hitting a hole or the goal.
        self.max_steps = max_steps
        self.steps = 0

    def get_state(self):
        return self.agent_row * self.n_cols + self.agent_col

    def reset(self):
        self.agent_row = self.start_row
        self.agent_col = self.start_col

        self.done = False
        self.goal_reached = False
        self.steps = 0

        return self.get_state()

    def is_terminal(self):
        return self.done

    def step(self, action):
        move_row, move_col = self.actions[action]
        new_row = self.agent_row + move_row
        new_col = self.agent_col + move_col

        # Enforce grid boundaries (bumping a wall just keeps the agent in place)
        new_row = max(0, min(self.n_rows - 1, new_row))
        new_col = max(0, min(self.n_cols - 1, new_col))

        self.agent_row = new_row
        self.agent_col = new_col
        self.steps += 1

        cell = self.grid[new_row][new_col]

        if cell == "H":
            reward = -1.0
            self.done = True
            self.goal_reached = False

        elif cell == "G":
            reward = 10.0
            self.done = True
            self.goal_reached = True

        else:
            reward = -0.01
            self.done = False
            self.goal_reached = False

        # Truncate over-long episodes (counted as a non-goal failure)
        if self.steps >= self.max_steps and not self.done:
            self.done = True

        return self.get_state(), reward, self.done

    def render(self):
        for r in range(self.n_rows):
            row = ""
            for c in range(self.n_cols):
                if r == self.agent_row and c == self.agent_col:
                    row += "A "
                else:
                    row += self.grid[r][c] + " "
            print(row)
        print()