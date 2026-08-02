"""Headless reimplementation of Puzzle A (game.html) as an ARC-AGI-3-style environment.

Faithful port of the inline JS in game.html:
- moveAvatar / isValidPosition / checkAdjacentCorners / getColorAtPosition / checkWinCondition
- Levels 1-5 hardcoded exactly as in game.html level1()..level5()

ARC-AGI-3 alignment:
- The agent interacts through step(action) with an abstract action vocabulary
  (ACTION1..ACTION4, RESET) whose semantics are never revealed.
- Observations are "frames": the 6x6 grid of visual colors exactly as rendered
  to human participants (renderGrid), plus score (levels completed) and state.
- Completing a level increments score and auto-advances to the next level.
"""

import copy

GRID_SIZE = 6

# Visual rendering colors (COLORS in game.html): empty cells are white; the teal
# reset block *looks* teal even though touching it makes the avatar grey.
EMPTY_COLOR = "white"

# Abstract action vocabulary -> movement deltas (dx, dy). Never shown to the agent.
ACTION_DELTAS = {
    "ACTION1": (0, -1),   # up
    "ACTION2": (0, 1),    # down
    "ACTION3": (-1, 0),   # left
    "ACTION4": (1, 0),    # right
}
ACTIONS = list(ACTION_DELTAS.keys()) + ["RESET"]

# Map abstract actions to the arrow-key names used in the human logs (for replay).
KEY_TO_ACTION = {
    "ArrowUp": "ACTION1",
    "ArrowDown": "ACTION2",
    "ArrowLeft": "ACTION3",
    "ArrowRight": "ACTION4",
}


def _lvl(avatar, center, corners, target_color="blue", initial_avatar_color="grey"):
    return {
        "avatar": avatar,
        "center": center,
        "corners": corners,
        "targetColor": target_color,
        "initialAvatarColor": initial_avatar_color,
    }


LEVELS = [
    _lvl(  # level 1
        avatar=(2, 1), center=(3, 3),
        corners=[(0, 0, "teal"), (5, 0, "yellow"), (0, 5, "pink"), (5, 5, "blue")],
        target_color="blue",
    ),
    _lvl(  # level 2
        avatar=(2, 5), center=(2, 4),
        corners=[(0, 0, "blue"), (5, 0, "lime"), (0, 5, "teal"), (5, 5, "pink")],
        target_color="lime", initial_avatar_color="blue",
    ),
    _lvl(  # level 3
        avatar=(5, 2), center=(1, 2),
        corners=[(0, 0, "yellow"), (5, 0, "blue"), (0, 5, "pink"), (5, 5, "teal")],
        target_color="pink", initial_avatar_color="yellow",
    ),
    _lvl(  # level 4
        avatar=(5, 4), center=(2, 0),
        corners=[(0, 0, "yellow"), (5, 0, "pink"), (0, 5, "teal"), (5, 5, "blue")],
        target_color="yellow",
    ),
    _lvl(  # level 5
        avatar=(1, 0), center=(4, 3),
        corners=[(5, 0, "yellow"), (0, 5, "blue"), (2, 5, "pink"), (2, 4, "teal"), (0, 0, "lime")],
        target_color="pink", initial_avatar_color="lime",
    ),
]


class PuzzleAEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        """Full game reset: back to level 1, score 0."""
        self.level_index = 0
        self.score = 0
        self.game_over = False
        self.actions_per_level = [0] * len(LEVELS)
        self._load_level(0)
        return self.observation()

    def _load_level(self, idx):
        level = LEVELS[idx]
        self.avatar = list(level["avatar"])
        self.center = tuple(level["center"])
        self.corners = copy.deepcopy(level["corners"])
        self.target_color = level["targetColor"]
        self.avatar_color = level["initialAvatarColor"]
        self.color_history = [level["initialAvatarColor"]]
        self.level_complete = False

    # --- ports of the JS helpers -------------------------------------------

    def _color_at(self, x, y):
        """getColorAtPosition: teal maps to grey (reset); center is special."""
        for cx, cy, color in self.corners:
            if (x, y) == (cx, cy):
                return "grey" if color == "teal" else color
        if (x, y) == self.center:
            return "center"
        return None

    def _is_corner_block(self, x, y):
        return any((x, y) == (cx, cy) for cx, cy, _ in self.corners)

    def _adjacent_color(self, x, y):
        """checkAdjacentCorners: first match in left, right, up, down order."""
        for ax, ay in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            c = self._color_at(ax, ay)
            if c and c != "center":
                return c
        return None

    def _is_valid(self, x, y):
        if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
            return False
        if self._is_corner_block(x, y):
            return False
        if (x, y) == self.center:
            return False
        return True

    def _check_win(self):
        dx = abs(self.avatar[0] - self.center[0])
        dy = abs(self.avatar[1] - self.center[1])
        if not ((dx == 1 and dy == 0) or (dx == 0 and dy == 1)):
            return False
        if self.avatar_color != self.target_color:
            return False
        last_grey = -1
        for i in range(len(self.color_history) - 1, -1, -1):
            if self.color_history[i] == "grey":
                last_grey = i
                break
        if last_grey == -1:
            return False
        return all(c == self.target_color for c in self.color_history[last_grey + 1:])

    # --- public API ---------------------------------------------------------

    def step(self, action):
        """Apply one action. Counts every action, including invalid/no-op moves,
        matching how human keypresses are logged. Returns the new observation."""
        if self.game_over:
            return self.observation()

        self.actions_per_level[self.level_index] += 1

        if action == "RESET":
            self._load_level(self.level_index)
            return self.observation()

        dx, dy = ACTION_DELTAS[action]
        nx, ny = self.avatar[0] + dx, self.avatar[1] + dy
        if self._is_valid(nx, ny):
            self.avatar = [nx, ny]
            adj = self._adjacent_color(nx, ny)
            if adj:
                self.avatar_color = adj
                self.color_history.append(adj)
            if self._check_win():
                self.score += 1
                if self.level_index + 1 < len(LEVELS):
                    self.level_index += 1
                    self._load_level(self.level_index)
                else:
                    self.game_over = True
        return self.observation()

    def frame(self):
        """The grid exactly as renderGrid paints it for humans."""
        grid = [[EMPTY_COLOR] * GRID_SIZE for _ in range(GRID_SIZE)]
        for cx, cy, color in self.corners:
            grid[cy][cx] = color
        cx, cy = self.center
        if not self._is_corner_block(cx, cy):
            grid[cy][cx] = self.target_color
        grid[self.avatar[1]][self.avatar[0]] = self.avatar_color
        return grid

    def observation(self):
        return {
            "frame": self.frame(),
            "score": self.score,
            "state": "WIN" if self.game_over else "NOT_FINISHED",
        }

    # --- debug/replay helpers ----------------------------------------------

    def letter_matrix(self):
        """generateGameStateMatrix: the E/G/C/M/R/A encoding used in human logs."""
        m = [["E"] * GRID_SIZE for _ in range(GRID_SIZE)]
        for cx, cy, color in self.corners:
            if color == "teal":
                code = "R"
            elif color == self.target_color:
                code = "M"
            else:
                code = "C"
            m[cy][cx] = code
        m[self.center[1]][self.center[0]] = "G"
        m[self.avatar[1]][self.avatar[0]] = "A"
        return m
