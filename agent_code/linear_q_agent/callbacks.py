import os
import pickle
import random
import numpy as np
from collections import deque

from .model import Linear_QModel

DIRECTIONS = [
    (0, -1),    # UP
    (0, 1),     # DOWN
    (-1, 0),    # LEFT
    (1, 0)      # RIGHT
]

TASK1_ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT']
TASK2_ACTIONS = TASK1_ACTIONS + ["BOMB"]

EPSILON_START = 1.0
FEATURE_SIZES = {"f0": 7, "f1": 11, "f2": 25}

BOMB_POWER = 3

def setup(self):
    """
    This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows to share trained agent
    with other students, without revealing training code.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Seed
    seed_value = os.getenv("EXPERIMENT_SEED", "0")
    self.experiment_seed = int(seed_value)
    self.rng = random.Random(self.experiment_seed)
    self.logger.info(f"Experiment seed: {self.experiment_seed}")

    # Model Start Mode
    self.model_start_mode = os.getenv("MODEL_START_MODE", "resume")

    # Check if the model_start_mode is either "resume" or "fresh"
    if self.model_start_mode not in {"resume", "fresh"}:
        raise ValueError("MODEL_START_MODE must be either 'resume' or 'fresh'.")

    self.feature_mode = os.getenv("FEATURE_MODE", "f1")

    if self.feature_mode not in FEATURE_SIZES:
        raise ValueError("FEATURE_MODE must be one of 'f0', 'f1', or 'f2'.")

    self.feature_size = FEATURE_SIZES[self.feature_mode]
    self.logger.info(f"Feature mode: {self.feature_mode} ({self.feature_size} features)")

    self.actions = actions_for_feature_mode(self.feature_mode)

    # Check if file exists
    checkpoint_exists = os.path.isfile("my-saved-model.pt")

    if self.train and self.model_start_mode == "fresh":
        # Initialize fresh model
        self.logger.info("Setting up model from scratch.")
        self.model = Linear_QModel(input_size=self.feature_size, output_size=len(self.actions), seed=self.experiment_seed)
        self.epsilon = EPSILON_START
    else:
        if not checkpoint_exists:
            raise FileNotFoundError("'my-saved-model.pt' does not exist.")

        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            checkpoint = pickle.load(file)

        if isinstance(checkpoint, dict) and "model" in checkpoint:
            self.model = checkpoint["model"]
            self.epsilon = float(checkpoint.get("epsilon", EPSILON_START))
        else:
            # Support model-only files created before epsilon was persisted.
            self.model = checkpoint
            self.epsilon = EPSILON_START

        if self.model.input_size != self.feature_size:
            raise ValueError(f"Checkpoint expects {self.model.input_size} features, but FEATURE_MODE='{self.feature_mode}' uses {self.feature_size}.")

        if self.model.output_size != len(self.actions):
            raise ValueError(f"Checkpoint expects {self.model.output_size} actions, but FEATURE_MODE='{self.feature_mode}' uses {len(self.actions)}.")


def act(self, game_state: dict) -> str:
    """
    Agent should parse the input, think, and take a decision.
    """
    features = state_to_features(game_state, self.feature_mode)

    # Exploration during training
    if self.train and self.rng.random() < self.epsilon:
        self.logger.debug("Choosing action purely at random.")
        return self.rng.choice(self.actions)

    # Exploitation
    q_values = self.model.predict(features)

    # Choose action with highest Q-value
    action_index = int(np.argmax(q_values))
    action = self.actions[action_index]

    self.logger.debug("Choosing action with the highest Q-value.")

    return action


def state_to_features(game_state: dict, feature_mode: str) -> np.ndarray:
    """
    Converts the game state to a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    if feature_mode not in FEATURE_SIZES:
        raise ValueError("feature_mode must be one of 'f0', 'f1', or 'f2'.")

    # Get the current location of the agent
    field = game_state["field"] # np.ndarray
    # bombs = game_state["bombs"] # (x, y), timer
    coins = game_state["coins"] # x, y
    agent = game_state["self"] # name, score, bombs_left, (x, y)

    agent_x, agent_y = agent[3]

    field_x, field_y = field.shape

    # Create a feature vector
    # F0: [1, free_U, free_D, free_L, free_R, coin_dx, coin_dy]
    # F1: F0 + [path_U, path_D, path_L, path_R]
    # F2: F1 + [bomb_available, 
    #           adjacent_crate_U, adjacent_crate_D, adjacent_crate_L, adjacent_crate_R, 
    #           crate_path_U, crate_path_D, crate_path_L, crate_path_R, 
    #           in_bomb_danger, 
    #           escape_U, escape_D, escape_L, escape_R]
    feature_size = FEATURE_SIZES[feature_mode]
    features = np.zeros(feature_size)

    # Set bias as 1
    features[0] = 1

    # Check the obstacle location around the agent
    UP = field[agent_x, agent_y - 1]
    DOWN = field[agent_x, agent_y + 1]
    LEFT = field[agent_x - 1, agent_y]
    RIGHT = field[agent_x + 1, agent_y]

    if UP == 0:
        features[1] = 1
    if DOWN == 0:
        features[2] = 1
    if LEFT == 0:
        features[3] = 1
    if RIGHT == 0:
        features[4] = 1

    # Calculate the distance (dx, dy) between the agent and the coin, and normalize dx, dy
    closest_distance = field_x + field_y
    closest_coin = None

    if len(coins) == 0:
        features[5] = 0
        features[6] = 0
    else:
        for coin in coins:
            coin_dx = (coin[0] - agent_x)
            coin_dy = (coin[1] - agent_y)

            # Search the coin with the smallest Manhattan distance
            manhattan_distance = np.abs(coin_dx) + np.abs(coin_dy)

            if closest_distance > manhattan_distance:
                features[5] = coin_dx / (field_x - 1)
                features[6] = coin_dy / (field_y - 1)

                closest_distance = manhattan_distance
                closest_coin = coin

        if closest_coin is not None and feature_mode in {"f1", "f2"}:
            path_directions = shortest_path_directions(field, agent[3], closest_coin)
            features[7:11] = path_directions

    if feature_mode == "f2":
        # 11: bomb available
        features[11] = float(agent[2] > 0)

        # 12:16: adjacent crates [UP, DOWN, LEFT, RIGHT]
        for i, (dx, dy) in enumerate(DIRECTIONS):
            nx = agent_x + dx
            ny = agent_y + dy

            if field[nx, ny] == 1:
                features[12 + i] = 1.0

        # 16:20: path to crates [UP, DOWN, LEFT, RIGHT]
        crate_targets = crate_placement_targets(field)
        features[16:20] = shortest_path_directions_to_any(field, agent[3], crate_targets)

        # 20: bomb_danger
        bombs = game_state["bombs"]
        explosion_map = game_state.get("explosion_map")

        danger_tiles = bomb_danger_tiles(field, bombs, explosion_map)
        features[20] = float(agent[3] in danger_tiles)

        # 21:25: escape direction [UP, DOWN, LEFT, RIGHT]
        safe_targets = set()

        for x in range(field.shape[0]):
            for y in range(field.shape[1]):
                if field[x, y] == 0 and (x, y) not in danger_tiles:
                    safe_targets.add((x, y))

        escape_field = field.copy()

        for (bomb_x, bomb_y), _timer in bombs:
            if (bomb_x, bomb_y) != agent[3]:
                escape_field[bomb_x, bomb_y] = -1

        if features[20] == 1.0:
            features[21:25] = shortest_path_directions_to_any(escape_field, agent[3], safe_targets)

    # Return the final feature vector
    return features


def shortest_path_directions(field, start, target):
    """Return valid first-step directions along shortest paths from start to target."""
    return shortest_path_directions_to_any(field, start, {target})


def actions_for_feature_mode(feature_mode):
    if feature_mode == "f2":
        return TASK2_ACTIONS
    return TASK1_ACTIONS


def crate_placement_targets(field):
    targets = set()

    crate_positions = np.argwhere(field == 1)

    for crate_x, crate_y in crate_positions:
        for dx, dy in DIRECTIONS:
            nx = crate_x + dx
            ny = crate_y + dy

            if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1] and field[nx, ny] == 0:
                targets.add((nx, ny))

    return targets


def shortest_path_directions_to_any(field, start, targets):
    """"""

    # No movement needed if already at target
    if not targets:
        return np.zeros(4)

    if start in targets:
        return np.zeros(4)

    # Distance from each tile to the target
    distances = np.full(field.shape, -1)
    queue = deque()

    for target in targets:
        distances[target] = 0
        queue.append(target)

    # BFS starting from the target
    while queue:
        x, y = queue.popleft()

        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy

            # Check field boundaries
            if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
                continue

            # Task 1: only free tiles are walkable
            if field[nx, ny] != 0:
                continue

            # Already visited
            if distances[nx, ny] != -1:
                continue

            distances[nx, ny] = distances[x, y] + 1
            queue.append((nx, ny))

    # Target cannot be reached from start
    if distances[start] == -1:
        return np.zeros(4)

    current_distance = distances[start]
    path_directions = np.zeros(4)

    # Check which neighboring tiles reduce the shortest-path distance by 1
    for i, (dx, dy) in enumerate(DIRECTIONS):
        nx = start[0] + dx
        ny = start[1] + dy

        if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
            continue

        if distances[nx, ny] == current_distance - 1:
            path_directions[i] = 1.0

    return path_directions


def bomb_danger_tiles(field, bombs, explosion_map=None):
    danger_tiles = set()

    for (bomb_x, bomb_y), _timer in bombs:
        danger_tiles.add((bomb_x, bomb_y))

        for dx, dy in DIRECTIONS:
            for distance in range(1, BOMB_POWER + 1):
                nx = bomb_x + dx * distance
                ny = bomb_y + dy * distance

                if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
                    break

                if field[nx, ny] == -1:
                    break

                danger_tiles.add((nx, ny))

    if explosion_map is not None:
        xs, ys = np.where(explosion_map > 0)
        danger_tiles.update(zip(xs, ys))

    return danger_tiles