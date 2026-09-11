import os
import pickle
import random
import numpy as np
from collections import deque

from .model import Linear_QModel


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
EPSILON_START = 1.0
MODEL_START_MODE = "resume"  # Set to "fresh" to ignore an existing checkpoint.
FEATURE_SIZE = 11


def setup(self):
    """
    This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows to share trained agent
    with other students, without revealing training code.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if MODEL_START_MODE not in {"resume", "fresh"}:
        raise ValueError("MODEL_START_MODE must be either 'resume' or 'fresh'.")

    if MODEL_START_MODE == "fresh" or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.model = Linear_QModel(input_size=FEATURE_SIZE, output_size=len(ACTIONS))
        self.epsilon = EPSILON_START
    else:
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

def act(self, game_state: dict) -> str:
    """
    Agent should parse the input, think, and take a decision.
    """

    features = state_to_features(game_state)

    # Exploration during training
    if self.train and random.random() < self.epsilon:
        self.logger.debug("Choosing action purely at random.")
        return random.choice(ACTIONS)

    # Exploitation
    q_values = self.model.predict(features)

    # Choose action with highest Q-value
    action_index = int(np.argmax(q_values))
    action = ACTIONS[action_index]

    self.logger.debug("Choosing action with the highest Q-value.")

    return action


def state_to_features(game_state: dict) -> np.ndarray:
    """
    Converts the game state to the input of model, i.e. a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # Get the current location of the agent
    field = game_state["field"] # np.ndarray
    # bombs = game_state["bombs"] # (x, y), timer
    coins = game_state["coins"] # x, y
    agent = game_state["self"] # name, score, bombs_left, (x, y)

    agent_x, agent_y = agent[3]

    field_x, field_y = field.shape

    # Create a feature vector
    features = np.zeros(11) # [1, free_U, free_D, free_L, free_R, coin_dx, coin_dy, path_U, path_D, path_L, path_R]

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

        if closest_coin is not None:
            path_directions = shortest_path_directions(field, agent[3], coin)
            features[7:11] = path_directions

    # Return the final feature vector
    return features


def shortest_path_directions(field, start, target):
    """Return valid first-step directions along shortest paths from start to target."""

    directions = [
        (0, -1),    # UP
        (0, 1),     # DOWN
        (-1, 0),    # LEFT
        (1, 0)      # RIGHT
    ]

    # No movement needed if already at target
    if start == target:
        return np.zeros(4)

    # Distance from each tile to the target
    distances = np.full(field.shape, -1)

    queue = deque([target])
    distances[target] = 0

    # BFS starting from the target
    while queue:
        x, y = queue.popleft()

        for dx, dy in directions:
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
    for i, (dx, dy) in enumerate(directions):
        nx = start[0] + dx
        ny = start[1] + dy

        if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
            continue

        if distances[nx, ny] == current_distance - 1:
            path_directions[i] = 1

    return path_directions