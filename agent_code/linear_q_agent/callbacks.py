import os
import pickle
import random

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows to share trained agent
    with other students, without revealing training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # to do: Exploration vs exploitation
    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")
    return np.random.choice(ACTIONS, p=self.model)


def state_to_features(game_state: dict) -> np.array:
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
    bombs = game_state["bombs"] # (x, y), timer
    coins = game_state["coins"] # x, y
    agent = game_state["self"] # name, score, bombs_left, (x, y)

    agent_x, agent_y = agent[3]

    field_x, field_y = field.shape

    # Create a feature vector
    features = np.zeros(7) # [1, free_U, free_D, free_L, free_R, coin_dx, coin_dy]

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
    
    # Return the final feature vector
    return features
