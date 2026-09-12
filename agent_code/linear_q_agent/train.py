import os
from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995

# Events
# PLACEHOLDER_EVENT = "PLACEHOLDER"
MOVED_TOWARDS_COIN = "MOVED_TOWARDS_COIN"
MOVED_AWAY_FROM_COIN = "MOVED_AWAY_FROM_COIN"
UNNECESSARILY_WAITED = "UNNECESSARILY_WAITED"
OSCILLATION = "OSCILLATION"

ESCAPED_BOMB_DANGER = "ESCAPED_BOMB_DANGER"
STAYED_IN_BOMB_DANGER = "STAYED_IN_BOMB_DANGER"
MOVED_TOWARDS_CRATE = "MOVED_TOWARDS_CRATE"
MOVED_AWAY_FROM_CRATE = "MOVED_AWAY_FROM_CRATE"

SPARSE_REWARDS = {
    e.COIN_COLLECTED: +10
}

BASIC_EXTRA_REWARDS = {
    e.INVALID_ACTION: -2,
    e.CRATE_DESTROYED: +2,
    e.COIN_FOUND: +3,
    e.KILLED_SELF: -20,
}

SHAPING_EXTRA_REWARDS = {
    MOVED_TOWARDS_COIN: +1,
    MOVED_AWAY_FROM_COIN: -1,
    UNNECESSARILY_WAITED: -0.5,
    ESCAPED_BOMB_DANGER: +3,
    STAYED_IN_BOMB_DANGER: -2,

    MOVED_TOWARDS_CRATE: +1,
    MOVED_AWAY_FROM_CRATE: -1,
}

REWARD_CONFIGS = {
    "sparse": SPARSE_REWARDS, 
    "basic": {**SPARSE_REWARDS, **BASIC_EXTRA_REWARDS}, 
    "shaped": {**SPARSE_REWARDS, **BASIC_EXTRA_REWARDS, **SHAPING_EXTRA_REWARDS}
}

ACTION_TO_INDEX = {
    "UP": 0,
    "DOWN": 1,
    "LEFT": 2,
    "RIGHT": 3,
    "WAIT": 4,
    "BOMB": 5,
}


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    self.epsilon = getattr(self, "epsilon", EPSILON_START)
    self.epsilon_min = EPSILON_MIN
    self.epsilon_decay = EPSILON_DECAY

    # Movement tracking
    self.last_action = None
    self.previous_action = None
    self.last_distance = None

    # Reward configuration
    self.reward_mode = os.getenv("REWARD_MODE", "basic").lower()

    if self.reward_mode not in REWARD_CONFIGS:
        raise ValueError(f"Invalid reward mode: {self.reward_mode}.\n Choose from {list(REWARD_CONFIGS.keys())}.")

    self.logger.info(f"Reward mode: {self.reward_mode}")

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to the agent that occurred during the previous step.

    This is *one* of the places where the agent could be updated.

    :param self: This object is passed to all callbacks and can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that tookplace.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # state_to_features is defined in callbacks.py
    state = state_to_features(old_game_state, self.feature_mode)
    next_state = state_to_features(new_game_state, self.feature_mode)

    # Bomb danger status
    old_in_danger = self.feature_mode in {"f2", "f3"} and state[20] == 1.0
    new_in_danger = self.feature_mode in {"f2", "f3"} and next_state[20]  == 1.0

    # Custom event: escaped bomb danger
    if old_in_danger and not new_in_danger:
        events.append(ESCAPED_BOMB_DANGER)

    # Custom event: move along crate path
    if self.feature_mode in {"f2", "f3"}:
        crate_path = state[16:20]

        # Only search for crates when there is no visible coin and escaping a bomb is not currently more important.
        if not old_in_danger and not old_game_state["coins"] and crate_path.any():
            old_x, old_y = old_game_state["self"][3]
            new_x, new_y = new_game_state["self"][3]

            dx = new_x - old_x
            dy = new_y - old_y

            direction_to_index = {
                (0, -1): 0, # UP
                (0, 1): 1,  # DOWN
                (-1, 0): 2, # LEFT
                (1, 0): 3   # RIGHT
            }

            moved_index = direction_to_index.get((dx, dy))

            # Only shape successful movement, not WAIT/BOMB/invalid movement.
            if moved_index is not None:
                if crate_path[moved_index] == 1.0:
                    events.append(MOVED_TOWARDS_CRATE)
                else:
                    events.append(MOVED_AWAY_FROM_CRATE)

    # Custom events based on coin proximity and movement
    # Coin distance
    old_dx, old_dy = state[5], state[6]
    new_dx, new_dy = next_state[5], next_state[6]

    old_distance = abs(old_dx) + abs(old_dy)
    new_distance = abs(new_dx) + abs(new_dy)

    # Movement toward / away from coin
    # Skip distance shaping when a coin was collected, because the nearest target coin may have changed.
    if not old_in_danger and old_distance > 0 and e.COIN_COLLECTED not in events:
        if new_distance < old_distance:
            events.append(MOVED_TOWARDS_COIN)

        elif new_distance > old_distance:
            events.append(MOVED_AWAY_FROM_COIN)

    # Penalize unnecessary WAIT
    if not old_in_danger and self_action == "WAIT" and old_distance > 0:
        events.append(UNNECESSARILY_WAITED)

    # Penalize oscillation
    opposite = {
        "UP": "DOWN",
        "DOWN": "UP",
        "LEFT": "RIGHT",
        "RIGHT": "LEFT",
    }

    previous_action = getattr(self, "previous_action", None)
    last_action = getattr(self, "last_action", None)

    if (
        previous_action is not None
        and last_action is not None
        and self_action == previous_action
        and last_action == opposite.get(self_action)
    ):
        events.append(OSCILLATION)

    # Update action history
    self.previous_action = last_action
    self.last_action = self_action
    self.last_distance = old_distance

    # Model Learn
    reward = reward_from_events(self, events)
    action = ACTION_TO_INDEX[self_action]
    self.model.update(state, action, reward, next_state)
    self.transitions.append(Transition(state, self_action, next_state, reward))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during the agent's final step.

    This is *one* of the places where the agent could be updated and stored.

    :param self: The same object that is passed to all of the callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    state = state_to_features(last_game_state, self.feature_mode)
    reward = reward_from_events(self, events)
    action = ACTION_TO_INDEX[last_action]
    self.model.update(state, action, reward, None)
    self.transitions.append(Transition(state, last_action, None, reward))

    # Reset each round
    self.previous_action = None
    self.last_action = None
    self.last_distance = None
    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(
            {
                "model": self.model,
                "epsilon": self.epsilon,
            },
            file,
        )


def reward_from_events(self, events: List[str]) -> float:
    """
    Here we can modify the rewards the agent get so as to en/discourage certain behavior.
    """
    game_rewards = REWARD_CONFIGS[self.reward_mode]

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
            
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")

    return reward_sum