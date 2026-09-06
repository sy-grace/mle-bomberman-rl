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
PLACEHOLDER_EVENT = "PLACEHOLDER"
ACTION_TO_INDEX = {
    "UP": 0,
    "RIGHT": 1,
    "DOWN": 2,
    "LEFT": 3,
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

    # Idea: Add own events to hand out rewards
    if ...:
        events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    state = state_to_features(old_game_state)
    next_state = state_to_features(new_game_state)
    reward = reward_from_events(self, events)
    action = ACTION_TO_INDEX[self_action]
    self.model.update(state, action, reward, next_state)
    self.transitions.append(Transition(state, self_action, next_state, reward))
    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


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
    state = state_to_features(last_game_state)
    reward = reward_from_events(self, events)
    action = ACTION_TO_INDEX[last_action]
    self.model.update(state, action, reward, None)
    self.transitions.append(Transition(state, last_action, None, reward))
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


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure the code.*

    Here we can modify the rewards the agent get so as to en/discourage certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
