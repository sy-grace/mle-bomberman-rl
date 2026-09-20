import os
from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features, select_action, DIRECTIONS, bomb_danger_tiles

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.99

# Events
# PLACEHOLDER_EVENT = "PLACEHOLDER"
MOVED_TOWARDS_COIN = "MOVED_TOWARDS_COIN"
MOVED_AWAY_FROM_COIN = "MOVED_AWAY_FROM_COIN"
UNNECESSARILY_WAITED = "UNNECESSARILY_WAITED"

ESCAPED_BOMB_DANGER = "ESCAPED_BOMB_DANGER"
MOVED_TOWARDS_CRATE = "MOVED_TOWARDS_CRATE"
MOVED_AWAY_FROM_CRATE = "MOVED_AWAY_FROM_CRATE"
SAFE_USEFUL_BOMB_DROPPED = "SAFE_USEFUL_BOMB_DROPPED"

MOVED_TOWARDS_OPPONENT = "MOVED_TOWARDS_OPPONENT"
SAFE_OPPONENT_BOMB_DROPPED = "SAFE_OPPONENT_BOMB_DROPPED"
MOVED_AWAY_FROM_OPPONENT = "MOVED_AWAY_FROM_OPPONENT"
MOVED_TOWARDS_OPPONENT_HUNT = "MOVED_TOWARDS_OPPONENT_HUNT"
SAFE_OPPONENT_BOMB_DROPPED_HUNT = "SAFE_OPPONENT_BOMB_DROPPED_HUNT"
MOVED_AWAY_FROM_OPPONENT_HUNT = "MOVED_AWAY_FROM_OPPONENT_HUNT"

SPARSE_REWARDS = {
    e.COIN_COLLECTED: +10
}

BASIC_EXTRA_REWARDS = {
    e.INVALID_ACTION: -2,
    e.CRATE_DESTROYED: +2,
    e.COIN_FOUND: +3,
    e.KILLED_SELF: -20,
    e.KILLED_OPPONENT: +20,
}

SHAPING_EXTRA_REWARDS = {
    MOVED_TOWARDS_COIN: +1,
    MOVED_AWAY_FROM_COIN: -1,
    UNNECESSARILY_WAITED: -0.5,

    ESCAPED_BOMB_DANGER: +3,

    MOVED_TOWARDS_CRATE: +1,
    MOVED_AWAY_FROM_CRATE: -1,
    SAFE_USEFUL_BOMB_DROPPED: +2,

    MOVED_TOWARDS_OPPONENT: +1.0,
    SAFE_OPPONENT_BOMB_DROPPED: +0.5,
    MOVED_AWAY_FROM_OPPONENT: -0.5,
    MOVED_TOWARDS_OPPONENT_HUNT: +3.0,
    SAFE_OPPONENT_BOMB_DROPPED_HUNT: +1.0,
}

HUNT_EXTRA_REWARDS = {MOVED_AWAY_FROM_OPPONENT_HUNT: -1.0}

REWARD_CONFIGS = {
    "sparse": SPARSE_REWARDS, 
    "basic": {**SPARSE_REWARDS, **BASIC_EXTRA_REWARDS}, 
    "shaped": {**SPARSE_REWARDS, **BASIC_EXTRA_REWARDS, **SHAPING_EXTRA_REWARDS},
    "hunt_extra": {**SPARSE_REWARDS, **BASIC_EXTRA_REWARDS, **SHAPING_EXTRA_REWARDS, **HUNT_EXTRA_REWARDS},
}

ACTION_TO_INDEX = {
    "UP": 0,
    "DOWN": 1,
    "LEFT": 2,
    "RIGHT": 3,
    "WAIT": 4,
    "BOMB": 5,
}


def has_actionable_navigation_move(game_state, state, feature_mode):
    """Return True if at least one currently relevant navigation path points to a tile that the agent can actually enter now."""
    if game_state is None:
        return False

    path_vectors = []

    # Coin navigation
    coin_path = state[7:11]
    if coin_path.any():
        path_vectors.append(coin_path)

    # Crate navigation is used only when no visible coin exists, matching the existing crate reward shaping.
    if feature_mode in {"f2", "f3", "f4", "f5", "f6", "f7", "f8"} and not game_state["coins"]:
        crate_path = state[16:20]

        if crate_path.any():
            path_vectors.append(crate_path)

    # Opponent hunting
    if feature_mode in {"f5", "f6", "f7", "f8"}:
        opponent_path = state[31:35]

        if opponent_path.any():
            path_vectors.append(opponent_path)

        if not path_vectors:
            return False

    field = game_state["field"]
    explosion_map = game_state.get("explosion_map")
    x, y = game_state["self"][3]

    bomb_positions = {position for position, _ in game_state.get("bombs", [])}

    opponent_positions = {other[3] for other in game_state.get("others", [])}

    # Tiles that are lethal during this action: currently active explosions; bombs with timer 0
    imminent_bombs = [(position, timer) for position, timer in game_state.get("bombs", []) if timer <= 0]

    immediate_danger_tiles = bomb_danger_tiles(field, imminent_bombs, explosion_map)

    for i, (dx, dy) in enumerate(DIRECTIONS):

        # No relevant navigation feature points this way.
        if not any(path[i] == 1.0 for path in path_vectors):
            continue

        nx = x + dx
        ny = y + dy
        next_position = (nx, ny)

        if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
            continue

        # Wall/crate
        if field[nx, ny] != 0:
            continue

        # Active explosion
        if explosion_map is not None and explosion_map[nx, ny] > 0:
            continue

        # Bomb occupies tile
        if next_position in bomb_positions:
            continue

        # Opponent occupies tile
        if next_position in opponent_positions:
            continue

        # Moving there would kill the agent during this turn.
        if next_position in immediate_danger_tiles:
            continue

        return True

    return False


def setup_training(self):
    """
    Initialize self for training purpose.

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
    self.pending_action = None

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

    # Custom event: opponent hunting
    if self.feature_mode in {"f5", "f6", "f7", "f8"}:
        # Reward moving along the shortest path toward an opponent.
        # F5 features 31:35 = [UP, DOWN, LEFT, RIGHT]
        opponent_path = state[31:35]

        if state[20] == 0.0 and opponent_path.any():
            old_x, old_y = old_game_state["self"][3]
            new_x, new_y = new_game_state["self"][3]

            dx = new_x - old_x
            dy = new_y - old_y

            direction_to_index = {
                (0, -1): 0, # UP
                (0, 1): 1, # DOWN
                (-1, 0): 2, # LEFT
                (1, 0): 3 # RIGHT
            }

            moved_index = direction_to_index.get((dx, dy))

            # Only shape an actual successful movement.
            if moved_index is not None:
                hunt_mode = self.feature_mode in {"f7", "f8"} and is_hunt_mode(old_game_state)

                if opponent_path[moved_index] == 1.0:
                    if hunt_mode:
                        events.append(MOVED_TOWARDS_OPPONENT_HUNT)
                    else:
                        events.append(MOVED_TOWARDS_OPPONENT)
                elif hunt_mode:
                    events.append(MOVED_AWAY_FROM_OPPONENT_HUNT)

        # Reward a bomb that currently threatens an opponent and still leaves an escape route.
        if self_action == "BOMB" and state[35] == 1.0:
            if self.feature_mode in {"f7", "f8"} and is_hunt_mode(old_game_state):
                events.append(SAFE_OPPONENT_BOMB_DROPPED_HUNT)
            else:
                events.append(SAFE_OPPONENT_BOMB_DROPPED)

    # Custom event: safe and useful bomb placement
    if self.feature_mode in {"f4", "f5", "f6", "f7", "f8"} and self_action == "BOMB" and state[26] == 1.0:
        events.append(SAFE_USEFUL_BOMB_DROPPED)

    # Bomb danger status
    old_in_danger = self.feature_mode in {"f2", "f3", "f4", "f5", "f6", "f7", "f8"} and state[20] == 1.0
    new_in_danger = self.feature_mode in {"f2", "f3", "f4", "f5", "f6", "f7", "f8"} and next_state[20]  == 1.0

    # Custom event: escaped bomb danger
    if old_in_danger and not new_in_danger:
        events.append(ESCAPED_BOMB_DANGER)

    # Custom event: move along crate path
    if self.feature_mode in {"f2", "f3", "f4", "f5", "f6", "f7", "f8"}:
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

    # Penalize  WAIT only when the agent is safe and has an immediately actionable navigation move.
    if not old_in_danger and self_action == "WAIT" and has_actionable_navigation_move(old_game_state, state, self.feature_mode):
        events.append(UNNECESSARILY_WAITED)

    # Model Learn
    reward = reward_from_events(self, events)
    action = ACTION_TO_INDEX[self_action]

    next_action = select_action(self, next_state, new_game_state)
    self.pending_action = next_action

    self.model.update(state, action, reward, next_state, ACTION_TO_INDEX[next_action])
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
    self.model.reset_traces()
    self.pending_action = None # Movement / SARSA action tracking
    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    self.feature_previous_action = None
    self.cached_features = None
    self.escape_bomb_position = None

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(
            {
                "model": self.model,
                "epsilon": self.epsilon,
                "feature_mode": self.feature_mode,
                "actions": list(self.actions),
            },
            file,
        )


def reward_from_events(self, events: List[str]) -> float:
    """Modify the rewards the agent get so as to en/discourage certain behavior."""
    game_rewards = REWARD_CONFIGS[self.reward_mode]

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
            
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")

    return reward_sum


def is_hunt_mode( game_state):
    """Return True when no currently collectable coins are visible and at least one opponent is still alive."""
    if game_state is None:
        return False

    return len(game_state.get("coins", [])) == 0 and len(game_state.get("others", [])) > 0


