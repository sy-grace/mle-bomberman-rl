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
FEATURE_SIZES = {
    "f0": 7, "f1": 11, "f2": 25, "f3": 26, "f4": 27, "f5": 31, "f6": 32,
    "f7": 41,
}

BOMB_POWER = 3
BOMB_TIMER = 4

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
        raise ValueError("FEATURE_MODE must be one of 'f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', or 'f7'.")

    self.feature_size = FEATURE_SIZES[self.feature_mode]
    self.logger.info(f"Feature mode: {self.feature_mode} ({self.feature_size} features)")

    self.actions = actions_for_feature_mode(self.feature_mode)

    self.feature_previous_action = None
    self.cached_features = None
    self.recent_positions = deque(maxlen=12)
    self.escape_plan = []

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

        saved_actions = checkpoint.get("actions") if isinstance(checkpoint, dict) else None

        if saved_actions is not None and saved_actions != self.actions:
            raise ValueError(f"Checkpoint action order {saved_actions} does not match current action order {self.actions}.")

        if self.model.input_size != self.feature_size:
            raise ValueError(f"Checkpoint expects {self.model.input_size} features, but FEATURE_MODE='{self.feature_mode}' uses {self.feature_size}.")

        if self.model.output_size != len(self.actions):
            raise ValueError(f"Checkpoint expects {self.model.output_size} actions, but FEATURE_MODE='{self.feature_mode}' uses {len(self.actions)}.")


def act(self, game_state: dict) -> str:
    """
    Agent should parse the input, think, and take a decision.
    """
    if game_state["step"] == 1:
        self.feature_previous_action = None
        self.cached_features = None
        if hasattr(self, "recent_positions"):
            self.recent_positions.clear()

    features = state_to_features(game_state, self.feature_mode, previous_action=self.feature_previous_action)
    self.cached_features = features.copy()
    position = tuple(game_state["self"][3])
    if not hasattr(self, "recent_positions"):
        self.recent_positions = deque(maxlen=12)
    self.recent_positions.append(position)

    opponent_positions = [
        tuple(opponent[3])
        for opponent in game_state.get("others", [])
    ]
    opponent_path = None
    opponent_bomb_target = False
    if self.feature_mode == "f6":
        opponent_path = shortest_path_directions_to_any(
            game_state["field"],
            position,
            opponent_bomb_targets(game_state["field"], opponent_positions),
        )
        opponent_bomb_target = any(
            bomb_would_hit_opponent(game_state["field"], position, opponent_position)
            for opponent_position in opponent_positions
        )

    allowed_actions = action_candidates(
        features,
        self.actions,
        self.feature_mode,
        position=position,
        recent_positions=self.recent_positions,
        field=game_state["field"],
        bombs=game_state["bombs"],
        explosion_map=game_state.get("explosion_map"),
        opponent_path=opponent_path,
        opponent_bomb_target=opponent_bomb_target,
    )
    escape_plan = getattr(self, "escape_plan", [])
    if self.feature_mode in {"f6", "f7"} and features[20] == 1.0 and escape_plan:
        if escape_plan[0] in allowed_actions:
            allowed_actions = [escape_plan[0]]
        else:
            escape_plan = escape_path(
                game_state["field"],
                position,
                game_state["bombs"],
                game_state.get("explosion_map"),
            )
            self.escape_plan = escape_plan
            if escape_plan and escape_plan[0] in allowed_actions:
                allowed_actions = [escape_plan[0]]
            else:
                escape_plan.clear()

    # Exploration during training
    if self.train and self.rng.random() < self.epsilon:
        self.logger.debug("Choosing action purely at random.")
        action = self.rng.choice(allowed_actions)
    else:
        # Exploitation
        q_values = self.model.predict(features)

        # F6 can reject actions that are known to be invalid or counterproductive
        # even when an old checkpoint assigns them a high Q-value.
        masked_q_values = np.full_like(q_values, -np.inf, dtype=float)
        for action in allowed_actions:
            masked_q_values[self.actions.index(action)] = q_values[self.actions.index(action)]

        # Choose the highest-Q action among the safe candidates.
        action_index = int(np.argmax(masked_q_values))
        action = self.actions[action_index]
        self.logger.debug("Choosing action with the highest Q-value.")

    self.feature_previous_action = action
    if action == "BOMB" and self.feature_mode in {"f6", "f7"}:
        self.escape_plan = escape_path(
            game_state["field"],
            position,
            list(game_state["bombs"]) + [(position, BOMB_TIMER)],
            game_state.get("explosion_map"),
        )
    elif escape_plan:
        if escape_plan[0] == action:
            escape_plan.pop(0)
        else:
            escape_plan.clear()
    return action


def action_candidates(
    features: np.ndarray,
    actions,
    feature_mode: str,
    position=None,
    recent_positions=(),
    field=None,
    bombs=(),
    explosion_map=None,
    opponent_path=None,
    opponent_bomb_target=False,
):
    """Return valid and safety-filtered actions for F6/F7 states."""
    if feature_mode not in {"f6", "f7"}:
        return actions

    candidates = list(actions)
    in_danger = features[20] == 1.0

    # A bomb is useful only when F6's safety and crate-yield calculation found
    # a reachable escape and at least one crate in the blast area.
    opponent_bomb_available = feature_mode == "f7" and features[39] > 0.0
    if features[31] <= 0.0 and not opponent_bomb_target and not opponent_bomb_available and "BOMB" in candidates:
        candidates.remove("BOMB")
    elif "BOMB" in candidates and field is not None and position is not None:
        if not can_escape_after_bomb(field, position, bombs, explosion_map):
            candidates.remove("BOMB")

    # Never deliberately choose a blocked movement. Keep WAIT as a fallback.
    movement_features = {
        "UP": 1,
        "DOWN": 2,
        "LEFT": 3,
        "RIGHT": 4,
    }
    candidates = [
        action for action in candidates
        if action not in movement_features or features[movement_features[action]] == 1.0
    ]

    if field is not None and position is not None and not in_danger:
        danger_tiles = bomb_danger_tiles(field, bombs, explosion_map)
        offsets = {
            "UP": (0, -1),
            "DOWN": (0, 1),
            "LEFT": (-1, 0),
            "RIGHT": (1, 0),
        }
        candidates = [
            action for action in candidates
            if action not in offsets
            or (
                position[0] + offsets[action][0],
                position[1] + offsets[action][1],
            ) not in danger_tiles
        ]

    if in_danger:
        danger_tiles = bomb_danger_tiles(field, bombs, explosion_map) if field is not None else set()
        offsets = {
            "UP": (0, -1),
            "DOWN": (0, 1),
            "LEFT": (-1, 0),
            "RIGHT": (1, 0),
        }
        immediate_escape_actions = [
            action for action in candidates
            if action in offsets
            and position is not None
            and (
                position[0] + offsets[action][0],
                position[1] + offsets[action][1],
            ) not in danger_tiles
        ]
        if immediate_escape_actions:
            return immediate_escape_actions

        if bombs and position is not None:
            perpendicular_actions = set()
            for (bomb_x, bomb_y), _timer in bombs:
                if position[0] == bomb_x and abs(position[1] - bomb_y) <= BOMB_POWER:
                    perpendicular_actions.update(("LEFT", "RIGHT"))
                if position[1] == bomb_y and abs(position[0] - bomb_x) <= BOMB_POWER:
                    perpendicular_actions.update(("UP", "DOWN"))
            corridor_actions = [
                action for action in candidates
                if action in perpendicular_actions
            ]
            if corridor_actions:
                return corridor_actions

        escape_actions = [
            action for index, action in enumerate(("UP", "DOWN", "LEFT", "RIGHT"))
            if features[21 + index] == 1.0 and action in candidates
        ]
        if escape_actions:
            # During an explosion threat, learned preferences and exploration
            # must not override the computed route to a safe tile.
            return escape_actions

        if bombs and position is not None:
            bomb_positions = [bomb_position for bomb_position, _timer in bombs]
            current_distance = min(
                abs(position[0] - bomb_x) + abs(position[1] - bomb_y)
                for bomb_x, bomb_y in bomb_positions
            )
            non_backtracking_actions = []
            for action in candidates:
                if action not in offsets:
                    continue
                next_position = (
                    position[0] + offsets[action][0],
                    position[1] + offsets[action][1],
                )
                next_distance = min(
                    abs(next_position[0] - bomb_x) + abs(next_position[1] - bomb_y)
                    for bomb_x, bomb_y in bomb_positions
                )
                if next_distance >= current_distance:
                    non_backtracking_actions.append(action)
            if non_backtracking_actions:
                return non_backtracking_actions

        # If no route was found, do not wait or place another bomb.
        emergency_actions = [
            action for action in candidates if action not in {"WAIT", "BOMB"}
        ]
        return emergency_actions or candidates

    # Outside bomb danger, avoid an immediate reversal when another action is
    # available. In danger, the escape policy must be allowed to reverse.
    if not in_danger:
        coin_path = features[7:11]
        crate_path = features[16:20]
        if feature_mode == "f6" and opponent_path is not None and opponent_path.any():
            target_path = opponent_path
        elif feature_mode == "f7":
            opponent_feature_path = features[34:38]
            target_path = (
                opponent_feature_path
                if opponent_feature_path.any()
                else coin_path if coin_path.any() else crate_path
            )
        else:
            target_path = coin_path if coin_path.any() else crate_path

        # Preserve F6's goal-directed movement while allowing F7 to prefer
        # opponent attack routes when they are available.
        if target_path.any():
            path_actions = [
                action for index, action in enumerate(("UP", "DOWN", "LEFT", "RIGHT"))
                if target_path[index] == 1.0 and action in candidates
            ]
            if path_actions:
                candidates = path_actions
                if feature_mode == "f6" and opponent_bomb_target and "BOMB" in actions:
                    candidates.append("BOMB")
                if feature_mode == "f7" and opponent_bomb_available and "BOMB" in actions:
                    candidates.append("BOMB")

        opposite = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
        previous_action = next(
            (action for action, index in {
                "UP": 27, "DOWN": 28, "LEFT": 29, "RIGHT": 30
            }.items() if features[index] == 1.0),
            None,
        )
        reverse = opposite.get(previous_action)
        alternative_movement_exists = any(
            action != reverse and action in movement_features
            for action in candidates
        )
        if reverse in candidates and alternative_movement_exists:
            candidates.remove(reverse)

    # Avoid short position cycles such as UP-RIGHT-DOWN-LEFT. Do not apply
    # this while escaping, and keep the valid candidates as a fallback when
    # every available tile has recently been visited.
    if not in_danger and position is not None:
        offsets = {
            "UP": (0, -1),
            "DOWN": (0, 1),
            "LEFT": (-1, 0),
            "RIGHT": (1, 0),
        }
        visited = set(recent_positions)
        fresh_candidates = [
            action for action in candidates
            if action not in offsets
            or (position[0] + offsets[action][0], position[1] + offsets[action][1]) not in visited
        ]
        if any(action in movement_features for action in fresh_candidates):
            candidates = fresh_candidates

    return candidates or list(actions)


def state_to_features(game_state: dict, feature_mode: str, previous_action=None) -> np.ndarray:
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
        raise ValueError("feature_mode must be one of 'f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', or 'f7'.")

    # Get the current location of the agent
    field = game_state["field"] # np.ndarray
    bombs = game_state["bombs"] # (x, y), timer
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
    # F3: F2 + [safe_to_bomb]
    # F4: F3 + [safe_and_useful_bomb]
    # F5: F4 + [previous_UP, previous_DOWN, previous_LEFT, previous_RIGHT]
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

        if closest_coin is not None and feature_mode in {"f1", "f2", "f3", "f4", "f5", "f6", "f7"}:
            path_directions = shortest_path_directions(field, agent[3], closest_coin)
            features[7:11] = path_directions

    if feature_mode in {"f2", "f3", "f4", "f5", "f6", "f7"}:
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
            features[21:25] = escape_directions(
                escape_field,
                agent[3],
                bombs,
                explosion_map,
                danger_tiles,
            )

        # 25: safe to bomb
        if feature_mode in {"f3", "f4", "f5", "f6", "f7"}:
            safe_to_bomb = bool(agent[2] and can_escape_after_bomb(field, agent[3], bombs, explosion_map))
            features[25] = float(safe_to_bomb)

        # 26: safe and useful bomb
        if feature_mode in {"f4", "f5", "f6", "f7"}:
            features[26] = float(safe_to_bomb and bomb_would_destroy_crate(field, agent[3]))

        # 27:31: previous action [UP, DOWN, LEFT, RIGHT]
        if feature_mode in {"f5", "f6", "f7"}:
            previous_action_to_index = {"UP": 27, "DOWN": 28, "LEFT": 29, "RIGHT": 30}
            index = previous_action_to_index.get(previous_action)
            if index is not None:
                features[index] = 1.0

        # 31: bomb crate-yield feature
        if feature_mode in {"f6", "f7"}:
            features[31] = safe_bomb_crate_yield(field, agent[3], bombs, explosion_map, agent[2])

        if feature_mode == "f7":
            opponents = [tuple(opponent[3]) for opponent in game_state.get("others", [])]
            closest_opponent = min(
                opponents,
                key=lambda opponent: abs(opponent[0] - agent_x) + abs(opponent[1] - agent_y),
                default=None,
            )
            if closest_opponent is not None:
                features[32] = (closest_opponent[0] - agent_x) / (field_x - 1)
                features[33] = (closest_opponent[1] - agent_y) / (field_y - 1)
                features[34:38] = shortest_path_directions_to_any(
                    field,
                    agent[3],
                    opponent_bomb_targets(field, opponents) - {agent[3]},
                )
                features[38] = float(
                    any(bomb_would_hit_opponent(field, agent[3], opponent) for opponent in opponents)
                )
                features[39] = float(
                    safe_to_bomb
                    and any(bomb_would_hit_opponent(field, agent[3], opponent) for opponent in opponents)
                )
                features[40] = 1.0

    # Return the final feature vector
    return features


def shortest_path_directions(field, start, target):
    """Return valid first-step directions along shortest paths from start to target."""
    return shortest_path_directions_to_any(field, start, {target})


def opponent_bomb_targets(field, opponents):
    """Return free tiles from which a bomb can reach any visible opponent."""
    targets = set()
    for opponent in opponents:
        for dx, dy in DIRECTIONS:
            for distance in range(1, BOMB_POWER + 1):
                x = opponent[0] + dx * distance
                y = opponent[1] + dy * distance
                if not (0 <= x < field.shape[0] and 0 <= y < field.shape[1]):
                    break
                if field[x, y] == -1:
                    break
                if field[x, y] == 0:
                    targets.add((x, y))
                if field[x, y] == 1:
                    break
    return targets


def bomb_would_hit_opponent(field, start, opponent):
    """Return whether a bomb at start can reach the opponent through the board."""
    if start == opponent:
        return True
    dx = opponent[0] - start[0]
    dy = opponent[1] - start[1]
    if dx != 0 and dy != 0:
        return False
    direction = (0, 0)
    distance = abs(dx) + abs(dy)
    if dx:
        direction = (1 if dx > 0 else -1, 0)
    elif dy:
        direction = (0, 1 if dy > 0 else -1)
    if distance > BOMB_POWER:
        return False
    for step in range(1, distance + 1):
        x = start[0] + direction[0] * step
        y = start[1] + direction[1] * step
        if field[x, y] == -1:
            return False
        if field[x, y] == 1 and step < distance:
            return False
    return True


def actions_for_feature_mode(feature_mode):
    if feature_mode in {"f2", "f3", "f4", "f5", "f6", "f7"}:
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

            # Only free tiles are walkable
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


def can_escape_after_bomb(field, start, bombs, explosion_map=None):
    """Return True if a bomb placed at start still allows escape within BOMB_TIMER moves."""
    hypothetical_bombs = list(bombs) + [(start, BOMB_TIMER)]
    danger_tiles = bomb_danger_tiles(field, hypothetical_bombs, explosion_map)
    escape_field = field.copy()
    for position, _timer in bombs:
        if position != start:
            escape_field[position] = -1
    return bool(escape_directions(
        escape_field,
        start,
        hypothetical_bombs,
        explosion_map,
        danger_tiles,
    ).any())


def escape_directions(field, start, bombs, explosion_map=None, danger_tiles=None):
    """Return first moves that reach a safe tile before the relevant bomb explodes."""
    if danger_tiles is None:
        danger_tiles = bomb_danger_tiles(field, bombs, explosion_map)

    relevant_timers = [
        timer for bomb, timer in bombs
        if start in bomb_danger_tiles(field, [(bomb, timer)])
    ]
    if explosion_map is not None and explosion_map[start] > 0:
        relevant_timers.append(1)
    if not relevant_timers:
        return np.zeros(4)

    max_distance = min(relevant_timers)
    if max_distance < 1:
        return np.zeros(4)

    existing_bomb_tiles = {position for position, _timer in bombs}
    escape_directions = np.zeros(4)
    queue = deque()
    visited = {start}

    for index, (dx, dy) in enumerate(DIRECTIONS):
        nx = start[0] + dx
        ny = start[1] + dy
        next_pos = (nx, ny)
        if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
            continue
        if field[nx, ny] != 0 or next_pos in existing_bomb_tiles:
            continue
        if explosion_map is not None and explosion_map[nx, ny] > 0:
            continue
        queue.append((next_pos, 1, index))
        visited.add(next_pos)

    while queue:
        (x, y), distance, first_direction = queue.popleft()
        if (x, y) not in danger_tiles:
            escape_directions[first_direction] = 1.0
            continue
        if distance >= max_distance:
            continue

        for dx, dy in DIRECTIONS:
            nx = x + dx
            ny = y + dy
            next_pos = (nx, ny)
            if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
                continue
            if field[nx, ny] != 0 or next_pos in existing_bomb_tiles:
                continue
            if explosion_map is not None and explosion_map[nx, ny] > 0:
                continue
            if next_pos in visited:
                continue
            visited.add(next_pos)
            queue.append((next_pos, distance + 1, first_direction))

    return escape_directions


def escape_path(field, start, bombs, explosion_map=None):
    """Return a shortest timed path to safety, expressed as movement actions."""
    danger_tiles = bomb_danger_tiles(field, bombs, explosion_map)
    relevant_timers = [
        timer for bomb, timer in bombs
        if start in bomb_danger_tiles(field, [(bomb, timer)])
    ]
    if explosion_map is not None and explosion_map[start] > 0:
        relevant_timers.append(1)
    if not relevant_timers:
        return []

    max_distance = min(relevant_timers)
    existing_bomb_tiles = {position for position, _timer in bombs}
    offsets_to_actions = dict(zip(DIRECTIONS, ("UP", "DOWN", "LEFT", "RIGHT")))
    queue = deque([(start, 0, [])])
    visited = {start}

    while queue:
        (x, y), distance, path = queue.popleft()
        if distance > 0 and (x, y) not in danger_tiles:
            return path
        if distance >= max_distance:
            continue

        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            next_position = (nx, ny)
            if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
                continue
            if field[nx, ny] != 0 or next_position in existing_bomb_tiles:
                continue
            if explosion_map is not None and explosion_map[nx, ny] > 0:
                continue
            if next_position in visited:
                continue
            visited.add(next_position)
            queue.append((
                next_position,
                distance + 1,
                path + [offsets_to_actions[(dx, dy)]],
            ))

    return []


def bomb_would_destroy_crate(field, start):
    """Return True if a bomb placed at start would hit at least one crate."""
    start_x, start_y = start

    for dx, dy in DIRECTIONS:
        for distance in range(1, BOMB_POWER + 1):
            nx = start_x + dx * distance
            ny = start_y + dy * distance

            if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
                break

            # Stone walls block the blast.
            if field[nx, ny] == -1:
                break

            if field[nx, ny] == 1:
                return True

    return False


def safe_bomb_crate_yield(field, start, bombs, explosion_map, bomb_available):
    """Return the normalized number of crates that a bomb placed at the current position would destroy, provided that placing the bomb is safe."""
    # No bomb can be placed.
    if not bomb_available:
        return 0.0

    # Do not encourage bombing if the agent cannot escape safely.
    if not can_escape_after_bomb(field, start, bombs, explosion_map): 
        return 0.0

    crate_count = 0
    for dx, dy in DIRECTIONS:
        for distance in range(1, BOMB_POWER + 1):
            x = start[0] + dx * distance
            y = start[1] + dy * distance

            if field[x, y] == -1:
                break

            if field[x, y] == 1:
                crate_count += 1

    return crate_count / float(4 * BOMB_POWER)
