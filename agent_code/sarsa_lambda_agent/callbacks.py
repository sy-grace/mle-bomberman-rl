import os
import pickle
import random
import numpy as np
from collections import deque

from .model import Linear_SARSAModel

DIRECTIONS = [
    (0, -1),    # UP
    (0, 1),     # DOWN
    (-1, 0),    # LEFT
    (1, 0)      # RIGHT
]

TASK1_ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT']
TASK2_ACTIONS = TASK1_ACTIONS + ["BOMB"]

EPSILON_START = 1.0
FEATURE_SIZES = {"f0": 7, "f1": 11, "f2": 25, "f3": 26, "f4": 27, "f5": 38, "f6": 38}

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
        raise ValueError("FEATURE_MODE must be one of 'f0', 'f1', 'f2', 'f3', 'f4', 'f5', or 'f6'.")

    self.feature_size = FEATURE_SIZES[self.feature_mode]
    self.logger.info(f"Feature mode: {self.feature_mode} ({self.feature_size} features)")

    self.actions = actions_for_feature_mode(self.feature_mode)

    # Persistent escape controller
    self.escape_bomb_position = None

    # Check if file exists
    checkpoint_exists = os.path.isfile("my-saved-model.pt")

    if self.train and self.model_start_mode == "fresh":
        # Initialize fresh model
        self.sarsa_lambda = float(os.getenv("SARSA_LAMBDA", "0.8"))
        self.logger.info(f"SARSA lambda: {self.sarsa_lambda}")
        self.logger.info("Setting up model from scratch.")
        self.model = Linear_SARSAModel(input_size=self.feature_size, output_size=len(self.actions), seed=self.experiment_seed, lambda_=self.sarsa_lambda)
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
    """Return the pending SARSA action or select a new action."""
    if game_state["step"] == 1:
        self.pending_action = None
        self.escape_bomb_position = None

    # Compute the features actually observed at this decision step.
    features = state_to_features(game_state, self.feature_mode)

    # Reuse the action already selected for the SARSA target.
    if self.train and getattr(self, "pending_action", None) is not None:
        action = self.pending_action
        self.pending_action = None
    else:
        action = select_action(self, features, game_state)

    return action


def select_action(self, features: np.ndarray, game_state=None) -> str:
    """Select an action using the escpae controller first, then the SARSA policy."""
    # F5/F6: keep persistent protection from the agent's own recently placed bomb until that bomb and its explosion have disappeared.
    if self.feature_mode in {"f5", "f6"} and game_state is not None:
        escape_action = persistent_escape_action(self, game_state, features)

        if escape_action is not None:
            return escape_action

    # F6: escape bombs that are not covered by the persistent own-bomb controller. In normal play these are opponent bombs.
    if self.feature_mode in {"f6"} and game_state is not None:
        opponent_escape_action = opponent_bomb_aviodance_action(self, game_state, features)
        if opponent_escape_action is not None:
            return opponent_escape_action

    action = policy_action(self, features)

    # If the policy decides to place a valid bomb, remember its position.
    # The bomb will appear in the next game state.
    if self.feature_mode in {"f5", "f6"} and game_state is not None and action == "BOMB" and game_state["self"][2]:
        self.escape_bomb_position = game_state["self"][3]

    return action


def policy_action(self, features: np.ndarray, allowed_actions=None) -> str:
    """Select an epsilon-greedy SARSA action, optionally from a restricted action set."""
    allowed_actions = list(allowed_actions or self.actions)

    if not allowed_actions:
        return "WAIT"

    # Exploration
    if self.train and self.rng.random() < self.epsilon:
        self.logger.debug("Choosing action purely at random.")
        return self.rng.choice(allowed_actions)

    # Exploitation
    q_values = self.model.predict(features)

    allowed_indices = [self.actions.index(action) for action in allowed_actions]

    best_index = max(allowed_indices, key=lambda index: q_values[index])

    self.logger.debug("Choosing action with the highest Q-value.")
    return self.actions[best_index]


def state_to_features(game_state: dict, feature_mode: str) -> np.ndarray:
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

    if feature_mode not in FEATURE_SIZES:
        raise ValueError("feature_mode must be one of 'f0', 'f1', 'f2', 'f3', 'f4', 'f5', or 'f6'.")

    # Get the current location of the agent
    field = game_state["field"] # np.ndarray
    bombs = game_state["bombs"] # (x, y), timer
    coins = game_state["coins"] # x, y
    agent = game_state["self"] # name, score, bombs_left, (x, y)
    others = game_state.get("others", [])

    explosion_map = game_state.get("explosion_map")

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
    # F5: F4 + [opponent_UP, opponent_DOWN, opponent_LEFT, opponent_RIGHT,
    #           opponent_path_UP, opponent_path_DOWN, opponent_path_LEFT, opponent_path_RIGHT,
    #           safe_and_useful_opponent_bomb, bomb_urgency, normalized_crate_distance]
    # F6: same 38-dimensional representation as F5 + opponent bomb avoidance controller
    feature_size = FEATURE_SIZES[feature_mode]
    features = np.zeros(feature_size)

    # Set bias as 1
    features[0] = 1

    UP_POS = (agent_x, agent_y - 1)
    DOWN_POS = (agent_x, agent_y + 1)
    LEFT_POS = (agent_x - 1, agent_y)
    RIGHT_POS = (agent_x + 1, agent_y)

    # Check the walkable locations around the agent
    UP = field[UP_POS]
    DOWN = field[DOWN_POS]
    LEFT = field[LEFT_POS]
    RIGHT = field[RIGHT_POS]

    if UP == 0 and explosion_map[UP_POS] == 0:
        features[1] = 1
    if DOWN == 0 and explosion_map[DOWN_POS] == 0:
        features[2] = 1
    if LEFT == 0 and explosion_map[LEFT_POS] == 0:
        features[3] = 1
    if RIGHT == 0 and explosion_map[RIGHT_POS] == 0:
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

        if closest_coin is not None and feature_mode in {"f1", "f2", "f3", "f4", "f5", "f6"}:
            path_directions = shortest_path_directions(field, agent[3], closest_coin)
            features[7:11] = path_directions

    if feature_mode in {"f2", "f3", "f4", "f5", "f6"}:
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

        if crate_targets:
            crate_distances = shortest_path_distance_map(field, crate_targets)
            features[16:20] = shortest_path_directions_to_any(field, agent[3], crate_targets, distances=crate_distances)

        danger_tiles = bomb_danger_tiles(field, bombs, explosion_map)
        features[20] = float(agent[3] in danger_tiles)

        # 21:25: escape direction [UP, DOWN, LEFT, RIGHT]
        if feature_mode in {"f5", "f6"}:
            opponent_positions = {other[3] for other in others}
        else:
            opponent_positions = set()

        safe_targets = set()

        for x in range(field.shape[0]):
            for y in range(field.shape[1]):
                if field[x, y] == 0 and (x, y) not in danger_tiles and (x, y) not in opponent_positions:
                    safe_targets.add((x, y))

        escape_field = field.copy()

        for (bomb_x, bomb_y), _ in bombs:
            if (bomb_x, bomb_y) != agent[3]:
                escape_field[bomb_x, bomb_y] = -1

        if feature_mode in {"f5", "f6"}:
            for ox, oy in opponent_positions:
                escape_field[ox, oy] = -1

        if features[20] == 1.0:
            features[21:25] = shortest_path_directions_to_any(escape_field, agent[3], safe_targets)

        # 25: safe to bomb
        if feature_mode in {"f3", "f4", "f5", "f6"}:
            safe_to_bomb = bool(agent[2] and can_escape_after_bomb(field, agent[3], bombs, explosion_map, blocked_positions=opponent_positions))
            features[25] = float(safe_to_bomb)

            # 26: safe and useful bomb
            if feature_mode in {"f4", "f5", "f6"}:
                features[26] = float(safe_to_bomb and bomb_would_destroy_crate(field, agent[3]))

                if feature_mode in {"f5", "f6"}:
                    # 27:31: adjacent opponents [UP, DOWN, LEFT, RIGHT]
                    for i, (dx, dy) in enumerate(DIRECTIONS):
                        nx = agent_x + dx
                        ny = agent_y + dy

                        if (nx, ny) in opponent_positions:
                            features[27 + i] = 1.0

                    # 31:35: path towards an opponent
                    opponent_targets = opponent_approach_targets(field, others)
                    features[31:35] = shortest_path_directions_to_any(field, agent[3], opponent_targets)

                    # 35: safe and useful bomb against opponent
                    features[35] = float(safe_to_bomb and bomb_would_hit_opponent(field, agent[3], opponent_positions))

                    # 36: urgency of current bomb danger
                    features[36] = bomb_danger_urgency(field, agent[3], bombs, explosion_map)

                    # 37: normalized shortest-path distance to a crate placement target
                    if crate_targets:
                        crate_distance = crate_distances[agent[3]]

                        if crate_distance >= 0:
                            max_distance = field.shape[0] + field.shape[1] - 2
                            features[37] = crate_distance / max_distance

    # Return the final feature vector
    return features


def actions_for_feature_mode(feature_mode):
    if feature_mode in {"f2", "f3", "f4", "f5", "f6"}:
        return TASK2_ACTIONS
    return TASK1_ACTIONS


def shortest_path_directions(field, start, target):
    """Return valid first-step directions along shortest paths from start to target."""
    return shortest_path_directions_to_any(field, start, {target})


def shortest_path_directions_to_any(field, start, targets, distances=None):
    """Return valid first-step directions along shortest paths to any target."""

    # No movement needed if already at target
    if not targets:
        return np.zeros(4)

    if start in targets:
        return np.zeros(4)

    if distances is None:
        distances = shortest_path_distance_map(field, targets)

    # Target cannot be reached from start
    if distances[start] == -1:
        return np.zeros(4)

    current_distance = distances[start]
    path_directions = np.zeros(4)

    for i, (dx, dy) in enumerate(DIRECTIONS):
        nx = start[0] + dx
        ny = start[1] + dy

        if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
            continue

        if distances[nx, ny] == current_distance - 1:
            path_directions[i] = 1

    return path_directions


def shortest_path_distance_map(field, targets):
    """Return BFS distance from every reachable free tile to the nearest target."""
    distances = np.full(field.shape, -1, dtype=int)
    queue = deque()

    for target in targets:
        distances[target] = 0
        queue.append(target)

    while queue:
        x, y = queue.popleft()

        for dx, dy in DIRECTIONS:
            nx = x + dx
            ny = y + dy

            if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
                continue

            # Only free tiles are walkable
            if field[nx, ny] != 0:
                continue

            if distances[nx, ny] != -1:
                continue

            distances[nx, ny] = distances[x, y] + 1
            queue.append((nx, ny))

    return distances


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


def bomb_danger_tiles(field, bombs, explosion_map=None):
    danger_tiles = set()

    for (bomb_x, bomb_y), _ in bombs:
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


def bomb_danger_urgency(field, position, bombs, explosion_map=None):
    """
    Return urgency of bomb danger at position.

    0.00 = no currnet bomb danger
    0.25 = dangerous bomb with timer 3
    0.50 = dangerous bomb with timer 2
    0.75 = dangerous bomb with timer 1
    1.00 = dangerous bomb with timer 0 / active explosion
    """
    x, y = position

    # Active explosion is maximally urgent.
    if explosion_map is not None and explosion_map[x, y] > 0:
        return 1.0

    min_timer = None

    for (bomb_x, bomb_y), timer in bombs:
        blast_tiles = {(bomb_x, bomb_y)}

        for dx, dy in DIRECTIONS:
            for distance in range(1, BOMB_POWER + 1):
                nx = bomb_x + dx * distance
                ny = bomb_y + dy * distance

                if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
                    break

                # Only stone walls stop the blast.
                if field[nx, ny] == -1:
                    break

                blast_tiles.add((nx, ny))

        if position in blast_tiles:
            if min_timer is None or timer < min_timer:
                min_timer = timer

    if min_timer is None:
        return 0.0

    return (BOMB_TIMER - min_timer) / BOMB_TIMER


def can_escape_after_bomb(field, start, bombs, explosion_map=None, blocked_positions=None):
    """
    Return True if a bomb placed at start still allows escape within BOMB_TIMER moves.
    blocked_positions contains temporarily occupied tiles, such as opponent positions, that cannot be used as part of an escape route.
    """
    hypothetical_bombs = list(bombs) + [(start, BOMB_TIMER)]
    danger_tiles = bomb_danger_tiles(field, hypothetical_bombs, explosion_map)
    existing_bomb_tiles = {position for position, _ in bombs}
    blocked_positions = set(blocked_positions or [])

    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        (x, y), distance = queue.popleft()

        # A reachable tile outside the future blast zone is a valid escape.
        if distance > 0 and (x, y) not in danger_tiles:
            return True

        # No more movement is possible before the hypothetical bomb explodes.
        if distance >= BOMB_TIMER:
            continue

        for dx, dy in DIRECTIONS:
            nx = x + dx
            ny = y + dy
            next_pos = (nx, ny)

            if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
                continue

            # Walls and crates are not walkable.
            if field[nx, ny] != 0:
                continue

            # Existing bombs are obstacles.
            if next_pos in existing_bomb_tiles:
                continue

            # Other agents temporarily occupy their tiles.
            if next_pos in blocked_positions:
                continue

            # After leaving the newly placed bomb tile, the agent cannot walk back onto it.
            if next_pos == start:
                continue

            # Never walk through an active explosion.
            if explosion_map is not None and explosion_map[nx, ny] > 0:
                continue

            if next_pos in visited:
                continue

            visited.add(next_pos)
            queue.append((next_pos, distance + 1))

    return False


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


def opponent_approach_targets(field, opponents):
    """Return free tiles adjacent to any opponent."""
    targets = set()

    for opponent in opponents:
        ox, oy = opponent[3]

        for dx, dy in DIRECTIONS:
            nx = ox + dx
            ny = oy + dy

            if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1] and field[nx, ny] == 0:
                targets.add((nx, ny))

    return targets


def bomb_would_hit_opponent(field, start, opponent_positions):
    """Return True if a bomb at start would hit an opponent."""
    start_x, start_y = start

    for dx, dy in DIRECTIONS:
        for distance in range(1, BOMB_POWER + 1):
            nx = start_x + dx * distance
            ny = start_y + dy * distance

            if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
                break

            # Stone wall blocks blast.
            if field[nx, ny] == -1:
                break

            if (nx, ny) in opponent_positions:
                return True

    return False


def escape_path(field, start, bombs, explosion_map=None, blocked_positions=None):
    """
    Return a shortest path out of bomb danger.

    Bomb timer t means that t + 1 movement actions are still possible before the bomb explodes in this environment.
    """
    blocked_positions = set(blocked_positions or [])

    danger_tiles = bomb_danger_tiles(field, bombs, explosion_map)

    relevant_deadlines = []

    for bomb_position, timer in bombs:
        bomb_danger = bomb_danger_tiles(field, [(bomb_position, timer)])

        if start in bomb_danger:
            # The environment executes the action first. A bomb with timer == 0 explodes after this action.
            relevant_deadlines.append(timer + 1)

    if explosion_map is not None and explosion_map[start] > 0:
        relevant_deadlines.append(1)

    if not relevant_deadlines:
        return []

    max_distance = min(relevant_deadlines)

    bomb_positions = {position for position, _timer in bombs}

    offsets_to_actions = dict(zip(DIRECTIONS, ("UP", "DOWN", "LEFT", "RIGHT")))

    queue = deque([(start, 0, [])])

    visited = {start}

    while queue:
        (x, y), distance, path = queue.popleft()

        # We have left all currently relevant blast areas.
        if distance > 0 and (x, y) not in danger_tiles:
            return path

        if distance >= max_distance:
            continue

        for dx, dy in DIRECTIONS:
            nx = x + dx
            ny = y + dy
            next_position = (nx, ny)

            if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
                continue

            # Wall / crate
            if field[nx, ny] != 0:
                continue

            # Cannot walk onto a bomb.
            if next_position in bomb_positions:
                continue

            # Cannot walk through another agent.
            if next_position in blocked_positions:
                continue

            # Do not enter an already active explosion.
            if explosion_map is not None and explosion_map[nx, ny] > 0:
                continue

            if next_position in visited:
                continue

            visited.add(next_position)

            queue.append((next_position, distance + 1, path + [offsets_to_actions[(dx, dy)]]))

    return []


def persistent_escape_action(self, game_state, features):
    """Keep the agent safe from its own recently placed bomb until the bomb and its explosion are no longer dangerous."""
    tracked_bomb = getattr(self, "escape_bomb_position", None)

    if tracked_bomb is None:
        return None

    field = game_state["field"]
    bombs = game_state.get("bombs", [])
    explosion_map = game_state.get("explosion_map")
    current_position = game_state["self"][3]

    opponent_positions = {other[3] for other in game_state.get("others", [])}

    # Blast area of the bomb that activated escape mode.
    tracked_blast = bomb_danger_tiles(field, [(tracked_bomb, 0)], explosion_map=None)
    bomb_still_exists = any(position == tracked_bomb for position, _timer in bombs)
    explosion_still_active = explosion_map is not None and any(explosion_map[x, y] > 0 for x, y in tracked_blast)

    # Escape mode ends only after both the bomb and its explosion disappear.
    if not bomb_still_exists and not explosion_still_active:
        self.escape_bomb_position = None
        return None

    # If still inside bomb danger, deterministically follow the shortest escape path.
    current_danger = bomb_danger_tiles(field, bombs, explosion_map)

    if current_position in current_danger:
        path = escape_path(field, current_position, bombs, explosion_map, blocked_positions=opponent_positions)

        if path:
            return path[0]

    # We may already have escaped the blast, but the bomb is still alive. Do not allow the learned policy to walk back into the tracked blast.
    allowed_actions = safe_actions_during_escape(game_state, tracked_blast)

    return policy_action(self, features, allowed_actions=allowed_actions)


def safe_actions_during_escape(game_state, tracked_blast):
    """Return actions that cannot immediately re-enter the tracked bomb blast."""
    field = game_state["field"]
    bombs = game_state.get("bombs", [])
    explosion_map = game_state.get("explosion_map")
    x, y = game_state["self"][3]

    bomb_positions = {position for position, _timer in bombs}
    opponent_positions = {other[3] for other in game_state.get("others", [])}

    imminent_bombs = [(position, timer) for position, timer in bombs if timer <= 0]
    immediate_danger = bomb_danger_tiles(field, imminent_bombs, explosion_map)

    action_offsets = {
        "UP": (0, -1),
        "DOWN": (0, 1),
        "LEFT": (-1, 0),
        "RIGHT": (1, 0)
    }

    allowed = []

    for action, (dx, dy) in action_offsets.items():
        next_position = (x + dx, y + dy)
        nx, ny = next_position

        if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
            continue

        if field[nx, ny] != 0:
            continue

        if next_position in bomb_positions:
            continue
        
        if next_position in opponent_positions:
            continue
        
        if next_position in immediate_danger:
            continue

        if next_position in tracked_blast:
            continue

        allowed.append(action)

    # Waiting is allowed only if the current tile itself is safe.
    curernt_position = (x, y)

    if curernt_position not in tracked_blast and curernt_position not in immediate_danger:
        allowed.append("WAIT")

    return allowed


def opponent_bomb_aviodance_action(self, game_state, features):
    """
    F6 controller for bombs not handled by the persistent own-bomb controller.
    The framework does not expose bomb ownership in game_state, so this controller reacts 
    to any current bomb danger when no tracked own-bomb controller has already taken over.
    """
    field = game_state["field"]
    bombs = game_state.get("bombs", [])
    explosion_map = game_state.get("explosion_map")
    curernt_position = game_state["self"][3]

    opponent_positions = {other[3] for other in game_state.get("others", [])}

    current_danger = bomb_danger_tiles(field, bombs, explosion_map)

    # No bomb currently threatens the agent.
    if curernt_position not in current_danger:
        return None

    # No bomb currently threatens the agent.
    if curernt_position not in current_danger:
        return None

    # Deterministically follow a shortest feasible escape path.
    path = escape_path(field, curernt_position, bombs, explosion_map, blocked_positions=opponent_positions)

    if path:
        return path[0]

    # No complete escape route was found. At least avoid actions that are immediately lethal in the current step.
    allowed_actions = safe_actions_during_escape(game_state, tracked_blast=set())

    if allowed_actions:
        return policy_action(self, features, allowed_actions=allowed_actions)

    # No survivable movement is known.
    return "WAIT"
