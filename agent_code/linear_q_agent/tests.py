import os
import pickle
import numpy as np
import unittest
import tempfile
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import events as game_events

from . import callbacks, train
from .callbacks import state_to_features
from .model import Linear_QModel


@contextmanager
def temporary_working_directory(directory):
    previous_directory = os.getcwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(previous_directory)


class LinearQAgentTest(unittest.TestCase):
    @staticmethod
    def _game_state():
        field = np.full((7, 7), -1, dtype=int)
        field[1:-1, 1:-1] = 0
        return {
            "field": field,
            "bombs": [],
            "coins": [(5, 4)],
            "self": ("player", 0, 1, (3, 3)),
            "explosion_map": np.zeros((7, 7)),
            "step": 1,
        }


    def test_features_all_directions_free(self):
        """Feature Test A: All directions are free."""
        state = self._game_state()
        features = state_to_features(state, "f1")

        expected = [1, 1, 1, 1, 1]

        np.testing.assert_array_equal(features[:5], expected)


    def test_features_up_blocked(self):
        """Feature Test B: Up tile is blocked."""
        state = self._game_state()

        # UP of agent (3, 3) is (3, 2)
        state["field"][3, 2] = 1
        features = state_to_features(state, "f1")

        expected = [1, 0, 1, 1, 1]
        np.testing.assert_array_equal(features[:5], expected)


    def test_features_coin_to_the_right(self):
        """Feature Test C: When there is a coin on the right"""
        state = self._game_state()

        # There is a coin at (5, 3), and the agent is at (3, 3)
        state["coins"] = [(5, 3)]
        features = state_to_features(state, "f1")

        expected = [2/6, 0]
        np.testing.assert_allclose(features[5:7], expected)


    def test_features_nearest_coin_select(self):
        """Feature Test D: Check if the agent chooses the nearest coin among all coins."""
        state = self._game_state()

        # There are two coins: one at (5, 4), and the other at (2, 2)
        state["coins"] = [(5, 4), (2, 2)]
        features = state_to_features(state, "f1")

        expected = [-1/6, -1/6]
        np.testing.assert_allclose(features[5:7], expected)


    def test_features_no_coin(self):
        """Feature Test E: No coin in the field."""
        state = self._game_state()

        # There is no coin in the field
        state["coins"] = []
        features = state_to_features(state, "f1")

        expected = [0, 0]
        np.testing.assert_allclose(features[5:7], expected)
        

    def test_features_agent_in_corner(self):
        """Feature Test F: Agent is at a walkable corner next to border walls."""
        state = self._game_state()

        # The agent is in the corner
        state["self"] = ("player", 0, 1, (5, 5))
        features = state_to_features(state, "f1")

        expected = [1, 1, 0, 1, 0]
        np.testing.assert_array_equal(features[:5], expected)
        

    def test_f2_feature_vector_has_twenty_five_features(self):
        """Feature Test G: Verify that F2 produces a 25-dimensional feature vector."""
        state = self._game_state()
        features = state_to_features(state, "f2")

        expected = (25,)
        self.assertEqual(features.shape, expected)


    def test_f1_uses_task1_action_space(self):
        """Feature Test H: Verify that F1 uses the five Task 1 actions without BOMB."""
        actions = callbacks.actions_for_feature_mode("f1")

        expected = ["UP", "DOWN", "LEFT", "RIGHT", "WAIT"]
        self.assertEqual(actions, expected)


    def test_f2_uses_task2_action_space_with_bomb(self):
        """Feature Test I: Verify that F2 uses the six Task 2 actions including BOMB."""
        actions = callbacks.actions_for_feature_mode("f2")

        expected = ["UP", "DOWN", "LEFT", "RIGHT", "WAIT", "BOMB"]
        self.assertEqual(actions, expected)


    def test_f2_fresh_model_uses_twenty_five_inputs_and_six_outputs(self):
        """Feature Test J: Verify that fresh F2 training creates a 25-input, 6-action model."""
        agent = SimpleNamespace(
            train=True,
            logger=Mock()
        )

        with patch.dict(os.environ, {"MODEL_START_MODE": "fresh", "FEATURE_MODE": "f2"}, clear=True):
            callbacks.setup(agent)

        self.assertEqual(agent.model.input_size, 25)
        self.assertEqual(agent.model.output_size, 6)


    def test_f2_marks_available_bomb(self):
        """Feature Test K: Verify that F2 marks an available bomb."""
        state = self._game_state()

        name, score, _, position = state["self"]
        state["self"] = (name, score, True, position)

        features = state_to_features(state, "f2")

        expected = 1.0
        np.testing.assert_array_equal(features[11], expected)


    def test_f2_detects_adjacent_crate(self):
        """Feature Test L: Verify that F2 detects a crate next to the agent."""
        state = self._game_state()

        x, y = 3, 3
        name, score, bombs_left, _ = state["self"]
        state["self"] = (name, score, bombs_left, (x, y))

        field = state["field"].copy()
        field[x, y] = 0
        field[x, y - 1] = 1 # crate above agent
        state["field"] = field

        features = state_to_features(state, "f2")

        expected = np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(features[12:16], expected)
        

    def test_f2_points_toward_nearest_crate_placement_tile(self):
        """Feature Test M: Verify that F2 points toward a reachable bomb-placement tile near a crate."""
        state = self._game_state()

        name, score, bombs_left, _ = state["self"]
        state["self"] = (name, score, bombs_left, (2, 3))

        field = state["field"].copy()

        # Remove existing crates from the test area if necessary.
        field[field == 1] = 0

        field[2, 3] = 0
        field[5, 3] = 1
        state["field"] = field

        features = state_to_features(state, "f2")

        expected = np.array([0.0, 0.0, 0.0, 1.0])
        np.testing.assert_array_equal(features[16:20], expected)


    def test_f2_crate_path_is_zero_when_no_crates_exists(self):
        """Feature Test N: Verify that crate-path features remain zero when no crates exist."""
        state = self._game_state()

        field = state["field"].copy()
        field[field == 1] = 0
        state["field"] = field

        features = state_to_features(state, "f2")

        expected = np.zeros(4)
        np.testing.assert_array_equal(features[16:20], expected)


    def test_f2_detects_bomb_danger(self):
        """Feature Test O: Verify that F2 marks the agent as endangered by a bomb."""
        state = self._game_state()

        state["self"] = ("player", 0, 0, (3, 3))
        state["bombs"] = [((3, 5), 3)]

        features = state_to_features(state, "f2")

        self.assertEqual(features[20], 1.0)


    def test_f2_escape_is_zero_when_not_in_bomb_danger(self):
        """Feature Test P: Verify that escape features remain zero when the agent is safe."""
        state = self._game_state()
        state["bombs"] = []

        features = state_to_features(state, "f2")

        self.assertEqual(features[20], 0.0)

        expected = np.zeros(4)
        np.testing.assert_array_equal(features[21:25], expected)


    def test_f2_escape_points_toward_safe_tile(self):
        """Feature Test Q: Verify that F2 points toward a reachable safe tile when in bomb danger."""
        state = self._game_state()

        state["self"] = ("player", 0, 0, (3, 3))
        state["bombs"] = [((3, 5), 3)]

        # Block LEFT so RIGHT is the only immediate safe direction.
        state["field"][2, 3] = -1

        features = state_to_features(state, "f2")

        expected = np.array([0.0, 0.0, 0.0, 1.0])
        np.testing.assert_array_equal(features[21:25], expected)


    def test_f2_wall_blocks_bomb_danger(self):
        """Feature Test R: Verify that a wall blocks a bomb blast before it reaches the agent."""
        state = self._game_state()

        state["self"] = ("player", 0, 1, (3, 3))
        state["bombs"] = [((3, 5), 3)]

        # Wall between agent and bomb
        state["field"][3, 4] = -1

        features = state_to_features(state, "f2")

        self.assertEqual(features[20], 0.0)


    def test_f2_can_escape_while_standing_on_own_bomb(self):
        """Feature Test S: Verify that an agent standing on its own bomb still gets an escape direction."""
        state = self._game_state()

        state["coins"] = []
        state["self"] = ("player", 0, False, (1, 1))
        state["bombs"] = [((1, 1), 3)]

        # Force RIGHT as the escape route.
        state["field"][1, 2] = -1

        features = state_to_features(state, "f2")

        self.assertEqual(features[20], 1.0)
        self.assertEqual(features[24], 1.0) # RIGHT


    def test_f3_feature_vector_has_twenty_six_features(self):
        """Feature Test T: Verify that F3 produces a 26-dimensional feature vector."""
        state = self._game_state()
        features = state_to_features(state, "f3")
        self.assertEqual(features.shape, (26,))


    def test_f3_preserves_all_f2_features(self):
        """Feature Test U: F3 must preserve the complete F2 representation."""
        state = self._game_state()

        f2 = state_to_features(state, "f2")
        f3 = state_to_features(state, "f3")

        np.testing.assert_array_equal(f3[:25], f2)


    def test_f3_uses_task2_action_space_with_bomb(self):
        """Feature Test V: F3 uses the six Task 2 actions."""
        actions = callbacks.actions_for_feature_mode("f3")
        expected = ["UP", "DOWN", "LEFT", "RIGHT", "WAIT", "BOMB"]
        self.assertEqual(actions, expected)


    def test_f3_fresh_model_uses_twenty_six_inputs_and_six_outputs(self):
        """Feature Test W: Fresh F3 training creates a 26-input, 6-action model."""
        agent = SimpleNamespace(train=True, logger=Mock())

        with patch.dict(os.environ, {"MODEL_START_MODE": "fresh", "FEATURE_MODE": "f3"}, clear=True):
            callbacks.setup(agent)

        self.assertEqual(agent.model.input_size, 26)
        self.assertEqual(agent.model.output_size, 6)


    def test_f3_marks_safe_bomb_placement_in_open_space(self):
        """Feature Test X: Bomb placement is safe when an escape route exists."""
        state = self._game_state()
        state["self"] = ("player", 0, True, (3, 3))
        features = state_to_features(state, "f3")
        self.assertEqual(features[25], 1.0)


    def test_f3_safe_to_bomb_is_zero_when_bomb_unavailable(self):
        """Feature Test Y: safe_to_bomb is zero when BOMB cannot be placed."""
        state = self._game_state()
        state["self"] = ("player", 0, False, (3, 3))
        features = state_to_features(state, "f3")
        self.assertEqual(features[25], 0.0)


    def test_f3_detects_unsafe_bomb_in_dead_end(self):
        """Feature Test Z: Bomb placement is unsafe when no escape route exists."""
        state = self._game_state()
        field = np.full((7, 7), -1, dtype=int)

        # Agent and a corridor that stays entirely inside the bomb blast.
        field[3, 3] = 0
        field[3, 2] = 0
        field[3, 1] = 0

        state["field"] = field
        state["self"] = ("player", 0, True, (3, 3))
        state["coins"] = []
        state["explosion_map"] = np.zeros((7, 7))

        features = state_to_features(state, "f3")

        self.assertEqual(features[25], 0.0)


    def test_f3_allows_escape_in_exactly_four_moves(self):
        """Feature Test AA: Escape on the fourth movement step is still safe."""
        field = np.full((9, 9), -1, dtype=int)

        start = (4, 4)

        # Three steps upward remain inside the blast.
        field[4, 4] = 0
        field[4, 3] = 0
        field[4, 2] = 0
        field[4, 1] = 0

        # Fourth step leaves the blast line.
        field[5, 1] = 0

        safe = callbacks.can_escape_after_bomb(field, start, bombs=[], explosion_map=np.zeros((9, 9)))

        self.assertTrue(safe)


    def test_f3_crate_blocks_escape_route(self):
        """Feature Test AB: Crates cannot be crossed while escaping."""
        field = np.full((9, 9), -1, dtype=int)

        start = (4, 4)

        field[4, 4] = 0
        field[4, 3] = 0
        field[4, 2] = 1 # crate blocks the only route
        field[4, 1] = 0
        field[5, 1] = 0

        safe = callbacks.can_escape_after_bomb(field, start, bombs=[], explosion_map=np.zeros((9, 9)))

        self.assertFalse(safe)


    def test_f3_active_explosion_blocks_escape_route(self):
        """Feature Test AC: Active explosions cannot be used as escape paths."""
        field = np.full((9, 9), -1, dtype=int)

        start = (4, 4)

        field[4, 4] = 0
        field[4, 3] = 0
        field[4, 2] = 0
        field[4, 1] = 0
        field[5, 1] = 0

        explosion_map = np.zeros((9, 9))
        explosion_map[4, 3] = 1

        safe = callbacks.can_escape_after_bomb(field, start, bombs=[], explosion_map=explosion_map)

        self.assertFalse(safe)


    def test_predict_returns_one_value_per_action(self):
        model = Linear_QModel(input_size=7, output_size=len(callbacks.actions_for_feature_mode("f0")), seed=1)
        features = np.ones(7)

        q_values = model.predict(features)

        self.assertEqual(q_values.shape, (len(callbacks.actions_for_feature_mode("f0")),))
        self.assertTrue(np.isfinite(q_values).all())


    def test_terminal_update_changes_only_selected_action(self):
        model = Linear_QModel(
            input_size=3,
            output_size=2,
            learning_rate=0.1,
            gamma=0.9,
            seed=1,
        )
        state = np.array([1.0, 2.0, -1.0])
        old_weights = model.weights.copy()
        old_q = model.predict(state)[1]

        td_error = model.update(
            state=state,
            action=1,
            reward=2.0,
            next_state=None,
        )

        expected_error = 2.0 - old_q
        expected_weights = old_weights[:, 1] + 0.1 * expected_error * state
        np.testing.assert_allclose(td_error, expected_error)
        np.testing.assert_allclose(model.weights[:, 1], expected_weights)
        np.testing.assert_allclose(model.weights[:, 0], old_weights[:, 0])


    def test_non_terminal_update_uses_next_state_value(self):
        model = Linear_QModel(
            input_size=2,
            output_size=2,
            learning_rate=0.1,
            gamma=0.5,
            seed=1,
        )
        model.weights[:] = 0.0
        model.weights[:, 1] = [1.0, 0.0]
        state = np.array([1.0, 2.0])
        next_state = np.array([3.0, 0.0])

        td_error = model.update(
            state=state,
            action=0,
            reward=1.0,
            next_state=next_state,
        )

        # max(Q(next_state)) is 3, so target is 1 + 0.5 * 3 = 2.5.
        self.assertAlmostEqual(td_error, 2.5)
        np.testing.assert_allclose(model.weights[:, 0], [0.25, 0.5])


    def test_act_explores_when_random_value_is_below_epsilon(self):
        agent = SimpleNamespace(
            train=True,
            epsilon=0.5,
            model=Mock(),
            logger=Mock(),
            feature_mode="f1",
            actions=callbacks.actions_for_feature_mode("f1"),
            rng=Mock()
        )

        agent.rng.random.return_value = 0.1
        agent.rng.choice.return_value = "WAIT"
        action = callbacks.act(agent, self._game_state())

        self.assertEqual(action, "WAIT")
        agent.rng.choice.assert_called_once_with(callbacks.actions_for_feature_mode("f1"))
        agent.model.predict.assert_not_called()


    def test_act_exploits_highest_q_value_when_not_exploring(self):
        agent = SimpleNamespace(
            train=True,
            epsilon=0.5,
            model=Mock(),
            logger=Mock(),
            feature_mode="f1",
            actions=callbacks.actions_for_feature_mode("f1"),
            rng=Mock()
        )
        agent.model.predict.return_value = np.array([1.0, 2.0, 0.0, 4.0, 3.0])

        agent.rng.random.return_value = 0.9
        action = callbacks.act(agent, self._game_state())

        self.assertEqual(action, "RIGHT")
        agent.model.predict.assert_called_once()


    def test_game_event_updates_selected_action_without_decaying_epsilon(self):
        agent = SimpleNamespace(
            model=Mock(),
            logger=Mock(),
            epsilon=0.5,
            epsilon_min=0.05,
            epsilon_decay=0.995,
            transitions=[],
            feature_mode="f1"
        )
        old_state = self._game_state()
        new_state = self._game_state()

        with patch.object(train, "reward_from_events", return_value=2.0):
            train.game_events_occurred(agent, old_state, "LEFT", new_state, [])

        state, action, next_state, reward = agent.transitions[-1]
        self.assertEqual(action, "LEFT")
        self.assertEqual(reward, 2.0)
        np.testing.assert_allclose(state, state_to_features(old_state, "f1"))
        np.testing.assert_allclose(next_state, state_to_features(new_state, "f1"))
        agent.model.update.assert_called_once()
        update_state, update_action, update_reward, update_next_state = (
            agent.model.update.call_args.args
        )
        np.testing.assert_allclose(update_state, state)
        self.assertEqual(update_action, train.ACTION_TO_INDEX["LEFT"])
        self.assertEqual(update_reward, 2.0)
        np.testing.assert_allclose(update_next_state, next_state)
        self.assertAlmostEqual(agent.epsilon, 0.5)


    def test_terminal_event_updates_with_no_next_state(self):
        agent = SimpleNamespace(
            model=Mock(),
            logger=Mock(),
            epsilon=0.5,
            epsilon_min=0.05,
            epsilon_decay=0.995,
            transitions=[],
            feature_mode="f1"
        )

        with patch.object(train, "reward_from_events", return_value=-1.0), patch(
            "builtins.open"
        ), patch.object(train.pickle, "dump"):
            train.end_of_round(agent, self._game_state(), "RIGHT", [])

        agent.model.update.assert_called_once()
        _, action, reward, next_state = agent.model.update.call_args.args
        self.assertEqual(action, train.ACTION_TO_INDEX["RIGHT"])
        self.assertEqual(reward, -1.0)
        self.assertIsNone(next_state)
        self.assertAlmostEqual(agent.epsilon, 0.5 * 0.995)


    def test_reward_mode_default_to_basic(self):
        """Reward Test A: Test that the default reward mode is "basic" when no environment is set."""
        agent = SimpleNamespace(logger=Mock())

        with patch.dict(os.environ, {}, clear=True):
            train.setup_training(agent)

        self.assertEqual(agent.reward_mode, "basic")


    def test_reward_from_environment(self):
        """Reward Test B: Test that the reward mode is read from the environment variable."""
        agent = SimpleNamespace(logger=Mock())

        with patch.dict(os.environ, {"REWARD_MODE": "sparse"}, clear=True):
            train.setup_training(agent)

        self.assertEqual(agent.reward_mode, "sparse")


    def test_invalid_reward_mode_raises(self):
        """Reward Test C: Test that an invalid reward mode raises a ValueError."""
        agent = SimpleNamespace(logger=Mock())

        with patch.dict(os.environ, {"REWARD_MODE": "grape"}, clear=True):
            with self.assertRaises(ValueError):
                train.setup_training(agent)


    def test_reward_values_differ_by_mode(self):
        """Reward Test D: Test reward calculation for sparse, basic, and shaped modes."""
        agent = SimpleNamespace(logger=Mock())

        events = [
            game_events.COIN_COLLECTED, # +10
            game_events.INVALID_ACTION, # -2
            train.MOVED_TOWARDS_COIN,   # +1
            train.UNNECESSARILY_WAITED  # -0.5
        ]

        agent.reward_mode = "sparse"
        reward_sparse = train.reward_from_events(agent, events)

        agent.reward_mode = "basic"
        reward_basic = train.reward_from_events(agent, events)
        
        agent.reward_mode = "shaped"
        reward_shaped = train.reward_from_events(agent, events)
        
        self.assertAlmostEqual(reward_sparse, 10)
        self.assertAlmostEqual(reward_basic, 8)
        self.assertAlmostEqual(reward_shaped, 8.5)


    def test_excluded_custom_events_have_zero_reward(self):
        """Reward Test E: Test that excluded custom event contributes zero reward in all modes."""
        agent = SimpleNamespace(logger=Mock())

        events = [
            train.OSCILLATION
        ]

        agent.reward_mode = "sparse"
        reward_sparse = train.reward_from_events(agent, events)

        agent.reward_mode = "basic"
        reward_basic = train.reward_from_events(agent, events)
        
        agent.reward_mode = "shaped"
        reward_shaped = train.reward_from_events(agent, events)
        
        self.assertEqual(reward_sparse, 0)
        self.assertEqual(reward_basic, 0)
        self.assertEqual(reward_shaped, 0)


    def test_coin_collection_skips_distance_shaping(self):
        """Reward Test F: Test that no distance-shaping event is added when a coin is collected."""
        agent = SimpleNamespace(model=Mock(), logger=Mock(), transitions=[], feature_mode="f1")

        old_state = self._game_state()
        new_state = self._game_state()

        old_state["coins"] = [(4, 3), (1, 1)]

        new_state["self"] = ("player", 1, 1, (4, 3))
        new_state["coins"] = [(1, 1)]

        events = [game_events.COIN_COLLECTED]

        with patch.object(train, "reward_from_events", return_value=0.0):
            train.game_events_occurred(agent, old_state, "RIGHT", new_state, events)

        self.assertNotIn(train.MOVED_AWAY_FROM_COIN, events)
        self.assertNotIn(train.MOVED_TOWARDS_COIN, events)


    def test_basic_reward_includes_crate_destruction_and_coin_found(self):
        """Reward Test G: Verify that basic reward values useful crate outcomes."""
        agent = SimpleNamespace(logger=Mock())

        events = [
            game_events.CRATE_DESTROYED, # +2
            game_events.COIN_FOUND, # +3
        ]

        agent.reward_mode = "basic"
        reward = train.reward_from_events(agent, events)
        
        self.assertAlmostEqual(reward, 5)


    def test_basic_reward_penalizes_self_kill(self):
        """Reward Test H: Verify that basic reward strongly penalizes self-destruction."""
        agent = SimpleNamespace(logger=Mock())

        events = [
            game_events.KILLED_SELF, # -20
        ]

        agent.reward_mode = "basic"
        reward = train.reward_from_events(agent, events)
        
        self.assertAlmostEqual(reward, -20)


    def test_sparse_reward_ignores_task2_auxiliary_events(self):
        """Reward Test I: Verify that sparse mode ignores Task 2 auxiliary events."""
        agent = SimpleNamespace(logger=Mock())

        events = [
            game_events.CRATE_DESTROYED, # +2
            game_events.COIN_FOUND, # +3
            game_events.KILLED_SELF, # -20
        ]

        agent.reward_mode = "sparse"
        reward = train.reward_from_events(agent, events)
        
        self.assertAlmostEqual(reward, 0)


    def test_basic_reward_does_not_double_penalize_self_kill(self):
        """Reward Test J: Verify that self-destruction is not penalized twice."""
        agent = SimpleNamespace(logger=Mock())

        events = [
            game_events.KILLED_SELF, # -20
            game_events.GOT_KILLED,
        ]

        agent.reward_mode = "basic"
        reward = train.reward_from_events(agent, events)
        
        self.assertAlmostEqual(reward, -20)


    def test_f2_adds_escape_event_when_leaving_bomb_danger(self):
        """Reward Test K: Verify that leaving bomb danger creates an escape event."""
        agent = SimpleNamespace(
            model=Mock(),
            logger=Mock(),
            transitions=[],
            feature_mode="f2",
        )

        old_state = self._game_state()
        new_state = self._game_state()

        old_state["coins"] = []
        new_state["coins"] = []

        # Agent starts inside the bomb's blast line.
        old_state["self"] = ("player", 0, False, (3, 3))
        old_state["bombs"] = [((3, 5), 3)]

        # Agent moves RIGHT to a safe tile.
        new_state["self"] = ("player", 0, False, (4, 3))
        new_state["bombs"] = [((3, 5), 2)]
        new_state["step"] = 2

        events = []

        with patch.object(train, "reward_from_events", return_value=0.0):
            train.game_events_occurred(agent, old_state, "RIGHT", new_state, events)

        self.assertIn(train.ESCAPED_BOMB_DANGER, events)


    def test_f2_does_not_add_escape_event_when_remaining_safe(self):
        """Reward Test L: Verify that safe-to-safe movement does not create an escape event."""
        agent = SimpleNamespace(
            model=Mock(),
            logger=Mock(),
            transitions=[],
            feature_mode="f2",
        )

        old_state = self._game_state()
        new_state = self._game_state()

        old_state["coins"] = []
        new_state["coins"] = []

        old_state["bombs"] = []
        new_state["bombs"] = []

        old_state["self"] = ("player", 0, True, (3, 3))
        new_state["self"] = ("player", 0, True, (4, 3))
        new_state["step"] = 2

        events = []

        with patch.object(train, "reward_from_events", return_value=0.0):
            train.game_events_occurred(agent, old_state, "RIGHT", new_state, events)
            
        self.assertNotIn(train.ESCAPED_BOMB_DANGER, events)


    def test_shaped_reward_rewards_escaping_bomb_danger(self):
        """Reward Test M: Verify that shaped mode rewards escaping bomb danger."""
        agent = SimpleNamespace(logger=Mock())

        events = [train.ESCAPED_BOMB_DANGER] # +3

        agent.reward_mode = "shaped"
        reward = train.reward_from_events(agent, events)
        
        self.assertAlmostEqual(reward, 3)


    def test_f2_adds_towards_crate_event_for_recommended_move(self):
        """Reward Test N: Verify that following the crate path creates a positive shaping event."""
        agent = SimpleNamespace(
            model=Mock(),
            logger=Mock(),
            transitions=[],
            feature_mode="f2",
        )

        old_state = self._game_state()
        new_state = self._game_state()

        old_state["coins"] = []
        new_state["coins"] = []

        # Agent at (2, 3), crate at (5, 3): nearest bomb-placement tile is (4, 3), so RIGHT is recommended.
        old_state["field"][5, 3] = 1
        new_state["field"][5, 3] = 1

        old_state["self"] = ("player", 0, True, (2, 3))
        new_state["self"] = ("player", 0, True, (3, 3))
        new_state["step"] = 2

        events = []

        with patch.object(train, "reward_from_events", return_value=0.0):
            train.game_events_occurred(agent, old_state, "RIGHT", new_state, events)
            
        self.assertIn(train.MOVED_TOWARDS_CRATE, events)


    def test_f2_adds_away_from_crate_event_for_wrong_move(self):
        """Reward Test O: Verify that moving away from the crate path creates a penalty event."""
        agent = SimpleNamespace(
            model=Mock(),
            logger=Mock(),
            transitions=[],
            feature_mode="f2",
        )

        old_state = self._game_state()
        new_state = self._game_state()

        old_state["coins"] = []
        new_state["coins"] = []

        # Agent at (2, 3), crate at (5, 3): nearest bomb-placement tile is (4, 3), so LEFT causes penalty.
        old_state["field"][5, 3] = 1
        new_state["field"][5, 3] = 1

        old_state["self"] = ("player", 0, True, (2, 3))
        new_state["self"] = ("player", 0, True, (1, 3))
        new_state["step"] = 2

        events = []

        with patch.object(train, "reward_from_events", return_value=0.0):
            train.game_events_occurred(agent, old_state, "LEFT", new_state, events)
            
        self.assertIn(train.MOVED_AWAY_FROM_CRATE, events)


    def test_f2_does_not_shape_crate_navigation_while_in_bomb_danger(self):
        """Reward Test P: Bomb escape takes priority over crate navigation."""
        agent = SimpleNamespace(
            model=Mock(),
            logger=Mock(),
            transitions=[],
            feature_mode="f2",
        )

        old_state = self._game_state()
        new_state = self._game_state()

        old_state["coins"] = []
        new_state["coins"] = []

        # Crate to the RIGHT -> crate path recommends RIGHT.
        old_state["field"][5, 3] = 1
        new_state["field"][5, 3] = 1

        # Agent is inside bomb danger.
        old_state["self"] = ("player", 0, False, (3, 3))
        old_state["bombs"] = [((3, 5), 3)]

        # RIGHT also escapes the blast line.
        new_state["self"] = ("player", 0, False, (4, 3))
        new_state["bombs"] = [((3, 5), 2)]
        new_state["step"] = 2

        events = []

        with patch.object(train, "reward_from_events", return_value=0.0):
            train.game_events_occurred(agent, old_state, "RIGHT", new_state, events)

        self.assertIn(train.ESCAPED_BOMB_DANGER, events)
        self.assertNotIn(train.MOVED_TOWARDS_CRATE, events)
        self.assertNotIn(train.MOVED_AWAY_FROM_CRATE, events)


    def test_f2_does_not_shape_crate_navigation_when_coin_is_visible(self):
        """Reward Test Q: Visible coin navigation takes priority over crate search."""
        agent = SimpleNamespace(
            model=Mock(),
            logger=Mock(),
            transitions=[],
            feature_mode="f2",
        )

        old_state = self._game_state()
        new_state = self._game_state()

        # Crate path would recommend RIGHT.
        old_state["field"][5, 3] = 1
        new_state["field"][5, 3] = 1

        old_state["self"] = ("player", 0, True, (2, 3))
        new_state["self"] = ("player", 0, True, (3, 3))

        # But a visible coin already exists.
        old_state["coins"] = [(2, 1)]
        new_state["coins"] = [(2, 1)]
        new_state["step"] = 2

        events = []

        with patch.object(train, "reward_from_events", return_value=0.0):
            train.game_events_occurred(agent, old_state, "RIGHT", new_state, events)

        self.assertNotIn(train.MOVED_TOWARDS_CRATE, events)
        self.assertNotIn(train.MOVED_AWAY_FROM_CRATE, events)


    def test_shortest_path_direction_right(self):
        """Path Test A: Test that a target directly to the right returns RIGHT as the valid first step."""
        state = self._game_state()
        state["coins"] = [(4, 3)]

        features = state_to_features(state, "f1")
        
        expected = [0, 0, 0, 1]

        np.testing.assert_array_equal(features[7:11], expected)


    def test_shortest_path_avoids_obstacle(self):
        """Path Test B: Test that the shortest path direction accounts for obstacles in the field."""
        state = self._game_state()
        state["coins"] = [(5, 3)]

        state["field"][3, 2] = 1
        state["field"][4, 3] = 1
        state["field"][4, 4] = 1

        features = state_to_features(state, "f1")

        expected = [0, 1, 0, 0]

        np.testing.assert_array_equal(features[7:11], expected)


    def test_shortest_path_multiple_first_steps(self):
        """Path Test C: Test that all valid first steps are returned when multiple shortest paths exist."""
        state = self._game_state()
        state["coins"] = [(4, 4)]

        features = state_to_features(state, "f1")

        expected = [0, 1, 0, 1]

        np.testing.assert_array_equal(features[7:11], expected)


    def test_shortest_path_unreachable_target(self):
        """Path Test D: Test that an unreachable target returns no valid path directions."""
        state = self._game_state()
        state["coins"] = [(3, 3)]

        state["field"] = np.array([
            [1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1]
        ])

        state["self"] = ("player", 0, 1, (1, 1))

        features = state_to_features(state, "f1")

        expected = [0, 0, 0, 0]

        np.testing.assert_array_equal(features[7:11], expected)


    def test_shortest_path_start_equals_target(self):
        """Path Test E: Test that no direction is returned when start and target are identical."""
        state = self._game_state()
        state["coins"] = [(3, 3)]

        features = state_to_features(state, "f1")

        expected = [0, 0, 0, 0]

        np.testing.assert_array_equal(features[7:11], expected)


    def test_shortest_path_uses_nearest_coin(self):
        """Path Test F: Test that path directions are computed for the nearest coin."""
        state = self._game_state()
        state["coins"] = [(2, 3), (5, 3)]

        features = state_to_features(state, "f1")

        expected = [0, 0, 1, 0]

        np.testing.assert_array_equal(features[7:11], expected)


    def test_model_start_mode_defaults_to_resume(self):
        """Model Start Test A: Test that the default model start mode is 'resume'."""
        with tempfile.TemporaryDirectory() as directory, temporary_working_directory(directory):
            model = Linear_QModel(input_size=callbacks.FEATURE_SIZES["f1"], output_size=len(callbacks.actions_for_feature_mode("f1")), seed=1)
            model.weights[:] = 42.0

            with open("my-saved-model.pt", "wb") as file:
                pickle.dump({"model": model, "epsilon": 0.25}, file)

            agent = SimpleNamespace(train=True, logger=Mock())

            with patch.dict(os.environ, {}, clear=True):
                callbacks.setup(agent)

            self.assertEqual(agent.model_start_mode, "resume")


    def test_model_start_mode_from_environment(self):
        """Model Start Test B: Test that the model start mode is read from the environment variable."""
        with tempfile.TemporaryDirectory() as directory, temporary_working_directory(directory):
            agent = SimpleNamespace(train=True, logger=Mock())

            with patch.dict(os.environ, {"MODEL_START_MODE": "fresh"}, clear=True):
                callbacks.setup(agent)

            self.assertEqual(agent.model_start_mode, "fresh")


    def test_invalid_model_start_mode_raises(self):
        """Model Start Test C: Test that unsupported model start modes raise a ValueError."""
        with tempfile.TemporaryDirectory() as directory, temporary_working_directory(directory):
            agent = SimpleNamespace(train=True, logger=Mock())

            with patch.dict(os.environ, {"MODEL_START_MODE": "grape"}, clear=True):
                with self.assertRaises(ValueError):
                    callbacks.setup(agent)


    def test_fresh_training_ignores_existing_checkpoint(self):
        """Model Start Test D: Test that fresh training ignores an existing checkpoint and initializes a new model."""
        with tempfile.TemporaryDirectory() as directory, temporary_working_directory(directory):
            model = Linear_QModel(input_size=callbacks.FEATURE_SIZES["f1"], output_size=len(callbacks.actions_for_feature_mode("f1")), seed=1)
            model.weights[:] = 42.0

            with open("my-saved-model.pt", "wb") as file:
                pickle.dump({"model": model, "epsilon": 0.25}, file)

            agent = SimpleNamespace(train=True, logger=Mock())

            with patch.dict(os.environ, {"MODEL_START_MODE": "fresh"}, clear=True):
                callbacks.setup(agent)

            self.assertFalse(np.allclose(agent.model.weights, model.weights))
            self.assertEqual(agent.epsilon, callbacks.EPSILON_START)


    def test_resume_training_loads_existing_checkpoint(self):
        """Model Start Test E: Test that resume training loads an existing checkpoint when available."""
        with tempfile.TemporaryDirectory() as directory, temporary_working_directory(directory):
            model = Linear_QModel(input_size=callbacks.FEATURE_SIZES["f1"], output_size=len(callbacks.actions_for_feature_mode("f1")), seed=1)
            model.weights[:] = 42.0

            with open("my-saved-model.pt", "wb") as file:
                pickle.dump({"model": model, "epsilon": 0.25}, file)

            agent = SimpleNamespace(train=True, logger=Mock())

            with patch.dict(os.environ, {"MODEL_START_MODE": "resume"}, clear=True):
                callbacks.setup(agent)

            np.testing.assert_allclose(agent.model.weights, model.weights)
            self.assertEqual(agent.epsilon, 0.25)


    def test_evaluation_loads_checkpoint_independent_of_start_mode(self):
        """Model Start Test F: Test that evaluation loads the checkpoint regardless of the training start mode."""
        with tempfile.TemporaryDirectory() as directory, temporary_working_directory(directory):
            model = Linear_QModel(input_size=callbacks.FEATURE_SIZES["f1"], output_size=len(callbacks.actions_for_feature_mode("f1")), seed=1)
            model.weights[:] = 42.0

            with open("my-saved-model.pt", "wb") as file:
                pickle.dump({"model": model, "epsilon": 0.25}, file)

            agent = SimpleNamespace(train=False, logger=Mock())

            with patch.dict(os.environ, {"MODEL_START_MODE": "fresh"}, clear=True):
                callbacks.setup(agent)

            np.testing.assert_allclose(agent.model.weights, model.weights)
            self.assertEqual(agent.epsilon, 0.25)


    def test_resume_training_without_checkpoint_raises_file_not_found_error(self):
        """Model Start Test G: Test that resume training raises FileNotFoundError when no checkpoint is available."""
        with tempfile.TemporaryDirectory() as directory, temporary_working_directory(directory):
            agent = SimpleNamespace(train=True, logger=Mock())

            with patch.dict(os.environ, {"MODEL_START_MODE": "resume"}, clear=True):
                with self.assertRaises(FileNotFoundError):
                    callbacks.setup(agent)


    def test_feature_mode_defaults_to_f1(self):
        """Feature Mode Test A: Test that the default feature mode is F1 with 11 features."""
        with tempfile.TemporaryDirectory() as directory, temporary_working_directory(directory):
            model = Linear_QModel(input_size=callbacks.FEATURE_SIZES["f1"], output_size=len(callbacks.actions_for_feature_mode("f1")), seed=1)

            with open("my-saved-model.pt", "wb") as file:
                pickle.dump({"model": model, "epsilon": 0.25}, file)

            agent = SimpleNamespace(train=True, logger=Mock())

            with patch.dict(os.environ, {}, clear=True):
                callbacks.setup(agent)

            self.assertEqual(agent.feature_mode, "f1")
            self.assertEqual(agent.feature_size, 11)


    def test_feature_mode_f0_creates_seven_feature_model(self):
        """Feature Mode Test B: Test that F0 uses 7 features for a fresh model."""
        with tempfile.TemporaryDirectory() as directory, temporary_working_directory(directory):
            agent = SimpleNamespace(train=True, logger=Mock())

            with patch.dict(os.environ, {"MODEL_START_MODE": "fresh", "FEATURE_MODE": "f0"}, clear=True):
                callbacks.setup(agent)

            self.assertEqual(agent.feature_mode, "f0")
            self.assertEqual(agent.feature_size, 7)
            self.assertEqual(agent.model.input_size, 7)


    def test_invalid_feature_mode_raises(self):
        """Feature Mode Test C: Test that unsupported feature modes raise a ValueError."""
        with tempfile.TemporaryDirectory() as directory, temporary_working_directory(directory):
            agent = SimpleNamespace(train=True, logger=Mock())

            with patch.dict(os.environ, {"MODEL_START_MODE": "fresh", "FEATURE_MODE": "grape"}, clear=True):
                with self.assertRaises(ValueError):
                    callbacks.setup(agent)


    def test_f0_feature_vector_has_seven_features(self):
        """Feature Mode Test D: Test that F0 returns a 7-dimensional feature vector."""
        state = self._game_state()
        features = state_to_features(state, "f0")
        self.assertEqual(features.shape, (7,))


    def test_f1_feature_vector_has_eleven_features(self):
        """Feature Mode Test E: Test that F1 returns an 11-dimensional feature vector."""
        state = self._game_state()
        features = state_to_features(state, "f1")
        self.assertEqual(features.shape, (11,))


    def test_f0_matches_first_seven_f1_features(self):
        """Feature Mode Test F: Test that F0 matches the first seven features of F1."""
        state = self._game_state()

        f0 = state_to_features(state, "f0")
        f1 = state_to_features(state, "f1")

        np.testing.assert_allclose(f0, f1[:7])


    def test_checkpoint_feature_size_mismatch_raises(self):
        """Feature Mode Test G: Test that a checkpoint with a mismatched feature size raises a ValueError."""
        with tempfile.TemporaryDirectory() as directory, temporary_working_directory(directory):
            # F1 checkpoint: 11 inputs
            model = Linear_QModel(input_size=callbacks.FEATURE_SIZES["f1"], output_size=len(callbacks.actions_for_feature_mode("f1")), seed=1)

            with open("my-saved-model.pt", "wb") as file:
                pickle.dump({"model": model, "epsilon": 0.25}, file)

            agent = SimpleNamespace(train=True, logger=Mock())

            with patch.dict(os.environ, {"MODEL_START_MODE": "resume", "FEATURE_MODE": "f0"}, clear=True):
                with self.assertRaises(ValueError):
                    callbacks.setup(agent)


    def test_same_experiment_seed_reproduces_model_and_rng(self):
        """Experiment Seed Test A: Test that the same experiment seed reproduces model initialization and RNG sequence."""
        with tempfile.TemporaryDirectory() as directory, temporary_working_directory(directory):
            agent_a = SimpleNamespace(train=True, logger=Mock())
            agent_b = SimpleNamespace(train=True, logger=Mock())

            with patch.dict(os.environ, {"MODEL_START_MODE": "fresh", "EXPERIMENT_SEED": "123"}, clear=True):
                callbacks.setup(agent_a)
                callbacks.setup(agent_b)

            self.assertEqual(agent_a.experiment_seed, 123)
            self.assertEqual(agent_b.experiment_seed, 123)

            np.testing.assert_allclose(agent_a.model.weights, agent_b.model.weights)
            self.assertEqual(agent_a.rng.random(), agent_b.rng.random())


    def test_different_experiment_seeds_produce_different_model_initializations(self):
        """Experiment Seed Test B: Test that different experiment seeds produce different model initializations."""
        with tempfile.TemporaryDirectory() as directory, temporary_working_directory(directory):
            agent_a = SimpleNamespace(train=True, logger=Mock())
            agent_b = SimpleNamespace(train=True, logger=Mock())

            with patch.dict(os.environ, {"MODEL_START_MODE": "fresh", "EXPERIMENT_SEED": "123"}, clear=True):
                callbacks.setup(agent_a)

            with patch.dict(os.environ, {"MODEL_START_MODE": "fresh", "EXPERIMENT_SEED": "456"}, clear=True):
                callbacks.setup(agent_b)

            self.assertFalse(np.allclose(agent_a.model.weights, agent_b.model.weights))


    def test_task2_action_indices_match_action_space(self):
        """Action Test A: Verify that Task 2 action indices match the model output order."""
        actions = callbacks.actions_for_feature_mode("f2")

        for index, action in enumerate(actions):
            self.assertEqual(train.ACTION_TO_INDEX[action], index)