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
        field = np.ones((7, 7), dtype=int)
        field[1:-1, 1:-1] = 0
        return {
            "field": field,
            "bombs": [],
            "coins": [(5, 4)],
            "self": ("player", 0, 1, (3, 3)),
            "step": 1,
        }


    def test_features_all_directions_free(self):
        """Test A: All directions are free."""
        state = self._game_state()
        features = state_to_features(state, "f1")

        expected = [1, 1, 1, 1, 1]

        np.testing.assert_array_equal(features[:5], expected)


    def test_features_up_blocked(self):
        """Test B: Up tile is blocked."""
        state = self._game_state()

        # UP of agent (3, 3) is (3, 2)
        state["field"][3, 2] = 1
        features = state_to_features(state, "f1")

        expected = [1, 0, 1, 1, 1]
        np.testing.assert_array_equal(features[:5], expected)


    def test_features_coin_to_the_right(self):
        """Test C: When there is a coin on the right"""
        state = self._game_state()

        # There is a coin at (5, 3), and the agent is at (3, 3)
        state["coins"] = [(5, 3)]
        features = state_to_features(state, "f1")

        expected = [2/6, 0]
        np.testing.assert_allclose(features[5:7], expected)


    def test_features_nearest_coin_select(self):
        """Test D: Check if the agent chooses the nearest coin among all coins."""
        state = self._game_state()

        # There are two coins: one at (5, 4), and the other at (2, 2)
        state["coins"] = [(5, 4), (2, 2)]
        features = state_to_features(state, "f1")

        expected = [-1/6, -1/6]
        np.testing.assert_allclose(features[5:7], expected)


    def test_features_no_coin(self):
        """Test E: No coin in the field."""
        state = self._game_state()

        # There is no coin in the field
        state["coins"] = []
        features = state_to_features(state, "f1")

        expected = [0, 0]
        np.testing.assert_allclose(features[5:7], expected)
        

    def test_features_agent_in_corner(self):
        """Test F: Agent is at a walkable corner next to border walls."""
        state = self._game_state()

        # The agent is in the corner
        state["self"] = ("player", 0, 1, (5, 5))
        features = state_to_features(state, "f1")

        expected = [1, 1, 0, 1, 0]
        np.testing.assert_array_equal(features[:5], expected)
        

    def test_predict_returns_one_value_per_action(self):
        model = Linear_QModel(input_size=7, output_size=len(callbacks.ACTIONS), seed=1)
        features = np.ones(7)

        q_values = model.predict(features)

        self.assertEqual(q_values.shape, (len(callbacks.ACTIONS),))
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
            rng=Mock()
        )

        agent.rng.random.return_value = 0.1
        agent.rng.choice.return_value = "WAIT"
        action = callbacks.act(agent, self._game_state())

        self.assertEqual(action, "WAIT")
        agent.rng.choice.assert_called_once_with(callbacks.ACTIONS)
        agent.model.predict.assert_not_called()


    def test_act_exploits_highest_q_value_when_not_exploring(self):
        agent = SimpleNamespace(
            train=True,
            epsilon=0.5,
            model=Mock(),
            logger=Mock(),
            feature_mode="f1",
            rng=Mock()
        )
        agent.model.predict.return_value = np.array([1.0, 4.0, 2.0, 0.0, 3.0])

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
            model = Linear_QModel(input_size=callbacks.FEATURE_SIZES["f1"], output_size=len(callbacks.ACTIONS), seed=1)
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
            model = Linear_QModel(input_size=callbacks.FEATURE_SIZES["f1"], output_size=len(callbacks.ACTIONS), seed=1)
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
            model = Linear_QModel(input_size=callbacks.FEATURE_SIZES["f1"], output_size=len(callbacks.ACTIONS), seed=1)
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
            model = Linear_QModel(input_size=callbacks.FEATURE_SIZES["f1"], output_size=len(callbacks.ACTIONS), seed=1)
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
            model = Linear_QModel(input_size=callbacks.FEATURE_SIZES["f1"], output_size=len(callbacks.ACTIONS), seed=1)

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
            model = Linear_QModel(input_size=callbacks.FEATURE_SIZES["f1"], output_size=len(callbacks.ACTIONS), seed=1)

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