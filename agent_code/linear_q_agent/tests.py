import os
import numpy as np
import unittest
import tempfile
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import events as game_events

from . import callbacks, train
from .callbacks import state_to_features
from .model import Linear_QNet


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


    def test_features(self):
        field = np.ones((7, 7), dtype=int)  # walls/crates
        field[1:-1, 1:-1] = 0               # free interior
        field[2, 3] = 1                     # wall to the LEFT of agent

        game_state = {
            "field": field,
            "bombs": [],
            "coins": [(1, 1), (5, 4)],
            "self": ("player", 0, 1, (3, 3)),
        }

        features = state_to_features(game_state)
        print(features)

        expected = np.array([
            1,      # bias
            1,      # UP is free
            1,      # DOWN is free
            0,      # LEFT is blocked
            1,      # RIGHT is free
            2 / 6,  # nearest coin dx: (5 - 3) / (7 - 1)
            1 / 6,  # nearest coin dy: (4 - 3) / (7 - 1)
        ])

        np.testing.assert_allclose(features, expected)
        assert features.shape == (7,)
        assert np.isfinite(features).all()

        print("state_to_features works correctly")

        assert state_to_features(None) is None

        state_without_coins = dict(game_state)
        state_without_coins["coins"] = []
        assert np.array_equal(
            state_to_features(state_without_coins),
            np.array([1, 1, 1, 0, 1, 0, 0])
        )


    def test_features_all_directions_free(self):
        """Test A: All directions are free."""
        state = self._game_state()
        features = state_to_features(state)

        expected = [1, 1, 1, 1, 1]

        np.testing.assert_array_equal(features[:5], expected)


    def test_features_up_blocked(self):
        """Test B: Up tile is blocked."""
        state = self._game_state()

        # UP of agent (3, 3) is (3, 2)
        state["field"][3, 2] = 1
        features = state_to_features(state)

        expected = [1, 0, 1, 1, 1]
        np.testing.assert_array_equal(features[:5], expected)


    def test_features_coin_to_the_right(self):
        """Test C: When there is a coin on the right"""
        state = self._game_state()

        # There is a coin at (5, 3), and the agent is at (3, 3)
        state["coins"] = [(5, 3)]
        features = state_to_features(state)

        expected = [2/6, 0]
        np.testing.assert_allclose(features[5:], expected)


    def test_features_nearest_coin_select(self):
        """Test D: Check if the agent chooses the nearest coin among all coins."""
        state = self._game_state()

        # There are two coins: one at (5, 4), and the other at (2, 2)
        state["coins"] = [(5, 4), (2, 2)]
        features = state_to_features(state)

        expected = [-1/6, -1/6]
        np.testing.assert_allclose(features[5:], expected)


    def test_features_no_coin(self):
        """Test E: No coin in the field."""
        state = self._game_state()

        # There is no coin in the field
        state["coins"] = []
        features = state_to_features(state)

        expected = [0, 0]
        np.testing.assert_allclose(features[5:], expected)
        

    def test_features_agent_in_corner(self):
        """Test F: Agent is at a walkable corner next to border walls."""
        state = self._game_state()

        # The agent is in the corner
        state["self"] = ("player", 0, 1, (5, 5))
        features = state_to_features(state)

        expected = [1, 1, 0, 1, 0]
        np.testing.assert_array_equal(features[:5], expected)
        

    def test_predict_returns_one_value_per_action(self):
        model = Linear_QNet(input_size=7, output_size=6, seed=1)
        features = np.ones(7)

        q_values = model.predict(features)

        self.assertEqual(q_values.shape, (6,))
        self.assertTrue(np.isfinite(q_values).all())


    def test_terminal_update_changes_only_selected_action(self):
        model = Linear_QNet(
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
        model = Linear_QNet(
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
        )

        with patch.object(callbacks.random, "random", return_value=0.1), patch.object(
            callbacks.random, "choice", return_value="WAIT"
        ) as choice:
            action = callbacks.act(agent, self._game_state())

        self.assertEqual(action, "WAIT")
        choice.assert_called_once_with(callbacks.ACTIONS)
        agent.model.predict.assert_not_called()


    def test_act_exploits_highest_q_value_when_not_exploring(self):
        agent = SimpleNamespace(
            train=True,
            epsilon=0.5,
            model=Mock(),
            logger=Mock(),
        )
        agent.model.predict.return_value = np.array([1.0, 4.0, 2.0, 0.0, 3.0, -1.0])

        with patch.object(callbacks.random, "random", return_value=0.9):
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
        )
        old_state = self._game_state()
        new_state = self._game_state()

        with patch.object(train, "reward_from_events", return_value=2.0):
            train.game_events_occurred(agent, old_state, "LEFT", new_state, [])

        state, action, next_state, reward = agent.transitions[-1]
        self.assertEqual(action, "LEFT")
        self.assertEqual(reward, 2.0)
        np.testing.assert_allclose(state, state_to_features(old_state))
        np.testing.assert_allclose(next_state, state_to_features(new_state))
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


    def test_training_checkpoint_reloads_for_evaluation(self):
        with tempfile.TemporaryDirectory() as directory, temporary_working_directory(
            directory
        ):
            training_agent = SimpleNamespace(train=True, logger=Mock())
            callbacks.setup(training_agent)
            train.setup_training(training_agent)
            training_agent.epsilon = 0.0

            state = self._game_state()
            action = callbacks.act(training_agent, state)
            weights_before_update = training_agent.model.weights.copy()

            train.game_events_occurred(
                training_agent,
                state,
                action,
                state,
                [game_events.COIN_COLLECTED],
            )
            train.end_of_round(
                training_agent,
                state,
                action,
                [],
            )

            self.assertFalse(
                np.array_equal(weights_before_update, training_agent.model.weights)
            )
            saved_weights = training_agent.model.weights.copy()
            saved_epsilon = training_agent.epsilon

            evaluation_agent = SimpleNamespace(train=False, logger=Mock())
            callbacks.setup(evaluation_agent)

            np.testing.assert_allclose(
                evaluation_agent.model.weights, saved_weights
            )
            self.assertEqual(evaluation_agent.epsilon, saved_epsilon)
            self.assertEqual(
                callbacks.act(evaluation_agent, state),
                callbacks.ACTIONS[
                    int(np.argmax(evaluation_agent.model.predict(
                        state_to_features(state)
                    )))
                ],
            )

    def test_reward_mode_default_to_basic(self):
        """Reward Test A: Test that the default reward mode is "basic" when no environment is set."""
        agent = SimpleNamespace(logger=Mock())

        with patch.dict(os.environ, {}, clear=True):
            train.setup_training(agent)

        self.assertEqual(agent.reward_mode, "basic")


    def test_reward_from_environment(self):
        """Reward Test B: Test that the reward mode is read from the environment variable."""
        agent = SimpleNamespace(logger=Mock())

        with patch.dict(os.environ, {"BOMBERMAN_REWARD_MODE": "sparse"}, clear=True):
            train.setup_training(agent)

        self.assertEqual(agent.reward_mode, "sparse")


    def test_invalid_reward_mode_raises(self):
        """Reward Test C: Test that an invalid reward mode raises a ValueError."""
        agent = SimpleNamespace(logger=Mock())

        with patch.dict(os.environ, {"BOMBERMAN_REWARD_MODE": "grape"}, clear=True):
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