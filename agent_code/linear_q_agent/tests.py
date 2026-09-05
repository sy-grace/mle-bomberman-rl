import numpy as np
import unittest
from .callbacks import state_to_features
from .model import Linear_QNet


class LinearQAgentTest(unittest.TestCase):
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