from __future__ import annotations

import numpy as np


class Linear_QModel:
    """Represent Q(s, a) as one linear weight vector per action (one Q-value per action)"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        learning_rate: float = 0.01,
        gamma: float = 0.9,
        seed: int | None = None,
    ) -> None:
        if input_size <= 0 or output_size <= 0:
            raise ValueError("input_size and output_size must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be between 0 and 1")

        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.gamma = gamma

        rng = np.random.default_rng(seed)
        self.weights = rng.normal(
            loc=0.0,
            scale=0.01,
            size=(input_size, output_size),
        )

    def __call__(self, features: np.ndarray) -> np.ndarray:
        # Return one Q-value for each action.
        features = self.validate_features(features)
        return features @ self.weights

    def predict(self, features: np.ndarray) -> np.ndarray:
        # Explicit alias for evaluating Q(s, a) for every action.
        return self(features)

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray | None,
    ) -> float:
        # Apply one Q-learning update and return the TD error.
        state = self.validate_features(state)
        if not 0 <= action < self.output_size:
            raise ValueError(f"action must be in [0, {self.output_size})")

        current_q = float(self(state)[action])
        if next_state is None:
            target_q = float(reward)
        else:
            next_state = self.validate_features(next_state)
            target_q = float(reward) + self.gamma * float(np.max(self(next_state)))

        td_error = target_q - current_q
        self.weights[:, action] += self.learning_rate * td_error * state
        return td_error

    def validate_features(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        if features.shape != (self.input_size,):
            raise ValueError(
                f"expected features with shape ({self.input_size},), "
                f"got {features.shape}"
            )
        if not np.isfinite(features).all():
            raise ValueError("features must contain only finite values")
        return features