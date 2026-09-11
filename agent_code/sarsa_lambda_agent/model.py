from __future__ import annotations

import numpy as np


class Linear_SARSAModel:
    """Represent Q(s, a) as one linear weight vector per action (one Q-value per action)"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        learning_rate: float = 0.01,
        gamma: float = 0.9,
        lambda_: float = 0.8,
        seed: int | None = None,
    ) -> None:
        if input_size <= 0 or output_size <= 0:
            raise ValueError("input_size and output_size must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be between 0 and 1")
        if not 0 <= lambda_ <= 1:
            raise ValueError("lambda_ must be between 0 and 1")

        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lambda_ = lambda_

        rng = np.random.default_rng(seed)
        self.weights = rng.normal(
            loc=0.0,
            scale=0.01,
            size=(input_size, output_size),
        )

        self.eligibility_traces = np.zeros_like(self.weights)

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
        next_action: int | None = None,
    ) -> float:
        # Apply one SARSA(lambda) update and return the TD error.
        state = self.validate_features(state)
        if not 0 <= action < self.output_size:
            raise ValueError(f"action must be in [0, {self.output_size})")

        current_q = float(self(state)[action])

        if next_state is None:
            target_q = float(reward)
        else:
            next_state = self.validate_features(next_state)

            if next_action is None:
                raise ValueError("next_action is required for a non-terminal SARSA update")

            if not 0 <= next_action < self.output_size:
                raise ValueError(f"next_action must be in [0, {self.output_size})")

            next_q = float(self(next_state)[next_action])
            target_q = float(reward) + self.gamma * next_q

        td_error = target_q - current_q

        self.eligibility_traces *= self.gamma * self.lambda_
        self.eligibility_traces[:, action] += state

        self.weights += self.learning_rate * td_error * self.eligibility_traces
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

    def reset_traces(self) -> None:
        self.eligibility_traces.fill(0.0)