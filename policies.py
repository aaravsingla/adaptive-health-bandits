"""Non-learning baselines. They share the agent interface (select/update)."""
from __future__ import annotations

import numpy as np

from config import EVENING, N_ACTIONS


class FixedPolicy:
    def __init__(self, slot: int = EVENING):
        if not 0 <= slot < N_ACTIONS:
            raise ValueError("invalid slot")
        self.slot = slot

    def select(self, context) -> int:
        return self.slot

    def update(self, action, context, reward) -> None:
        pass


class RandomPolicy:
    def __init__(self, rng: np.random.Generator, n_actions: int = N_ACTIONS):
        self.rng, self.n_actions = rng, n_actions

    def select(self, context) -> int:
        return int(self.rng.integers(self.n_actions))

    def update(self, action, context, reward) -> None:
        pass
