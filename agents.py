"""Disjoint LinUCB (Li et al., 2010) and its reduced-context ablation."""
from __future__ import annotations

import numpy as np

from config import N_ACTIONS
from context import N_FEATURES


class LinUCB:
    """One ridge-regression model per arm, with an upper-confidence exploration bonus.

    score_a = x^T theta_a + alpha * sqrt(x^T A_a^{-1} x),  theta_a = A_a^{-1} b_a
    After reward r for chosen arm a:  A_a += x x^T,  b_a += r x  (other arms untouched).

    `feature_idx` selects which context columns the agent uses; the reduced-context
    ablation uses only the bias column, so each arm learns a single mean reward.
    """

    def __init__(self, alpha: float = 0.5, lam: float = 1.0, feature_idx=None,
                 n_actions: int = N_ACTIONS, rng: np.random.Generator | None = None):
        if lam <= 0:
            raise ValueError("lam must be positive")
        if alpha < 0:
            raise ValueError("alpha must be nonnegative")
        self.feature_idx = np.arange(N_FEATURES) if feature_idx is None else np.asarray(feature_idx)
        d = len(self.feature_idx)
        self.alpha, self.lam, self.n_actions, self.d = alpha, lam, n_actions, d
        self.A = np.stack([lam * np.eye(d) for _ in range(n_actions)])
        self.b = np.zeros((n_actions, d))
        self.rng = rng if rng is not None else np.random.default_rng(0)

    def _x(self, context: np.ndarray) -> np.ndarray:
        return np.asarray(context, dtype=float)[self.feature_idx]

    def scores(self, context: np.ndarray):
        """Return (mu, bonus) arrays over arms. Uses Cholesky solves, not explicit inverses."""
        x = self._x(context)
        mu, bonus = np.empty(self.n_actions), np.empty(self.n_actions)
        for a in range(self.n_actions):
            L = np.linalg.cholesky(self.A[a])
            theta = np.linalg.solve(L.T, np.linalg.solve(L, self.b[a]))
            v = np.linalg.solve(L, x)                # x^T A^{-1} x = ||L^{-1} x||^2
            mu[a] = x @ theta
            bonus[a] = self.alpha * np.sqrt(max(v @ v, 0.0))
        return mu, bonus

    def select(self, context: np.ndarray) -> int:
        mu, bonus = self.scores(context)
        s = mu + bonus
        best = np.flatnonzero(np.isclose(s, s.max()))
        return int(self.rng.choice(best))

    def update(self, action: int, context: np.ndarray, reward: float) -> None:
        x = self._x(context)
        self.A[action] += np.outer(x, x)
        self.b[action] += reward * x

    def theta(self) -> np.ndarray:
        return np.stack([np.linalg.solve(self.A[a], self.b[a]) for a in range(self.n_actions)])

    def save(self, path: str) -> None:
        np.savez(path, A=self.A, b=self.b, alpha=self.alpha, lam=self.lam, feature_idx=self.feature_idx)


def make_full_linucb(alpha, lam, rng=None) -> LinUCB:
    return LinUCB(alpha, lam, feature_idx=None, rng=rng)


def make_reduced_linucb(alpha, lam, rng=None) -> LinUCB:
    """Ablation: intercept only. Per-arm disjoint models make this 'intercept + chosen action'."""
    return LinUCB(alpha, lam, feature_idx=[0], rng=rng)
