"""Synthetic health-app user simulator.

Everything in this module except `Observation` is hidden from the agent.

Engagement model for user u, day t, slot a:
    eff_pref   = (preferred_slot + weekend_shift * is_weekend) mod 3
    match(a)   = 1 - |a - eff_pref| / 2                       in {0, 0.5, 1}
    z          = baseline
               + pref_strength * (match(a) - 0.5)
               + activity_sens * ACTIVITY_WEIGHT[a] * (activity - 0.5)
               + receptivity
               - fatigue_sens * fatigue
    p_engage   = clip(sigmoid(z), 0.01, 0.99)

Fatigue event (only possible when the notification is ignored):
    q(a)       = clip(sigmoid(-2.5 + fatigue_sens * fatigue + (1 - match(a))), 0.01, 0.99)
    P(fatigue) = (1 - p_engage) * q(a)

Reward = engage_reward * engaged - fatigue_penalty * fatigue_event
Expected reward (used only for simulator-oracle regret):
    E[r | a]   = engage_reward * p_engage - fatigue_penalty * (1 - p_engage) * q(a)

State updates after each decision:
    fatigue     <- clip(0.6 * fatigue + ignored + 0.5 * fatigue_event, 0, 5)
    receptivity <- clip(r + 0.1 * (r0 + trend * t - r) + 0.05 * engaged
                       - 0.3 * fatigue_event + N(0, noise_sd^2), -2, 2)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import N_ACTIONS

# On active days, midday nudges land better; morning/evening slightly worse.
ACTIVITY_WEIGHT = np.array([-0.5, 1.0, -0.5])


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


@dataclass(frozen=True)
class UserParams:
    baseline: float
    preferred_slot: int
    pref_strength: float
    weekend_shift: int
    activity_sens: float
    fatigue_sens: float
    receptivity0: float
    receptivity_trend: float
    receptivity_noise: float
    activity_mean: float


@dataclass(frozen=True)
class Population:
    """Hidden user parameters plus exogenous observable paths (activity, weekday).

    Generated once per seed and shared by every policy so that all policies face
    the same users and the same activity/weekday sequences.
    """
    users: tuple
    activity: np.ndarray    # shape [n_days, n_users], in [0, 1]
    is_weekend: np.ndarray  # shape [n_days]

    @property
    def n_users(self):
        return len(self.users)

    @property
    def n_days(self):
        return len(self.is_weekend)


@dataclass(frozen=True)
class Observation:
    """The only per-decision information the environment reveals to the agent."""
    activity: float
    is_weekend: float


@dataclass(frozen=True)
class StepResult:
    engaged: int
    fatigue_event: int
    reward: float
    expected_rewards: np.ndarray  # oracle quantity: for regret only, never shown to agents


def generate_population(n_users: int, n_days: int, rng: np.random.Generator) -> Population:
    users = []
    for _ in range(n_users):
        users.append(UserParams(
            baseline=float(np.clip(rng.normal(-0.3, 0.5), -2.0, 1.0)),
            preferred_slot=int(rng.integers(N_ACTIONS)),
            pref_strength=float(rng.uniform(1.0, 2.5)),
            weekend_shift=int(rng.random() < 0.4),
            activity_sens=float(rng.uniform(0.0, 1.5)),
            fatigue_sens=float(rng.uniform(0.1, 0.6)),
            receptivity0=float(rng.uniform(-0.5, 0.5)),
            receptivity_trend=float(rng.uniform(-0.01, 0.01)),
            receptivity_noise=float(rng.uniform(0.02, 0.10)),
            activity_mean=float(rng.uniform(0.2, 0.8)),
        ))
    means = np.array([u.activity_mean for u in users])
    activity = np.clip(means[None, :] + rng.normal(0, 0.15, size=(n_days, n_users)), 0.0, 1.0)
    is_weekend = (np.arange(n_days) % 7 >= 5).astype(float)
    return Population(tuple(users), activity, is_weekend)


class NotificationEnv:
    def __init__(self, population: Population, rng: np.random.Generator,
                 engage_reward: float = 1.0, fatigue_penalty: float = 0.25):
        self._pop = population
        self._rng = rng
        self.engage_reward = engage_reward
        self.fatigue_penalty = fatigue_penalty
        self._receptivity = np.array([u.receptivity0 for u in population.users])
        self._fatigue = np.zeros(population.n_users)

    @property
    def n_users(self):
        return self._pop.n_users

    @property
    def n_days(self):
        return self._pop.n_days

    def observe(self, user: int, day: int) -> Observation:
        return Observation(activity=float(self._pop.activity[day, user]),
                           is_weekend=float(self._pop.is_weekend[day]))

    def _match(self, user: int, day: int) -> np.ndarray:
        u = self._pop.users[user]
        eff = (u.preferred_slot + u.weekend_shift * int(self._pop.is_weekend[day])) % N_ACTIONS
        return 1.0 - np.abs(np.arange(N_ACTIONS) - eff) / 2.0

    def probabilities(self, user: int, day: int):
        """Return (p_engage, q_fatigue_given_ignored) for all actions."""
        u = self._pop.users[user]
        match = self._match(user, day)
        act = self._pop.activity[day, user]
        z = (u.baseline + u.pref_strength * (match - 0.5)
             + u.activity_sens * ACTIVITY_WEIGHT * (act - 0.5)
             + self._receptivity[user] - u.fatigue_sens * self._fatigue[user])
        p = np.clip(sigmoid(z), 0.01, 0.99)
        q = np.clip(sigmoid(-2.5 + u.fatigue_sens * self._fatigue[user] + (1.0 - match)), 0.01, 0.99)
        return p, q

    def expected_rewards(self, user: int, day: int) -> np.ndarray:
        p, q = self.probabilities(user, day)
        return self.engage_reward * p - self.fatigue_penalty * (1.0 - p) * q

    def step(self, user: int, day: int, action: int) -> StepResult:
        if not 0 <= action < N_ACTIONS:
            raise ValueError(f"invalid action {action}")
        p, q = self.probabilities(user, day)
        exp_r = self.engage_reward * p - self.fatigue_penalty * (1.0 - p) * q
        engaged = int(self._rng.random() < p[action])
        fatigue_event = int((not engaged) and self._rng.random() < q[action])
        reward = self.engage_reward * engaged - self.fatigue_penalty * fatigue_event

        u = self._pop.users[user]
        self._fatigue[user] = np.clip(0.6 * self._fatigue[user] + (1 - engaged) + 0.5 * fatigue_event, 0.0, 5.0)
        r = self._receptivity[user]
        target = u.receptivity0 + u.receptivity_trend * day
        r = r + 0.1 * (target - r) + 0.05 * engaged - 0.3 * fatigue_event + self._rng.normal(0, u.receptivity_noise)
        self._receptivity[user] = np.clip(r, -2.0, 2.0)
        return StepResult(engaged, fatigue_event, float(reward), exp_r)
