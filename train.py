"""Online interaction loop shared by training and evaluation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from context import UserHistory, build_context
from environment import NotificationEnv


@dataclass
class EpisodeLog:
    """Per-decision outcomes, each array shaped [n_days, n_users]."""
    action: np.ndarray
    engaged: np.ndarray
    fatigue: np.ndarray
    reward: np.ndarray
    regret: np.ndarray   # simulator-oracle pseudo-regret: max_a E[r|a] - E[r|chosen]


def run_episode(env: NotificationEnv, policy, learn: bool = True) -> EpisodeLog:
    """Each day, every user gets one notification at the slot the policy chooses.

    The policy sees only the context vector; oracle expected rewards are logged for
    regret and never passed to the policy.
    """
    D, U = env.n_days, env.n_users
    log = EpisodeLog(*(np.zeros((D, U), dtype=dt) for dt in (int, int, int, float, float)))
    histories = [UserHistory() for _ in range(U)]
    for day in range(D):
        for u in range(U):
            x = build_context(env.observe(u, day), histories[u])
            a = policy.select(x)
            res = env.step(u, day, a)
            if learn:
                policy.update(a, x, res.reward)
            histories[u].record(a, res.engaged)
            log.action[day, u] = a
            log.engaged[day, u] = res.engaged
            log.fatigue[day, u] = res.fatigue_event
            log.reward[day, u] = res.reward
            log.regret[day, u] = res.expected_rewards.max() - res.expected_rewards[a]
    return log


def train_agent(agent, env: NotificationEnv) -> EpisodeLog:
    return run_episode(env, agent, learn=True)
