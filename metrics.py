"""Metric definitions and aggregation across evaluation seeds."""
from __future__ import annotations

import numpy as np
import pandas as pd

from train import EpisodeLog

METRICS = ("engagement_rate", "reward_per_user", "avg_reward_per_decision", "fatigue_rate",
           "engagements_per_user", "cumulative_regret_per_user")


def episode_metrics(log: EpisodeLog) -> dict:
    n = log.engaged.size
    U = log.engaged.shape[1]
    return {
        "engagement_rate": log.engaged.sum() / n,
        "reward_per_user": log.reward.sum() / U,
        "avg_reward_per_decision": log.reward.sum() / n,
        "fatigue_rate": log.fatigue.sum() / n,
        "engagements_per_user": log.engaged.sum() / U,
        "cumulative_regret_per_user": log.regret.sum() / U,
    }


def daily_curves(log: EpisodeLog) -> pd.DataFrame:
    return pd.DataFrame({
        "day": np.arange(log.engaged.shape[0]),
        "engagement_rate": log.engaged.mean(axis=1),
        "cum_reward_per_user": log.reward.mean(axis=1).cumsum(),
        "cum_regret_per_user": log.regret.mean(axis=1).cumsum(),
    })


def summarize(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Mean and std across seeds (the seed is the unit of analysis)."""
    g = per_seed.groupby("policy", sort=False)[list(METRICS)]
    out = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_std"))
    cols = [f"{m}_{s}" for m in METRICS for s in ("mean", "std")]
    return out[cols].reset_index()


def relative_improvement(adaptive: float, baseline: float) -> float:
    """100 * (adaptive - baseline) / baseline; NaN when the baseline is zero."""
    if baseline == 0:
        return float("nan")
    return 100.0 * (adaptive - baseline) / baseline
