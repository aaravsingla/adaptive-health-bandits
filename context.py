"""Observable context construction.

Feature schema (FEATURE_NAMES), all values in [0, 1]:
    0 bias               constant 1
    1 activity           today's activity level (observable before deciding)
    2 is_weekend         1 on Sat/Sun
    3 recent_engagement  engagement rate over the last 7 notifications (default 0.5)
    4 recent_ignored     ignored notifications in the last 7 days / 7 (default 0)
    5 slot_eng_morning   per-user smoothed engagement rate for the morning slot
    6 slot_eng_afternoon     "      afternoon slot
    7 slot_eng_evening       "      evening slot
Slot estimates use a Beta(1,1) prior: (engagements + 1) / (sends + 2), so the
default before any feedback is 0.5.

Only past outcomes enter the history; the current day's outcome is appended
via `UserHistory.record` *after* the decision has been made.
"""
from __future__ import annotations

import numpy as np

from config import N_ACTIONS
from environment import Observation

FEATURE_NAMES = ("bias", "activity", "is_weekend", "recent_engagement", "recent_ignored",
                 "slot_eng_morning", "slot_eng_afternoon", "slot_eng_evening")
N_FEATURES = len(FEATURE_NAMES)
WINDOW = 7


class UserHistory:
    """Interaction log for one user, built only from observed outcomes."""

    def __init__(self):
        self.actions: list[int] = []
        self.engaged: list[int] = []
        self.slot_sends = np.zeros(N_ACTIONS)
        self.slot_engagements = np.zeros(N_ACTIONS)

    def record(self, action: int, engaged: int) -> None:
        self.actions.append(action)
        self.engaged.append(engaged)
        self.slot_sends[action] += 1
        self.slot_engagements[action] += engaged


def build_context(obs: Observation, history: UserHistory) -> np.ndarray:
    recent = history.engaged[-WINDOW:]
    recent_eng = float(np.mean(recent)) if recent else 0.5
    recent_ign = (len(recent) - sum(recent)) / WINDOW
    slot_est = (history.slot_engagements + 1.0) / (history.slot_sends + 2.0)
    return np.array([1.0, obs.activity, obs.is_weekend, recent_eng, recent_ign, *slot_est])
