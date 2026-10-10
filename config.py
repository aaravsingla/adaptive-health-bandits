"""Experiment configuration. All defaults live here; CLI flags override them."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field

SLOT_NAMES = ("morning_08", "afternoon_13", "evening_19")
N_ACTIONS = len(SLOT_NAMES)
EVENING = 2


@dataclass
class Config:
    # Experiment size
    n_users: int = 100
    train_days: int = 60
    eval_days: int = 60
    n_eval_seeds: int = 10
    master_seed: int = 42

    # Reward definition
    engage_reward: float = 1.0
    fatigue_penalty: float = 0.25

    # LinUCB hyperparameters (fixed a priori, not tuned on evaluation data)
    alpha: float = 0.5
    lam: float = 1.0

    # Baselines
    fixed_slot: int = EVENING

    results_dir: str = "results"
    policies: tuple = field(default=("fixed_evening", "random", "linucb_reduced", "linucb_full"))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["policies"] = list(self.policies)
        d["slot_names"] = list(SLOT_NAMES)
        return d


def parse_args(argv=None) -> Config:
    cfg = Config()
    p = argparse.ArgumentParser(description="Adaptive notification timing via contextual bandits")
    p.add_argument("--users", type=int, default=cfg.n_users)
    p.add_argument("--train-days", type=int, default=cfg.train_days)
    p.add_argument("--eval-days", type=int, default=cfg.eval_days)
    p.add_argument("--seeds", type=int, default=cfg.n_eval_seeds, help="number of evaluation seeds")
    p.add_argument("--seed", type=int, default=cfg.master_seed, help="master seed")
    p.add_argument("--alpha", type=float, default=cfg.alpha)
    p.add_argument("--lam", type=float, default=cfg.lam)
    p.add_argument("--results-dir", default=cfg.results_dir)
    a = p.parse_args(argv)
    return Config(n_users=a.users, train_days=a.train_days, eval_days=a.eval_days,
                  n_eval_seeds=a.seeds, master_seed=a.seed, alpha=a.alpha, lam=a.lam,
                  results_dir=a.results_dir)
