"""Training + fair multi-seed evaluation of all policies."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from agents import make_full_linucb, make_reduced_linucb
from config import Config
from environment import NotificationEnv, generate_population
from metrics import daily_curves, episode_metrics
from policies import FixedPolicy, RandomPolicy
from train import train_agent

# Seed streams: (master, stream, ...) -> independent generators
TRAIN_POP, EVAL_POP, RESPONSE, POLICY = 0, 1, 2, 3


def _rng(*key) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(k) for k in key]))


def train_agents(cfg: Config) -> dict:
    """Train both LinUCB variants online on a separate training population."""
    pop = generate_population(cfg.n_users, cfg.train_days, _rng(cfg.master_seed, TRAIN_POP))
    agents = {}
    makers = {"linucb_reduced": make_reduced_linucb, "linucb_full": make_full_linucb}
    for i, (name, make) in enumerate(makers.items()):
        agent = make(cfg.alpha, cfg.lam, rng=_rng(cfg.master_seed, POLICY, 999, i))
        env = NotificationEnv(pop, _rng(cfg.master_seed, RESPONSE, 999, i), cfg.engage_reward, cfg.fatigue_penalty)
        train_agent(agent, env)
        agents[name] = agent
    return agents


def make_policy(name: str, cfg: Config, trained: dict, rng: np.random.Generator):
    if name == "fixed_evening":
        return FixedPolicy(cfg.fixed_slot)
    if name == "random":
        return RandomPolicy(rng)
    agent = copy.deepcopy(trained[name])   # each eval run starts from the same trained weights
    agent.rng = rng
    return agent


def evaluate(cfg: Config, trained: dict):
    """For each seed: fresh unseen users shared by all policies; each policy gets its own
    response RNG. LinUCB keeps learning online during evaluation."""
    rows, curves = [], []
    for s in range(cfg.n_eval_seeds):
        pop = generate_population(cfg.n_users, cfg.eval_days, _rng(cfg.master_seed, EVAL_POP, s))
        for p_idx, name in enumerate(cfg.policies):
            env = NotificationEnv(pop, _rng(cfg.master_seed, RESPONSE, s, p_idx),
                                  cfg.engage_reward, cfg.fatigue_penalty)
            policy = make_policy(name, cfg, trained, _rng(cfg.master_seed, POLICY, s, p_idx))
            log = train_agent(policy, env)  # same online loop; baselines' update is a no-op
            rows.append({"policy": name, "seed": s, "n_users": env.n_users, "n_days": env.n_days,
                         **episode_metrics(log),
                         **{f"frac_{k}": float((log.action == k).mean()) for k in range(3)}})
            c = daily_curves(log)
            c.insert(0, "seed", s)
            c.insert(0, "policy", name)
            curves.append(c)
    per_seed = pd.DataFrame(rows)
    raw = pd.concat(curves, ignore_index=True)
    g = raw.groupby(["policy", "day"], sort=False)
    lc = g.mean(numeric_only=True).drop(columns="seed").add_suffix("_mean").join(
        g.std(ddof=1, numeric_only=True).drop(columns="seed").add_suffix("_std")).reset_index()
    return per_seed, lc
