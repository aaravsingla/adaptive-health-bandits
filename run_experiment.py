"""End-to-end pipeline: python run_experiment.py [--users N --train-days D --seeds S --seed M]"""
from __future__ import annotations

import json
import os
import time

from config import parse_args
from evaluate import evaluate, train_agents
from metrics import relative_improvement, summarize
from plot_results import make_plots


def main(argv=None):
    cfg = parse_args(argv)
    os.makedirs(cfg.results_dir, exist_ok=True)
    t0 = time.time()
    print(f"Training LinUCB agents: {cfg.n_users} users x {cfg.train_days} days ...")
    trained = train_agents(cfg)
    print(f"Evaluating {len(cfg.policies)} policies on {cfg.n_eval_seeds} fresh seeds ...")
    per_seed, lc = evaluate(cfg, trained)
    summary = summarize(per_seed)

    per_seed.to_csv(os.path.join(cfg.results_dir, "per_seed_metrics.csv"), index=False)
    summary.to_csv(os.path.join(cfg.results_dir, "summary_metrics.csv"), index=False)
    lc.to_csv(os.path.join(cfg.results_dir, "learning_curves.csv"), index=False)
    with open(os.path.join(cfg.results_dir, "run_config.json"), "w") as f:
        json.dump({**cfg.to_dict(), "eval_seed_indices": list(range(cfg.n_eval_seeds))}, f, indent=2)
    make_plots(cfg.results_dir)

    s = summary.set_index("policy")
    print("\nPolicy                engage_rate       reward/user        fatigue_rate      regret/user")
    for p, r in s.iterrows():
        print(f"{p:<20}  {r.engagement_rate_mean:.3f} ± {r.engagement_rate_std:.3f}   "
              f"{r.reward_per_user_mean:6.2f} ± {r.reward_per_user_std:5.2f}   "
              f"{r.fatigue_rate_mean:.3f} ± {r.fatigue_rate_std:.3f}   "
              f"{r.cumulative_regret_per_user_mean:6.2f} ± {r.cumulative_regret_per_user_std:5.2f}")
    fixed = s.loc["fixed_evening", "engagement_rate_mean"]
    for p in ("linucb_full", "linucb_reduced", "random"):
        print(f"Relative engagement change vs fixed, {p}: "
              f"{relative_improvement(s.loc[p, 'engagement_rate_mean'], fixed):+.1f}%")
    print(f"\nArtifacts written to {cfg.results_dir}/ in {time.time() - t0:.1f}s")
    return summary


if __name__ == "__main__":
    main()
