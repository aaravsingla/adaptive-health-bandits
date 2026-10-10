"""Regenerate all figures from CSVs in the results directory.

Usage: python plot_results.py [--results-dir results]
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LABELS = {"fixed_evening": "Fixed (19:00)", "random": "Random",
          "linucb_reduced": "LinUCB (intercept only)", "linucb_full": "LinUCB (full context)"}
COLORS = {"fixed_evening": "#7f7f7f", "random": "#bcbd22", "linucb_reduced": "#1f77b4", "linucb_full": "#d62728"}


def _bar(summary, metric, title, ylabel, path, policies=None):
    df = summary if policies is None else summary[summary.policy.isin(policies)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([LABELS[p] for p in df.policy], df[f"{metric}_mean"], yerr=df[f"{metric}_std"],
           capsize=6, color=[COLORS[p] for p in df.policy])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Policy")
    plt.setp(ax.get_xticklabels(), rotation=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _curve(lc, col, title, ylabel, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for p, d in lc.groupby("policy", sort=False):
        m, s = d[f"{col}_mean"], d[f"{col}_std"]
        ax.plot(d.day, m, label=LABELS[p], color=COLORS[p], linestyle="--" if p == "fixed_evening" else "-", zorder=3 if p == "fixed_evening" else 2)
        ax.fill_between(d.day, m - s, m + s, color=COLORS[p], alpha=0.15)
    ax.set_title(title)
    ax.set_xlabel("Evaluation day")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_plots(results_dir: str) -> None:
    summary = pd.read_csv(os.path.join(results_dir, "summary_metrics.csv"))
    lc = pd.read_csv(os.path.join(results_dir, "learning_curves.csv"))
    j = lambda f: os.path.join(results_dir, f)
    _bar(summary, "engagement_rate", "Engagement rate (mean ± std over seeds)", "Engagements / notifications",
         j("engagement_comparison.png"))
    _bar(summary, "fatigue_rate", "Fatigue-event rate (mean ± std over seeds)", "Fatigue events / notifications",
         j("fatigue_comparison.png"))
    _curve(lc, "cum_reward_per_user", "Cumulative reward per user", "Cumulative reward", j("reward_curve.png"))
    _curve(lc, "cum_regret_per_user", "Cumulative simulator-oracle regret per user", "Cumulative regret",
           j("regret_curve.png"))
    _bar(summary, "engagement_rate", "Context ablation: full context vs intercept only",
         "Engagements / notifications", j("context_ablation.png"), policies=["linucb_reduced", "linucb_full"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    make_plots(ap.parse_args().results_dir)
