import os

import numpy as np
import pandas as pd

import evaluate as ev
from config import Config
from metrics import episode_metrics, relative_improvement, summarize
from run_experiment import main
from train import EpisodeLog

SMALL = Config(n_users=5, train_days=10, eval_days=8, n_eval_seeds=2, master_seed=7)


def test_matched_populations_and_horizons(monkeypatch):
    seen = []
    orig = ev.NotificationEnv

    def spy(pop, *a, **k):
        seen.append(pop)
        return orig(pop, *a, **k)

    monkeypatch.setattr(ev, "NotificationEnv", spy)
    per_seed, _ = ev.evaluate(SMALL, ev.train_agents(SMALL))
    n_pol = len(SMALL.policies)
    eval_pops = seen[-SMALL.n_eval_seeds * n_pol:]
    for s in range(SMALL.n_eval_seeds):
        group = eval_pops[s * n_pol:(s + 1) * n_pol]
        assert all(p is group[0] for p in group)       # identical users for every policy
    assert set(per_seed.n_days) == {SMALL.eval_days} and set(per_seed.n_users) == {SMALL.n_users}
    assert len(per_seed) == SMALL.n_eval_seeds * n_pol


def test_metrics_hand_checked():
    log = EpisodeLog(action=np.zeros((2, 2), int),
                     engaged=np.array([[1, 0], [1, 1]]),
                     fatigue=np.array([[0, 1], [0, 0]]),
                     reward=np.array([[1, -0.25], [1, 1]]),
                     regret=np.array([[0.0, 0.5], [0.1, 0.0]]))
    m = episode_metrics(log)
    assert m["engagement_rate"] == 0.75 and m["fatigue_rate"] == 0.25
    assert m["reward_per_user"] == 2.75 / 2 and m["avg_reward_per_decision"] == 2.75 / 4
    assert m["engagements_per_user"] == 1.5 and np.isclose(m["cumulative_regret_per_user"], 0.3)


def test_summarize_and_relative_improvement():
    df = pd.DataFrame({"policy": ["a", "a"], "engagement_rate": [0.2, 0.4], "reward_per_user": [1, 3],
                       "avg_reward_per_decision": [0, 0], "fatigue_rate": [0, 0],
                       "engagements_per_user": [0, 0], "cumulative_regret_per_user": [0, 0]})
    s = summarize(df).iloc[0]
    assert np.isclose(s.engagement_rate_mean, 0.3) and np.isclose(s.engagement_rate_std, np.sqrt(0.02))
    assert np.isclose(relative_improvement(0.36, 0.30), 20.0)
    assert np.isnan(relative_improvement(0.5, 0.0))


def test_regret_nonnegative():
    per_seed, lc = ev.evaluate(SMALL, ev.train_agents(SMALL))
    assert (per_seed.cumulative_regret_per_user >= 0).all()


def test_reproducible_with_same_seed():
    a, _ = ev.evaluate(SMALL, ev.train_agents(SMALL))
    b, _ = ev.evaluate(SMALL, ev.train_agents(SMALL))
    pd.testing.assert_frame_equal(a, b)


def test_end_to_end_clean_run(tmp_path):
    out = tmp_path / "res"
    main(["--users", "4", "--train-days", "7", "--eval-days", "7", "--seeds", "2", "--results-dir", str(out)])
    for f in ["summary_metrics.csv", "per_seed_metrics.csv", "learning_curves.csv", "run_config.json",
              "engagement_comparison.png", "reward_curve.png", "fatigue_comparison.png",
              "regret_curve.png", "context_ablation.png"]:
        assert os.path.getsize(out / f) > 0, f
