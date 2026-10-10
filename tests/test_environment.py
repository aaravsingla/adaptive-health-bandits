import dataclasses

import numpy as np

from environment import NotificationEnv, Observation, generate_population


def make_env(seed=0, users=10, days=20):
    pop = generate_population(users, days, np.random.default_rng(seed))
    return NotificationEnv(pop, np.random.default_rng(seed + 1)), pop


def test_probabilities_and_outcomes_bounded():
    env, _ = make_env()
    for day in range(env.n_days):
        for u in range(env.n_users):
            p, q = env.probabilities(u, day)
            assert np.all((p >= 0.01) & (p <= 0.99)) and np.all((q >= 0.01) & (q <= 0.99))
            res = env.step(u, day, day % 3)
            assert res.engaged in (0, 1) and res.fatigue_event in (0, 1)
            assert not (res.engaged and res.fatigue_event)
            assert res.reward in (1.0, 0.0, -0.25)
    assert np.all(np.abs(env._receptivity) <= 2) and np.all((env._fatigue >= 0) & (env._fatigue <= 5))


def test_engagement_is_stochastic_not_deterministic():
    env, _ = make_env(users=1, days=1)
    outcomes = {NotificationEnv(env._pop, np.random.default_rng(s)).step(0, 0, 0).engaged for s in range(200)}
    assert outcomes == {0, 1}


def test_observation_exposes_no_hidden_parameters():
    env, _ = make_env()
    obs = env.observe(0, 0)
    assert {f.name for f in dataclasses.fields(Observation)} == {"activity", "is_weekend"}
    assert 0 <= obs.activity <= 1


def test_users_are_heterogeneous():
    _, pop = make_env(users=60)
    assert {u.preferred_slot for u in pop.users} == {0, 1, 2}


def test_fatigue_reduces_future_engagement():
    env, _ = make_env(users=1)
    p_before, _ = env.probabilities(0, 0)
    env._fatigue[0] = 4.0
    p_after, _ = env.probabilities(0, 0)
    assert np.all(p_after < p_before)
