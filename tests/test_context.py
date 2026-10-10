import numpy as np

from context import FEATURE_NAMES, N_FEATURES, UserHistory, build_context
from environment import Observation


def test_default_context_before_any_feedback():
    x = build_context(Observation(0.3, 1.0), UserHistory())
    assert x.shape == (N_FEATURES,) == (len(FEATURE_NAMES),)
    np.testing.assert_allclose(x, [1, 0.3, 1, 0.5, 0, 0.5, 0.5, 0.5])


def test_history_features_use_past_outcomes_only():
    h = UserHistory()
    obs = Observation(0.5, 0.0)
    x0 = build_context(obs, h)
    h.record(action=2, engaged=1)        # outcome recorded after decision
    x1 = build_context(obs, h)
    assert x0[3] == 0.5 and x1[3] == 1.0
    np.testing.assert_allclose(x1[5:], [0.5, 0.5, 2 / 3])


def test_recent_window_hand_checked():
    h = UserHistory()
    for e in [1, 0, 0, 0, 0, 0, 0, 1, 1]:   # last 7: 0,0,0,0,0,1,1
        h.record(0, e)
    x = build_context(Observation(0.0, 0.0), h)
    assert np.isclose(x[3], 2 / 7) and np.isclose(x[4], 5 / 7)


def test_context_unaffected_by_hidden_state():
    """Changing hidden receptivity must not change the context the agent sees."""
    from environment import NotificationEnv, generate_population
    pop = generate_population(3, 5, np.random.default_rng(0))
    env = NotificationEnv(pop, np.random.default_rng(1))
    x_a = build_context(env.observe(0, 0), UserHistory())
    env._receptivity[:] = 2.0
    env._fatigue[:] = 5.0
    x_b = build_context(env.observe(0, 0), UserHistory())
    np.testing.assert_array_equal(x_a, x_b)
