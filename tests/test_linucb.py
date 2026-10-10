import numpy as np
import pytest

from agents import LinUCB, make_reduced_linucb
from context import N_FEATURES
from policies import FixedPolicy, RandomPolicy

X = np.linspace(0.1, 1.0, N_FEATURES)


def test_select_valid_action():
    a = LinUCB(rng=np.random.default_rng(0))
    assert a.select(X) in (0, 1, 2)


def test_update_only_chosen_arm():
    a = LinUCB()
    A0, b0 = a.A.copy(), a.b.copy()
    a.update(1, X, 1.0)
    np.testing.assert_allclose(a.A[1], A0[1] + np.outer(X, X))
    np.testing.assert_allclose(a.b[1], X)
    for k in (0, 2):
        np.testing.assert_array_equal(a.A[k], A0[k])
        np.testing.assert_array_equal(a.b[k], b0[k])


def test_bonus_nonnegative_and_shrinks():
    a = LinUCB(alpha=1.0)
    _, bonus0 = a.scores(X)
    assert np.all(bonus0 >= 0)
    for _ in range(20):
        a.update(0, X, 0.0)
    _, bonus1 = a.scores(X)
    assert np.all(bonus1 >= 0) and bonus1[0] < bonus0[0]


@pytest.mark.parametrize("r", [1.0, 0.0, -0.25])
def test_updates_with_any_reward_sign(r):
    a = LinUCB(lam=1.0, feature_idx=[0])
    a.update(0, X, r)
    x0 = X[0]
    assert np.isclose(a.theta()[0, 0], x0 * r / (1 + x0 ** 2))   # (lam + x^2)^-1 * r x


def test_exploits_better_arm():
    a = LinUCB(alpha=0.1, rng=np.random.default_rng(0))
    for _ in range(50):
        for arm, r in enumerate([0.0, 1.0, 0.0]):
            a.update(arm, X, r)
    assert a.select(X) == 1


def test_untried_arms_get_explored():
    a = LinUCB(alpha=1.0, rng=np.random.default_rng(0))
    for _ in range(10):
        a.update(0, X, 0.6)
    assert a.select(X) in (1, 2)


def test_reduced_uses_intercept_only():
    a = make_reduced_linucb(0.5, 1.0)
    assert a.d == 1 and a.A.shape == (3, 1, 1)


def test_invalid_hyperparams():
    with pytest.raises(ValueError):
        LinUCB(lam=0)


def test_fixed_baseline():
    p = FixedPolicy(2)
    assert all(p.select(X) == 2 for _ in range(50))


def test_random_baseline_valid():
    p = RandomPolicy(np.random.default_rng(0))
    acts = {p.select(X) for _ in range(300)}
    assert acts == {0, 1, 2}
