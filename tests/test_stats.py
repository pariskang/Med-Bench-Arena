"""Bootstrap CI / McNemar / sample-size-floor stats utilities (pure stdlib)."""
from __future__ import annotations

from medeval.stats import (
    MIN_HEADLINE_SAMPLES,
    below_min_sample_size,
    bootstrap_ci,
    ci_overlap,
    mcnemar_test,
    paired_bootstrap_diff_ci,
)


def test_bootstrap_ci_brackets_the_mean():
    vals = [1.0] * 20 + [0.0] * 20
    lo, hi = bootstrap_ci(vals, n_boot=500, seed=1)
    assert lo <= 0.5 <= hi
    assert lo < hi  # non-degenerate


def test_bootstrap_ci_degenerate_for_constant_values():
    lo, hi = bootstrap_ci([1.0] * 10, n_boot=200, seed=1)
    assert lo == hi == 1.0


def test_bootstrap_ci_empty_is_nan():
    lo, hi = bootstrap_ci([])
    assert lo != lo and hi != hi  # NaN != NaN


def test_bootstrap_ci_single_value():
    lo, hi = bootstrap_ci([0.7])
    assert lo == hi == 0.7


def test_bootstrap_ci_reproducible_with_same_seed():
    vals = [0.1, 0.9, 0.4, 0.6, 0.2, 0.8, 0.3]
    a = bootstrap_ci(vals, n_boot=300, seed=7)
    b = bootstrap_ci(vals, n_boot=300, seed=7)
    assert a == b


def test_paired_bootstrap_diff_ci_zero_when_identical():
    vals = [1.0, 0.0, 1.0, 1.0, 0.0]
    lo, hi = paired_bootstrap_diff_ci(vals, vals, n_boot=300, seed=3)
    assert lo == 0.0 == hi or (lo <= 0 <= hi)


def test_paired_bootstrap_diff_ci_shifts_positive():
    a = [1.0] * 20
    b = [0.0] * 20
    lo, hi = paired_bootstrap_diff_ci(a, b, n_boot=300, seed=3)
    assert lo == hi == 1.0


def test_mcnemar_exact_for_small_discordant_count():
    a = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1]
    b = [0, 1, 0, 0, 0, 1, 1, 1, 0, 1]
    r = mcnemar_test(a, b)
    assert r["method"] == "exact_binomial"
    assert 0.0 <= r["p_value"] <= 1.0
    assert r["n10"] >= 1 and r["n01"] == 0


def test_mcnemar_degenerate_when_no_discordant_pairs():
    a = [1, 0, 1, 0]
    b = [1, 0, 1, 0]
    r = mcnemar_test(a, b)
    assert r["method"] == "degenerate"
    assert r["p_value"] == 1.0


def test_mcnemar_chi2_for_large_discordant_count():
    a = [1] * 40 + [0] * 20
    b = [0] * 40 + [0] * 20
    r = mcnemar_test(a, b, exact_below=25)
    assert r["method"] == "chi2_corrected"
    assert r["p_value"] < 0.05  # a clearly beats b, 40 discordant pairs all one-way


def test_ci_overlap_true_and_false():
    assert ci_overlap((0.1, 0.3), (0.2, 0.4)) is True
    assert ci_overlap((0.1, 0.2), (0.5, 0.6)) is False


def test_below_min_sample_size():
    assert below_min_sample_size(5) is True
    assert below_min_sample_size(MIN_HEADLINE_SAMPLES) is False
    assert below_min_sample_size(MIN_HEADLINE_SAMPLES - 1) is True


if __name__ == "__main__":
    test_bootstrap_ci_brackets_the_mean()
    test_bootstrap_ci_degenerate_for_constant_values()
    test_bootstrap_ci_empty_is_nan()
    test_bootstrap_ci_single_value()
    test_bootstrap_ci_reproducible_with_same_seed()
    test_paired_bootstrap_diff_ci_zero_when_identical()
    test_paired_bootstrap_diff_ci_shifts_positive()
    test_mcnemar_exact_for_small_discordant_count()
    test_mcnemar_degenerate_when_no_discordant_pairs()
    test_mcnemar_chi2_for_large_discordant_count()
    test_ci_overlap_true_and_false()
    test_below_min_sample_size()
    print("OK: stats utility tests passed")
