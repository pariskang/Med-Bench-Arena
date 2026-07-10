"""Statistical rigor for the leaderboard: bootstrap confidence intervals,
paired significance testing, and a sample-size floor.

Pure stdlib (no numpy/scipy dependency, matching the rest of this codebase).
Bootstrap uses a fixed default seed so leaderboard CIs are reproducible across
runs of the same detail file, not just "close enough."
"""
from __future__ import annotations

import math
import random
from typing import Sequence

DEFAULT_N_BOOT = 2000
DEFAULT_SEED = 12345
# below this many graded samples, a headline point estimate is too noisy to
# rank on with confidence — the leaderboard flags it rather than hiding it.
MIN_HEADLINE_SAMPLES = 30


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of ``values``."""
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    if n == 1:
        return (float(values[0]), float(values[0]))
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    lo_idx = int((alpha / 2) * n_boot)
    hi_idx = min(n_boot - 1, int((1 - alpha / 2) * n_boot))
    return (means[lo_idx], means[hi_idx])


def paired_bootstrap_diff_ci(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """CI for the paired mean difference ``mean(a) - mean(b)``.

    ``a`` and ``b`` must be index-aligned (same sample, two models/rollouts) —
    resampling draws the same indices from both so the pairing is preserved.
    """
    n = len(a)
    if n == 0 or len(b) != n:
        return (float("nan"), float("nan"))
    if n == 1:
        d = a[0] - b[0]
        return (d, d)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        idxs = [rng.randrange(n) for _ in range(n)]
        ra = sum(a[i] for i in idxs) / n
        rb = sum(b[i] for i in idxs) / n
        diffs.append(ra - rb)
    diffs.sort()
    lo_idx = int((alpha / 2) * n_boot)
    hi_idx = min(n_boot - 1, int((1 - alpha / 2) * n_boot))
    return (diffs[lo_idx], diffs[hi_idx])


def _chi2_1dof_sf(x: float) -> float:
    """Survival function (1 - CDF) of the chi-square distribution, 1 dof.

    For 1 dof, chi2 is the square of a standard normal, so its SF is
    ``erfc(sqrt(x/2))`` — avoids depending on scipy for the gamma function.
    """
    if x <= 0:
        return 1.0
    return math.erfc(math.sqrt(x / 2.0))


def _binom_two_sided_p(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial test p-value for ``k`` successes in ``n`` trials."""
    if n == 0:
        return 1.0
    point = math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
    total = 0.0
    for i in range(n + 1):
        pi = math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
        # include outcomes no more likely than the observed one (standard
        # exact-binomial two-sided definition), with a small tolerance for
        # floating-point comparison noise.
        if pi <= point * (1 + 1e-9):
            total += pi
    return min(1.0, total)


def mcnemar_test(a: Sequence[int], b: Sequence[int], exact_below: int = 25) -> dict:
    """Paired significance test for two binary (correct/incorrect) rollouts.

    ``a``/``b`` are 0/1-valued and index-aligned (same sample). Discordant
    pairs (a=1,b=0 vs a=0,b=1) drive the test; concordant pairs carry no
    information about which model is better. Uses the exact binomial test
    when the discordant count is small (``exact_below``), else the
    continuity-corrected chi-square approximation.
    """
    n = len(a)
    if n == 0 or len(b) != n:
        return {"n": 0, "n01": 0, "n10": 0, "statistic": float("nan"),
                "p_value": float("nan"), "method": "none"}
    n10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n_disc = n10 + n01
    if n_disc == 0:
        return {"n": n, "n01": n01, "n10": n10, "statistic": 0.0,
                "p_value": 1.0, "method": "degenerate"}
    if n_disc < exact_below:
        p = _binom_two_sided_p(n10, n_disc, 0.5)
        return {"n": n, "n01": n01, "n10": n10, "statistic": float(n10),
                "p_value": p, "method": "exact_binomial"}
    stat = ((abs(n10 - n01) - 1) ** 2) / n_disc
    p = _chi2_1dof_sf(stat)
    return {"n": n, "n01": n01, "n10": n10, "statistic": stat,
            "p_value": p, "method": "chi2_corrected"}


def ci_overlap(ci_a: tuple[float, float], ci_b: tuple[float, float]) -> bool:
    """True if two confidence intervals overlap (a conservative "not clearly
    different" signal — NOT a substitute for a paired significance test, but
    cheap to compute for every leaderboard row pair)."""
    lo_a, hi_a = ci_a
    lo_b, hi_b = ci_b
    if any(v != v for v in (lo_a, hi_a, lo_b, hi_b)):  # NaN check
        return False
    return lo_a <= hi_b and lo_b <= hi_a


def below_min_sample_size(n: int, floor: int = MIN_HEADLINE_SAMPLES) -> bool:
    return n < floor
