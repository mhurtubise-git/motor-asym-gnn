"""
metrics_utils.py — Extended metrics for continuous prediction on small
cohorts.

Focus is on error-based metrics that reflect the clinical usefulness of
a predicted score ratio:

  * MAE               : primary error metric.
  * medAE             : median absolute error, robust to a few outliers.
  * pct_within_X      : fraction of subjects predicted within ±X
                         (clinical tolerance). Reported at X ∈ {0.02, 0.03}.
  * bootstrap 95 % CI : intervals for MAE and medAE.
  * bias              : mean(pred − true), systematic sign of the error.
  * CCC (Lin), ICC(2,1) : absolute agreement metrics.
  * Bland–Altman      : mean difference and limits of agreement.

Rank correlations (Spearman ρ, Kendall τ, Pearson r) are also computed
for completeness.
"""
import numpy as np
from scipy.stats import spearmanr, pearsonr, kendalltau


# Clinical tolerances for pct_within (ratios in [0, 1]).
TOLERANCES = (0.02, 0.03)


def ccc_lin(true, pred):
    """Lin's concordance correlation coefficient (1989).

    ``ρ_c = 2 · cov(t, p) / (σ_t² + σ_p² + (μ_t − μ_p)²)``

    Measures absolute agreement with the identity line ``y = x``.
    Penalises bias and variance mismatch, unlike Pearson r.
    """
    t = np.asarray(true, float)
    p = np.asarray(pred, float)
    if len(t) < 3 or np.std(t) < 1e-12 or np.std(p) < 1e-12:
        return float('nan')
    mean_t, mean_p = np.mean(t), np.mean(p)
    var_t, var_p = np.var(t, ddof=0), np.var(p, ddof=0)
    cov = np.mean((t - mean_t) * (p - mean_p))
    return float(2 * cov / (var_t + var_p + (mean_t - mean_p) ** 2))


def icc_2_1(true, pred):
    """ICC(2,1) absolute agreement, single rater (Shrout & Fleiss 1979).

    Treats ``(true, pred)`` as two raters over the same subjects.
    Returns 1 for perfect agreement, values ≤ 0 for no agreement.
    Penalises constant bias and variance mismatch.
    """
    t = np.asarray(true, float)
    p = np.asarray(pred, float)
    n = len(t)
    if n < 5:
        return float('nan')
    mat = np.vstack([t, p]).T                       # (n, 2)
    k = mat.shape[1]                                # 2 raters
    mean_per_subj = mat.mean(axis=1)
    mean_per_rater = mat.mean(axis=0)
    grand = mat.mean()
    ss_total = ((mat - grand) ** 2).sum()
    ss_subj = k * ((mean_per_subj - grand) ** 2).sum()
    ss_rater = n * ((mean_per_rater - grand) ** 2).sum()
    ss_err = ss_total - ss_subj - ss_rater
    ms_subj = ss_subj / (n - 1)
    ms_rater = ss_rater / (k - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))
    denom = ms_subj + (k - 1) * ms_err + k * (ms_rater - ms_err) / n
    if denom == 0 or not np.isfinite(denom):
        return float('nan')
    return float((ms_subj - ms_err) / denom)


def _bootstrap_ci(values, stat_fn, n_boot=10000, alpha=0.05, seed=42):
    """Percentile bootstrap 95 % confidence interval for ``stat_fn``."""
    arr = np.asarray(values, float)
    n = len(arr)
    if n < 3:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = stat_fn(arr[idx], axis=1)
    lo = float(np.percentile(boots, 100 * (alpha / 2)))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return lo, hi


def compute_extended_metrics(true, pred, tolerances=TOLERANCES,
                             bootstrap=True):
    """Compute every metric useful for evaluating a continuous prediction.

    Parameters
    ----------
    true, pred : 1-D array-like
        Ground-truth values and predicted values (same length).
    tolerances : tuple of float
        Absolute-error thresholds for ``pct_within_X``.
    bootstrap : bool
        If True, compute 95 % bootstrap CIs for MAE and medAE
        (10 000 resamples over the per-subject errors).

    Returns
    -------
    dict
        Keys: ``mae``, ``mae_ci_low``, ``mae_ci_high``, ``medae``,
        ``medae_ci_low``, ``medae_ci_high``, ``rmse``, ``bias``,
        ``max_err``, ``pct_within_{tol}``, ``ccc``, ``icc_2_1``, ``r2``,
        ``rho_spearman``, ``p_spearman``, ``tau_kendall``, ``p_kendall``,
        ``pearson_r``, ``p_pearson``, ``n``.
    """
    t = np.asarray(true, float)
    p = np.asarray(pred, float)
    n = len(t)

    out = {'n': int(n)}
    nan_keys = ('rho_spearman', 'p_spearman', 'tau_kendall', 'p_kendall',
                'pearson_r', 'p_pearson', 'ccc', 'icc_2_1', 'mae', 'rmse',
                'medae', 'max_err', 'r2', 'bias',
                'mae_ci_low', 'mae_ci_high',
                'medae_ci_low', 'medae_ci_high')
    if n < 3:
        for k in nan_keys:
            out[k] = float('nan')
        for tol in tolerances:
            out[f'pct_within_{tol:.2f}'] = float('nan')
        return out

    diff = p - t
    abs_err = np.abs(diff)

    out['mae'] = float(np.mean(abs_err))
    out['medae'] = float(np.median(abs_err))
    out['rmse'] = float(np.sqrt(np.mean(diff ** 2)))
    out['bias'] = float(np.mean(diff))
    out['max_err'] = float(np.max(abs_err))

    for tol in tolerances:
        out[f'pct_within_{tol:.2f}'] = float(100.0 * np.mean(abs_err <= tol))

    if bootstrap:
        lo, hi = _bootstrap_ci(
            abs_err, lambda a, axis: np.mean(a, axis=axis))
        out['mae_ci_low'], out['mae_ci_high'] = lo, hi
        lo, hi = _bootstrap_ci(
            abs_err, lambda a, axis: np.median(a, axis=axis))
        out['medae_ci_low'], out['medae_ci_high'] = lo, hi
    else:
        out['mae_ci_low'] = out['mae_ci_high'] = float('nan')
        out['medae_ci_low'] = out['medae_ci_high'] = float('nan')

    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((t - np.mean(t)) ** 2))
    out['r2'] = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    out['ccc'] = ccc_lin(t, p)
    out['icc_2_1'] = icc_2_1(t, p)

    rho, p_rho = spearmanr(t, p)
    tau, p_tau = kendalltau(t, p)
    r, p_r = pearsonr(t, p)
    out['rho_spearman'] = float(rho) if rho == rho else float('nan')
    out['p_spearman'] = float(p_rho) if p_rho == p_rho else float('nan')
    out['tau_kendall'] = float(tau) if tau == tau else float('nan')
    out['p_kendall'] = float(p_tau) if p_tau == p_tau else float('nan')
    out['pearson_r'] = float(r) if r == r else float('nan')
    out['p_pearson'] = float(p_r) if p_r == p_r else float('nan')

    return out


def metrics_to_dataframe(metrics_dict):
    """Convert a metrics dict to a long-format DataFrame for Excel export."""
    import pandas as pd
    rows = [{'metric': k, 'value': v} for k, v in metrics_dict.items()]
    return pd.DataFrame(rows)


def bland_altman_stats(true, pred):
    """Bland–Altman statistics.

    Returns
    -------
    dict
        ``mean_diff``, ``sd_diff``, ``loa_low``, ``loa_up``,
        ``pct_within_loa``, ``mean_arr``, ``diff_arr``.
    """
    t = np.asarray(true, float)
    p = np.asarray(pred, float)
    diff = p - t
    mean = (p + t) / 2.0
    mean_diff = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    loa_low = mean_diff - 1.96 * sd_diff
    loa_up = mean_diff + 1.96 * sd_diff
    pct = float(100.0 * np.mean((diff >= loa_low) & (diff <= loa_up)))
    return {
        'mean_diff': mean_diff,
        'sd_diff': sd_diff,
        'loa_low': loa_low,
        'loa_up': loa_up,
        'pct_within_loa': pct,
        'mean_arr': mean,
        'diff_arr': diff,
    }
