import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from scipy.stats import norm


def compute_cross_correlation(x, y, max_lag=None, detrend=True, normalize=True):
    """
    Computes cross-correlation between x and y for lags in [-max_lag, +max_lag].
    Returns:
        lags: array of lag indices
        corr: array of correlation values
    """
    x = np.asarray(x).astype(float)
    y = np.asarray(y).astype(float)

    # Optional detrending to remove slow drift
    if detrend:
        x = sig.detrend(x)
        y = sig.detrend(y)

    # Normalize to zero-mean, unit-variance
    if normalize:
        x = (x - x.mean()) / (x.std() + 1e-8)
        y = (y - y.mean()) / (y.std() + 1e-8)

    N = len(x)

    # Raw full correlation (2N-1 length)
    corr_full = np.correlate(x, y, mode='full')
    lags_full = np.arange(-N + 1, N)

    # Limit to requested lag window
    if max_lag is None:
        return lags_full, corr_full / N

    idx = np.where((lags_full >= -max_lag) & (lags_full <= max_lag))[0]
    corr = corr_full[idx] / N
    lags = lags_full[idx]
    return lags, corr


def estimate_lag(x, y, max_lag):
    """Return lag of maximum correlation."""
    lags, corr = compute_cross_correlation(x, y, max_lag=max_lag)
    i = np.argmax(np.abs(corr))
    return lags[i], corr[i], lags, corr


def lag_significance(corr, N_eff):
    """
    Compute approximate p-value and z-score for a correlation peak.
    Based on Fisher z-transform (approx).
    """
    r = corr
    if abs(r) >= 0.999:
        r = np.sign(r) * 0.999
    z = 0.5 * np.log((1 + r) / (1 - r))       # Fisher z-transform
    se = 1.0 / np.sqrt(N_eff - 3)             # standard error
    z_score = z / se
    p_value = 2 * (1 - norm.cdf(abs(z_score)))
    return z_score, p_value


def effective_sample_size(x, y):
    """
    Compute effective sample size for correlation significance,
    accounting for autocorrelation.
    (Bretherton et al. 1999)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)

    # Lag-1 autocorrelation
    r1x = np.corrcoef(x[:-1], x[1:])[0, 1]
    r1y = np.corrcoef(y[:-1], y[1:])[0, 1]
    Neff = n * (1 - r1x * r1y) / (1 + r1x * r1y)
    Neff = max(5, Neff)  # safety floor
    return Neff


def plot_correlogram(lags, corr, best_lag, title="Cross-correlation"):
    plt.figure(figsize=(10, 4))
    plt.plot(lags, corr, '-k')
    plt.axvline(best_lag, color='red', linestyle='--', label=f"best lag = {best_lag}")
    plt.xlabel("Lag (steps)")
    plt.ylabel("Correlation")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_shift_overlay(x, y, lag, title="Shift overlay"):
    """
    Visualize x(t) and y(t-lag) overlaid after applying the estimated shift.
    """
    T = len(x)
    if lag > 0:
        y_shifted = np.zeros_like(y)
        y_shifted[lag:] = y[:-lag]
    elif lag < 0:
        y_shifted = np.zeros_like(y)
        y_shifted[:lag] = y[-lag:]
    else:
        y_shifted = y.copy()

    plt.figure(figsize=(12, 4))
    plt.plot(x, label='local signal')
    plt.plot(y_shifted, label=f'global signal shifted by lag={lag}')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
