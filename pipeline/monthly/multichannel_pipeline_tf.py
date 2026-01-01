# multichannel_pipeline_tf.py
# -*- coding: utf-8 -*-
"""
Multichannel Time-Series Pipeline (TensorFlow/Keras)
===================================================

Author: M365 Copilot (for Mirko Lukovic)
Date: 2025-12-28

This single file implements an end-to-end pipeline for:

- Preprocessing & cleaning 30-day, 10-min resolution segments with a 10-day overlap
- Optional misalignment (lag) correction using cross-correlation
- Outlier detection (Hampel filter) and optional drift trend extraction
- Training a multi-task temporal model (TCN-style) that:
    * Reconstructs/imputes the 10-min multichannel input (focus on local channels)
    * Predicts cleaned hourly outputs for the 3 local targets (T, RH, Stem)
- Evaluation with baselines (linear interpolation for imputation; persistence for hourly targets)
- Residual-based drift detection for RH using Page–Hinkley
- Physics-aware post-processing hooks (RH bounds) and provenance flags
- Overlap-aware consensus of drift alarms across adjacent segments

Assumptions & Data Format
-------------------------
- Inputs are normalized per segment to [0,1]. Keep the per-channel scalers for inverse transform if needed.
- Shapes:
    X_train: (N_train, 4320, 11)
    y_train: (N_train, 720, 3)
    X_test:  (N_test,  4320, 11)
    y_test:  (N_test,  720, 3)
- Channels (zero-based index):
    0: local mean temperature (°C), 10-min, raw before normalization
    1: local relative humidity (%), 10-min, raw before normalization
    2: local tree stem radius change (µm), 10-min, raw before normalization
    3: GLOBAL mean temperature (°C), daily cleaned
    4: GLOBAL min temperature (°C), daily cleaned
    5: GLOBAL max temperature (°C), daily cleaned
    6: GLOBAL relative humidity (%), daily cleaned
    7: GLOBAL vapor pressure deficit (kPa), daily cleaned
    8: GLOBAL precipitation (mm), daily cleaned
    9: GLOBAL solar radiation (W m^-2), daily cleaned
    10: day of year (DOY), daily

- Targets y_* (hourly, cleaned): columns [local T (°C), local RH (%), local Stem (µm)] in normalized domain.

Usage
-----
1) Replace the placeholder loaders in `if __name__ == "__main__":` with your actual arrays.
2) Tune configuration flags (e.g., AUGMENT_WITH_GAPS, use_misalignment_correction).
3) Run: `python multichannel_pipeline_tf.py` to train and evaluate.

"""

from __future__ import annotations
import os
import typing as t
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

# Optional: try to import statsmodels for LOWESS; fall back to rolling-median trend if unavailable
try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

# =====================
# Configuration
# =====================
SEQ_LEN_10MIN: int = 4320  # 30 days * 24 h/day * 6 steps/h
HOUR_STEPS: int = 720      # 30 days * 24 h
STRIDE_PER_HOUR: int = 6
N_CHANNELS: int = 11
N_TARGETS: int = 3

# Channel indices (zero-based)
CH = {
    'local_T_10m': 0,
    'local_RH_10m': 1,
    'local_stem_10m': 2,
    'global_T_mean_daily': 3,
    'global_T_min_daily': 4,
    'global_T_max_daily': 5,
    'global_RH_daily': 6,
    'global_VPD_daily': 7,
    'global_precip_daily': 8,
    'global_rad_daily': 9,
    'DOY_daily': 10,
}

# Training & augmentation
AUGMENT_WITH_GAPS: bool = True         # on-the-fly random gaps during training
GAP_CHANNEL_PROB: float = 0.5
N_GAPS_RANGE: t.Tuple[int, int] = (1, 3)
GAP_LEN_DAYS_RANGE: t.Tuple[int, int] = (1, 12)
RNG = np.random.default_rng(42)

# Misalignment correction
USE_MISALIGNMENT_CORRECTION: bool = True
MAX_LAG_STEPS: int = 6 * 24  # up to 24 hours of lag at 10-min resolution

# Cleaning
USE_HAMPEL_FILTER: bool = True
HAMPEL_WINDOW: int = 13
HAMPEL_SIGMAS: float = 3.0
USE_LOWESS_DETREND: bool = False   # set True to subtract slow trend from selected channels
LOWESS_FRAC: float = 0.02          # ~86 steps (~14.3 hours) over 4320
LOWESS_IT: int = 3

# Loss weights
W_RECON_MASKED: float = 1.0
W_RECON_UNMASKED: float = 0.05
W_HOURLY: float = 1.0

# Optimization
BATCH_SIZE: int = 32
LR: float = 3e-4
EPOCHS: int = 100
EARLY_STOP_PATIENCE: int = 10
REDUCE_LR_PATIENCE: int = 4
MIN_LR: float = 1e-6

# Drift detection (Page–Hinkley) on RH residuals
PH_DELTA: float = 0.003   # small allowed change (tune via training residuals)
PH_LAM: float = 0.02      # mean smoothing
PH_ALPHA: float = 30.0    # threshold (alarm when exceeded)

# Overlap consensus
OVERLAP_HOURS: int = 240  # 10 days * 24 h
CONSENSUS_MARGIN_HOURS: int = 6

# =====================
# Utilities
# =====================

def inject_random_gaps(x: np.ndarray, rng=RNG) -> t.Tuple[np.ndarray, np.ndarray]:
    """Inject gaps (set values to 0; mask=0) in randomly chosen channels and positions.
    Returns: x_masked, mask with shape (4320, 11), dtype float32.
    """
    x = x.copy().astype(np.float32)
    mask = np.ones_like(x, dtype=np.float32)

    n_gaps = rng.integers(N_GAPS_RANGE[0], N_GAPS_RANGE[1] + 1)
    chosen_channels = [c for c in range(x.shape[1]) if rng.random() < GAP_CHANNEL_PROB]
    if not chosen_channels:
        chosen_channels = [rng.integers(0, x.shape[1])]

    for _ in range(n_gaps):
        ch = rng.choice(chosen_channels)
        gap_len_days = rng.integers(GAP_LEN_DAYS_RANGE[0], GAP_LEN_DAYS_RANGE[1] + 1)
        gap_len = gap_len_days * 24 * 6
        gap_len = min(gap_len, SEQ_LEN_10MIN - 1)
        start = rng.integers(0, SEQ_LEN_10MIN - gap_len)
        end = start + gap_len
        x[start:end, ch] = 0.0
        mask[start:end, ch] = 0.0
    return x, mask


def hampel_filter(x: np.ndarray, window_size: int = HAMPEL_WINDOW, n_sigmas: float = HAMPEL_SIGMAS) -> np.ndarray:
    """Hampel filter to detect spikes; returns boolean flags of outlier positions.
    Implemented with rolling median and MAD; no external dependencies.
    """
    x = np.asarray(x)
    T = len(x)
    k = int(window_size)
    flags = np.zeros(T, dtype=bool)
    for t in range(T):
        lo = max(0, t - k)
        hi = min(T, t + k + 1)
        window = x[lo:hi]
        med = np.median(window)
        mad = np.median(np.abs(window - med)) + 1e-8
        if np.abs(x[t] - med) / (1.4826 * mad) > n_sigmas:
            flags[t] = True
    return flags


def lowess_detrend(x: np.ndarray, frac: float = LOWESS_FRAC, it: int = LOWESS_IT) -> t.Tuple[np.ndarray, np.ndarray]:
    """LOWESS detrending (subtract smooth trend). Falls back to rolling median if statsmodels is unavailable.
    Returns: x_detrended, trend
    """
    T = len(x)
    xi = np.arange(T)
    if HAS_STATSMODELS:
        trend = sm.nonparametric.lowess(x, xi, frac=frac, it=it, return_sorted=False)
    else:
        # Fallback: rolling median as a robust trend estimator
        win = max(5, int(frac * T))
        if win % 2 == 0:
            win += 1
        pad = win // 2
        xpad = np.pad(x, (pad, pad), mode='edge')
        trend = np.array([np.median(xpad[i:i+win]) for i in range(T)])
    x_detr = x - trend
    return x_detr.astype(np.float32), trend.astype(np.float32)


def estimate_fixed_lag(x: np.ndarray, y: np.ndarray, max_lag_steps: int = MAX_LAG_STEPS, robust: bool = True) -> t.Tuple[int, float]:
    """Estimate lag (in steps) to best align x with y via cross-correlation.
    Positive lag means x should be shifted forward.
    Returns: best_lag, best_corr
    """
    x_ = x.copy()
    y_ = y.copy()
    if robust:
        # Remove linear trend (approx via first differences) and median
        x_ = np.diff(x_, prepend=x_[0]) - np.median(np.diff(x_, prepend=x_[0]))
        y_ = np.diff(y_, prepend=y_[0]) - np.median(np.diff(y_, prepend=y_[0]))
    lags = np.arange(-max_lag_steps, max_lag_steps + 1)
    corr = []
    for L in lags:
        if L >= 0:
            xc = x_[L:]
            yc = y_[:len(x_) - L]
        else:
            xc = x_[:len(x_) + L]
            yc = y_[-L:]
        if len(xc) < 10:
            corr.append(-np.inf)
        else:
            c = np.corrcoef(xc, yc)[0, 1]
            corr.append(c)
    best_idx = int(np.argmax(corr))
    best_lag = int(lags[best_idx])
    return best_lag, float(corr[best_idx])


def shift_series(x: np.ndarray, lag: int) -> np.ndarray:
    """Shift series by lag steps with zero-fill at edges."""
    T = len(x)
    y = np.zeros_like(x)
    if lag > 0:
        y[lag:] = x[:T - lag]
    elif lag < 0:
        y[:T + lag] = x[-lag:]
    else:
        y[:] = x
    return y


def apply_preclean(x: np.ndarray,
                   use_hampel: bool = USE_HAMPEL_FILTER,
                   use_lowess: bool = USE_LOWESS_DETREND,
                   lowess_channels: t.Optional[t.Sequence[int]] = None) -> t.Tuple[np.ndarray, np.ndarray, dict]:
    """Apply pre-cleaning: Hampel spikes -> set missing; optional LOWESS detrend.
    Args:
        x: (4320, C) normalized input
        use_hampel: whether to flag spikes
        use_lowess: whether to subtract slow trend
        lowess_channels: list of channel indices to detrend (e.g., [CH['local_RH_10m']])
    Returns:
        x_clean: (4320, C)
        mask: (4320, C) where 1=observed, 0=missing after cleaning
        trends: dict {channel_index: trend_series}
    """
    x = x.copy().astype(np.float32)
    T, C = x.shape
    mask = np.ones_like(x, dtype=np.float32)
    trends: dict = {}

    for c in range(C):
        xc = x[:, c]
        if use_hampel:
            spikes = hampel_filter(xc, window_size=HAMPEL_WINDOW, n_sigmas=HAMPEL_SIGMAS)
            if spikes.any():
                x[spikes, c] = 0.0
                mask[spikes, c] = 0.0
        if use_lowess and (lowess_channels is not None) and (c in lowess_channels):
            x_detr, trend = lowess_detrend(xc, frac=LOWESS_FRAC, it=LOWESS_IT)
            x[:, c] = x_detr
            trends[c] = trend
    return x, mask, trends


# =====================
# Dataset builders
# =====================

def make_example(x_10m: np.ndarray,
                 y_hourly: np.ndarray,
                 training: bool = True,
                 preclean: bool = True,
                 misalign_correction: bool = USE_MISALIGNMENT_CORRECTION) -> t.Tuple[dict, dict, dict]:
    """Build one training/eval example with optional pre-clean and misalignment correction.
    Returns:
      inputs: {'x_in': (4320, 11), 'mask_in': (4320, 11)}
      outputs: {'recon': (4320, 11), 'hourly': (720, 3)}
      sample_weights: {'recon': (4320, 11), 'hourly': (720, 3)}
    """
    x = x_10m.astype(np.float32)
    y = y_hourly.astype(np.float32)

    # Pre-clean (Hampel/outliers; optional LOWESS on selected channels)
    if preclean:
        x, mask_clean, _trends = apply_preclean(
            x,
            use_hampel=USE_HAMPEL_FILTER,
            use_lowess=USE_LOWESS_DETREND,
            lowess_channels=[CH['local_RH_10m']] if USE_LOWESS_DETREND else None,
        )
    else:
        mask_clean = np.ones_like(x, dtype=np.float32)

    # Misalignment correction (fixed-lag) for local vs global proxies
    if misalign_correction:
        # Align local T with global mean T
        lag_T, _ = estimate_fixed_lag(x[:, CH['local_T_10m']], x[:, CH['global_T_mean_daily']], MAX_LAG_STEPS)
        x[:, CH['local_T_10m']] = shift_series(x[:, CH['local_T_10m']], lag_T)
        # Align local RH with global RH (positive corr) or VPD (negative corr). Prefer RH.
        lag_RH, _ = estimate_fixed_lag(x[:, CH['local_RH_10m']], x[:, CH['global_RH_daily']], MAX_LAG_STEPS)
        x[:, CH['local_RH_10m']] = shift_series(x[:, CH['local_RH_10m']], lag_RH)
        # Align stem with radiation or mean T (heuristic)
        lag_stem, _ = estimate_fixed_lag(x[:, CH['local_stem_10m']], x[:, CH['global_rad_daily']], MAX_LAG_STEPS)
        x[:, CH['local_stem_10m']] = shift_series(x[:, CH['local_stem_10m']], lag_stem)

    # Training-time synthetic gaps augmentation
    if training and AUGMENT_WITH_GAPS:
        x_masked, aug_mask = inject_random_gaps(x)
        # combine preclean mask and augmentation mask via AND (both 1 => observed)
        mask = (mask_clean * aug_mask).astype(np.float32)
    else:
        x_masked = x
        mask = mask_clean

    # Reconstruction weights
    w_recon = W_RECON_UNMASKED * np.ones_like(mask, dtype=np.float32)
    w_recon[mask == 0.0] = W_RECON_MASKED
    # Hourly weights uniform
    w_hourly = W_HOURLY * np.ones_like(y, dtype=np.float32)

    inputs = {'x_in': x_masked, 'mask_in': mask}
    outputs = {'recon': x, 'hourly': y}
    sample_weights = {'recon': w_recon, 'hourly': w_hourly}
    return inputs, outputs, sample_weights


def make_tf_dataset(X: np.ndarray,
                    Y: np.ndarray,
                    batch_size: int = BATCH_SIZE,
                    training: bool = True,
                    shuffle: bool = True,
                    preclean: bool = True) -> tf.data.Dataset:
    """Create tf.data.Dataset that yields (inputs, outputs, sample_weights)."""
    def gen():
        for i in range(X.shape[0]):
            inp, out, sw = make_example(X[i], Y[i], training=training, preclean=preclean,
                                       misalign_correction=USE_MISALIGNMENT_CORRECTION)
            yield inp, out, sw

    output_signature = (
        {
            'x_in': tf.TensorSpec(shape=(SEQ_LEN_10MIN, N_CHANNELS), dtype=tf.float32),
            'mask_in': tf.TensorSpec(shape=(SEQ_LEN_10MIN, N_CHANNELS), dtype=tf.float32),
        },
        {
            'recon': tf.TensorSpec(shape=(SEQ_LEN_10MIN, N_CHANNELS), dtype=tf.float32),
            'hourly': tf.TensorSpec(shape=(HOUR_STEPS, N_TARGETS), dtype=tf.float32),
        },
        {
            'recon': tf.TensorSpec(shape=(SEQ_LEN_10MIN, N_CHANNELS), dtype=tf.float32),
            'hourly': tf.TensorSpec(shape=(HOUR_STEPS, N_TARGETS), dtype=tf.float32),
        },
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    if training and shuffle:
        ds = ds.shuffle(buffer_size=min(4096, X.shape[0]))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# =====================
# Model (TCN-style encoder + two heads)
# =====================

def TemporalBlock(filters: int, dilation_rate: int, dropout: float = 0.1, name: str | None = None):
    def f(x):
        y = layers.SeparableConv1D(filters, kernel_size=3, padding='same',
                                   dilation_rate=dilation_rate, name=f'{name}_sepconv')(x)
        y = layers.BatchNormalization(name=f'{name}_bn')(y)
        y = layers.Activation('gelu', name=f'{name}_act')(y)
        y = layers.SpatialDropout1D(dropout, name=f'{name}_drop')(y)
        if x.shape[-1] != filters:
            x_proj = layers.Conv1D(filters, kernel_size=1, padding='same', name=f'{name}_proj')(x)
        else:
            x_proj = x
        return layers.Add(name=f'{name}_add')([x_proj, y])
    return f


def build_model() -> tf.keras.Model:
    x_in = layers.Input(shape=(SEQ_LEN_10MIN, N_CHANNELS), name='x_in')
    m_in = layers.Input(shape=(SEQ_LEN_10MIN, N_CHANNELS), name='mask_in')

    # Concatenate mask to inputs so the model knows where values are missing
    z = layers.Concatenate(axis=-1, name='concat_mask')([x_in, m_in])  # (T, 22)
    z = layers.Conv1D(64, kernel_size=5, padding='same', name='emb_conv')(z)
    z = layers.Activation('gelu')(z)

    for i, d in enumerate([1, 2, 4, 8, 16, 32]):
        z = TemporalBlock(128, dilation_rate=d, dropout=0.1, name=f'block{i+1}')(z)

    feat_10m = layers.Conv1D(128, kernel_size=1, padding='same', name='shared_1x1')(z)
    feat_10m = layers.Activation('gelu')(feat_10m)

    # Reconstruction head (10-min)
    recon = layers.Conv1D(N_CHANNELS, kernel_size=1, padding='same', name='recon_head')(feat_10m)

    # Hourly head (pool 6 x 10-min -> 1 hour)
    hourly_feat = layers.AveragePooling1D(pool_size=STRIDE_PER_HOUR, strides=STRIDE_PER_HOUR,
                                          padding='valid', name='to_hourly')(feat_10m)  # (720, 128)
    hourly_feat = layers.Conv1D(64, kernel_size=1, activation='gelu', name='hourly_proj')(hourly_feat)
    hourly = layers.Conv1D(N_TARGETS, kernel_size=1, padding='same', name='hourly_head')(hourly_feat)

    return Model(inputs={'x_in': x_in, 'mask_in': m_in}, outputs={'recon': recon, 'hourly': hourly},
                 name='multitask_impute_hourly')


def masked_mse(y_true, y_pred):
    # Masking is handled through sample_weights provided in model.fit()
    return tf.reduce_mean(tf.square(y_true - y_pred))


# =====================
# Baselines
# =====================

def linear_interp_impute(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-channel linear interpolation along time for masked positions.
    Assumes x contains neutral fills at masked positions. Edge padding with nearest observed.
    """
    x_imp = x.copy()
    T, C = x.shape
    idx = np.arange(T)
    for c in range(C):
        m = mask[:, c].astype(bool)
        if m.all():
            continue
        # Edge handling
        if not m[0]:
            first_obs = np.argmax(m)
            x_imp[:first_obs, c] = x_imp[first_obs, c]
            m[:first_obs] = True
        if not m[-1]:
            last_obs = T - 1 - np.argmax(m[::-1])
            x_imp[last_obs+1:, c] = x_imp[last_obs, c]
            m[last_obs+1:] = True
        x_imp[~m, c] = np.interp(idx[~m], idx[m], x_imp[m, c])
    return x_imp


def persistence_hourly(y_true: np.ndarray) -> np.ndarray:
    """Persistence baseline: y[t] = y[t-1]; y[0] = y[0]."""
    y_pred = y_true.copy()
    y_pred[1:] = y_true[:-1]
    return y_pred


# =====================
# Drift detection (Page–Hinkley) on RH residuals
# =====================

def page_hinkley(residuals: np.ndarray, delta: float = PH_DELTA, lam: float = PH_LAM, alpha: float = PH_ALPHA) -> np.ndarray:
    """Page–Hinkley change detector for mean shifts.
    Returns indices (hours) where change is detected.
    """
    T = residuals.shape[0]
    mean_est = 0.0
    ph = 0.0
    min_ph = 0.0
    alarms = []
    for t in range(T):
        x = float(residuals[t])
        mean_est = (1.0 - lam) * mean_est + lam * x
        ph += (x - mean_est - delta)
        min_ph = min(min_ph, ph)
        if ph - min_ph > alpha:
            alarms.append(t)
            ph = 0.0
            min_ph = 0.0
            mean_est = 0.0
    return np.array(alarms, dtype=int)


def rh_residuals(model: tf.keras.Model, x_10m: np.ndarray, y_hourly: np.ndarray) -> np.ndarray:
    """Compute hourly RH residuals: true - predicted."""
    pred = model.predict({'x_in': x_10m[None, ...], 'mask_in': np.ones_like(x_10m)[None, ...]}, verbose=0)
    rh_true = y_hourly[:, 1]
    rh_hat = pred['hourly'][0, :, 1]
    return (rh_true - rh_hat)


def overlap_consensus(alarms_per_segment: t.List[np.ndarray], overlap_hours: int = OVERLAP_HOURS,
                      margin_hours: int = CONSENSUS_MARGIN_HOURS) -> t.List[t.Tuple[int, int]]:
    """Simple adjacency-based consensus across segments with fixed overlap.
    We assume consecutive segments overlap by `overlap_hours` at the end/beginning.
    If segment i has an alarm at hour h in its last overlap region,
    and segment i+1 has an alarm within ±margin_hours at its first overlap region,
    we record a high-confidence alarm (segment_index, hour).
    Returns list of (segment_index, hour) for consensus alarms.
    """
    consensus = []
    for i in range(len(alarms_per_segment) - 1):
        a_i = alarms_per_segment[i]
        a_j = alarms_per_segment[i + 1]
        if a_i.size == 0 or a_j.size == 0:
            continue
        # Overlap zones: last `overlap_hours` of i, first `overlap_hours` of j
        for h_i in a_i:
            if h_i >= (HOUR_STEPS - overlap_hours):
                # candidate in the tail of segment i
                for h_j in a_j:
                    if h_j <= overlap_hours and abs((h_i - (HOUR_STEPS - overlap_hours)) - h_j) <= margin_hours:
                        consensus.append((i, int(h_i)))
                        break
    return consensus


# =====================
# Post-processing & provenance
# =====================

def resample_10min_to_hourly_median(x_10m: np.ndarray) -> np.ndarray:
    """Resample (T=4320, C) to hourly (H=720, C) by median over 6 samples per hour."""
    H = HOUR_STEPS
    C = x_10m.shape[1]
    return np.median(x_10m.reshape(H, STRIDE_PER_HOUR, C), axis=1)


def make_provenance(mask_10m: np.ndarray,
                    drift_adjusted_flags: t.Optional[np.ndarray] = None,
                    capped_flags: t.Optional[np.ndarray] = None) -> dict:
    """Provenance flags/stats per hour.
    - frac_imputed_per_hour: average of (1-mask) across channels/time within the hour
    - drift_adjusted_any_hour: boolean per hour
    - capped_any_hour: boolean per hour
    """
    H = HOUR_STEPS
    imputed_10m = 1.0 - mask_10m
    frac_imp_hr = imputed_10m.reshape(H, STRIDE_PER_HOUR, mask_10m.shape[1]).mean(axis=(1, 2))
    prov = {
        'frac_imputed_per_hour': frac_imp_hr,
        'drift_adjusted_any_hour': None,
        'capped_any_hour': None,
    }
    if drift_adjusted_flags is not None:
        prov['drift_adjusted_any_hour'] = drift_adjusted_flags.reshape(H, STRIDE_PER_HOUR).any(axis=1)
    if capped_flags is not None:
        prov['capped_any_hour'] = capped_flags.reshape(H, STRIDE_PER_HOUR).any(axis=1)
    return prov


# =====================
# Training / Evaluation Harness
# =====================

def train_and_evaluate(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> tf.keras.Model:
    """Train the model with 80:20 split of training set; evaluate on test; print baselines."""
    # Train/val split
    N = X_train.shape[0]
    idx = np.arange(N)
    np.random.seed(123)
    np.random.shuffle(idx)
    split = int(0.8 * N)
    train_idx, val_idx = idx[:split], idx[split:]
    X_tr, Y_tr = X_train[train_idx], y_train[train_idx]
    X_val, Y_val = X_train[val_idx], y_train[val_idx]

    ds_tr = make_tf_dataset(X_tr, Y_tr, batch_size=BATCH_SIZE, training=True, shuffle=True, preclean=True)
    ds_val = make_tf_dataset(X_val, Y_val, batch_size=BATCH_SIZE, training=False, shuffle=False, preclean=True)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss={'recon': masked_mse, 'hourly': 'mse'},
        metrics={'recon': [tf.keras.metrics.MeanAbsoluteError(name='mae')],
                 'hourly': [tf.keras.metrics.MeanSquaredError(name='mse'),
                            tf.keras.metrics.MeanAbsoluteError(name='mae')]},
        loss_weights={'recon': 1.0, 'hourly': 1.0}
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_hourly_mse', patience=EARLY_STOP_PATIENCE, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_hourly_mse', factor=0.5, patience=REDUCE_LR_PATIENCE, min_lr=MIN_LR),
        tf.keras.callbacks.ModelCheckpoint('best_multitask.weights.h5', monitor='val_hourly_mse', save_best_only=True, save_weights_only=True)
    ]

    history = model.fit(ds_tr, validation_data=ds_val, epochs=EPOCHS, callbacks=callbacks)

    # Evaluate on held-out test (no synthetic gaps; preclean enabled)
    ds_test = make_tf_dataset(X_test, y_test, batch_size=BATCH_SIZE, training=False, shuffle=False, preclean=True)
    test_metrics = model.evaluate(ds_test, return_dict=True)
    print("\nTest metrics:", test_metrics)

    # Baseline comparison: persistence for hourly
    hourly_mae_model = []
    hourly_mae_pers = []
    for i in range(min(256, X_test.shape[0])):
        x = X_test[i]
        y = y_test[i]
        pred = model.predict({'x_in': x[None, ...], 'mask_in': np.ones_like(x)[None, ...]}, verbose=0)
        y_hat = pred['hourly'][0]  # (720, 3)
        hourly_mae_model.append(np.mean(np.abs(y_hat - y)))
        hourly_mae_pers.append(np.mean(np.abs(persistence_hourly(y) - y)))
    print("Hourly MAE — model:", np.mean(hourly_mae_model), "persistence:", np.mean(hourly_mae_pers))

    # Imputation baseline on injected gaps (evaluate MAE on masked positions)
    errs_model, errs_interp = [], []
    for i in range(min(128, X_test.shape[0])):
        x = X_test[i]
        # Apply preclean to simulate realistic mask
        x_pc, mask_pc, _ = apply_preclean(x, use_hampel=USE_HAMPEL_FILTER, use_lowess=USE_LOWESS_DETREND,
                                           lowess_channels=[CH['local_RH_10m']] if USE_LOWESS_DETREND else None)
        # Inject training-like gaps for fair comparison
        x_masked, aug_mask = inject_random_gaps(x_pc)
        mask = (mask_pc * aug_mask).astype(np.float32)

        pred = model.predict({'x_in': x_masked[None, ...], 'mask_in': mask[None, ...]}, verbose=0)
        recon_pred = pred['recon'][0]
        interp = linear_interp_impute(x_masked, mask)
        missing = (mask == 0.0)
        if missing.any():
            e_model = np.abs(recon_pred[missing] - x_pc[missing]).mean()
            e_interp = np.abs(interp[missing] - x_pc[missing]).mean()
            errs_model.append(e_model)
            errs_interp.append(e_interp)
    if errs_model:
        print("Imputation MAE on injected gaps — model:", np.mean(errs_model),
              "interp:", np.mean(errs_interp))

    return model


# =====================
# Main (replace loaders with your data)
# =====================
if __name__ == "__main__":
    # In production, load your arrays:
    #   X_train = np.load('X_train.npy')
    #   y_train = np.load('y_train.npy')
    #   X_test  = np.load('X_test.npy')
    #   y_test  = np.load('y_test.npy')

    if all(os.path.exists(f) for f in ['X_train.npy', 'y_train.npy', 'X_test.npy', 'y_test.npy']):
        X_train = np.load('X_train.npy')
        y_train = np.load('y_train.npy')
        X_test = np.load('X_test.npy')
        y_test = np.load('y_test.npy')
    else:
        print("Please provide X_train.npy, y_train.npy, X_test.npy, y_test.npy in the working directory.")
        print("Exiting without training.")
        raise SystemExit(0)

    # Sanity checks
    assert X_train.shape[1:] == (SEQ_LEN_10MIN, N_CHANNELS), "X_train shape mismatch"
    assert y_train.shape[1:] == (HOUR_STEPS, N_TARGETS), "y_train shape mismatch"
    assert X_test.shape[1:] == (SEQ_LEN_10MIN, N_CHANNELS), "X_test shape mismatch"
    assert y_test.shape[1:] == (HOUR_STEPS, N_TARGETS), "y_test shape mismatch"

    # Train and evaluate
    model = train_and_evaluate(X_train, y_train, X_test, y_test)

    # Example: RH drift detection on a sample of test segments + overlap consensus
    alarms_per_segment = []
    for i in range(min(50, X_test.shape[0])):
        x = X_test[i]
        y = y_test[i]
        res = rh_residuals(model, x, y)
        alarms = page_hinkley(res, delta=PH_DELTA, lam=PH_LAM, alpha=PH_ALPHA)
        alarms_per_segment.append(alarms)
    consensus_alarms = overlap_consensus(alarms_per_segment, overlap_hours=OVERLAP_HOURS,
                                         margin_hours=CONSENSUS_MARGIN_HOURS)
    print(f"Consensus RH drift alarms across segments (first 50): {consensus_alarms[:10]}")
