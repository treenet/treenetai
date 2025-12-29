# multichannel_pipeline_tf.py
# -*- coding: utf-8 -*-
"""
Multichannel Time-Series Pipeline (TensorFlow/Keras) with Per-Site Lag Correction
and Segment-Level Provenance Export

Author: M365 Copilot (for Mirko Lukovic)
Date: 2025-12-28

This pipeline:
- Applies per-site fixed lag correction using diagnostics from `lag_diagnostic.py`
- Cleans inputs (Hampel; optional LOWESS)
- Trains a TCN-style multitask model (10-min reconstruction + hourly targets)
- Evaluates against baselines
- Detects RH drift via Page–Hinkley with overlap consensus
- Exports segment-level provenance CSVs for train and test splits
"""

from __future__ import annotations
import os
import typing as t
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

# =====================
# Configuration
# =====================
SEQ_LEN_10MIN: int = 4320
HOUR_STEPS: int = 720
STRIDE_PER_HOUR: int = 6
N_CHANNELS: int = 11
N_TARGETS: int = 3

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

AUGMENT_WITH_GAPS: bool = True
GAP_CHANNEL_PROB: float = 0.5
N_GAPS_RANGE: t.Tuple[int, int] = (1, 3)
GAP_LEN_DAYS_RANGE: t.Tuple[int, int] = (1, 12)
RNG = np.random.default_rng(42)

USE_HAMPEL_FILTER: bool = True
HAMPEL_WINDOW: int = 13
HAMPEL_SIGMAS: float = 3.0
USE_LOWESS_DETREND: bool = False
LOWESS_FRAC: float = 0.02
LOWESS_IT: int = 3

USE_MISALIGNMENT_CORRECTION: bool = False
APPLY_SITE_SPECIFIC_LAG: bool = True
LAG_POLICY_FILE: str = os.path.join('diagnostics', 'lag_by_site.csv')

W_RECON_MASKED: float = 1.0
W_RECON_UNMASKED: float = 0.05
W_HOURLY: float = 1.0

BATCH_SIZE: int = 32
LR: float = 3e-4
EPOCHS: int = 100
EARLY_STOP_PATIENCE: int = 10
REDUCE_LR_PATIENCE: int = 4
MIN_LR: float = 1e-6

PH_DELTA: float = 0.003
PH_LAM: float = 0.02
PH_ALPHA: float = 30.0

OVERLAP_HOURS: int = 240
CONSENSUS_MARGIN_HOURS: int = 6

# =====================
# Utilities
# =====================

def inject_random_gaps(x: np.ndarray, rng=RNG) -> t.Tuple[np.ndarray, np.ndarray]:
    x = x.copy().astype(np.float32)
    mask = np.ones_like(x, dtype=np.float32)
    n_gaps = rng.integers(N_GAPS_RANGE[0], N_GAPS_RANGE[1] + 1)
    chosen_channels = [c for c in range(x.shape[1]) if rng.random() < GAP_CHANNEL_PROB]
    if not chosen_channels:
        chosen_channels = [rng.integers(0, x.shape[1])]
    for _ in range(n_gaps):
        ch = rng.choice(chosen_channels)
        gap_len_days = rng.integers(GAP_LEN_DAYS_RANGE[0], GAP_LEN_DAYS_RANGE[1] + 1)
        gap_len = min(gap_len_days * 24 * 6, SEQ_LEN_10MIN - 1)
        start = rng.integers(0, SEQ_LEN_10MIN - gap_len)
        end = start + gap_len
        x[start:end, ch] = 0.0
        mask[start:end, ch] = 0.0
    return x, mask


def hampel_filter(x: np.ndarray, window_size: int = HAMPEL_WINDOW, n_sigmas: float = HAMPEL_SIGMAS) -> np.ndarray:
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
    T = len(x)
    xi = np.arange(T)
    if HAS_STATSMODELS:
        trend = sm.nonparametric.lowess(x, xi, frac=frac, it=it, return_sorted=False)
    else:
        win = max(5, int(frac * T))
        if win % 2 == 0:
            win += 1
        pad = win // 2
        xpad = np.pad(x, (pad, pad), mode='edge')
        trend = np.array([np.median(xpad[i:i+win]) for i in range(T)])
    x_detr = x - trend
    return x_detr.astype(np.float32), trend.astype(np.float32)


def shift_series(x: np.ndarray, lag: int) -> np.ndarray:
    T = len(x)
    y = np.zeros_like(x)
    if lag > 0:
        y[lag:] = x[:T - lag]
    elif lag < 0:
        y[:T + lag] = x[-lag:]
    else:
        y[:] = x
    return y

# =====================
# Per-site lag policy
# =====================

def load_lag_policy(path: str) -> dict[int, dict[str, int]]:
    if not os.path.exists(path):
        print(f"Lag policy file not found: {path}. Skipping per-site lag correction.")
        return {}
    df = pd.read_csv(path)
    lag_map: dict[int, dict[str, int]] = {}
    for _, row in df.iterrows():
        try:
            sid = int(row['site_id'])
        except Exception:
            continue
        pair = str(row['pair'])
        rec = str(row.get('recommendation', 'SKIP_CORRECTION'))
        lag_steps = int(round(row.get('median_lag_steps', 0)))
        if rec == 'APPLY_FIXED_LAG_CORRECTION':
            d = lag_map.setdefault(sid, {})
            d[pair] = lag_steps
    print(f"Loaded lag policy for {len(lag_map)} sites from {path}.")
    return lag_map

PAIR_TO_LOCAL_CH = {
    'T_vs_Tmean': CH['local_T_10m'],
    'RH_vs_RHdaily': CH['local_RH_10m'],
    'STEM_vs_RAD': CH['local_stem_10m'],
}

# =====================
# Segment-level provenance utilities
# =====================

def compute_hampel_counts(x: np.ndarray, channels: t.Sequence[int]) -> dict:
    counts = {}
    for ch in channels:
        flags = hampel_filter(x[:, ch], window_size=HAMPEL_WINDOW, n_sigmas=HAMPEL_SIGMAS)
        counts[ch] = int(np.sum(flags))
    return counts


def segment_provenance_row(x_10m: np.ndarray, site_id: int, lag_map: dict[int, dict[str, int]],
                           training: bool = True) -> dict:
    x = x_10m.copy().astype(np.float32)
    sid = int(site_id)

    lag_T = lag_map.get(sid, {}).get('T_vs_Tmean', None)
    lag_RH = lag_map.get(sid, {}).get('RH_vs_RHdaily', None)
    lag_STEM = lag_map.get(sid, {}).get('STEM_vs_RAD', None)

    if lag_T is not None:
        x[:, CH['local_T_10m']] = shift_series(x[:, CH['local_T_10m']], int(lag_T))
    if lag_RH is not None:
        x[:, CH['local_RH_10m']] = shift_series(x[:, CH['local_RH_10m']], int(lag_RH))
    if lag_STEM is not None:
        x[:, CH['local_stem_10m']] = shift_series(x[:, CH['local_stem_10m']], int(lag_STEM))

    hampel_counts = compute_hampel_counts(x, [CH['local_T_10m'], CH['local_RH_10m'], CH['local_stem_10m']])

    x_pc, mask_pc, _ = apply_preclean(x, use_hampel=USE_HAMPEL_FILTER, use_lowess=USE_LOWESS_DETREND,
                                      lowess_channels=[CH['local_RH_10m']] if USE_LOWESS_DETREND else None)

    aug_applied = bool(training and AUGMENT_WITH_GAPS)
    aug_steps = 0
    mask_final = mask_pc
    if aug_applied:
        x_masked, aug_mask = inject_random_gaps(x_pc)
        mask_final = (mask_pc * aug_mask).astype(np.float32)
        aug_steps = int(np.sum(mask_pc == 1.0) - np.sum(mask_final == 1.0))

    frac_imputed_total = float(np.mean(1.0 - mask_final))

    row = {
        'site_id': sid,
        'lag_applied_T': bool(lag_T is not None),
        'lag_steps_T': int(lag_T) if lag_T is not None else 0,
        'lag_applied_RH': bool(lag_RH is not None),
        'lag_steps_RH': int(lag_RH) if lag_RH is not None else 0,
        'lag_applied_STEM': bool(lag_STEM is not None),
        'lag_steps_STEM': int(lag_STEM) if lag_STEM is not None else 0,
        'hampel_spikes_T': int(hampel_counts.get(CH['local_T_10m'], 0)),
        'hampel_spikes_RH': int(hampel_counts.get(CH['local_RH_10m'], 0)),
        'hampel_spikes_STEM': int(hampel_counts.get(CH['local_stem_10m'], 0)),
        'lowess_detrend_RH': bool(USE_LOWESS_DETREND),
        'augmentation_applied': aug_applied,
        'aug_total_steps_masked': aug_steps,
        'frac_imputed_total': frac_imputed_total,
    }
    return row


def export_provenance_table(X: np.ndarray, site_ids: np.ndarray, split_name: str, lag_map: dict[int, dict[str, int]],
                            training: bool) -> str:
    rows = []
    for i in range(X.shape[0]):
        r = segment_provenance_row(X[i], int(site_ids[i]) if site_ids is not None else -1, lag_map, training=training)
        r['segment_idx'] = i
        rows.append(r)
    df = pd.DataFrame(rows)
    out_dir = os.path.join('diagnostics')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'provenance_segments_{split_name}.csv')
    df.to_csv(out_path, index=False)
    print(f"Wrote segment-level provenance to {out_path} (rows={len(df)})")
    return out_path

# =====================
# Dataset builders
# =====================

def apply_site_fixed_lag(x: np.ndarray, site_id: int, lag_map: dict[int, dict[str, int]]) -> np.ndarray:
    if site_id in lag_map:
        for pair_name, lag_steps in lag_map[site_id].items():
            ch_idx = PAIR_TO_LOCAL_CH.get(pair_name)
            if ch_idx is not None:
                x[:, ch_idx] = shift_series(x[:, ch_idx], int(lag_steps))
    return x


def apply_preclean(x: np.ndarray,
                   use_hampel: bool = USE_HAMPEL_FILTER,
                   use_lowess: bool = USE_LOWESS_DETREND,
                   lowess_channels: t.Optional[t.Sequence[int]] = None) -> t.Tuple[np.ndarray, np.ndarray, dict]:
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


def make_example(x_10m: np.ndarray,
                 y_hourly: np.ndarray,
                 site_id: int,
                 lag_map: dict[int, dict[str, int]],
                 training: bool = True,
                 preclean: bool = True) -> t.Tuple[dict, dict, dict]:
    x = x_10m.astype(np.float32)
    y = y_hourly.astype(np.float32)

    if APPLY_SITE_SPECIFIC_LAG and lag_map:
        x = apply_site_fixed_lag(x, int(site_id), lag_map)

    if preclean:
        x, mask_clean, _ = apply_preclean(
            x,
            use_hampel=USE_HAMPEL_FILTER,
            use_lowess=USE_LOWESS_DETREND,
            lowess_channels=[CH['local_RH_10m']] if USE_LOWESS_DETREND else None,
        )
    else:
        mask_clean = np.ones_like(x, dtype=np.float32)

    if USE_MISALIGNMENT_CORRECTION:
        pass

    if training and AUGMENT_WITH_GAPS:
        x_masked, aug_mask = inject_random_gaps(x)
        mask = (mask_clean * aug_mask).astype(np.float32)
    else:
        x_masked = x
        mask = mask_clean

    w_recon = W_RECON_UNMASKED * np.ones_like(mask, dtype=np.float32)
    w_recon[mask == 0.0] = W_RECON_MASKED
    w_hourly = W_HOURLY * np.ones_like(y, dtype=np.float32)

    inputs = {'x_in': x_masked, 'mask_in': mask}
    outputs = {'recon': x, 'hourly': y}
    sample_weights = {'recon': w_recon, 'hourly': w_hourly}
    return inputs, outputs, sample_weights


def make_tf_dataset(X: np.ndarray,
                    Y: np.ndarray,
                    site_ids: np.ndarray,
                    lag_map: dict[int, dict[str, int]],
                    batch_size: int = BATCH_SIZE,
                    training: bool = True,
                    shuffle: bool = True,
                    preclean: bool = True) -> tf.data.Dataset:
    def gen():
        for i in range(X.shape[0]):
            sid = int(site_ids[i]) if site_ids is not None else -1
            inp, out, sw = make_example(X[i], Y[i], sid, lag_map, training=training, preclean=preclean)
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
# Model
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
    z = layers.Concatenate(axis=-1, name='concat_mask')([x_in, m_in])
    z = layers.Conv1D(64, kernel_size=5, padding='same', name='emb_conv')(z)
    z = layers.Activation('gelu')(z)
    for i, d in enumerate([1, 2, 4, 8, 16, 32]):
        z = TemporalBlock(128, dilation_rate=d, dropout=0.1, name=f'block{i+1}')(z)
    feat_10m = layers.Conv1D(128, kernel_size=1, padding='same', name='shared_1x1')(z)
    feat_10m = layers.Activation('gelu')(feat_10m)
    recon = layers.Conv1D(N_CHANNELS, kernel_size=1, padding='same', name='recon_head')(feat_10m)
    hourly_feat = layers.AveragePooling1D(pool_size=STRIDE_PER_HOUR, strides=STRIDE_PER_HOUR,
                                          padding='valid', name='to_hourly')(feat_10m)
    hourly_feat = layers.Conv1D(64, kernel_size=1, activation='gelu', name='hourly_proj')(hourly_feat)
    hourly = layers.Conv1D(N_TARGETS, kernel_size=1, padding='same', name='hourly_head')(hourly_feat)
    return Model(inputs={'x_in': x_in, 'mask_in': m_in}, outputs={'recon': recon, 'hourly': hourly},
                 name='multitask_impute_hourly')


def masked_mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# =====================
# Baselines & Drift
# =====================

def linear_interp_impute(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    x_imp = x.copy()
    T, C = x.shape
    idx = np.arange(T)
    for c in range(C):
        m = mask[:, c].astype(bool)
        if m.all():
            continue
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
    y_pred = y_true.copy()
    y_pred[1:] = y_true[:-1]
    return y_pred


def page_hinkley(residuals: np.ndarray, delta: float = PH_DELTA, lam: float = PH_LAM, alpha: float = PH_ALPHA) -> np.ndarray:
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
    pred = model.predict({'x_in': x_10m[None, ...], 'mask_in': np.ones_like(x_10m)[None, ...]}, verbose=0)
    rh_true = y_hourly[:, 1]
    rh_hat = pred['hourly'][0, :, 1]
    return (rh_true - rh_hat)


def overlap_consensus(alarms_per_segment: t.List[np.ndarray], overlap_hours: int = OVERLAP_HOURS,
                      margin_hours: int = CONSENSUS_MARGIN_HOURS) -> t.List[t.Tuple[int, int]]:
    consensus = []
    for i in range(len(alarms_per_segment) - 1):
        a_i = alarms_per_segment[i]
        a_j = alarms_per_segment[i + 1]
        if a_i.size == 0 or a_j.size == 0:
            continue
        for h_i in a_i:
            if h_i >= (HOUR_STEPS - overlap_hours):
                for h_j in a_j:
                    if h_j <= overlap_hours and abs((h_i - (HOUR_STEPS - overlap_hours)) - h_j) <= margin_hours:
                        consensus.append((i, int(h_i)))
                        break
    return consensus

# =====================
# Training / Evaluation
# =====================

def train_and_evaluate(X_train: np.ndarray, y_train: np.ndarray, site_ids_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray, site_ids_test: t.Optional[np.ndarray] = None) -> tf.keras.Model:
    # Load lag policy
    lag_map = load_lag_policy(LAG_POLICY_FILE) if APPLY_SITE_SPECIFIC_LAG else {}

    # Export provenance for TRAIN (pre-training)
    export_provenance_table(X_train, site_ids_train, split_name='train', lag_map=lag_map, training=True)

    # Split train/val
    N = X_train.shape[0]
    idx = np.arange(N)
    np.random.seed(123)
    np.random.shuffle(idx)
    split = int(0.8 * N)
    train_idx, val_idx = idx[:split], idx[split:]

    X_tr, Y_tr, SID_tr = X_train[train_idx], y_train[train_idx], site_ids_train[train_idx]
    X_val, Y_val, SID_val = X_train[val_idx], y_train[val_idx], site_ids_train[val_idx]

    ds_tr = make_tf_dataset(X_tr, Y_tr, SID_tr, lag_map, batch_size=BATCH_SIZE, training=True, shuffle=True, preclean=True)
    ds_val = make_tf_dataset(X_val, Y_val, SID_val, lag_map, batch_size=BATCH_SIZE, training=False, shuffle=False, preclean=True)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss={'recon': masked_mse, 'hourly': 'mse'},
        metrics={'recon': [tf.keras.metrics.MeanAbsoluteError(name='mae')],
                 'hourly': [tf.keras.metrics.MeanSquaredError(name='mse'),
                            tf.keras.metrics.MeanAbsoluteError(name='mae')]},
        loss_weights={'recon': 1.0, 'hourly': 1.0}
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_hourly_mse', patience=EARLY_STOP_PATIENCE, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_hourly_mse', factor=0.5, patience=REDUCE_LR_PATIENCE, min_lr=MIN_LR),
        tf.keras.callbacks.ModelCheckpoint('best_multitask.weights.h5', monitor='val_hourly_mse', save_best_only=True, save_weights_only=True)
    ]

    history = model.fit(ds_tr, validation_data=ds_val, epochs=EPOCHS, callbacks=callbacks)

    # Evaluate on test
    if site_ids_test is None:
        site_ids_test = np.zeros((X_test.shape[0],), dtype=int)
    ds_test = make_tf_dataset(X_test, y_test, site_ids_test, lag_map, batch_size=BATCH_SIZE, training=False, shuffle=False, preclean=True)
    test_metrics = model.evaluate(ds_test, return_dict=True)
    print("
Test metrics:", test_metrics)

    # Export provenance for TEST (no augmentation)
    export_provenance_table(X_test, site_ids_test, split_name='test', lag_map=lag_map, training=False)

    # Baseline comparison (persistence)
    hourly_mae_model, hourly_mae_pers = [], []
    for i in range(min(256, X_test.shape[0])):
        x = X_test[i]
        sid = int(site_ids_test[i])
        if APPLY_SITE_SPECIFIC_LAG and lag_map:
            x = apply_site_fixed_lag(x, sid, lag_map)
        y = y_test[i]
        pred = model.predict({'x_in': x[None, ...], 'mask_in': np.ones_like(x)[None, ...]}, verbose=0)
        y_hat = pred['hourly'][0]
        hourly_mae_model.append(np.mean(np.abs(y_hat - y)))
        hourly_mae_pers.append(np.mean(np.abs(persistence_hourly(y) - y)))
    print("Hourly MAE — model:", np.mean(hourly_mae_model), "persistence:", np.mean(hourly_mae_pers))

    # Imputation baseline on injected gaps
    errs_model, errs_interp = [], []
    for i in range(min(128, X_test.shape[0])):
        x = X_test[i]
        sid = int(site_ids_test[i])
        if APPLY_SITE_SPECIFIC_LAG and lag_map:
            x = apply_site_fixed_lag(x, sid, lag_map)
        x_pc, mask_pc, _ = apply_preclean(x, use_hampel=USE_HAMPEL_FILTER, use_lowess=USE_LOWESS_DETREND,
                                           lowess_channels=[CH['local_RH_10m']] if USE_LOWESS_DETREND else None)
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
        print("Imputation MAE on injected gaps — model:", np.mean(errs_model), "interp:", np.mean(errs_interp))

    return model

# =====================
# Main
# =====================
if __name__ == "__main__":
    required_files = ['X_train.npy', 'y_train.npy', 'site_ids_train.npy', 'X_test.npy', 'y_test.npy']
    if not all(os.path.exists(f) for f in required_files):
        print("Please provide X_train.npy, y_train.npy, site_ids_train.npy, X_test.npy, y_test.npy.")
        raise SystemExit(0)

    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    site_ids_train = np.load('site_ids_train.npy')
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')

    site_ids_test = None
    if os.path.exists('site_ids_test.npy'):
        site_ids_test = np.load('site_ids_test.npy')

    assert X_train.shape[1:] == (SEQ_LEN_10MIN, N_CHANNELS)
    assert y_train.shape[1:] == (HOUR_STEPS, N_TARGETS)
    assert site_ids_train.shape[0] == X_train.shape[0]
    assert X_test.shape[1:] == (SEQ_LEN_10MIN, N_CHANNELS)
    assert y_test.shape[1:] == (HOUR_STEPS, N_TARGETS)
    if site_ids_test is not None:
        assert site_ids_test.shape[0] == X_test.shape[0]

    model = train_and_evaluate(X_train, y_train, site_ids_train, X_test, y_test, site_ids_test)

    alarms_per_segment = []
    for i in range(min(50, X_test.shape[0])):
        x = X_test[i]
        sid = int(site_ids_test[i]) if site_ids_test is not None else -1
        lag_map = load_lag_policy(LAG_POLICY_FILE) if APPLY_SITE_SPECIFIC_LAG else {}
        if APPLY_SITE_SPECIFIC_LAG and lag_map:
            x = apply_site_fixed_lag(x, sid, lag_map)
        y = y_test[i]
        res = rh_residuals(model, x, y)
        alarms = page_hinkley(res, delta=PH_DELTA, lam=PH_LAM, alpha=PH_ALPHA)
        alarms_per_segment.append(alarms)
    consensus_alarms = overlap_consensus(alarms_per_segment, overlap_hours=OVERLAP_HOURS,
                                         margin_hours=CONSENSUS_MARGIN_HOURS)
    print(f"Consensus RH drift alarms across segments (first 50): {consensus_alarms[:10]}")
