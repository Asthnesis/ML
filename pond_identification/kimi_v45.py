from pathlib import Path
import pandas as pd
import numpy as np
import random
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
import lightgbm as lgb

data_dir = Path.cwd() / "data"
training_data = pd.read_csv(data_dir / 'Train.csv')
testing_data = pd.read_csv(data_dir / 'Test.csv')

train_clean = training_data.replace(-9999, np.nan)
test_clean = testing_data.replace(-9999, np.nan)

bands = ['VH','VV','blue','green','nir','nira','re1','re2','re3','red','swir1','swir2']
months = [f'{i:02d}' for i in range(1, 13)]
y = training_data['label'].values

# --- ROBUST AGGREGATION (median, mean, std only) ---
def aggregate_bands_robust(df, bands):
    out = pd.DataFrame(index=df.index)
    for b in bands:
        cols = [c for c in df.columns if c.startswith(b + '_')]
        vals = df[cols].values
        out[f'{b}_median'] = np.nanmedian(vals, axis=1)
        out[f'{b}_mean']   = np.nanmean(vals, axis=1)
        out[f'{b}_std']    = np.nanstd(vals, axis=1)
    return out

def simulate_test_missingness(raw_df, bands, seed=0):
    rng = random.Random(seed)
    masked = raw_df.copy()
    for idx in masked.index:
        window_len = rng.randint(4, 6)
        start = rng.randint(0, 12 - window_len)
        keep_months = months[start:start + window_len]
        drop_months = [m for m in months if m not in keep_months]
        cols_to_mask = [c for c in masked.columns 
                        if any(c.endswith(f'_{m}') for m in drop_months)]
        masked.loc[idx, cols_to_mask] = np.nan
    return masked

def compute_monthly_indices(df):
    out = pd.DataFrame(index=df.index)
    for m in months:
        nir = df.get(f'nir_{m}')
        red = df.get(f'red_{m}')
        green = df.get(f'green_{m}')
        swir1 = df.get(f'swir1_{m}')
        vh = df.get(f'VH_{m}')
        vv = df.get(f'VV_{m}')
        if nir is not None and red is not None:
            out[f'ndvi_{m}'] = (nir - red) / (nir + red + 1e-8)
        if green is not None and swir1 is not None:
            out[f'mndwi_{m}'] = (green - swir1) / (green + swir1 + 1e-8)
        if green is not None and nir is not None:
            out[f'ndwi_{m}'] = (green - nir) / (green + nir + 1e-8)
        if vh is not None and vv is not None:
            out[f'vh_vv_{m}'] = vh / (vv + 1e-8)
    return out

def aggregate_indices_robust(df_monthly, names):
    out = pd.DataFrame(index=df_monthly.index)
    for name in names:
        cols = [c for c in df_monthly.columns if re.match(rf'{name}_\d{{2}}', c)]
        if not cols:
            continue
        vals = df_monthly[cols].values
        out[f'{name}_median'] = np.nanmedian(vals, axis=1)
        out[f'{name}_mean']   = np.nanmean(vals, axis=1)
        out[f'{name}_std']    = np.nanstd(vals, axis=1)
    return out

idx_names = ['ndvi', 'mndwi', 'ndwi', 'vh_vv']

# Pre-build test features
test_band_agg = aggregate_bands_robust(test_clean, bands)
test_idx_monthly = compute_monthly_indices(test_clean)
test_idx_agg = aggregate_indices_robust(test_idx_monthly, idx_names)
test_final = pd.concat([test_band_agg, test_idx_agg], axis=1)

def build_augmented_training_set(raw_df, y_array, bands, idx_names, n_augments=4, seed=0):
    all_X, all_y = [], []
    for i in range(n_augments):
        masked = simulate_test_missingness(raw_df, bands, seed=seed + i)
        band_agg = aggregate_bands_robust(masked, bands)
        idx_monthly = compute_monthly_indices(masked)
        idx_agg = aggregate_indices_robust(idx_monthly, idx_names)
        all_X.append(pd.concat([band_agg, idx_agg], axis=1))
        all_y.append(y_array)
    X = pd.concat(all_X, axis=0, ignore_index=True)
    return X, np.concatenate(all_y)

# --- 5-FOLD CV WITH LOGGING ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_probs = np.zeros(len(train_clean))
test_probs_fold = np.zeros((5, len(test_clean)))

for fold, (tr_idx, val_idx) in enumerate(skf.split(train_clean, y)):
    print(f"\n=== Fold {fold+1} ===")
    
    raw_tr = train_clean.iloc[tr_idx].copy()
    raw_val = train_clean.iloc[val_idx].copy()
    y_tr_fold, y_val_fold = y[tr_idx], y[val_idx]
    
    X_tr_aug, y_tr_aug = build_augmented_training_set(
        raw_tr, y_tr_fold, bands, idx_names, n_augments=4, seed=fold*100
    )
    
    # Realistic validation
    raw_val_masked = simulate_test_missingness(raw_val, bands, seed=fold*1000)
    val_band_agg = aggregate_bands_robust(raw_val_masked, bands)
    val_idx_monthly = compute_monthly_indices(raw_val_masked)
    val_idx_agg = aggregate_indices_robust(val_idx_monthly, idx_names)
    X_val_real = pd.concat([val_band_agg, val_idx_agg], axis=1)
    
    common_cols = [c for c in X_tr_aug.columns if c in X_val_real.columns]
    X_tr_aug = X_tr_aug[common_cols]
    X_val_real = X_val_real[common_cols]
    
    model = lgb.LGBMClassifier(
        n_estimators=500,
        num_leaves=31,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1
    )
    model.fit(X_tr_aug, y_tr_aug)
    
    val_probs = model.predict_proba(X_val_real)[:, 1]
    oof_probs[val_idx] = val_probs
    prec, rec, thresholds = precision_recall_curve(y, oof_probs)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    best_idx = np.argmax(f1s)
    best_thresh = thresholds[best_idx]
    print(f"Best OOF F1: {f1s[best_idx]:.4f} at threshold {best_thresh:.4f}")
    
    print(f"  Fold AUC: {roc_auc_score(y_val_fold, val_probs):.4f}")
    print(f"  Fold F1  (t=0.5): {f1_score(y_val_fold, val_probs > 0.5):.4f}")
    
    # Feature importance from last fold
    if fold == 4:
        importance = pd.DataFrame({
            'feature': common_cols,
            'gain': model.booster_.feature_importance(importance_type='gain')
        }).sort_values('gain', ascending=False)
        print("\n  Top 10 features:")
        print(importance.head(10).to_string(index=False))
    
    # Test prediction
    X_test = test_final.reindex(columns=common_cols, fill_value=0)
    test_probs_fold[fold] = model.predict_proba(X_test)[:, 1]

# --- ENSEMBLE ---
test_probs = test_probs_fold.mean(axis=0)

print(f"\n{'='*40}")
print(f">>> OOF AUC: {roc_auc_score(y, oof_probs):.4f}")
print(f">>> OOF F1 (t=0.5): {f1_score(y, oof_probs > 0.5):.4f}")
print(f"{'='*40}")

# --- GENERATE TWO SUBMISSIONS ---
for thresh, suffix in [(0.50, 't50'), (0.45, 't45')]:
    test_preds = (test_probs > thresh).astype(int)
    submission = pd.DataFrame({
        'ID': testing_data['ID'],
        'TargetF1': test_preds,
        'TargetRAUC': test_probs
    })
    fname = f'submission_{suffix}.csv'
    submission.to_csv(fname, index=False)
    print(f"\nSaved: {fname} (threshold = {thresh})")
    print(submission.head(3).to_string(index=False))