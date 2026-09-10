from pathlib import Path
import os
PKG_ROOT = Path(__file__).resolve().parents[1]
SHARED = PKG_ROOT / 'shared_input'
RSCRIPT = os.environ.get('RSCRIPT', 'Rscript')
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
import copy
import warnings
warnings.filterwarnings('ignore')
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, GridSearchCV
from xgboost import XGBClassifier
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.weight'] = 'normal'
matplotlib.rcParams['axes.labelweight'] = 'normal'
matplotlib.rcParams['axes.titleweight'] = 'normal'
SEED = 2025
np.random.seed(SEED)
LAYER = 'Species'
MODEL_NAME = 'XGB'

def save_figure(fig, stem):
    for ext in ('pdf', 'svg', 'png', 'jpg'):
        fig.savefig(os.path.join(OUT, f'{stem}.{ext}'), bbox_inches='tight', dpi=300)
    plt.close(fig)

def save_table(df, stem):
    df.to_csv(os.path.join(OUT, f'{stem}.tsv'), sep='\t', index=False)
    df.to_csv(os.path.join(OUT, f'{stem}.csv'), index=False)

def clean_feat_names(feats):
    return [f.replace('[', '(').replace(']', ')').replace('|', '_').replace('<', '_') for f in feats]
meta = pd.read_csv(os.path.join(OUT, 'meta.csv'))
meta['SampleID'] = meta['SampleID'].astype(str)
meta = meta.set_index('SampleID')
samples = meta.index.to_numpy()
y = (meta['Group'].astype(str).values == 'PCOS').astype(int)
cohort = meta['Cohort'].astype(str).values
cohort_map = {c: i for i, c in enumerate(sorted(np.unique(cohort)))}
cohort_int = np.array([cohort_map[c] for c in cohort])
strat_label = np.array([f'{yi}_{ci}' for yi, ci in zip(y, cohort_int)])
neg_pos_ratio = (y == 0).sum() / max((y == 1).sum(), 1)
pipeline = Pipeline([('clf', XGBClassifier(objective='binary:logistic', scale_pos_weight=neg_pos_ratio, eval_metric='logloss', n_jobs=1, random_state=SEED, verbosity=0))])
param_grid = {'clf__max_depth': [3, 5], 'clf__learning_rate': [0.05, 0.1], 'clf__n_estimators': [100, 300], 'clf__subsample': [0.8, 1.0]}

def load_fold_clr(fold):
    fold_root = os.path.join(FOLD_DIR, LAYER, f'fold{fold}')
    tr = pd.read_csv(os.path.join(fold_root, 'train_clr.csv'))
    te = pd.read_csv(os.path.join(fold_root, 'test_clr.csv'))
    feats = [ln.strip() for ln in open(os.path.join(fold_root, 'features.txt')) if ln.strip()]
    tr = tr.set_index('SampleID').reindex(columns=feats)
    te = te.set_index('SampleID').reindex(columns=feats)
    return (tr.values.astype(np.float32), te.values.astype(np.float32), clean_feat_names(feats), tr.index.to_numpy(), te.index.to_numpy())

def compute_shap_xgb(pipe, X_raw):
    clf = pipe.named_steps['clf']
    exp = shap.TreeExplainer(clf)
    sv = exp.shap_values(X_raw)
    return (np.asarray(sv), X_raw)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
fold_records = []
abs_sum = defaultdict(float)
abs_cnt = defaultdict(int)
mean_sum = defaultdict(float)
print(f'[SHAP] {LAYER} | {MODEL_NAME}')
for fold, (train_idx, test_idx) in enumerate(outer_cv.split(np.zeros(len(y)), strat_label)):
    X_tr, X_te, feats, id_tr, id_te = load_fold_clr(fold)
    y_tr = (meta.loc[id_tr, 'Group'].astype(str).values == 'PCOS').astype(int)
    coh_tr = np.array([cohort_map[c] for c in meta.loc[id_tr, 'Cohort'].astype(str)])
    inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    gs = GridSearchCV(copy.deepcopy(pipeline), param_grid, cv=inner_cv, scoring='roc_auc', n_jobs=4, refit=True)
    gs.fit(X_tr, y_tr, groups=coh_tr)
    pipe = gs.best_estimator_
    print(f'  fold{fold}: n_train={X_tr.shape[0]} feats={X_tr.shape[1]} best={gs.best_params_}')
    sv, X_plot = compute_shap_xgb(pipe, X_tr)
    mas = np.abs(sv).mean(axis=0)
    ms = sv.mean(axis=0)
    for j, f in enumerate(feats):
        abs_sum[f] += float(mas[j])
        abs_cnt[f] += 1
        mean_sum[f] += float(ms[j])
    fold_records.append({'fold': fold, 'pipeline': pipe, 'X_tr': X_tr, 'feats': feats, 'sv': sv})
feats_sorted = sorted(abs_cnt.keys(), key=lambda f: abs_sum[f] / abs_cnt[f], reverse=True)
mean_abs = np.array([abs_sum[f] / abs_cnt[f] for f in feats_sorted])
mean_shap = np.array([mean_sum[f] / abs_cnt[f] for f in feats_sorted])
fi = pd.DataFrame({'Feature': feats_sorted, 'Mean_AbsSHAP': mean_abs, 'Mean_SHAP': mean_shap, 'Direction': np.where(mean_shap > 0, 'PCOS-enriched', 'Healthy-enriched')})
fi.to_csv(os.path.join(OUT, f'feature_importance_{LAYER}.csv'), index=False)
save_table(fi, f'feature_importance_{LAYER}')
top20 = feats_sorted[:20]
mas20 = mean_abs[:20]
msh20 = mean_shap[:20]
fig, ax = plt.subplots(figsize=(8, 6.5))
order = list(range(20))[::-1]
colors_bar = ['#D6604D' if msh20[i] > 0 else '#4393C3' for i in order]
ax.barh(range(20), mas20[order], color=colors_bar, edgecolor='white', alpha=0.85)
ax.set_yticks(range(20))
ax.set_yticklabels([top20[i] for i in order], fontsize=9, style='italic')
ax.set_xlabel('Mean |SHAP| (avg across outer folds)')
ax.set_title(f'Top 20 features — {LAYER} | {MODEL_NAME}')
ax.grid(axis='x', alpha=0.3)
ax.legend(handles=[mpatches.Patch(color='#D6604D', alpha=0.85, label='Higher in PCOS'), mpatches.Patch(color='#4393C3', alpha=0.85, label='Higher in Healthy')], fontsize=9, loc='lower right', frameon=False)
fig.tight_layout()
save_figure(fig, f'Fig6_SHAP_importance_{LAYER}')
save_table(pd.DataFrame({'Feature': top20, 'Mean_AbsSHAP': mas20, 'Mean_SHAP': msh20}), f'Fig6_SHAP_importance_{LAYER}')
print(f'  wrote Fig6_SHAP_importance_{LAYER}')
rec_last = fold_records[-1]
sv_last = rec_last['sv']
X_last = rec_last['X_tr']
feats_last = rec_last['feats']
idx_map, names_plot = ([], [])
for f in top20:
    if f in feats_last:
        idx_map.append(feats_last.index(f))
        names_plot.append(f)
if len(idx_map) < 5:
    raise RuntimeError(f'Too few top20 features present in last fold: {len(idx_map)}')
plt.figure(figsize=(9, 7))
shap.summary_plot(sv_last[:, idx_map], X_last[:, idx_map], feature_names=names_plot, plot_type='dot', show=False, max_display=len(names_plot))
plt.title(f'SHAP beeswarm — {LAYER} | {MODEL_NAME} (outer fold example)', fontsize=11)
plt.tight_layout()
fig = plt.gcf()
ax = plt.gca()
for lab in ax.get_yticklabels():
    lab.set_fontstyle('italic')
    lab.set_fontweight('normal')
save_figure(fig, f'Fig5_SHAP_beeswarm_{LAYER}')
rows = []
for j, fname in enumerate(names_plot):
    for i in range(sv_last.shape[0]):
        rows.append({'Feature': fname, 'SHAP': float(sv_last[i, idx_map[j]]), 'FeatureValue': float(X_last[i, idx_map[j]])})
save_table(pd.DataFrame(rows), f'Fig5_SHAP_beeswarm_{LAYER}')
print(f'  wrote Fig5_SHAP_beeswarm_{LAYER}')
print('done')
