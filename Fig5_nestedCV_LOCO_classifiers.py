from pathlib import Path
import os
PKG_ROOT = Path(__file__).resolve().parents[1]
SHARED = PKG_ROOT / 'shared_input'
RSCRIPT = os.environ.get('RSCRIPT', 'Rscript')
ROOT = str(SHARED)
OUT = str(PKG_ROOT / 'output' / 'Fig5_ml')
FOLD_DIR = str(Path(OUT) / 'fold_corrected')
R_HELPER = str(PKG_ROOT / 'code' / 'Fig5_mmuphin_fit_apply.R')
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
import copy
import json
import subprocess
import tempfile
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, accuracy_score, roc_curve, confusion_matrix
from xgboost import XGBClassifier
import shap
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.weight'] = 'normal'
matplotlib.rcParams['axes.labelweight'] = 'normal'
matplotlib.rcParams['axes.titleweight'] = 'normal'
SEED = 2025
np.random.seed(SEED)
os.makedirs(OUT, exist_ok=True)
os.makedirs(FOLD_DIR, exist_ok=True)

def save_figure(fig, stem):
    for ext in ('pdf', 'svg', 'png', 'jpg'):
        path = os.path.join(OUT, f'{stem}.{ext}')
        fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)

def save_tsv(df, stem):
    path = os.path.join(OUT, f'{stem}.tsv')
    df.to_csv(path, sep='\t', index=False)
    return path

def load_abd_table(path):
    df = pd.read_csv(path, sep='\t', index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df

def build_meta():
    sgb_info = pd.read_csv(os.path.join(ROOT, 'sgb_genome_annotation.tsv'), sep='\t')
    sgb_info['SampleID'] = sgb_info['ID'].astype(str).str.replace('_bin\\..*$', '', regex=True)
    meta = sgb_info[['SampleID', 'BIOPROJECT', 'Group']].drop_duplicates('SampleID').dropna(subset=['Group']).rename(columns={'BIOPROJECT': 'Cohort'})
    meta = meta.set_index('SampleID')
    return meta
genus_raw = load_abd_table(os.path.join(ROOT, 'metaphlan_genus_abundance.xls'))
species_raw = load_abd_table(os.path.join(ROOT, 'metaphlan_species_abundance.tsv'))
sgb_raw = load_abd_table(os.path.join(ROOT, 'sgb_bin_abundance.tsv'))
meta_all = build_meta()
common = sorted(set(genus_raw.columns) & set(species_raw.columns) & set(sgb_raw.columns) & set(meta_all.index))
print(f'Common samples: {len(common)}')
genus_raw = genus_raw[common]
species_raw = species_raw[common]
sgb_raw = sgb_raw[common]
meta = meta_all.loc[common].copy()
meta['Group'] = pd.Categorical(meta['Group'], categories=['Healthy', 'PCOS'])
meta['Cohort'] = meta['Cohort'].astype(str)
abd_paths = {}
for name, mat in [('Genus', genus_raw), ('Species', species_raw), ('SGB', sgb_raw)]:
    p = os.path.join(OUT, f'raw_{name}.tsv')
    mat.to_csv(p, sep='\t')
    abd_paths[name] = p
meta_csv = os.path.join(OUT, 'meta.csv')
meta.reset_index().to_csv(meta_csv, index=False)
y = (meta['Group'].astype(str).values == 'PCOS').astype(int)
cohort = meta['Cohort'].values
samples = np.array(common)
cohort_map = {c: i for i, c in enumerate(sorted(np.unique(cohort)))}
cohort_int = np.array([cohort_map[c] for c in cohort])
strat_label = np.array([f'{yi}_{ci}' for yi, ci in zip(y, cohort_int)])
print(pd.crosstab(meta['Cohort'], meta['Group']))
print(f'Healthy/PCOS ratio = {(y == 0).sum() / (y == 1).sum():.3f}')
neg_pos_ratio = (y == 0).sum() / max((y == 1).sum(), 1)
models = {'RF': Pipeline([('clf', RandomForestClassifier(n_estimators=300, class_weight='balanced', n_jobs=1, random_state=SEED))]), 'LASSO': Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', max_iter=2000, random_state=SEED))]), 'XGB': Pipeline([('clf', XGBClassifier(objective='binary:logistic', scale_pos_weight=neg_pos_ratio, eval_metric='logloss', n_jobs=1, random_state=SEED, verbosity=0))]), 'SVM': Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=SEED))]), 'LR': Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(penalty='l2', solver='lbfgs', class_weight='balanced', max_iter=2000, random_state=SEED))])}
param_grids = {'RF': {'clf__max_features': ['sqrt', 'log2'], 'clf__min_samples_leaf': [1, 3, 5], 'clf__n_estimators': [100, 300]}, 'LASSO': {'clf__C': [0.001, 0.01, 0.1, 1.0, 10.0]}, 'XGB': {'clf__max_depth': [3, 5], 'clf__learning_rate': [0.05, 0.1], 'clf__n_estimators': [100, 300], 'clf__subsample': [0.8, 1.0]}, 'SVM': {'clf__C': [0.1, 1.0, 10.0, 100.0], 'clf__gamma': ['scale', 'auto', 0.01, 0.001]}, 'LR': {'clf__C': [0.001, 0.01, 0.1, 1.0, 10.0]}}
MODEL_ORDER = ['RF', 'LASSO', 'XGB', 'SVM', 'LR']
model_colors = {'RF': '#E41A1C', 'LASSO': '#377EB8', 'XGB': '#FF7F00', 'SVM': '#984EA3', 'LR': '#4DAF4A'}
LAYERS = ['Genus', 'Species', 'SGB']

def clean_feat_names(feats):
    return [f.replace('[', '(').replace(']', ')').replace('|', '_').replace('<', '_') for f in feats]

def run_mmuphin_fold(layer, fold_id, train_ids, test_ids):
    fold_root = os.path.join(FOLD_DIR, layer, f'fold{fold_id}')
    os.makedirs(fold_root, exist_ok=True)
    train_ids_path = os.path.join(fold_root, 'train_ids.txt')
    test_ids_path = os.path.join(fold_root, 'test_ids.txt')
    out_train = os.path.join(fold_root, 'train_clr.csv')
    out_test = os.path.join(fold_root, 'test_clr.csv')
    out_feats = os.path.join(fold_root, 'features.txt')
    if os.path.exists(out_train) and os.path.exists(out_test) and os.path.exists(out_feats):
        return (out_train, out_test, out_feats)
    with open(train_ids_path, 'w') as f:
        f.write('\n'.join(train_ids) + '\n')
    with open(test_ids_path, 'w') as f:
        f.write('\n'.join(test_ids) + '\n')
    cmd = [RSCRIPT, R_HELPER, abd_paths[layer], meta_csv, train_ids_path, test_ids_path, out_train, out_test, out_feats]
    print('  R:', ' '.join(cmd[-7:]))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise RuntimeError(f'MMUPHin fold failed: {layer} fold{fold_id}')
    print('   ', proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else 'OK')
    return (out_train, out_test, out_feats)

def load_fold_clr(out_train, out_test, out_feats):
    tr = pd.read_csv(out_train)
    te = pd.read_csv(out_test)
    feats = [ln.strip() for ln in open(out_feats) if ln.strip()]
    feats_c = clean_feat_names(feats)
    feat_cols_tr = [c for c in tr.columns if c != 'SampleID']
    feat_cols_te = [c for c in te.columns if c != 'SampleID']
    tr = tr.set_index('SampleID')
    te = te.set_index('SampleID')
    tr = tr.reindex(columns=feats)
    te = te.reindex(columns=feats)
    X_tr = tr.values.astype(np.float32)
    X_te = te.values.astype(np.float32)
    return (X_tr, X_te, feats_c, tr.index.to_numpy(), te.index.to_numpy())

def nested_cv_leakage_safe(layer, model_name, pipeline, param_grid, n_outer=5, n_inner=3):
    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=SEED)
    oof_probs = np.zeros(len(y), dtype=float)
    oof_preds = np.zeros(len(y), dtype=int)
    best_params_list = []
    fold_aucs = []
    fold_shap_records = []
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(np.zeros(len(y)), strat_label)):
        train_ids = samples[train_idx].tolist()
        test_ids = samples[test_idx].tolist()
        out_train, out_test, out_feats = run_mmuphin_fold(layer, fold, train_ids, test_ids)
        X_tr, X_te, feats, id_tr, id_te = load_fold_clr(out_train, out_test, out_feats)
        y_tr = (meta.loc[id_tr, 'Group'].astype(str).values == 'PCOS').astype(int)
        y_te = (meta.loc[id_te, 'Group'].astype(str).values == 'PCOS').astype(int)
        coh_tr = np.array([cohort_map[c] for c in meta.loc[id_tr, 'Cohort'].astype(str)])
        inner_cv = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=42)
        gs = GridSearchCV(copy.deepcopy(pipeline), param_grid, cv=inner_cv, scoring='roc_auc', n_jobs=4, refit=True)
        gs.fit(X_tr, y_tr, groups=coh_tr)
        best_params_list.append(gs.best_params_)
        fold_prob = gs.predict_proba(X_te)[:, 1]
        fold_pred = gs.predict(X_te)
        pos = {s: i for i, s in enumerate(samples)}
        for sid, pr, pd_ in zip(id_te, fold_prob, fold_pred):
            oof_probs[pos[sid]] = pr
            oof_preds[pos[sid]] = pd_
        fold_auc = float('nan')
        if len(np.unique(y_te)) == 2:
            fold_auc = float(roc_auc_score(y_te, fold_prob))
            fold_aucs.append(fold_auc)
        print(f'    [{layer}|{model_name}] fold{fold} AUC={fold_auc:.4f} feats={X_tr.shape[1]}')
        fold_shap_records.append({'fold': fold, 'pipeline': gs.best_estimator_, 'X_tr': X_tr, 'feats': feats, 'y_tr': y_tr})
    metrics = {'AUC': roc_auc_score(y, oof_probs), 'AUC_std': float(np.std(fold_aucs, ddof=1)) if len(fold_aucs) > 1 else 0.0, 'F1': f1_score(y, oof_preds), 'MCC': matthews_corrcoef(y, oof_preds), 'ACC': accuracy_score(y, oof_preds)}
    return (metrics, oof_probs, best_params_list, fold_aucs, fold_shap_records)
print('\n' + '=' * 60)
print('Leakage-safe nested CV (MMUPHin inside each outer fold)')
print('=' * 60)
outer_cv_pre = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
print('\n--- Precomputing fold-wise MMUPHin corrections ---')
for layer in LAYERS:
    for fold, (train_idx, test_idx) in enumerate(outer_cv_pre.split(np.zeros(len(y)), strat_label)):
        run_mmuphin_fold(layer, fold, samples[train_idx].tolist(), samples[test_idx].tolist())
all_results = []
shap_cache = {}
for layer in LAYERS:
    for model_name, pipeline in models.items():
        print(f'\n  [{layer}] {model_name} ...')
        metrics, oof_probs, best_params, fold_aucs, fold_shap = nested_cv_leakage_safe(layer, model_name, copy.deepcopy(pipeline), param_grids[model_name])
        print(f"    AUC={metrics['AUC']:.4f}±{metrics['AUC_std']:.4f}  F1={metrics['F1']:.4f}  MCC={metrics['MCC']:.4f}  ACC={metrics['ACC']:.4f}")
        all_results.append({'Layer': layer, 'Model': model_name, **metrics, 'oof_probs': oof_probs, 'fold_aucs': fold_aucs, 'best_params': best_params})
        shap_cache[layer, model_name] = fold_shap
results_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ('oof_probs', 'fold_aucs', 'best_params')} for r in all_results])
results_df.to_csv(os.path.join(OUT, 'results_summary.csv'), index=False)
save_tsv(results_df, 'Table1_results_summary')
print('\n=== AUC pivot ===')
print(results_df.pivot_table(index='Model', columns='Layer', values='AUC').round(4))
orig_path = os.path.join(ROOT, '..', 'sgb_machine_learning', 'ml_output', 'results_summary.csv')
if os.path.exists(orig_path):
    orig = pd.read_csv(orig_path)
    cmp = results_df.merge(orig[['Layer', 'Model', 'AUC']], on=['Layer', 'Model'], suffixes=('_nested', '_original'))
    cmp = cmp.rename(columns={'AUC': 'AUC_nested'})
    if 'AUC_original' not in cmp.columns:
        pass
    cmp2 = results_df[['Layer', 'Model', 'AUC']].rename(columns={'AUC': 'AUC_nested'}).merge(orig[['Layer', 'Model', 'AUC']].rename(columns={'AUC': 'AUC_original'}), on=['Layer', 'Model'])
    cmp2['delta_AUC'] = cmp2['AUC_nested'] - cmp2['AUC_original']
    cmp2.to_csv(os.path.join(OUT, 'auc_original_vs_nested.csv'), index=False)
    save_tsv(cmp2, 'TableS_auc_original_vs_nested')
    print('\n=== Original vs nested AUC ===')
    print(cmp2.to_string(index=False))
model_styles = {'RF': 'solid', 'LASSO': 'dashed', 'XGB': 'dotted', 'SVM': 'dashdot', 'LR': (0, (3, 1, 1, 1))}
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
roc_rows = []
for ax, layer in zip(axes, LAYERS):
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5)
    for r in all_results:
        if r['Layer'] != layer:
            continue
        fpr, tpr, _ = roc_curve(y, r['oof_probs'])
        ax.plot(fpr, tpr, linestyle=model_styles[r['Model']], color=model_colors[r['Model']], lw=1.8, label=f"{r['Model']} (AUC={r['AUC']:.3f})")
        for fp, tp in zip(fpr, tpr):
            roc_rows.append({'Layer': layer, 'Model': r['Model'], 'FPR': fp, 'TPR': tp, 'AUC': r['AUC']})
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate', title=layer, xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=7, loc='lower right', frameon=False)
    ax.grid(alpha=0.3)
fig.suptitle('ROC — nested CV with fold-wise MMUPHin', fontsize=12, y=1.02)
fig.tight_layout()
save_figure(fig, 'Fig2_ROC_all_layers')
save_tsv(pd.DataFrame(roc_rows), 'Fig2_ROC_all_layers')
fig, ax = plt.subplots(figsize=(10, 4.5))
n_models = len(MODEL_ORDER)
width = 0.15
x = np.arange(3)
offsets = np.linspace(-(n_models - 1) / 2 * width, (n_models - 1) / 2 * width, n_models)
res_lookup = {(r['Layer'], r['Model']): r for r in all_results}
bar_rows = []
for i, model in enumerate(MODEL_ORDER):
    means, stds = ([], [])
    for layer in LAYERS:
        rec = res_lookup[layer, model]
        means.append(rec['AUC'])
        stds.append(rec['AUC_std'])
        bar_rows.append({'Layer': layer, 'Model': model, 'AUC': rec['AUC'], 'AUC_std': rec['AUC_std']})
    ax.bar(x + offsets[i], means, width, yerr=stds, label=model, color=model_colors[model], alpha=0.85, edgecolor='white', capsize=3, error_kw=dict(elinewidth=1.0, ecolor='#333333'))
ax.set_xticks(x)
ax.set_xticklabels(LAYERS)
ax.set_ylabel('AUC (OOF global ± fold SD)')
ax.set_ylim(0.35, 1.05)
ax.axhline(0.5, color='gray', lw=0.8, linestyle='--', alpha=0.6)
ax.legend(title='Model', fontsize=8, ncol=5, frameon=False, loc='upper right')
ax.set_title('Model performance — fold-wise MMUPHin nested CV')
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
save_figure(fig, 'Fig2b_AUC_errorbars')
save_tsv(pd.DataFrame(bar_rows), 'Fig2b_AUC_errorbars')

def delong_test(y_true, prob1, prob2):
    n1 = int(np.sum(y_true == 1))
    n0 = int(np.sum(y_true == 0))
    pos1 = prob1[y_true == 1]
    neg1 = prob1[y_true == 0]
    pos2 = prob2[y_true == 1]
    neg2 = prob2[y_true == 0]
    v10_1 = np.array([np.mean(p > neg1) + 0.5 * np.mean(p == neg1) for p in pos1])
    v01_1 = np.array([np.mean(n < pos1) + 0.5 * np.mean(n == pos1) for n in neg1])
    v10_2 = np.array([np.mean(p > neg2) + 0.5 * np.mean(p == neg2) for p in pos2])
    v01_2 = np.array([np.mean(n < pos2) + 0.5 * np.mean(n == pos2) for n in neg2])
    var1 = np.var(v10_1, ddof=1) / n1 + np.var(v01_1, ddof=1) / n0
    var2 = np.var(v10_2, ddof=1) / n1 + np.var(v01_2, ddof=1) / n0
    cov = np.cov(v10_1, v10_2)[0, 1] / n1 + np.cov(v01_1, v01_2)[0, 1] / n0
    auc1 = roc_auc_score(y_true, prob1)
    auc2 = roc_auc_score(y_true, prob2)
    denom = np.sqrt(max(var1 + var2 - 2 * cov, 1e-12))
    z = (auc1 - auc2) / denom
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return (auc1, auc2, z, p)
best_by_layer = {}
for layer in LAYERS:
    layer_res = [r for r in all_results if r['Layer'] == layer]
    best_by_layer[layer] = max(layer_res, key=lambda x: x['AUC'])
    print(f"Best [{layer}]: {best_by_layer[layer]['Model']}  AUC={best_by_layer[layer]['AUC']:.4f}")
delong_records = []
for i in range(len(LAYERS)):
    for j in range(i + 1, len(LAYERS)):
        L1, L2 = (LAYERS[i], LAYERS[j])
        a1, a2, z, p = delong_test(y, best_by_layer[L1]['oof_probs'], best_by_layer[L2]['oof_probs'])
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        delong_records.append({'Comparison': f'{L1} vs {L2}', 'AUC1': a1, 'AUC2': a2, 'z': z, 'p': p, 'sig': sig})
        print(f'  {L1} vs {L2}: {a1:.4f} vs {a2:.4f}  z={z:.3f} p={p:.4f} {sig}')
pd.DataFrame(delong_records).to_csv(os.path.join(OUT, 'delong_test.csv'), index=False)
save_tsv(pd.DataFrame(delong_records), 'Table2_delong_test')
print('\n=== LOCO ===')
cohort_labels = np.unique(cohort)
loco_records = []
for layer in LAYERS:
    best_model_name = best_by_layer[layer]['Model']
    loco_aucs = []
    for test_cohort in cohort_labels:
        train_mask = cohort != test_cohort
        test_mask = cohort == test_cohort
        if len(np.unique(y[test_mask])) < 2:
            print(f'  [{layer}] {test_cohort}: single class, skip')
            continue
        train_ids = samples[train_mask].tolist()
        test_ids = samples[test_mask].tolist()
        fold_tag = f'loco_{test_cohort}'
        out_train, out_test, out_feats = run_mmuphin_fold(layer, fold_tag, train_ids, test_ids)
        X_tr, X_te, feats, id_tr, id_te = load_fold_clr(out_train, out_test, out_feats)
        y_tr = (meta.loc[id_tr, 'Group'].astype(str).values == 'PCOS').astype(int)
        y_te = (meta.loc[id_te, 'Group'].astype(str).values == 'PCOS').astype(int)
        coh_tr = np.array([cohort_map[c] for c in meta.loc[id_tr, 'Cohort'].astype(str)])
        n_groups = len(np.unique(coh_tr))
        if n_groups >= 3:
            inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
            fit_params = {'groups': coh_tr}
        else:
            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            fit_params = {}
        gs = GridSearchCV(copy.deepcopy(models[best_model_name]), param_grids[best_model_name], cv=inner_cv, scoring='roc_auc', n_jobs=4, refit=True)
        gs.fit(X_tr, y_tr, **fit_params)
        prob_te = gs.predict_proba(X_te)[:, 1]
        auc_te = roc_auc_score(y_te, prob_te)
        loco_aucs.append(auc_te)
        print(f'  [{layer}] test={test_cohort}  AUC={auc_te:.4f}')
        loco_records.append({'Layer': layer, 'Model': best_model_name, 'Test_Cohort': test_cohort, 'AUC': auc_te})
    if loco_aucs:
        print(f'  [{layer}] LOCO mean={np.mean(loco_aucs):.4f} ± {np.std(loco_aucs):.4f}')
loco_df = pd.DataFrame(loco_records)
loco_df.to_csv(os.path.join(OUT, 'loco_results.csv'), index=False)
save_tsv(loco_df, 'Table3_loco_results')
fig, ax = plt.subplots(figsize=(7.5, 4.5))
pivot_loco = loco_df.pivot(index='Test_Cohort', columns='Layer', values='AUC')
for col in LAYERS:
    if col not in pivot_loco.columns:
        pivot_loco[col] = np.nan
pivot_loco[LAYERS].plot(kind='bar', ax=ax, color=['#E41A1C', '#377EB8', '#4DAF4A'], alpha=0.85, edgecolor='white', width=0.7)
ax.set(xlabel='Test cohort (left-out)', ylabel='AUC', title='LOCO with fold-wise MMUPHin', ylim=(0.3, 1.0))
ax.axhline(0.5, color='gray', lw=0.8, linestyle='--')
ax.legend(title='Layer', fontsize=9, frameon=False)
ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right')
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
save_figure(fig, 'Fig4_LOCO_generalization')
save_tsv(loco_df, 'Fig4_LOCO_generalization')
loco_mean_by_layer = {l: loco_df.loc[loco_df['Layer'] == l, 'AUC'].mean() for l in LAYERS}
best_layer = max(LAYERS, key=lambda l: 0.5 * best_by_layer[l]['AUC'] + 0.5 * loco_mean_by_layer.get(l, 0))
print(f'\nBest layer (CV+LOCO): {best_layer}')
for l in LAYERS:
    score = 0.5 * best_by_layer[l]['AUC'] + 0.5 * loco_mean_by_layer.get(l, 0)
    print(f"  {l}: CV={best_by_layer[l]['AUC']:.4f}  LOCO={loco_mean_by_layer.get(l, 0):.4f}  score={score:.4f}")

def compute_shap_values(pipeline, model_name, X_raw):
    clf = pipeline.named_steps['clf']
    if model_name in ('LASSO', 'LR'):
        X_sc = pipeline.named_steps['scaler'].transform(X_raw)
        exp = shap.LinearExplainer(clf, X_sc)
        sv = exp.shap_values(X_sc)
        return (np.asarray(sv), X_sc)
    if model_name == 'RF':
        exp = shap.TreeExplainer(clf)
        sv = exp.shap_values(X_raw)
        sv = sv[1] if isinstance(sv, list) else sv
        return (np.asarray(sv), X_raw)
    if model_name == 'XGB':
        exp = shap.TreeExplainer(clf)
        sv = exp.shap_values(X_raw)
        return (np.asarray(sv), X_raw)
    if model_name == 'SVM':
        X_sc = pipeline.named_steps['scaler'].transform(X_raw)
        bg_idx = np.random.RandomState(SEED).choice(X_sc.shape[0], min(100, X_sc.shape[0]), replace=False)
        bg = shap.kmeans(X_sc[bg_idx], k=min(20, len(bg_idx)))
        exp = shap.KernelExplainer(clf.predict_proba, bg)
        sv_all = exp.shap_values(X_sc, nsamples=100)
        sv = sv_all[1] if isinstance(sv_all, list) else sv_all
        return (np.asarray(sv), X_sc)
    raise ValueError(model_name)

def aggregate_fold_shap(layer, model_name):
    records = shap_cache[layer, model_name]
    from collections import defaultdict
    abs_sum = defaultdict(float)
    abs_cnt = defaultdict(int)
    mean_sum = defaultdict(float)
    example_X = {}
    example_sv = {}
    for rec in records:
        pipe, X_tr, feats = (rec['pipeline'], rec['X_tr'], rec['feats'])
        try:
            sv, X_plot = compute_shap_values(pipe, model_name, X_tr)
        except Exception as e:
            print(f"  SHAP skip fold{rec['fold']}: {e}")
            continue
        mas = np.abs(sv).mean(axis=0)
        ms = sv.mean(axis=0)
        for j, f in enumerate(feats):
            abs_sum[f] += float(mas[j])
            abs_cnt[f] += 1
            mean_sum[f] += float(ms[j])
            example_X[f] = X_plot[:, j]
            example_sv[f] = sv[:, j]
    feats_sorted = sorted(abs_cnt.keys(), key=lambda f: abs_sum[f] / abs_cnt[f], reverse=True)
    mean_abs = np.array([abs_sum[f] / abs_cnt[f] for f in feats_sorted])
    mean_shap = np.array([mean_sum[f] / abs_cnt[f] for f in feats_sorted])
    return (feats_sorted, mean_abs, mean_shap)
layer_artifacts = {}
for layer in LAYERS:
    mn = best_by_layer[layer]['Model']
    print(f'\n[SHAP aggregate] {layer} | {mn}')
    feats_s, mas, msh = aggregate_fold_shap(layer, mn)
    layer_artifacts[layer] = {'model_name': mn, 'feats': feats_s, 'mean_abs_shap': mas, 'mean_shap': msh, 'top30_feats': feats_s[:30]}
    fi = pd.DataFrame({'Feature': feats_s, 'Mean_AbsSHAP': mas, 'Mean_SHAP': msh, 'Direction': np.where(msh > 0, 'PCOS-enriched', 'Healthy-enriched')})
    fi.to_csv(os.path.join(OUT, f'feature_importance_{layer}.csv'), index=False)
    save_tsv(fi, f'feature_importance_{layer}')
for plot_layer in LAYERS:
    best_model_name = best_by_layer[plot_layer]['Model']
    art = layer_artifacts[plot_layer]
    top20 = art['top30_feats'][:20]
    mas20 = art['mean_abs_shap'][:20]
    msh20 = art['mean_shap'][:20]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    order = list(range(20))[::-1]
    colors_bar = ['#D6604D' if msh20[i] > 0 else '#4393C3' for i in order]
    ax.barh(range(20), mas20[order], color=colors_bar, edgecolor='white', alpha=0.85)
    ax.set_yticks(range(20))
    ytick_style = 'italic' if plot_layer in ('Species', 'SGB') else 'normal'
    ax.set_yticklabels([top20[i] for i in order], fontsize=9, style=ytick_style)
    ax.set_xlabel('Mean |SHAP| (avg across outer folds)')
    ax.set_title(f'Top 20 features — {plot_layer} | {best_model_name}')
    ax.grid(axis='x', alpha=0.3)
    ax.legend(handles=[mpatches.Patch(color='#D6604D', alpha=0.85, label='Higher in PCOS'), mpatches.Patch(color='#4393C3', alpha=0.85, label='Higher in Healthy')], fontsize=9, loc='lower right', frameon=False)
    fig.tight_layout()
    save_figure(fig, f'Fig6_SHAP_importance_{plot_layer}')
    save_tsv(pd.DataFrame({'Feature': top20, 'Mean_AbsSHAP': mas20, 'Mean_SHAP': msh20}), f'Fig6_SHAP_importance_{plot_layer}')
    recs = shap_cache[plot_layer, best_model_name]
    rec_last = recs[-1]
    sv_last, X_last = compute_shap_values(rec_last['pipeline'], best_model_name, rec_last['X_tr'])
    feats_last = rec_last['feats']
    idx_map = []
    names_plot = []
    for f in top20:
        if f in feats_last:
            idx_map.append(feats_last.index(f))
            names_plot.append(f)
    if len(idx_map) >= 5:
        fig, _ = plt.subplots(figsize=(9, 7))
        shap.summary_plot(sv_last[:, idx_map], X_last[:, idx_map], feature_names=names_plot, plot_type='dot', show=False, max_display=len(names_plot))
        plt.title(f'SHAP beeswarm — {plot_layer} | {best_model_name} (outer fold example)', fontsize=11)
        plt.tight_layout()
        fig = plt.gcf()
        if plot_layer in ('Species', 'SGB'):
            for lab in plt.gca().get_yticklabels():
                lab.set_fontstyle('italic')
                lab.set_fontweight('normal')
        save_figure(fig, f'Fig5_SHAP_beeswarm_{plot_layer}')
        rows = []
        for j, fname in enumerate(names_plot):
            for i in range(sv_last.shape[0]):
                rows.append({'Feature': fname, 'SHAP': float(sv_last[i, idx_map[j]]), 'FeatureValue': float(X_last[i, idx_map[j]])})
        save_tsv(pd.DataFrame(rows), f'Fig5_SHAP_beeswarm_{plot_layer}')
best_model_name = best_by_layer[best_layer]['Model']
y_pred_best = (best_by_layer[best_layer]['oof_probs'] >= 0.5).astype(int)
cm = confusion_matrix(y, y_pred_best)
fig, ax = plt.subplots(figsize=(4.5, 3.8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred Healthy', 'Pred PCOS'], yticklabels=['True Healthy', 'True PCOS'], ax=ax)
ax.set_title(f'Confusion matrix — {best_layer} | {best_model_name}')
fig.tight_layout()
save_figure(fig, f'Fig8_confusion_matrix_{best_layer}')
save_tsv(pd.DataFrame(cm, index=['True Healthy', 'True PCOS'], columns=['Pred Healthy', 'Pred PCOS']).reset_index(), f'Fig8_confusion_matrix_{best_layer}')
note = f"# Leakage-safe re-analysis summary\n\n- Original: MMUPHin on the full dataset with disease status as covariate, then nested CV.\n- Revised: within each outer CV fold, prevalence filtering + MMUPHin::adjust_batch\n  were fit on the training split only (covariates = NULL). Test samples were adjusted\n  using train-derived batch location/scale shifts, then CLR-transformed.\n- Nested hyperparameter search (inner StratifiedGroupKFold) used only fold-corrected\n  training features.\n\n- Best layer (0.5*CV + 0.5*LOCO): {best_layer}\n- Best model at that layer: {best_model_name}\n- Nested OOF AUC: {best_by_layer[best_layer]['AUC']:.4f} ± {best_by_layer[best_layer]['AUC_std']:.4f}\n\nSee ml_output/results_summary.csv and Table1_results_summary.tsv\nSee ml_output/auc_original_vs_nested.csv for delta vs original pipeline.\n"
with open(os.path.join(OUT, 'REVISION_NOTE.md'), 'w') as f:
    f.write(note)
print('\n' + '=' * 60)
print('Done. Key outputs in:', OUT)
print(results_df.sort_values('AUC', ascending=False).to_string(index=False))
print(f"\nRecommended: {best_layer} | {best_model_name} | AUC={best_by_layer[best_layer]['AUC']:.4f}")
loco_best = loco_df[loco_df['Layer'] == best_layer]['AUC']
if len(loco_best):
    print(f'LOCO: {loco_best.mean():.4f} ± {loco_best.std():.4f}')
Path(OUT).mkdir(parents=True, exist_ok=True)
Path(FOLD_DIR).mkdir(parents=True, exist_ok=True)
