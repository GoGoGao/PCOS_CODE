#!/usr/bin/env python3

import os, sys, re, math, copy, pickle, warnings, argparse, logging
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, fisher_exact
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf

warnings.filterwarnings('ignore')


# ============================================================
# Module 1: Alpha Diversity
# ============================================================

def calculate_alpha_diversity(abundance_df):
    results = {}
    for sample in abundance_df.columns:
        x = abundance_df[sample].values
        x = x[x > 0]
        observed = len(x)
        p = x / x.sum()
        shannon = -np.sum(p * np.log(p))
        simpson = 1 - np.sum(p ** 2)
        inv_simpson = 1 / np.sum(p ** 2)
        pielou = shannon / np.log(observed) if observed > 1 else 0
        n = x.sum()
        fisher_alpha = observed / np.log(n / observed) if n > observed else observed
        results[sample] = {
            'Observed': observed, 'Shannon': shannon,
            'Simpson': simpson, 'InvSimpson': inv_simpson,
            'Pielou': pielou, 'Fisher': fisher_alpha
        }
    return pd.DataFrame(results).T


def stratified_wilcoxon(data, metric, group_col, strata_col):
    strata = data[strata_col].unique()
    layer_stats = []
    for stratum in strata:
        stratum_data = data[data[strata_col] == stratum]
        grp1 = stratum_data[stratum_data[group_col] == 'PCOS'][metric].values
        grp2 = stratum_data[stratum_data[group_col] == 'Healthy'][metric].values
        if len(grp1) >= 3 and len(grp2) >= 3:
            stat, pval = mannwhitneyu(grp1, grp2, alternative='two-sided')
            n1, n2 = len(grp1), len(grp2)
            layer_stats.append({
                'stratum': stratum, 'n1': n1, 'n2': n2,
                'U': stat, 'p_value': pval,
                'effect_size': stat / (n1 * n2)
            })
    if layer_stats:
        p_values = [s['p_value'] for s in layer_stats]
        chi2_stat = -2 * np.sum(np.log(p_values))
        combined_p = 1 - stats.chi2.cdf(chi2_stat, df=2 * len(p_values))
        weights = [s['n1'] * s['n2'] for s in layer_stats]
        weighted_effect = np.average(
            [s['effect_size'] for s in layer_stats], weights=weights)
        return {'layer_stats': layer_stats, 'combined_p': combined_p,
                'weighted_effect': weighted_effect}
    return None


def run_alpha_lmm(alpha_div, metrics):
    results = {}
    for metric in metrics:
        model_data = alpha_div[['Sample', metric, 'Group', 'Bioproject']].copy()
        model_data['Group_binary'] = (model_data['Group'] == 'PCOS').astype(int)
        model = smf.mixedlm(
            f"{metric} ~ Group_binary", model_data,
            groups=model_data['Bioproject'])
        result = model.fit(method='powell')
        results[metric] = {
            'coefficient': result.fe_params['Group_binary'],
            'p_value': result.pvalues['Group_binary'],
            'ci_low': result.conf_int().loc['Group_binary'][0],
            'ci_high': result.conf_int().loc['Group_binary'][1]
        }
    return results


def plot_alpha_boxplots(alpha_div, metrics, metric_labels, colors_group,
                        colors_batch, lmm_results, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for idx, (ax, metric, label) in enumerate(
            zip(axes.flat, metrics, metric_labels)):
        bp = ax.boxplot(
            [alpha_div[alpha_div['Group'] == 'PCOS'][metric],
             alpha_div[alpha_div['Group'] == 'Healthy'][metric]],
            positions=[1, 2], widths=0.6, patch_artist=True)
        bp['boxes'][0].set_facecolor(colors_group['PCOS'])
        bp['boxes'][1].set_facecolor(colors_group['Healthy'])
        for box in bp['boxes']:
            box.set_alpha(0.7)
        for grp_idx, grp in enumerate(['PCOS', 'Healthy']):
            grp_data = alpha_div[alpha_div['Group'] == grp]
            x_pos = grp_idx + 1
            for batch in colors_batch:
                batch_data = grp_data[grp_data['Bioproject'] == batch]
                jitter = np.random.normal(0, 0.08, len(batch_data))
                ax.scatter(x_pos + jitter, batch_data[metric],
                           c=colors_batch[batch], s=30, alpha=0.6,
                           edgecolors='white', linewidths=0.5)
        pval = lmm_results[metric]['p_value']
        sig_text = '***' if pval < 0.001 else '**' if pval < 0.01 \
            else '*' if pval < 0.05 else 'ns'
        y_max = alpha_div[metric].max()
        y_range = y_max - alpha_div[metric].min()
        ax.plot([1, 1, 2, 2],
                [y_max + y_range * 0.05, y_max + y_range * 0.08,
                 y_max + y_range * 0.08, y_max + y_range * 0.05],
                'k-', linewidth=1)
        ax.text(1.5, y_max + y_range * 0.12, f'{sig_text}\np={pval:.3f}',
                ha='center', va='bottom', fontsize=10)
        n_pcos = (alpha_div['Group'] == 'PCOS').sum()
        n_healthy = (alpha_div['Group'] == 'Healthy').sum()
        ax.set_xticks([1, 2])
        ax.set_xticklabels([f'PCOS\n(n={n_pcos})', f'Healthy\n(n={n_healthy})'])
        ax.set_ylabel(label)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(f"{output_dir}/alpha_diversity_boxplot.pdf", bbox_inches='tight')
    plt.close(fig)


# ============================================================
# Module 2: Beta Diversity
# ============================================================

def permanova_test(distance_matrix, grouping, n_permutations=999, seed=42):
    groups = np.asarray(grouping)
    unique_groups = np.unique(groups)
    n = len(groups)

    def calc_ss(dist_mat, grps):
        ss_total = np.sum(dist_mat ** 2) / (2 * n)
        ss_within = 0
        for g in np.unique(grps):
            mask = grps == g
            n_g = np.sum(mask)
            if n_g > 1:
                ss_within += np.sum(
                    dist_mat[np.ix_(mask, mask)] ** 2) / (2 * n_g)
        return ss_total - ss_within, ss_within, ss_total

    ss_between, ss_within, ss_total = calc_ss(distance_matrix, groups)
    n_groups = len(unique_groups)
    df_between = n_groups - 1
    df_within = n - n_groups
    f_stat = (ss_between / df_between) / (ss_within / df_within)
    r_squared = ss_between / ss_total

    np.random.seed(seed)
    f_perms = []
    for _ in range(n_permutations):
        perm_groups = np.random.permutation(groups)
        ss_b, ss_w, _ = calc_ss(distance_matrix, perm_groups)
        f_perms.append((ss_b / df_between) / (ss_w / df_within))
    p_value = (np.sum(np.array(f_perms) >= f_stat) + 1) / (n_permutations + 1)
    return {'R2': r_squared, 'F': f_stat, 'df1': df_between,
            'df2': df_within, 'p_value': p_value}


def permanova_two_factor(distance_matrix, factor1, factor2,
                         n_permutations=999, seed=42):
    n = len(factor1)
    groups1, groups2 = np.asarray(factor1), np.asarray(factor2)
    ss_total = np.sum(distance_matrix ** 2) / (2 * n)
    ss_within1 = 0
    for g in np.unique(groups1):
        mask = groups1 == g
        n_g = np.sum(mask)
        if n_g > 1:
            ss_within1 += np.sum(
                distance_matrix[np.ix_(mask, mask)] ** 2) / (2 * n_g)
    ss_factor1 = ss_total - ss_within1
    ss_within_both = 0
    for g1 in np.unique(groups1):
        for g2 in np.unique(groups2):
            mask = (groups1 == g1) & (groups2 == g2)
            n_g = np.sum(mask)
            if n_g > 1:
                ss_within_both += np.sum(
                    distance_matrix[np.ix_(mask, mask)] ** 2) / (2 * n_g)
    ss_factor2 = ss_total - ss_factor1 - ss_within_both
    ss_residual = ss_within_both
    df2 = len(np.unique(groups2)) - 1
    df_resid = n - (len(np.unique(groups1)) - 1) - df2 - 1
    f_stat = (ss_factor2 / df2) / (ss_residual / df_resid) \
        if (df_resid > 0 and ss_residual > 0) else 0
    np.random.seed(seed)
    f_perms = []
    for _ in range(n_permutations):
        perm_factor2 = pd.Series(groups2.copy())
        for g1 in np.unique(groups1):
            mask = groups1 == g1
            perm_factor2.iloc[np.where(mask)[0]] = np.random.permutation(
                perm_factor2.iloc[np.where(mask)[0]].values)
        ss_wp = 0
        for g1 in np.unique(groups1):
            for g2 in np.unique(perm_factor2):
                mask = (groups1 == g1) & (perm_factor2.values == g2)
                n_g = np.sum(mask)
                if n_g > 1:
                    ss_wp += np.sum(
                        distance_matrix[np.ix_(mask, mask)] ** 2) / (2 * n_g)
        ss_f2p = ss_total - ss_factor1 - ss_wp
        f_perms.append(
            (ss_f2p / df2) / (ss_wp / df_resid) if (df_resid > 0 and ss_wp > 0) else 0)
    p_value = (np.sum(np.array(f_perms) >= f_stat) + 1) / (n_permutations + 1)
    return {'Factor1_R2': ss_factor1 / ss_total,
            'Factor2_R2': max(0, ss_factor2 / ss_total),
            'Factor2_F': f_stat, 'Factor2_p': p_value}


def anosim_test(distance_matrix, grouping, n_permutations=999, seed=42):
    groups = np.asarray(grouping)
    n = len(groups)
    ranks = stats.rankdata(squareform(distance_matrix))
    rank_matrix = squareform(ranks)
    within_ranks, between_ranks = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if groups[i] == groups[j]:
                within_ranks.append(rank_matrix[i, j])
            else:
                between_ranks.append(rank_matrix[i, j])
    r_w, r_b = np.mean(within_ranks), np.mean(between_ranks)
    n_comp = n * (n - 1) / 2
    r_stat = (r_b - r_w) / (n_comp / 2)
    np.random.seed(seed)
    r_perms = []
    for _ in range(n_permutations):
        pg = np.random.permutation(groups)
        wr, br = [], []
        for i in range(n):
            for j in range(i + 1, n):
                (wr if pg[i] == pg[j] else br).append(rank_matrix[i, j])
        r_perms.append((np.mean(br) - np.mean(wr)) / (n_comp / 2))
    p_value = (np.sum(np.array(r_perms) >= r_stat) + 1) / (n_permutations + 1)
    return {'R': r_stat, 'p_value': p_value}


def perform_pcoa(dist_matrix):
    n = dist_matrix.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (dist_matrix ** 2) @ H
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    pos_mask = eigenvalues > 0
    eigenvalues = eigenvalues[pos_mask]
    eigenvectors = eigenvectors[:, pos_mask]
    coords = eigenvectors[:, :2] * np.sqrt(eigenvalues[:2])
    var_explained = eigenvalues[:2] / np.sum(np.abs(eigenvalues)) * 100
    return coords, var_explained


# ============================================================
# Module 3: Differential Analysis (LMM)
# ============================================================

def run_differential_lmm(species_clr, aligned_info, fdr_threshold=0.05):
    species_clr_t = species_clr.T
    lmm_results = []
    for species in species_clr.index:
        try:
            model_data = pd.DataFrame({
                'abundance': species_clr_t[species].values,
                'Group': aligned_info['Group'].values,
                'Batch': aligned_info['Bioproject'].values,
            })
            model_data['Group_binary'] = (model_data['Group'] == 'PCOS').astype(int)
            model = smf.mixedlm(
                "abundance ~ Group_binary", model_data,
                groups=model_data['Batch'])
            result = model.fit(method='powell', disp=False)
            lmm_results.append({
                'Species': species,
                'LMM_coef': result.fe_params['Group_binary'],
                'LMM_p': result.pvalues['Group_binary'],
            })
        except Exception:
            lmm_results.append({
                'Species': species, 'LMM_coef': np.nan, 'LMM_p': np.nan})
    lmm_df = pd.DataFrame(lmm_results)
    valid = ~lmm_df['LMM_p'].isna()
    lmm_df.loc[valid, 'LMM_FDR'] = multipletests(
        lmm_df.loc[valid, 'LMM_p'], method='fdr_bh')[1]
    lmm_df['Significant_LMM'] = lmm_df['LMM_FDR'] < fdr_threshold
    lmm_df['Direction'] = 'NS'
    lmm_df.loc[(lmm_df['Significant_LMM']) & (lmm_df['LMM_coef'] > 0),
               'Direction'] = 'PCOS_enriched'
    lmm_df.loc[(lmm_df['Significant_LMM']) & (lmm_df['LMM_coef'] < 0),
               'Direction'] = 'Healthy_enriched'
    return lmm_df


def run_wilcoxon_test(species_tss, pcos_samples, healthy_samples):
    results = []
    pseudo = 0.001
    for species in species_tss.index:
        pcos_vals = species_tss.loc[species, pcos_samples].values
        healthy_vals = species_tss.loc[species, healthy_samples].values
        stat, pval = mannwhitneyu(pcos_vals, healthy_vals, alternative='two-sided')
        log2fc = np.log2(
            (pcos_vals.mean() + pseudo) / (healthy_vals.mean() + pseudo))
        results.append({
            'Species': species,
            'PCOS_mean': pcos_vals.mean(),
            'Healthy_mean': healthy_vals.mean(),
            'PCOS_prevalence': np.sum(pcos_vals > 0) / len(pcos_vals),
            'Healthy_prevalence': np.sum(healthy_vals > 0) / len(healthy_vals),
            'Log2FC': log2fc,
            'Wilcoxon_stat': stat,
            'Wilcoxon_p': pval,
        })
    df = pd.DataFrame(results)
    df['Wilcoxon_FDR'] = multipletests(df['Wilcoxon_p'], method='fdr_bh')[1]
    return df


# ============================================================
# Module 4: Core Microbiome
# ============================================================

def identify_core_microbiome(abundance_df, samples,
                             prevalence_threshold=0.5):
    subset = abundance_df[samples]
    prevalence = (subset > 0).sum(axis=1) / len(samples)
    mean_abundance = subset.mean(axis=1)
    core_species = prevalence[prevalence >= prevalence_threshold].index.tolist()
    return core_species, prevalence, mean_abundance


# ============================================================
# Module 5: Machine Learning (Nested CV)
# ============================================================

def setup_ml_env():
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def load_corrected_data(path):
    df = pd.read_csv(path)
    meta_cols = ['SampleID', 'Group', 'Cohort']
    feat_cols = [c for c in df.columns if c not in meta_cols]
    return df, feat_cols


def clean_feat_names(feats):
    return [f.replace('[', '(').replace(']', ')').replace('|', '_').replace('<', '_')
            for f in feats]


def nested_cv(X, y, cohort_int, strat_label, pipeline, param_grid,
              n_outer=5, n_inner=3):
    from sklearn.model_selection import (StratifiedKFold, StratifiedGroupKFold,
                                         GridSearchCV)
    from sklearn.metrics import (roc_auc_score, f1_score, matthews_corrcoef,
                                 accuracy_score)
    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=2025)
    oof_probs = np.zeros(len(y))
    oof_preds = np.zeros(len(y), dtype=int)
    fold_aucs = []
    for fold, (train_idx, test_idx) in enumerate(
            outer_cv.split(X, strat_label)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        inner_cv = StratifiedGroupKFold(
            n_splits=n_inner, shuffle=True, random_state=42)
        gs = GridSearchCV(
            copy.deepcopy(pipeline), param_grid,
            cv=inner_cv, scoring='roc_auc', n_jobs=4, refit=True)
        gs.fit(X_tr, y_tr, groups=cohort_int[train_idx])
        fold_prob = gs.predict_proba(X_te)[:, 1]
        oof_probs[test_idx] = fold_prob
        oof_preds[test_idx] = gs.predict(X_te)
        if len(np.unique(y_te)) == 2:
            fold_aucs.append(roc_auc_score(y_te, fold_prob))
    metrics = {
        'AUC': roc_auc_score(y, oof_probs),
        'AUC_std': float(np.std(fold_aucs, ddof=1)) if len(fold_aucs) > 1 else 0.0,
        'F1': f1_score(y, oof_preds),
        'MCC': matthews_corrcoef(y, oof_preds),
        'ACC': accuracy_score(y, oof_preds)
    }
    return metrics, oof_probs, fold_aucs


def delong_test(y_true, prob1, prob2):
    from sklearn.metrics import roc_auc_score
    n1 = int(np.sum(y_true == 1))
    n0 = int(np.sum(y_true == 0))
    pos1, neg1 = prob1[y_true == 1], prob1[y_true == 0]
    pos2, neg2 = prob2[y_true == 1], prob2[y_true == 0]
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
    return auc1, auc2, z, p


# ============================================================
# Module 6: Functional (KO) Differential Analysis
# ============================================================

def prevalence_filter(freq_series, n_total, thresh=0.05):
    min_count = math.ceil(n_total * thresh)
    mask = (freq_series >= min_count) & (freq_series <= n_total - min_count)
    return freq_series[mask].index, min_count


def run_fisher_test(sko_df, pcos_samples, healthy_samples, prev_thresh=0.05):
    n_p, n_h = len(pcos_samples), len(healthy_samples)
    n_total = n_p + n_h
    freq = sko_df[pcos_samples + healthy_samples].sum(axis=1)
    testable, min_count = prevalence_filter(freq, n_total, prev_thresh)
    results = []
    for fid in testable:
        kid = fid.split(':')[0]
        gene = fid.split(':')[1] if ':' in fid else ''
        pp = int(sko_df.loc[fid, pcos_samples].sum())
        hp = int(sko_df.loc[fid, healthy_samples].sum())
        pn, hn = n_p - pp, n_h - hp
        od, p = fisher_exact([[pp, pn], [hp, hn]])
        log2or = np.log2(od) if od > 0 and np.isfinite(od) else (
            5 if od == np.inf else -5)
        log2or = np.clip(log2or, -5, 5)
        se = np.sqrt(1 / max(pp, 0.5) + 1 / max(pn, 0.5) +
                     1 / max(hp, 0.5) + 1 / max(hn, 0.5)) / np.log(2)
        results.append({
            'KO': kid, 'full_id': fid, 'gene': gene,
            'PCOS_n': pp, 'Healthy_n': hp,
            'PCOS_pct': pp / n_p * 100, 'Healthy_pct': hp / n_h * 100,
            'OR': od, 'log2OR': log2or, 'SE_log2OR': se,
            'CI_lo': log2or - 1.96 * se, 'CI_hi': log2or + 1.96 * se,
            'P': p,
            'direction': 'PCOS-enriched' if od > 1 else 'Healthy-enriched',
        })
    df = pd.DataFrame(results)
    if len(df) > 0:
        _, df['FDR'], _, _ = multipletests(df['P'], method='fdr_bh')
        df['neglog10P'] = -np.log10(df['P'].clip(lower=1e-10))
        df['sig'] = 'NS'
        df.loc[df['P'] < 0.05, 'sig'] = 'P<0.05'
        df.loc[df['FDR'] < 0.2, 'sig'] = 'FDR<0.2'
    return df, testable, min_count


def run_gseapy_ora(query_kos, background_kos, pathway_db):
    try:
        import gseapy as gp
    except ImportError:
        return pd.DataFrame()
    q = [k for k in query_kos if k in set(background_kos)]
    if len(q) == 0:
        return pd.DataFrame()
    try:
        enr = gp.enrich(
            gene_list=q, gene_sets=pathway_db,
            background=list(background_kos),
            outdir=None, no_plot=True, verbose=False)
        df = enr.results.copy()
        if df.empty:
            return df
        df[['overlap_k', 'path_n']] = (
            df['Overlap'].str.split('/', expand=True).astype(int))
        df['enrichment_ratio'] = (
            (df['overlap_k'] / len(q)) /
            (df['path_n'] / len(background_kos))
        ).replace([np.inf], 0)
        df = df.rename(columns={
            'P-value': 'P', 'Adjusted P-value': 'FDR',
            'Odds Ratio': 'OR', 'Genes': 'overlap_KOs'})
        df['-log10P'] = -np.log10(df['P'].clip(lower=1e-10))
        return df.sort_values('P').reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


# ============================================================
# Module 7: Visualization Utilities
# ============================================================

def save_figure(fig, base_path, dpi=300):
    fig.savefig(f"{base_path}.pdf", format='pdf', bbox_inches='tight')
    fig.savefig(f"{base_path}.svg", format='svg', bbox_inches='tight')
    fig.savefig(f"{base_path}.jpg", format='jpg', dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def panel_label(ax, label, x=-0.12, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top', ha='left')


def to_log(x, pseudo=0.001):
    return np.log10(np.asarray(x, dtype=float) + pseudo)


def draw_species_boxplot(ax, plot_df, title_str, group_colors,
                         cohort_colors, fdr_val):
    groups_order = ['PCOS', 'Healthy']
    max_val = plot_df['Abundance'].max()
    box_data = [to_log(plot_df[plot_df['Group'] == g]['Abundance'].values)
                for g in groups_order]
    bp = ax.boxplot(box_data, positions=[1, 2], widths=0.45,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color='#E69F00', linewidth=1.8))
    for i, g in enumerate(groups_order):
        bp['boxes'][i].set_facecolor(group_colors[g])
        bp['boxes'][i].set_alpha(0.35)
    np.random.seed(42)
    for g in groups_order:
        sub = plot_df[plot_df['Group'] == g]
        pos = groups_order.index(g) + 1
        jitter = np.random.normal(0, 0.07, len(sub))
        cvals = [cohort_colors.get(b, '#888888') for b in sub['Cohort']]
        ax.scatter(pos + jitter, to_log(sub['Abundance'].values),
                   c=cvals, s=16, alpha=0.85,
                   edgecolors='white', linewidths=0.25, zorder=3)
    sig_str = '***' if fdr_val < 0.001 else '**' if fdr_val < 0.01 \
        else '*' if fdr_val < 0.05 else 'ns'
    y_top = to_log(max_val) + 0.15
    ax.annotate('', xy=(2, y_top), xytext=(1, y_top),
                arrowprops=dict(arrowstyle='-', color='#333333', lw=0.8))
    ax.text(1.5, y_top + 0.03, sig_str, ha='center', va='bottom', fontsize=9)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(groups_order)
    ax.set_xlim(0.4, 2.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(title_str, fontsize=8.5, fontweight='bold')


if __name__ == '__main__':
    print("PCOS multi-cohort metagenome analysis pipeline")
    print("Import this module or run individual analysis steps.")
