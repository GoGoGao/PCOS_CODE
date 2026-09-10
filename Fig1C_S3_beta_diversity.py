from pathlib import Path
import os
PKG_ROOT = Path(__file__).resolve().parents[1]
SHARED = PKG_ROOT / 'shared_input'
RSCRIPT = os.environ.get('RSCRIPT', 'Rscript')
INPUT_DIR = str(SHARED)
OUTPUT_DIR = str(PKG_ROOT / 'output' / 'Fig1C_S3')
import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform, braycurtis
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import matplotlib.lines as mlines
import seaborn as sns
from datetime import datetime
import warnings
import sys
import os
import pickle
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
WORK_DIR = os.environ.get('PCOS_WORK', str(__import__('pathlib').Path(__file__).resolve().parents[1] / '_runtime'))
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = f'{WORK_DIR}/logs/03_beta_diversity_log.txt'

class Logger:

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()
sys.stdout = Logger(log_file)
print('=' * 70)
print('Microbiome Beta Diversity Analysis Log')
print(f"Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print('=' * 70)
print()
print('[STEP 1] Loading preprocessed data')
print('-' * 50)
with open(f'{INPUT_DIR}/metaphlan_preprocessed.pkl', 'rb') as f:
    data = pickle.load(f)
species_tss = data['species_tss']
species_clr = data['species_clr']
group_info = data['group_info']
colors_group = data['colors_group']
colors_batch = data['colors_batch']
print(f'  Species count: {species_tss.shape[0]}')
print(f'  Sample count:  {species_tss.shape[1]}')
print('\n[STEP 2] Computing beta diversity distance matrices')
print('-' * 50)
species_tss_t = species_tss.T
species_clr_t = species_clr.T
bc_dist = pdist(species_tss_t.values, metric='braycurtis')
bc_matrix = squareform(bc_dist)
bc_df = pd.DataFrame(bc_matrix, index=species_tss_t.index, columns=species_tss_t.index)
print('  Bray-Curtis distance [DONE]')

def jaccard_distance(u, v):
    u_binary = (u > 0).astype(int)
    v_binary = (v > 0).astype(int)
    intersection = np.sum(u_binary & v_binary)
    union = np.sum(u_binary | v_binary)
    return 1 - intersection / union if union > 0 else 0
jaccard_dist = pdist(species_tss_t.values, metric=jaccard_distance)
jaccard_matrix = squareform(jaccard_dist)
print('  Jaccard distance [DONE]')
aitchison_dist = pdist(species_clr_t.values, metric='euclidean')
aitchison_matrix = squareform(aitchison_dist)
print('  Aitchison distance [DONE]')
print('\n[STEP 3] PERMANOVA analysis (accounting for batch effects)')
print('-' * 50)

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
                ss_within += np.sum(dist_mat[np.ix_(mask, mask)] ** 2) / (2 * n_g)
        ss_between = ss_total - ss_within
        return (ss_between, ss_within, ss_total)
    ss_between, ss_within, ss_total = calc_ss(distance_matrix, groups)
    n_groups = len(unique_groups)
    df_between = n_groups - 1
    df_within = n - n_groups
    f_stat = ss_between / df_between / (ss_within / df_within)
    r_squared = ss_between / ss_total
    np.random.seed(seed)
    f_perms = []
    for _ in range(n_permutations):
        perm_groups = np.random.permutation(groups)
        ss_b, ss_w, _ = calc_ss(distance_matrix, perm_groups)
        f_perm = ss_b / df_between / (ss_w / df_within)
        f_perms.append(f_perm)
    p_value = (np.sum(np.array(f_perms) >= f_stat) + 1) / (n_permutations + 1)
    return {'R2': r_squared, 'F': f_stat, 'df1': df_between, 'df2': df_within, 'p_value': p_value, 'n_permutations': n_permutations}
aligned_info = group_info.copy()
aligned_info.index = aligned_info['Sample']
aligned_info = aligned_info.loc[species_tss_t.index]
print('\n[3.1] Single-factor PERMANOVA (all samples):')
results_single = {}
for dist_name, dist_mat in [('Bray-Curtis', bc_matrix), ('Jaccard', jaccard_matrix), ('Aitchison', aitchison_matrix)]:
    res_group = permanova_test(dist_mat, aligned_info['Group'])
    res_batch = permanova_test(dist_mat, aligned_info['Bioproject'])
    results_single[dist_name] = {'Group': res_group, 'Batch': res_batch}
    print(f'  {dist_name}:')
    print(f"    Group: R2={res_group['R2']:.4f}, F={res_group['F']:.2f}, p={res_group['p_value']:.4f}")
    print(f"    Batch: R2={res_batch['R2']:.4f}, F={res_batch['F']:.2f}, p={res_batch['p_value']:.4f}")
print('\n[3.2] Two-factor PERMANOVA (Batch + Group):')

def permanova_two_factor(distance_matrix, factor1, factor2, n_permutations=999, seed=42):
    n = len(factor1)
    groups1 = np.asarray(factor1)
    groups2 = np.asarray(factor2)
    ss_total = np.sum(distance_matrix ** 2) / (2 * n)
    ss_within1 = 0
    for g in np.unique(groups1):
        mask = groups1 == g
        n_g = np.sum(mask)
        if n_g > 1:
            ss_within1 += np.sum(distance_matrix[np.ix_(mask, mask)] ** 2) / (2 * n_g)
    ss_factor1 = ss_total - ss_within1
    ss_within_both = 0
    for g1 in np.unique(groups1):
        for g2 in np.unique(groups2):
            mask = (groups1 == g1) & (groups2 == g2)
            n_g = np.sum(mask)
            if n_g > 1:
                ss_within_both += np.sum(distance_matrix[np.ix_(mask, mask)] ** 2) / (2 * n_g)
    ss_factor2 = ss_total - ss_factor1 - ss_within_both
    ss_residual = ss_within_both
    r2_factor1 = ss_factor1 / ss_total
    r2_factor2 = max(0, ss_factor2 / ss_total)
    df1 = len(np.unique(groups1)) - 1
    df2 = len(np.unique(groups2)) - 1
    df_resid = n - df1 - df2 - 1
    f_stat = ss_factor2 / df2 / (ss_residual / df_resid) if df_resid > 0 and ss_residual > 0 else 0
    np.random.seed(seed)
    f_perms = []
    for _ in range(n_permutations):
        perm_factor2 = pd.Series(groups2.copy())
        for g1 in np.unique(groups1):
            mask = groups1 == g1
            perm_factor2.iloc[np.where(mask)[0]] = np.random.permutation(perm_factor2.iloc[np.where(mask)[0]].values)
        ss_wp = 0
        for g1 in np.unique(groups1):
            for g2 in np.unique(perm_factor2):
                mask = (groups1 == g1) & (perm_factor2.values == g2)
                n_g = np.sum(mask)
                if n_g > 1:
                    ss_wp += np.sum(distance_matrix[np.ix_(mask, mask)] ** 2) / (2 * n_g)
        ss_f2p = ss_total - ss_factor1 - ss_wp
        f_perm = ss_f2p / df2 / (ss_wp / df_resid) if df_resid > 0 and ss_wp > 0 else 0
        f_perms.append(f_perm)
    p_value = (np.sum(np.array(f_perms) >= f_stat) + 1) / (n_permutations + 1)
    return {'Factor1_R2': r2_factor1, 'Factor2_R2': r2_factor2, 'Factor2_F': f_stat, 'Factor2_p': p_value}
results_two = {}
for dist_name, dist_mat in [('Bray-Curtis', bc_matrix), ('Jaccard', jaccard_matrix), ('Aitchison', aitchison_matrix)]:
    res = permanova_two_factor(dist_mat, aligned_info['Bioproject'], aligned_info['Group'])
    results_two[dist_name] = res
    print(f'  {dist_name}:')
    print(f"    Batch R2={res['Factor1_R2']:.4f}")
    print(f"    Group (adjusted): R2={res['Factor2_R2']:.4f}, F={res['Factor2_F']:.2f}, p={res['Factor2_p']:.4f}")
print('\n[3.3] Per-cohort ADONIS (PCOS vs Healthy within each Bioproject):')
print('-' * 50)
batch_order = ['PRJNA530971', 'PRJNA549764', 'PRJNA791492']
adonis_percohort_rows = []
for dist_name, dist_mat in [('Bray-Curtis', bc_matrix), ('Jaccard', jaccard_matrix), ('Aitchison', aitchison_matrix)]:
    for batch in batch_order:
        batch_mask = aligned_info['Bioproject'] == batch
        batch_idx = np.where(batch_mask)[0]
        batch_samples = aligned_info.index[batch_mask]
        sub_dist = dist_mat[np.ix_(batch_idx, batch_idx)]
        sub_group = aligned_info.loc[batch_samples, 'Group']
        n_pcos = (sub_group == 'PCOS').sum()
        n_healthy = (sub_group == 'Healthy').sum()
        n_total = len(sub_group)
        if n_pcos < 2 or n_healthy < 2:
            print(f'  [WARN] {batch} / {dist_name}: insufficient samples (PCOS={n_pcos}, Healthy={n_healthy}), skipped.')
            adonis_percohort_rows.append({'Bioproject': batch, 'Distance_metric': dist_name, 'n_PCOS': n_pcos, 'n_Healthy': n_healthy, 'n_total': n_total, 'R2': np.nan, 'F_statistic': np.nan, 'df1': np.nan, 'df2': np.nan, 'p_value': np.nan, 'significance': 'skipped'})
            continue
        res = permanova_test(sub_dist, sub_group, n_permutations=999, seed=42)
        pv = res['p_value']
        if pv < 0.001:
            sig = '***'
        elif pv < 0.01:
            sig = '**'
        elif pv < 0.05:
            sig = '*'
        else:
            sig = 'ns'
        print(f"  {batch} | {dist_name}: n={n_total} (PCOS={n_pcos}, Healthy={n_healthy})  R2={res['R2']:.4f}  F={res['F']:.2f}  p={pv:.4f}  {sig}")
        adonis_percohort_rows.append({'Bioproject': batch, 'Distance_metric': dist_name, 'n_PCOS': n_pcos, 'n_Healthy': n_healthy, 'n_total': n_total, 'R2': res['R2'], 'F_statistic': res['F'], 'df1': res['df1'], 'df2': res['df2'], 'p_value': pv, 'significance': sig})
adonis_percohort_df = pd.DataFrame(adonis_percohort_rows, columns=['Bioproject', 'Distance_metric', 'n_PCOS', 'n_Healthy', 'n_total', 'R2', 'F_statistic', 'df1', 'df2', 'p_value', 'significance'])
adonis_percohort_tsv = f'{OUTPUT_DIR}/adonis_per_cohort_PCOS_vs_Healthy.tsv'
adonis_percohort_df.to_csv(adonis_percohort_tsv, sep='\t', index=False, float_format='%.6e')
print(f'\n  [DONE] Per-cohort ADONIS saved -> adonis_per_cohort_PCOS_vs_Healthy.tsv')
print('\n[STEP 4] ANOSIM analysis')
print('-' * 50)

def anosim_test(distance_matrix, grouping, n_permutations=999, seed=42):
    groups = np.asarray(grouping)
    n = len(groups)
    ranks = stats.rankdata(squareform(distance_matrix))
    rank_matrix = squareform(ranks)
    within_ranks = []
    between_ranks = []
    for i in range(n):
        for j in range(i + 1, n):
            if groups[i] == groups[j]:
                within_ranks.append(rank_matrix[i, j])
            else:
                between_ranks.append(rank_matrix[i, j])
    r_w = np.mean(within_ranks)
    r_b = np.mean(between_ranks)
    n_comparisons = n * (n - 1) / 2
    r_stat = (r_b - r_w) / (n_comparisons / 2)
    np.random.seed(seed)
    r_perms = []
    for _ in range(n_permutations):
        perm_groups = np.random.permutation(groups)
        wr, br = ([], [])
        for i in range(n):
            for j in range(i + 1, n):
                if perm_groups[i] == perm_groups[j]:
                    wr.append(rank_matrix[i, j])
                else:
                    br.append(rank_matrix[i, j])
        r_perms.append((np.mean(br) - np.mean(wr)) / (n_comparisons / 2))
    p_value = (np.sum(np.array(r_perms) >= r_stat) + 1) / (n_permutations + 1)
    return {'R': r_stat, 'p_value': p_value}
for dist_name, dist_mat in [('Bray-Curtis', bc_matrix)]:
    res_group = anosim_test(dist_mat, aligned_info['Group'])
    res_batch = anosim_test(dist_mat, aligned_info['Bioproject'])
    print(f'  {dist_name}:')
    print(f"    Group: R={res_group['R']:.4f}, p={res_group['p_value']:.4f}")
    print(f"    Batch: R={res_batch['R']:.4f}, p={res_batch['p_value']:.4f}")
print('\n[STEP 5] Within/between group distance analysis')
print('-' * 50)

def calculate_group_distances(dist_matrix, groups):
    n = len(groups)
    within_pcos, within_healthy, between_groups = ([], [], [])
    for i in range(n):
        for j in range(i + 1, n):
            gi, gj = (groups.iloc[i], groups.iloc[j])
            if gi == 'PCOS' and gj == 'PCOS':
                within_pcos.append(dist_matrix[i, j])
            elif gi == 'Healthy' and gj == 'Healthy':
                within_healthy.append(dist_matrix[i, j])
            else:
                between_groups.append(dist_matrix[i, j])
    return (within_pcos, within_healthy, between_groups)
within_pcos, within_healthy, between = calculate_group_distances(bc_matrix, aligned_info['Group'])
print(f'  PCOS within-group:    {np.mean(within_pcos):.4f} +/- {np.std(within_pcos):.4f}')
print(f'  Healthy within-group: {np.mean(within_healthy):.4f} +/- {np.std(within_healthy):.4f}')
print(f'  Between-group:        {np.mean(between):.4f} +/- {np.std(between):.4f}')
stat, pval = stats.mannwhitneyu(within_pcos, within_healthy)
print(f'  Within-group comparison (Wilcoxon): p={pval:.4f}')
print('\n[STEP 6] PCoA visualization')
print('-' * 50)

def perform_pcoa(dist_matrix, sample_names):
    n = dist_matrix.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ dist_matrix ** 2 @ H
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    pos_mask = eigenvalues > 0
    eigenvalues = eigenvalues[pos_mask]
    eigenvectors = eigenvectors[:, pos_mask]
    coords = eigenvectors[:, :2] * np.sqrt(eigenvalues[:2])
    var_explained = eigenvalues[:2] / np.sum(np.abs(eigenvalues)) * 100
    return (coords, var_explained)
pcoa_coords, var_exp = perform_pcoa(bc_matrix, species_tss_t.index)
pcoa_df = pd.DataFrame({'Sample': species_tss_t.index, 'PC1': pcoa_coords[:, 0], 'PC2': pcoa_coords[:, 1]})
pcoa_df = pcoa_df.merge(group_info[['Sample', 'Bioproject', 'Group']], on='Sample')
batch_markers = {'PRJNA530971': 'o', 'PRJNA549764': 's', 'PRJNA791492': '^'}
_extra_markers = ['D', 'v', 'P', 'X', 'h']
_ei = 0
for _b in pcoa_df['Bioproject'].unique():
    if _b not in batch_markers:
        batch_markers[_b] = _extra_markers[_ei % len(_extra_markers)]
        _ei += 1
fig, ax = plt.subplots(figsize=(9, 7))
for grp in ['PCOS', 'Healthy']:
    for batch in batch_order:
        mask = (pcoa_df['Group'] == grp) & (pcoa_df['Bioproject'] == batch)
        if mask.sum() == 0:
            continue
        ax.scatter(pcoa_df.loc[mask, 'PC1'], pcoa_df.loc[mask, 'PC2'], c=colors_group[grp], marker=batch_markers[batch], s=80, alpha=0.75, edgecolors='white', linewidths=0.4, zorder=3)
for grp in ['PCOS', 'Healthy']:
    mask = pcoa_df['Group'] == grp
    x = pcoa_df.loc[mask, 'PC1'].values
    y = pcoa_df.loc[mask, 'PC2'].values
    if len(x) > 2:
        cov = np.cov(x, y)
        evals, evecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(evecs[1, 1], evecs[0, 1]))
        w, h = 2 * 1.96 * np.sqrt(evals)
        ell = Ellipse(xy=(np.mean(x), np.mean(y)), width=w, height=h, angle=angle, facecolor=colors_group[grp], alpha=0.12, edgecolor=colors_group[grp], linestyle='-', linewidth=1.5)
        ax.add_patch(ell)
perm_res = results_two['Bray-Curtis']
stats_text = f"PERMANOVA (adjusted for batch):\nR$^2$ = {perm_res['Factor2_R2']:.3f}, p = {perm_res['Factor2_p']:.3f}"
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.4)
ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.4)
ax.set_xlabel(f'PCoA1 ({var_exp[0]:.1f}%)', fontsize=11)
ax.set_ylabel(f'PCoA2 ({var_exp[1]:.1f}%)', fontsize=11)
ax.set_title('PCoA Analysis (Bray-Curtis Distance)\nPCOS vs Healthy', fontsize=13, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
legend_group = [mpatches.Patch(facecolor=colors_group['PCOS'], label='PCOS'), mpatches.Patch(facecolor=colors_group['Healthy'], label='Healthy')]
legend_batch = [mlines.Line2D([], [], color='gray', marker=batch_markers[b], linestyle='None', markersize=7, label=b) for b in batch_order if b in batch_markers]
leg1 = ax.legend(handles=legend_group, title='Group', loc='upper right', framealpha=0.9, fontsize=9)
ax.add_artist(leg1)
ax.legend(handles=legend_batch, title='Bioproject', loc='lower right', framealpha=0.9, fontsize=8)
plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/pcoa_group_comparison.pdf', bbox_inches='tight', dpi=300)
fig.savefig(f'{OUTPUT_DIR}/pcoa_group_comparison.jpg', bbox_inches='tight', dpi=300)
fig.savefig(f'{OUTPUT_DIR}/pcoa_group_comparison.svg', bbox_inches='tight')
plt.close()
pcoa_df.to_csv(f'{OUTPUT_DIR}/pcoa_group_comparison_plotdata.tsv', sep='\t', index=False)
print('  [DONE] pcoa_group_comparison.pdf / .jpg / .svg')
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
ax = axes[0]
for grp in ['PCOS', 'Healthy']:
    mask = pcoa_df['Group'] == grp
    ax.scatter(pcoa_df.loc[mask, 'PC1'], pcoa_df.loc[mask, 'PC2'], c=colors_group[grp], s=70, alpha=0.7, label=grp, edgecolors='white', linewidths=0.5)
    x = pcoa_df.loc[mask, 'PC1'].values
    y = pcoa_df.loc[mask, 'PC2'].values
    if len(x) > 2:
        cov = np.cov(x, y)
        evals, evecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(evecs[1, 1], evecs[0, 1]))
        w, h = 2 * 1.96 * np.sqrt(evals)
        ell = Ellipse(xy=(np.mean(x), np.mean(y)), width=w, height=h, angle=angle, facecolor=colors_group[grp], alpha=0.15, edgecolor=colors_group[grp], linestyle='-', linewidth=2)
        ax.add_patch(ell)
ax.set_xlabel(f'PCoA1 ({var_exp[0]:.1f}%)', fontsize=11)
ax.set_ylabel(f'PCoA2 ({var_exp[1]:.1f}%)', fontsize=11)
ax.set_title('Colored by Group', fontsize=12, fontweight='bold')
ax.legend(title='Group', loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax = axes[1]
for batch in colors_batch:
    mask = pcoa_df['Bioproject'] == batch
    ax.scatter(pcoa_df.loc[mask, 'PC1'], pcoa_df.loc[mask, 'PC2'], c=colors_batch[batch], s=70, alpha=0.7, label=batch, edgecolors='white', linewidths=0.5)
    x = pcoa_df.loc[mask, 'PC1'].values
    y = pcoa_df.loc[mask, 'PC2'].values
    if len(x) > 2:
        cov = np.cov(x, y)
        evals, evecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(evecs[1, 1], evecs[0, 1]))
        w, h = 2 * 1.96 * np.sqrt(evals)
        ell = Ellipse(xy=(np.mean(x), np.mean(y)), width=w, height=h, angle=angle, facecolor=colors_batch[batch], alpha=0.15, edgecolor=colors_batch[batch], linestyle='-', linewidth=2)
        ax.add_patch(ell)
ax.set_xlabel(f'PCoA1 ({var_exp[0]:.1f}%)', fontsize=11)
ax.set_ylabel(f'PCoA2 ({var_exp[1]:.1f}%)', fontsize=11)
ax.set_title('Colored by Batch', fontsize=12, fontweight='bold')
ax.legend(title='Bioproject', loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.suptitle('PCoA Analysis - Bray-Curtis Distance', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/pcoa_group_batch.pdf', bbox_inches='tight', dpi=300)
fig.savefig(f'{OUTPUT_DIR}/pcoa_group_batch.jpg', bbox_inches='tight', dpi=300)
fig.savefig(f'{OUTPUT_DIR}/pcoa_group_batch.svg', bbox_inches='tight')
plt.close()
print('  [DONE] pcoa_group_batch.pdf / .jpg / .svg')
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, batch in enumerate(batch_order):
    ax = axes[idx]
    batch_data = pcoa_df[pcoa_df['Bioproject'] == batch]
    for grp in ['PCOS', 'Healthy']:
        mask = batch_data['Group'] == grp
        ax.scatter(batch_data.loc[mask, 'PC1'], batch_data.loc[mask, 'PC2'], c=colors_group[grp], s=80, alpha=0.7, label=grp, edgecolors='white', linewidths=0.5)
    n_pcos = (batch_data['Group'] == 'PCOS').sum()
    n_healthy = (batch_data['Group'] == 'Healthy').sum()
    row = adonis_percohort_df[(adonis_percohort_df['Bioproject'] == batch) & (adonis_percohort_df['Distance_metric'] == 'Bray-Curtis')]
    if len(row) and (not np.isnan(row['R2'].values[0])):
        ann = f"PERMANOVA:\nR$^2$={row['R2'].values[0]:.3f}\np={row['p_value'].values[0]:.3f} {row['significance'].values[0]}"
        ax.text(0.02, 0.98, ann, transform=ax.transAxes, fontsize=8, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.75))
    ax.set_xlabel(f'PCoA1 ({var_exp[0]:.1f}%)', fontsize=10)
    ax.set_ylabel(f'PCoA2 ({var_exp[1]:.1f}%)', fontsize=10)
    ax.set_title(f'{batch}\n(PCOS={n_pcos}, Healthy={n_healthy})', fontsize=11, fontweight='bold')
    ax.legend(title='Group', loc='upper right', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
plt.suptitle('PCoA by Batch - Group Separation within Each Study', fontsize=13, fontweight='bold', y=1.05)
plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/pcoa_by_batch.pdf', bbox_inches='tight', dpi=300)
fig.savefig(f'{OUTPUT_DIR}/pcoa_by_batch.jpg', bbox_inches='tight', dpi=300)
fig.savefig(f'{OUTPUT_DIR}/pcoa_by_batch.svg', bbox_inches='tight')
plt.close()
print('  [DONE] pcoa_by_batch.pdf / .jpg / .svg')
fig, ax = plt.subplots(figsize=(8, 6))
bp = ax.boxplot([within_pcos, within_healthy, between], positions=[1, 2, 3], widths=0.6, patch_artist=True)
for patch, color in zip(bp['boxes'], [colors_group['PCOS'], colors_group['Healthy'], '#888888']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['Within\nPCOS', 'Within\nHealthy', 'Between\nGroups'], fontsize=11)
ax.set_ylabel('Bray-Curtis Distance', fontsize=12)
ax.set_title('Within-group vs Between-group Distances', fontsize=13, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
stat_wh, pval_wh = stats.mannwhitneyu(within_pcos, within_healthy)
y_max = max(max(within_pcos), max(within_healthy))
ax.plot([1, 1, 2, 2], [y_max + 0.02, y_max + 0.03, y_max + 0.03, y_max + 0.02], 'k-', linewidth=1)
sig_text = '***' if pval_wh < 0.001 else '**' if pval_wh < 0.01 else '*' if pval_wh < 0.05 else 'ns'
ax.text(1.5, y_max + 0.04, f'{sig_text}\np={pval_wh:.3f}', ha='center', fontsize=10)
plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/distance_comparison.pdf', bbox_inches='tight', dpi=300)
fig.savefig(f'{OUTPUT_DIR}/distance_comparison.jpg', bbox_inches='tight', dpi=300)
fig.savefig(f'{OUTPUT_DIR}/distance_comparison.svg', bbox_inches='tight')
dist_comp_df = pd.DataFrame({'Distance': within_pcos + within_healthy + between, 'Comparison': ['Within_PCOS'] * len(within_pcos) + ['Within_Healthy'] * len(within_healthy) + ['Between_Groups'] * len(between)})
dist_comp_df.to_csv(f'{OUTPUT_DIR}/distance_comparison_plotdata.tsv', sep='\t', index=False)
plt.close()
print('  [DONE] distance_comparison.pdf / .jpg / .svg')
print('\n  Running NMDS...')
nmds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, max_iter=1000, n_init=10, normalized_stress='auto')
nmds_coords = nmds.fit_transform(bc_matrix)
stress = nmds.stress_
fig, ax = plt.subplots(figsize=(10, 8))
for grp in ['PCOS', 'Healthy']:
    mask = aligned_info['Group'] == grp
    ax.scatter(nmds_coords[mask, 0], nmds_coords[mask, 1], c=colors_group[grp], s=80, alpha=0.7, label=grp, edgecolors='white', linewidths=0.5)
    x = nmds_coords[mask, 0]
    y = nmds_coords[mask, 1]
    if len(x) > 2:
        cov = np.cov(x, y)
        evals, evecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(evecs[1, 1], evecs[0, 1]))
        w, h = 2 * 1.96 * np.sqrt(evals)
        ell = Ellipse(xy=(np.mean(x), np.mean(y)), width=w, height=h, angle=angle, facecolor=colors_group[grp], alpha=0.15, edgecolor=colors_group[grp], linestyle='-', linewidth=2)
        ax.add_patch(ell)
ax.set_xlabel('NMDS1', fontsize=12)
ax.set_ylabel('NMDS2', fontsize=12)
ax.set_title(f'NMDS Analysis (Bray-Curtis Distance)\nStress = {stress:.3f}', fontsize=14, fontweight='bold')
ax.legend(title='Group', loc='upper right', framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/nmds_analysis.pdf', bbox_inches='tight', dpi=300)
fig.savefig(f'{OUTPUT_DIR}/nmds_analysis.jpg', bbox_inches='tight', dpi=300)
fig.savefig(f'{OUTPUT_DIR}/nmds_analysis.svg', bbox_inches='tight')
plt.close()
print('  [DONE] nmds_analysis.pdf / .jpg / .svg')
print('\n[STEP 7] Saving analysis results')
print('-' * 50)
bc_df.to_csv(f'{OUTPUT_DIR}/bray_curtis_distance_matrix.csv')
pcoa_df.to_csv(f'{OUTPUT_DIR}/pcoa_coordinates.csv', index=False)
stats_results = []
for dist_name in results_single:
    stats_results.append({'Distance_metric': dist_name, 'Group_R2': results_single[dist_name]['Group']['R2'], 'Group_F': results_single[dist_name]['Group']['F'], 'Group_p': results_single[dist_name]['Group']['p_value'], 'Batch_R2': results_single[dist_name]['Batch']['R2'], 'Batch_F': results_single[dist_name]['Batch']['F'], 'Batch_p': results_single[dist_name]['Batch']['p_value'], 'Group_adjusted_R2': results_two[dist_name]['Factor2_R2'], 'Group_adjusted_p': results_two[dist_name]['Factor2_p']})
stats_df = pd.DataFrame(stats_results)
stats_df.to_csv(f'{OUTPUT_DIR}/permanova_results.csv', index=False)
beta_data = {'bc_matrix': bc_matrix, 'jaccard_matrix': jaccard_matrix, 'aitchison_matrix': aitchison_matrix, 'pcoa_df': pcoa_df, 'permanova_single': results_single, 'permanova_two': results_two, 'adonis_per_cohort': adonis_percohort_df}
with open(f'{OUTPUT_DIR}/beta_diversity_data.pkl', 'wb') as f:
    pickle.dump(beta_data, f)
print('  Saved files:')
print('    - bray_curtis_distance_matrix.csv')
print('    - pcoa_coordinates.csv')
print('    - permanova_results.csv')
print('    - adonis_per_cohort_PCOS_vs_Healthy.tsv   [NEW]')
print('    - pcoa_group_comparison_plotdata.tsv       [NEW]')
print('    - distance_comparison_plotdata.tsv         [NEW]')
print('    - beta_diversity_data.pkl')
print('\n' + '=' * 70)
print('Beta Diversity Analysis Summary')
print('=' * 70)
print('\n[PERMANOVA Results] (Bray-Curtis)')
print('  Single-factor:')
print(f"    Group: R2={results_single['Bray-Curtis']['Group']['R2']:.4f}, p={results_single['Bray-Curtis']['Group']['p_value']:.4f}")
print(f"    Batch: R2={results_single['Bray-Curtis']['Batch']['R2']:.4f}, p={results_single['Bray-Curtis']['Batch']['p_value']:.4f}")
print('  Two-factor (batch-adjusted):')
print(f"    Group: R2={results_two['Bray-Curtis']['Factor2_R2']:.4f}, p={results_two['Bray-Curtis']['Factor2_p']:.4f}")
print('\n[Per-cohort ADONIS Summary] (Bray-Curtis)')
for _, row in adonis_percohort_df[adonis_percohort_df['Distance_metric'] == 'Bray-Curtis'].iterrows():
    print(f"  {row['Bioproject']}: n={int(row['n_total'])} R2={row['R2']:.4f}  p={row['p_value']:.4e}  {row['significance']}")
print('\n[Within/Between Distances]')
print(f'  PCOS within:    {np.mean(within_pcos):.4f} +/- {np.std(within_pcos):.4f}')
print(f'  Healthy within: {np.mean(within_healthy):.4f} +/- {np.std(within_healthy):.4f}')
print(f'  Between groups: {np.mean(between):.4f} +/- {np.std(between):.4f}')
print('\n' + '=' * 70)
print(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print('=' * 70)
