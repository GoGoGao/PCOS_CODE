from pathlib import Path
import os
PKG_ROOT = Path(__file__).resolve().parents[1]
SHARED = PKG_ROOT / 'shared_input'
RSCRIPT = os.environ.get('RSCRIPT', 'Rscript')
INPUT_DIR = str(SHARED)
OUTPUT_DIR = str(PKG_ROOT / 'output' / 'FigS4')
import os
import pickle
import subprocess
import tempfile
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from scipy.spatial.distance import pdist, squareform
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Nimbus Roman', 'Liberation Serif', 'DejaVu Serif']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.weight'] = 'normal'
matplotlib.rcParams['axes.labelweight'] = 'normal'
matplotlib.rcParams['axes.titleweight'] = 'normal'
OUT_DIR = Path(__file__).resolve().parent
WORK = Path('.')
PKL = WORK / '01_preprocessing' / 'metaphlan_preprocessed.pkl'
BATCH_ORDER = ['PRJNA530971', 'PRJNA549764', 'PRJNA791492']
GROUP_ORDER = ['Healthy', 'PCOS']

def perform_pcoa(dist_matrix: np.ndarray):
    n = dist_matrix.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ dist_matrix ** 2 @ H
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    pos_mask = eigenvalues > 1e-12
    eigenvalues = eigenvalues[pos_mask]
    eigenvectors = eigenvectors[:, pos_mask]
    coords = eigenvectors[:, :2] * np.sqrt(eigenvalues[:2])
    var_explained = eigenvalues[:2] / np.sum(np.abs(eigenvalues)) * 100
    return (coords, var_explained)

def permanova_test(distance_matrix, grouping, n_permutations=999, seed=42):
    groups = np.asarray(grouping)
    unique_groups = np.unique(groups)
    n = len(groups)

    def calc_ss(dist_mat, grps):
        ss_total = np.sum(dist_mat ** 2) / (2 * n)
        ss_within = 0.0
        for g in np.unique(grps):
            mask = grps == g
            n_g = int(np.sum(mask))
            if n_g > 1:
                ss_within += np.sum(dist_mat[np.ix_(mask, mask)] ** 2) / (2 * n_g)
        return (ss_total - ss_within, ss_within, ss_total)
    ss_between, ss_within, ss_total = calc_ss(distance_matrix, groups)
    df_between = len(unique_groups) - 1
    df_within = n - len(unique_groups)
    f_stat = ss_between / df_between / (ss_within / df_within)
    r_squared = ss_between / ss_total
    rng = np.random.default_rng(seed)
    f_perms = []
    for _ in range(n_permutations):
        perm_groups = rng.permutation(groups)
        ss_b, ss_w, _ = calc_ss(distance_matrix, perm_groups)
        f_perms.append(ss_b / df_between / (ss_w / df_within))
    p_value = (np.sum(np.asarray(f_perms) >= f_stat) + 1) / (n_permutations + 1)
    return {'R2': float(r_squared), 'F': float(f_stat), 'p_value': float(p_value)}

def permanova_two_factor(distance_matrix, factor1, factor2, n_permutations=999, seed=42):
    n = len(factor1)
    groups1 = np.asarray(factor1)
    groups2 = np.asarray(factor2)
    ss_total = np.sum(distance_matrix ** 2) / (2 * n)
    ss_within1 = 0.0
    for g in np.unique(groups1):
        mask = groups1 == g
        n_g = int(np.sum(mask))
        if n_g > 1:
            ss_within1 += np.sum(distance_matrix[np.ix_(mask, mask)] ** 2) / (2 * n_g)
    ss_factor1 = ss_total - ss_within1
    combo = np.array([f'{a}|{b}' for a, b in zip(groups1, groups2)])
    ss_within_combo = 0.0
    for g in np.unique(combo):
        mask = combo == g
        n_g = int(np.sum(mask))
        if n_g > 1:
            ss_within_combo += np.sum(distance_matrix[np.ix_(mask, mask)] ** 2) / (2 * n_g)
    ss_factor2 = ss_within1 - ss_within_combo
    df1 = len(np.unique(groups1)) - 1
    df2 = len(np.unique(combo)) - len(np.unique(groups1))
    df_res = n - len(np.unique(combo))
    f_stat = ss_factor2 / df2 / (ss_within_combo / df_res) if df_res > 0 else np.nan
    r2_f1 = ss_factor1 / ss_total
    r2_f2 = ss_factor2 / ss_total
    rng = np.random.default_rng(seed)
    f_perms = []
    for _ in range(n_permutations):
        perm2 = groups2.copy()
        for g in np.unique(groups1):
            mask = groups1 == g
            perm2[mask] = rng.permutation(groups2[mask])
        combo_p = np.array([f'{a}|{b}' for a, b in zip(groups1, perm2)])
        ss_w1 = 0.0
        for gg in np.unique(groups1):
            mask = groups1 == gg
            n_g = int(np.sum(mask))
            if n_g > 1:
                ss_w1 += np.sum(distance_matrix[np.ix_(mask, mask)] ** 2) / (2 * n_g)
        ss_wcombo = 0.0
        for gg in np.unique(combo_p):
            mask = combo_p == gg
            n_g = int(np.sum(mask))
            if n_g > 1:
                ss_wcombo += np.sum(distance_matrix[np.ix_(mask, mask)] ** 2) / (2 * n_g)
        ss_f2 = ss_w1 - ss_wcombo
        f_perms.append(ss_f2 / df2 / (ss_wcombo / df_res))
    p_value = (np.sum(np.asarray(f_perms) >= f_stat) + 1) / (n_permutations + 1)
    return {'Factor1_R2': float(r2_f1), 'Factor2_R2': float(r2_f2), 'Factor2_F': float(f_stat), 'Factor2_p': float(p_value)}

def apply_combat_r(clr_df: pd.DataFrame, batch: pd.Series, group: pd.Series) -> pd.DataFrame:
    with tempfile.TemporaryDirectory(prefix='combat_pcoa_') as tmp:
        tmp = Path(tmp)
        mat_path = tmp / 'clr.tsv'
        meta_path = tmp / 'meta.tsv'
        out_path = tmp / 'clr_combat.tsv'
        clr_df.to_csv(mat_path, sep='\t')
        meta = pd.DataFrame({'Sample': clr_df.columns.astype(str), 'Batch': batch.loc[clr_df.columns].astype(str).values, 'Group': group.loc[clr_df.columns].astype(str).values})
        meta.to_csv(meta_path, sep='\t', index=False)
        r_script = f"\nsuppressPackageStartupMessages(library(sva))\nmat <- as.matrix(read.delim('{mat_path}', row.names=1, check.names=FALSE))\nmeta <- read.delim('{meta_path}', stringsAsFactors=FALSE)\nstopifnot(identical(colnames(mat), meta$Sample))\nbatch <- factor(meta$Batch)\nmod <- model.matrix(~ Group, data=meta)\ncorrected <- ComBat(dat=mat, batch=batch, mod=mod, par.prior=TRUE, prior.plots=FALSE)\nwrite.table(corrected, file='{out_path}', sep='\\t', quote=FALSE, col.names=NA)\n"
        r_path = tmp / 'run_combat.R'
        r_path.write_text(r_script)
        proc = subprocess.run(['Rscript', str(r_path)], capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f'ComBat failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}')
        corrected = pd.read_csv(out_path, sep='\t', index_col=0)
        corrected.columns = corrected.columns.astype(str)
        return corrected

def confidence_ellipse(ax, x, y, color, n_std=1.96, **kwargs):
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = (vals[order], vecs[:, order])
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(np.maximum(vals, 0))
    ell = Ellipse(xy=(np.mean(x), np.mean(y)), width=width, height=height, angle=theta, facecolor='none', edgecolor=color, lw=1.2, **kwargs)
    ax.add_patch(ell)

def save_fig(fig, stem: Path):
    for ext in ('pdf', 'svg', 'png', 'jpg'):
        fig.savefig(f'{stem}.{ext}', dpi=300 if ext in ('png', 'jpg') else None, bbox_inches='tight', pad_inches=0.05, facecolor='white')

def main():
    os.chdir(OUT_DIR)
    print('[1] Load preprocessed data')
    with open(PKL, 'rb') as f:
        data = pickle.load(f)
    species_tss = data['species_tss']
    species_clr = data['species_clr']
    group_info = data['group_info'].copy()
    colors_group = data['colors_group']
    colors_batch = data['colors_batch']
    group_info = group_info.set_index('Sample', drop=False)
    samples = list(species_tss.columns.astype(str))
    group_info = group_info.loc[samples]
    batch = group_info['Bioproject'].astype(str)
    group = group_info['Group'].astype(str)
    print(f'  taxa={species_tss.shape[0]}, samples={len(samples)}')
    print('[2] Distances: uncorrected Bray-Curtis (TSS) and Aitchison (CLR)')
    tss_t = species_tss.T
    clr_t = species_clr.T
    bc_unc = squareform(pdist(tss_t.values, metric='braycurtis'))
    ait_unc = squareform(pdist(clr_t.values, metric='euclidean'))
    print('[3] ComBat on CLR (batch=Bioproject, mod=~Group)')
    clr_c = apply_combat_r(species_clr.copy(), batch, group)
    clr_c = clr_c[samples]
    ait_cor = squareform(pdist(clr_c.T.values, metric='euclidean'))
    clr_back = np.exp(clr_c.values)
    clr_back = clr_back / clr_back.sum(axis=0, keepdims=True)
    bc_cor = squareform(pdist(clr_back.T, metric='braycurtis'))
    print('[4] PERMANOVA')
    rows = []
    for label, dist in [('BrayCurtis_uncorrected', bc_unc), ('Aitchison_uncorrected', ait_unc), ('Aitchison_ComBat', ait_cor), ('BrayCurtis_from_ComBatCLR', bc_cor)]:
        g = permanova_test(dist, group.values)
        b = permanova_test(dist, batch.values)
        adj = permanova_two_factor(dist, batch.values, group.values)
        rows.append({'Distance': label, 'Group_R2': g['R2'], 'Group_F': g['F'], 'Group_p': g['p_value'], 'Batch_R2': b['R2'], 'Batch_F': b['F'], 'Batch_p': b['p_value'], 'Batch_then_Group_R2': adj['Factor2_R2'], 'Batch_then_Group_F': adj['Factor2_F'], 'Batch_then_Group_p': adj['Factor2_p'], 'Batch_marginal_R2': adj['Factor1_R2']})
        print(f"  {label}: Group R2={g['R2']:.4f} p={g['p_value']:.4f}; Batch R2={b['R2']:.4f} p={b['p_value']:.4f}; Group|Batch R2={adj['Factor2_R2']:.4f} p={adj['Factor2_p']:.4f}")
    permanova_df = pd.DataFrame(rows)
    permanova_df.to_csv(OUT_DIR / 'permanova_before_after_ComBat.tsv', sep='\t', index=False)
    print('[5] PCoA')
    coords_list = []
    var_map = {}
    for tag, dist in [('BrayCurtis_uncorrected', bc_unc), ('Aitchison_ComBat', ait_cor), ('BrayCurtis_from_ComBatCLR', bc_cor)]:
        coords, var_exp = perform_pcoa(dist)
        var_map[tag] = var_exp
        df = pd.DataFrame({'Sample': samples, 'PC1': coords[:, 0], 'PC2': coords[:, 1], 'Bioproject': batch.values, 'Group': group.values, 'Distance': tag, 'PC1_var_pct': var_exp[0], 'PC2_var_pct': var_exp[1]})
        coords_list.append(df)
    coords_all = pd.concat(coords_list, ignore_index=True)
    coords_all.to_csv(OUT_DIR / 'pcoa_coordinates_before_after.tsv', sep='\t', index=False)
    print('[6] Figures')
    batch_markers = {'PRJNA530971': 'o', 'PRJNA549764': 's', 'PRJNA791492': '^'}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.6))
    panel_defs = [(axes[0, 0], 'BrayCurtis_uncorrected', 'group', 'A. Uncorrected (color = Group)'), (axes[0, 1], 'BrayCurtis_uncorrected', 'batch', 'B. Uncorrected (color = Cohort)'), (axes[1, 0], 'Aitchison_ComBat', 'group', 'C. ComBat-corrected (color = Group)'), (axes[1, 1], 'Aitchison_ComBat', 'batch', 'D. ComBat-corrected (color = Cohort)')]
    plot_rows = []
    for ax, tag, mode, title in panel_defs:
        sub = coords_all[coords_all['Distance'] == tag].copy()
        v1, v2 = var_map[tag]
        if mode == 'group':
            for grp in GROUP_ORDER:
                for bname in BATCH_ORDER:
                    m = (sub['Group'] == grp) & (sub['Bioproject'] == bname)
                    if m.sum() == 0:
                        continue
                    ax.scatter(sub.loc[m, 'PC1'], sub.loc[m, 'PC2'], c=colors_group[grp], marker=batch_markers[bname], s=28, alpha=0.8, edgecolors='white', linewidths=0.3, zorder=3)
            for grp in GROUP_ORDER:
                m = sub['Group'] == grp
                confidence_ellipse(ax, sub.loc[m, 'PC1'].values, sub.loc[m, 'PC2'].values, colors_group[grp])
        else:
            for bname in BATCH_ORDER:
                m = sub['Bioproject'] == bname
                ax.scatter(sub.loc[m, 'PC1'], sub.loc[m, 'PC2'], c=colors_batch[bname], marker='o', s=28, alpha=0.8, edgecolors='white', linewidths=0.3, zorder=3)
            for bname in BATCH_ORDER:
                m = sub['Bioproject'] == bname
                confidence_ellipse(ax, sub.loc[m, 'PC1'].values, sub.loc[m, 'PC2'].values, colors_batch[bname])
        ax.set_xlabel(f'PCoA1 ({v1:.1f}%)', fontsize=9)
        ax.set_ylabel(f'PCoA2 ({v2:.1f}%)', fontsize=9)
        ax.set_title(title, fontsize=9, pad=4)
        ax.tick_params(labelsize=8)
        ax.set_aspect('equal', adjustable='datalim')
        for _, r in sub.iterrows():
            plot_rows.append({'panel': title[0], 'Distance': tag, 'color_by': mode, 'Sample': r['Sample'], 'PC1': r['PC1'], 'PC2': r['PC2'], 'Group': r['Group'], 'Bioproject': r['Bioproject'], 'PC1_var_pct': v1, 'PC2_var_pct': v2})
    g_handles = [mpatches.Patch(color=colors_group[g], label=g, alpha=0.85) for g in GROUP_ORDER]
    b_handles = [mpatches.Patch(color=colors_batch[b], label=b, alpha=0.85) for b in BATCH_ORDER]
    axes[0, 0].legend(handles=g_handles, frameon=False, fontsize=7, loc='best')
    axes[0, 1].legend(handles=b_handles, frameon=False, fontsize=6.5, loc='best')
    axes[1, 0].legend(handles=g_handles, frameon=False, fontsize=7, loc='best')
    axes[1, 1].legend(handles=b_handles, frameon=False, fontsize=6.5, loc='best')
    fig.tight_layout(pad=0.6)
    stem = OUT_DIR / 'Fig_PCoA_before_after_ComBat'
    save_fig(fig, stem)
    pd.DataFrame(plot_rows).to_csv(f'{stem}_plotdata.tsv', sep='\t', index=False)
    plt.close(fig)
    fig2, ax = plt.subplots(figsize=(4.8, 3.2))
    focus = permanova_df[permanova_df['Distance'].isin(['BrayCurtis_uncorrected', 'Aitchison_ComBat'])].copy()
    focus['label'] = focus['Distance'].map({'BrayCurtis_uncorrected': 'Uncorrected\nBray-Curtis', 'Aitchison_ComBat': 'ComBat-CLR\nAitchison'})
    x = np.arange(len(focus))
    w = 0.35
    bars1 = ax.bar(x - w / 2, focus['Batch_R2'] * 100, width=w, color='#3C5488', label='Cohort (Batch)', edgecolor='none')
    bars2 = ax.bar(x + w / 2, focus['Group_R2'] * 100, width=w, color='#E64B35', label='Group (PCOS/Healthy)', edgecolor='none')
    ax.set_xticks(x)
    ax.set_xticklabels(focus['label'], fontsize=8)
    ax.set_ylabel('PERMANOVA R² (%)', fontsize=9)
    ax.set_title('Variance explained by cohort vs disease', fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    ax.tick_params(labelsize=8)
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3, f'{h:.1f}', ha='center', va='bottom', fontsize=7)
    ax.set_ylim(0, max(focus['Batch_R2'].max(), focus['Group_R2'].max()) * 100 * 1.25)
    fig2.tight_layout(pad=0.4)
    stem2 = OUT_DIR / 'Fig_variance_Batch_vs_Group'
    save_fig(fig2, stem2)
    focus.to_csv(f'{stem2}_plotdata.tsv', sep='\t', index=False)
    plt.close(fig2)
    fig3, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))
    for ax, tag, title in [(axes[0], 'BrayCurtis_uncorrected', 'Uncorrected Bray-Curtis'), (axes[1], 'Aitchison_ComBat', 'ComBat-corrected Aitchison')]:
        sub = coords_all[coords_all['Distance'] == tag]
        v1, v2 = var_map[tag]
        for grp in GROUP_ORDER:
            for bname in BATCH_ORDER:
                m = (sub['Group'] == grp) & (sub['Bioproject'] == bname)
                if m.sum() == 0:
                    continue
                ax.scatter(sub.loc[m, 'PC1'], sub.loc[m, 'PC2'], c=colors_group[grp], marker=batch_markers[bname], s=32, alpha=0.82, edgecolors='white', linewidths=0.3, zorder=3)
        for grp in GROUP_ORDER:
            m = sub['Group'] == grp
            confidence_ellipse(ax, sub.loc[m, 'PC1'].values, sub.loc[m, 'PC2'].values, colors_group[grp])
        row = permanova_df[permanova_df['Distance'] == tag].iloc[0]
        txt = f"Group R²={row['Group_R2']:.3f}, P={row['Group_p']:.3f}\nCohort R²={row['Batch_R2']:.3f}, P={row['Batch_p']:.3f}"
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, va='top', ha='left', fontsize=7, family='serif')
        ax.set_xlabel(f'PCoA1 ({v1:.1f}%)', fontsize=9)
        ax.set_ylabel(f'PCoA2 ({v2:.1f}%)', fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=8)
    shape_handles = [plt.Line2D([0], [0], marker=batch_markers[b], color='gray', linestyle='None', markersize=6, label=b) for b in BATCH_ORDER]
    axes[1].legend(handles=g_handles + shape_handles, frameon=False, fontsize=6.5, loc='best')
    fig3.tight_layout(pad=0.5)
    stem3 = OUT_DIR / 'Fig_PCoA_main_before_after'
    save_fig(fig3, stem3)
    coords_all[coords_all['Distance'].isin(['BrayCurtis_uncorrected', 'Aitchison_ComBat'])].to_csv(f'{stem3}_plotdata.tsv', sep='\t', index=False)
    plt.close(fig3)
    clr_c.to_csv(OUT_DIR / 'species_CLR_ComBat.tsv', sep='\t')
    summary = {'n_samples': len(samples), 'n_taxa': int(species_tss.shape[0]), 'uncorrected_BrayCurtis_Batch_R2': float(permanova_df.loc[permanova_df['Distance'] == 'BrayCurtis_uncorrected', 'Batch_R2'].iloc[0]), 'uncorrected_BrayCurtis_Group_R2': float(permanova_df.loc[permanova_df['Distance'] == 'BrayCurtis_uncorrected', 'Group_R2'].iloc[0]), 'ComBat_Aitchison_Batch_R2': float(permanova_df.loc[permanova_df['Distance'] == 'Aitchison_ComBat', 'Batch_R2'].iloc[0]), 'ComBat_Aitchison_Group_R2': float(permanova_df.loc[permanova_df['Distance'] == 'Aitchison_ComBat', 'Group_R2'].iloc[0]), 'ComBat_Aitchison_Group_p': float(permanova_df.loc[permanova_df['Distance'] == 'Aitchison_ComBat', 'Group_p'].iloc[0]), 'manuscript_reported_Group_adj_R2': 0.039, 'manuscript_reported_Group_adj_p': 0.001}
    pd.Series(summary).to_csv(OUT_DIR / 'key_stats_summary.tsv', sep='\t', header=['value'])
    print('[DONE] outputs in', OUT_DIR)
if __name__ == '__main__':
    main()
