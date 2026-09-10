from pathlib import Path
import os
PKG_ROOT = Path(__file__).resolve().parents[1]
SHARED = PKG_ROOT / 'shared_input'
RSCRIPT = os.environ.get('RSCRIPT', 'Rscript')
import gzip
ROOT = PKG_ROOT / 'output' / 'FigS10_HUMAnN'
OUT = ROOT
FIG = OUT / 'figures'
FIG_P = FIG / 'panels'
SRC = OUT / 'source_data'
TAB = OUT / 'tables'
HUMANN = SHARED / 'humann_ko_cpm_named.tsv.gz'
TECH = SHARED / 'L1_bias' / 'sample_technical_metrics.tsv'
GROUP_INFO = SHARED / 'sample_group_info_aligned.csv'

def _open_text(path):
    path = Path(path)
    if str(path).endswith('.gz'):
        return gzip.open(path, 'rt')
    return open(path, 'r')
from pathlib import Path
import os
import gzip
PKG_ROOT = Path(__file__).resolve().parents[1]
SHARED = PKG_ROOT / 'shared_input'
import math
import os
import re
import subprocess
import tempfile
import warnings
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from scipy.spatial.distance import pdist, squareform
warnings.filterwarnings('ignore')
matplotlib.use('Agg')
matplotlib.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'font.family': 'serif', 'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif', 'DejaVu Serif'], 'axes.unicode_minus': False, 'font.weight': 'normal', 'axes.labelweight': 'normal', 'axes.titleweight': 'normal'})
PREV = 0.05
BATCH_ORDER = ['PRJNA530971', 'PRJNA549764', 'PRJNA791492']
GROUP_ORDER = ['Healthy', 'PCOS']
COLORS_GROUP = {'PCOS': '#E64B35', 'Healthy': '#4DBBD5'}
COLORS_BATCH = {'PRJNA530971': '#00A087', 'PRJNA549764': '#3C5488', 'PRJNA791492': '#F39B7F'}
BATCH_MARKERS = {'PRJNA530971': 'o', 'PRJNA549764': 's', 'PRJNA791492': '^'}

def ensure_dirs():
    for d in (FIG, FIG_P, SRC, TAB):
        d.mkdir(parents=True, exist_ok=True)

def extract_ko_id(name: str) -> str | None:
    m = re.match('^(K\\d{5})', name.strip().strip('"'))
    return m.group(1) if m else None

def load_humann_community() -> pd.DataFrame:
    print('[1] Load HUMAnN community KO table')
    sample_cols = None
    data_rows = []
    ko_ids = []
    with _open_text(HUMANN) as f:
        header = f.readline().rstrip('\n').split('\t')
        sample_cols = [h.replace('_350.nohost_Abundance-RPKs', '') for h in header[1:]]
        for line in f:
            parts = line.rstrip('\n').split('\t')
            gf = parts[0].strip('"')
            if not gf.startswith('K') or '|' in gf:
                continue
            kid = extract_ko_id(gf)
            if not kid:
                continue
            vals = np.array([float(x) if x else 0.0 for x in parts[1:]], dtype=float)
            ko_ids.append(kid)
            data_rows.append(vals)
    mat = pd.DataFrame(data_rows, index=ko_ids, columns=sample_cols)
    mat = mat.groupby(level=0).max()
    print(f'  raw: {mat.shape[0]} KOs x {mat.shape[1]} samples')
    return mat

def load_meta(samples: list[str]) -> pd.DataFrame:
    tech = pd.read_csv(TECH, sep='\t')
    tech = tech.rename(columns={'sample': 'Sample', 'BIOPROJECT': 'Bioproject'})
    gi = pd.read_csv(GROUP_INFO)
    meta = tech.merge(gi[['Sample', 'Bioproject', 'Group']], on='Sample', how='inner', suffixes=('', '_gi'))
    if 'Bioproject_gi' in meta.columns:
        meta['Bioproject'] = meta['Bioproject'].fillna(meta['Bioproject_gi'])
        meta = meta.drop(columns=['Bioproject_gi'])
    if 'Group_gi' in meta.columns:
        meta['Group'] = meta['Group'].fillna(meta['Group_gi'])
        meta = meta.drop(columns=['Group_gi'])
    meta = meta.set_index('Sample')
    common = [s for s in samples if s in meta.index]
    meta = meta.loc[common].copy()
    meta['log_depth'] = np.log1p(meta['depth_gb'].astype(float))
    meta['Bioproject'] = meta['Bioproject'].astype(str)
    meta['Group'] = meta['Group'].astype(str)
    return meta

def filter_and_tss(mat: pd.DataFrame, samples: list[str]) -> pd.DataFrame:
    mat = mat[samples]
    n = len(samples)
    lo = math.ceil(n * PREV)
    freq = (mat > 0).sum(axis=1)
    keep = freq >= lo
    mat = mat.loc[keep]
    colsum = mat.sum(axis=0)
    mat = mat.div(colsum.replace(0, np.nan), axis=1).fillna(0.0)
    print(f'  after prev>={PREV:.0%}: {mat.shape[0]} KOs x {mat.shape[1]} samples')
    return mat

def clr_transform(tss: pd.DataFrame) -> pd.DataFrame:
    pos = tss.values[tss.values > 0]
    if len(pos) == 0:
        raise ValueError('No positive abundances for CLR')
    pc = float(pos.min()) / 2.0
    logx = np.log(tss.values + pc)
    clr = logx - logx.mean(axis=0, keepdims=True)
    return pd.DataFrame(clr, index=tss.index, columns=tss.columns)

def apply_combat_r(clr_df: pd.DataFrame, batch: pd.Series, group: pd.Series, log_depth: pd.Series) -> pd.DataFrame:
    with tempfile.TemporaryDirectory(prefix='combat_humann_') as tmp:
        tmp = Path(tmp)
        mat_path = tmp / 'clr.tsv'
        meta_path = tmp / 'meta.tsv'
        out_path = tmp / 'clr_combat.tsv'
        clr_df.to_csv(mat_path, sep='\t')
        meta = pd.DataFrame({'Sample': clr_df.columns.astype(str), 'Batch': batch.loc[clr_df.columns].astype(str).values, 'Group': group.loc[clr_df.columns].astype(str).values, 'log_depth': log_depth.loc[clr_df.columns].astype(float).values})
        meta.to_csv(meta_path, sep='\t', index=False)
        r_script = f"\nsuppressPackageStartupMessages(library(sva))\nmat <- as.matrix(read.delim('{mat_path}', row.names=1, check.names=FALSE))\nmeta <- read.delim('{meta_path}', stringsAsFactors=FALSE)\nstopifnot(identical(colnames(mat), meta$Sample))\nbatch <- factor(meta$Batch)\nmod <- model.matrix(~ Group + log_depth, data=meta)\ncorrected <- ComBat(dat=mat, batch=batch, mod=mod, par.prior=TRUE, prior.plots=FALSE)\nwrite.table(corrected, file='{out_path}', sep='\\t', quote=FALSE, col.names=NA)\n"
        r_path = tmp / 'run_combat.R'
        r_path.write_text(r_script)
        proc = subprocess.run([RSCRIPT, str(r_path)], capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f'ComBat failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}')
        corrected = pd.read_csv(out_path, sep='\t', index_col=0)
        corrected.columns = corrected.columns.astype(str)
        return corrected

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

def save_fig(fig, stem: Path, also_panels: bool=False):
    for folder in [FIG, FIG_P] if also_panels else [FIG]:
        folder.mkdir(parents=True, exist_ok=True)
        out = folder / stem.name
        for ext in ('pdf', 'svg', 'png', 'jpg'):
            fig.savefig(f'{out}.{ext}', dpi=300 if ext in ('png', 'jpg') else None, bbox_inches='tight', pad_inches=0.05, facecolor='white')

def save_tsv_csv(df: pd.DataFrame, stem: Path):
    df.to_csv(f'{stem}.tsv', sep='\t', index=False)
    df.to_csv(f'{stem}.csv', index=False)

def plot_pcoa_panel(ax, sub: pd.DataFrame, v1: float, v2: float, mode: str, title: str, adonis_txt: str | None=None):
    if mode == 'group':
        for grp in GROUP_ORDER:
            for bname in BATCH_ORDER:
                m = (sub['Group'] == grp) & (sub['Bioproject'] == bname)
                if m.sum() == 0:
                    continue
                ax.scatter(sub.loc[m, 'PC1'], sub.loc[m, 'PC2'], c=COLORS_GROUP[grp], marker=BATCH_MARKERS[bname], s=32, alpha=0.82, edgecolors='white', linewidths=0.3, zorder=3)
        for grp in GROUP_ORDER:
            m = sub['Group'] == grp
            confidence_ellipse(ax, sub.loc[m, 'PC1'].values, sub.loc[m, 'PC2'].values, COLORS_GROUP[grp])
    else:
        for bname in BATCH_ORDER:
            m = sub['Bioproject'] == bname
            ax.scatter(sub.loc[m, 'PC1'], sub.loc[m, 'PC2'], c=COLORS_BATCH[bname], marker='o', s=32, alpha=0.82, edgecolors='white', linewidths=0.3, zorder=3)
        for bname in BATCH_ORDER:
            m = sub['Bioproject'] == bname
            confidence_ellipse(ax, sub.loc[m, 'PC1'].values, sub.loc[m, 'PC2'].values, COLORS_BATCH[bname])
    ax.set_xlabel(f'PCoA1 ({v1:.1f}%)', fontsize=9)
    ax.set_ylabel(f'PCoA2 ({v2:.1f}%)', fontsize=9)
    ax.set_title(title, fontsize=9, pad=4)
    ax.tick_params(labelsize=8)
    ax.set_aspect('equal', adjustable='datalim')
    if adonis_txt:
        ax.text(0.02, 0.02, adonis_txt, transform=ax.transAxes, va='bottom', ha='left', fontsize=7, family='serif', bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='0.75', alpha=0.9))

def main():
    ensure_dirs()
    os.chdir(OUT)
    hum = load_humann_community()
    meta = load_meta(list(hum.columns.astype(str)))
    samples = list(meta.index.astype(str))
    hum = filter_and_tss(hum, samples)
    batch = meta['Bioproject']
    group = meta['Group']
    log_depth = meta['log_depth']
    print('[2] CLR + distances (uncorrected)')
    clr = clr_transform(hum)
    bc_unc = squareform(pdist(hum.T.values, metric='braycurtis'))
    ait_unc = squareform(pdist(clr.T.values, metric='euclidean'))
    print('[3] ComBat (batch=Bioproject, mod=~Group+log_depth)')
    clr_c = apply_combat_r(clr.copy(), batch, group, log_depth)
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
    save_tsv_csv(permanova_df, TAB / 'permanova_before_after_ComBat')
    save_tsv_csv(permanova_df, SRC / 'permanova_before_after_ComBat')
    print('[5] PCoA coordinates')
    coords_list = []
    var_map = {}
    for tag, dist in [('BrayCurtis_uncorrected', bc_unc), ('Aitchison_uncorrected', ait_unc), ('Aitchison_ComBat', ait_cor), ('BrayCurtis_from_ComBatCLR', bc_cor)]:
        coords, var_exp = perform_pcoa(dist)
        var_map[tag] = var_exp
        coords_list.append(pd.DataFrame({'Sample': samples, 'PC1': coords[:, 0], 'PC2': coords[:, 1], 'Bioproject': batch.values, 'Group': group.values, 'log_depth': log_depth.values, 'Distance': tag, 'PC1_var_pct': var_exp[0], 'PC2_var_pct': var_exp[1]}))
    coords_all = pd.concat(coords_list, ignore_index=True)
    save_tsv_csv(coords_all, TAB / 'pcoa_coordinates_before_after')
    save_tsv_csv(coords_all, SRC / 'pcoa_coordinates_before_after')
    print('[6] Figures')
    tag_c = 'Aitchison_ComBat'
    sub_c = coords_all[coords_all['Distance'] == tag_c].copy()
    v1, v2 = var_map[tag_c]
    row_c = permanova_df[permanova_df['Distance'] == tag_c].iloc[0]
    adonis_c = f"Group R2={row_c['Group_R2']:.3f}, P={row_c['Group_p']:.3f}\nCohort R2={row_c['Batch_R2']:.3f}, P={row_c['Batch_p']:.3f}"
    adonis_label = f"Group R2={row_c['Group_R2']:.3f}, P={row_c['Group_p']:.3f}; Cohort R2={row_c['Batch_R2']:.3f}, P={row_c['Batch_p']:.3f}"
    fig_c, ax = plt.subplots(figsize=(4.6, 4.2))
    plot_pcoa_panel(ax, sub_c, v1, v2, 'group', 'ComBat-corrected HUMAnN KO (color = Group)', adonis_txt=adonis_c)
    g_handles = [mpatches.Patch(color=COLORS_GROUP[g], label=g, alpha=0.85) for g in GROUP_ORDER]
    shape_handles = [plt.Line2D([0], [0], marker=BATCH_MARKERS[b], color='gray', linestyle='None', markersize=6, label=b) for b in BATCH_ORDER]
    leg_g = ax.legend(handles=g_handles, frameon=False, fontsize=7, loc='upper left')
    ax.add_artist(leg_g)
    ax.legend(handles=shape_handles, frameon=False, fontsize=6.5, loc='upper right')
    fig_c.tight_layout(pad=0.4)
    stem_c = Path('Fig_HUMAnN_PCoA_ComBat_C')
    save_fig(fig_c, stem_c, also_panels=True)
    plot_c = sub_c.copy()
    plot_c['panel'] = 'C'
    plot_c['color_by'] = 'group'
    plot_c['shape_by'] = 'cohort'
    plot_c['legend_left'] = 'upper left (Group color)'
    plot_c['legend_right'] = 'upper right (Cohort shape)'
    plot_c['Group_R2'] = row_c['Group_R2']
    plot_c['Group_p'] = row_c['Group_p']
    plot_c['Batch_R2'] = row_c['Batch_R2']
    plot_c['Batch_p'] = row_c['Batch_p']
    plot_c['adonis_label'] = adonis_label
    plot_c['method_note'] = 'CLR + ComBat(batch=Bioproject, mod=~Group+log_depth) + Aitchison PCoA'
    save_tsv_csv(plot_c, SRC / 'Fig_HUMAnN_PCoA_ComBat_C_plotdata')
    save_tsv_csv(plot_c, FIG / 'Fig_HUMAnN_PCoA_ComBat_C_plotdata')
    plt.close(fig_c)
    fig_d, ax = plt.subplots(figsize=(4.6, 4.2))
    adonis_d = f"Cohort R2={row_c['Batch_R2']:.3f}, P={row_c['Batch_p']:.3f}\nGroup R2={row_c['Group_R2']:.3f}, P={row_c['Group_p']:.3f}"
    plot_pcoa_panel(ax, sub_c, v1, v2, 'batch', 'ComBat-corrected HUMAnN KO (color = Cohort)', adonis_txt=adonis_d)
    b_handles = [mpatches.Patch(color=COLORS_BATCH[b], label=b, alpha=0.85) for b in BATCH_ORDER]
    g_shape = [plt.Line2D([0], [0], marker='o' if g == 'Healthy' else '^', color='gray', linestyle='None', markersize=6, label=g) for g in GROUP_ORDER]
    ax.legend(handles=b_handles, frameon=False, fontsize=6.5, loc='upper left')
    fig_d.tight_layout(pad=0.4)
    stem_d = Path('Fig_HUMAnN_PCoA_ComBat_D')
    save_fig(fig_d, stem_d, also_panels=True)
    plot_d = sub_c.copy()
    plot_d['panel'] = 'D'
    plot_d['color_by'] = 'batch'
    plot_d['shape_by'] = 'group'
    plot_d['Group_R2'] = row_c['Group_R2']
    plot_d['Group_p'] = row_c['Group_p']
    plot_d['Batch_R2'] = row_c['Batch_R2']
    plot_d['Batch_p'] = row_c['Batch_p']
    plot_d['adonis_label'] = f"Cohort R2={row_c['Batch_R2']:.3f}, P={row_c['Batch_p']:.3f}; Group R2={row_c['Group_R2']:.3f}, P={row_c['Group_p']:.3f}"
    save_tsv_csv(plot_d, SRC / 'Fig_HUMAnN_PCoA_ComBat_D_plotdata')
    save_tsv_csv(plot_d, FIG / 'Fig_HUMAnN_PCoA_ComBat_D_plotdata')
    plt.close(fig_d)
    adonis_cd = pd.DataFrame([{'panel': 'C_D', 'Distance': tag_c, 'Group_R2': row_c['Group_R2'], 'Group_p': row_c['Group_p'], 'Batch_R2': row_c['Batch_R2'], 'Batch_p': row_c['Batch_p'], 'Batch_then_Group_R2': row_c['Batch_then_Group_R2'], 'Batch_then_Group_p': row_c['Batch_then_Group_p'], 'n_samples': len(samples), 'n_KOs': int(hum.shape[0]), 'combat_mod': '~Group+log_depth', 'combat_batch': 'Bioproject'}])
    save_tsv_csv(adonis_cd, TAB / 'Fig_HUMAnN_PCoA_ComBat_CD_adonis')
    save_tsv_csv(adonis_cd, SRC / 'Fig_HUMAnN_PCoA_ComBat_CD_adonis')
    save_tsv_csv(adonis_cd, FIG / 'Fig_HUMAnN_PCoA_ComBat_CD_adonis')
    fig3, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    for ax, tag, title in [(axes[0], 'BrayCurtis_uncorrected', 'Uncorrected Bray-Curtis'), (axes[1], 'Aitchison_ComBat', 'ComBat-corrected Aitchison')]:
        sub = coords_all[coords_all['Distance'] == tag]
        vv1, vv2 = var_map[tag]
        row = permanova_df[permanova_df['Distance'] == tag].iloc[0]
        txt = f"Group R2={row['Group_R2']:.3f}, P={row['Group_p']:.3f}\nCohort R2={row['Batch_R2']:.3f}, P={row['Batch_p']:.3f}"
        plot_pcoa_panel(ax, sub, vv1, vv2, 'group', title, adonis_txt=txt)
    leg_g2 = axes[1].legend(handles=g_handles, frameon=False, fontsize=6.5, loc='upper left')
    axes[1].add_artist(leg_g2)
    axes[1].legend(handles=shape_handles, frameon=False, fontsize=6.0, loc='upper right')
    fig3.tight_layout(pad=0.5)
    stem3 = Path('Fig_HUMAnN_PCoA_main_before_after')
    save_fig(fig3, stem3, also_panels=True)
    save_tsv_csv(coords_all[coords_all['Distance'].isin(['BrayCurtis_uncorrected', 'Aitchison_ComBat'])], SRC / 'Fig_HUMAnN_PCoA_main_before_after_plotdata')
    plt.close(fig3)
    fig2, ax = plt.subplots(figsize=(4.8, 3.2))
    focus = permanova_df[permanova_df['Distance'].isin(['BrayCurtis_uncorrected', 'Aitchison_ComBat'])].copy()
    focus['label'] = focus['Distance'].map({'BrayCurtis_uncorrected': 'Uncorrected\nBray-Curtis', 'Aitchison_ComBat': 'ComBat-CLR\nAitchison'})
    x = np.arange(len(focus))
    w = 0.35
    bars1 = ax.bar(x - w / 2, focus['Batch_R2'] * 100, width=w, color='#3C5488', label='Cohort (Batch)', edgecolor='none')
    bars2 = ax.bar(x + w / 2, focus['Group_R2'] * 100, width=w, color='#E64B35', label='Group (PCOS/Healthy)', edgecolor='none')
    ax.set_xticks(x)
    ax.set_xticklabels(focus['label'], fontsize=8)
    ax.set_ylabel('PERMANOVA R2 (%)', fontsize=9)
    ax.set_title('HUMAnN KO: variance by cohort vs disease', fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    ax.tick_params(labelsize=8)
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3, f'{h:.1f}', ha='center', va='bottom', fontsize=7)
    ymax = max(focus['Batch_R2'].max(), focus['Group_R2'].max()) * 100
    ax.set_ylim(0, ymax * 1.25 if ymax > 0 else 1)
    fig2.tight_layout(pad=0.4)
    stem2 = Path('Fig_HUMAnN_variance_Batch_vs_Group')
    save_fig(fig2, stem2, also_panels=True)
    save_tsv_csv(focus, SRC / 'Fig_HUMAnN_variance_Batch_vs_Group_plotdata')
    plt.close(fig2)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.6))
    panel_defs = [(axes[0, 0], 'BrayCurtis_uncorrected', 'group', 'A. Uncorrected (color = Group)'), (axes[0, 1], 'BrayCurtis_uncorrected', 'batch', 'B. Uncorrected (color = Cohort)'), (axes[1, 0], 'Aitchison_ComBat', 'group', 'C. ComBat-corrected (color = Group)'), (axes[1, 1], 'Aitchison_ComBat', 'batch', 'D. ComBat-corrected (color = Cohort)')]
    plot_rows = []
    for ax, tag, mode, title in panel_defs:
        sub = coords_all[coords_all['Distance'] == tag].copy()
        vv1, vv2 = var_map[tag]
        plot_pcoa_panel(ax, sub, vv1, vv2, mode, title)
        for _, r in sub.iterrows():
            plot_rows.append({'panel': title[0], 'Distance': tag, 'color_by': mode, 'Sample': r['Sample'], 'PC1': r['PC1'], 'PC2': r['PC2'], 'Group': r['Group'], 'Bioproject': r['Bioproject'], 'PC1_var_pct': vv1, 'PC2_var_pct': vv2})
    axes[0, 0].legend(handles=g_handles, frameon=False, fontsize=7, loc='best')
    axes[0, 1].legend(handles=b_handles, frameon=False, fontsize=6.5, loc='best')
    axes[1, 0].legend(handles=g_handles, frameon=False, fontsize=7, loc='best')
    axes[1, 1].legend(handles=b_handles, frameon=False, fontsize=6.5, loc='best')
    fig.tight_layout(pad=0.6)
    stem = Path('Fig_HUMAnN_PCoA_before_after_ComBat')
    save_fig(fig, stem, also_panels=True)
    save_tsv_csv(pd.DataFrame(plot_rows), SRC / 'Fig_HUMAnN_PCoA_before_after_ComBat_plotdata')
    plt.close(fig)
    clr_c.to_csv(TAB / 'HUMAnN_KO_CLR_ComBat.tsv', sep='\t')
    clr.to_csv(TAB / 'HUMAnN_KO_CLR_uncorrected.tsv', sep='\t')
    summary = pd.Series({'n_samples': len(samples), 'n_KOs_used': int(hum.shape[0]), 'prevalence_threshold': PREV, 'combat_batch': 'Bioproject', 'combat_mod': '~Group+log_depth', 'uncorrected_BrayCurtis_Batch_R2': float(permanova_df.loc[permanova_df['Distance'] == 'BrayCurtis_uncorrected', 'Batch_R2'].iloc[0]), 'uncorrected_BrayCurtis_Group_R2': float(permanova_df.loc[permanova_df['Distance'] == 'BrayCurtis_uncorrected', 'Group_R2'].iloc[0]), 'ComBat_Aitchison_Batch_R2': float(row_c['Batch_R2']), 'ComBat_Aitchison_Group_R2': float(row_c['Group_R2']), 'ComBat_Aitchison_Group_p': float(row_c['Group_p']), 'ComBat_Aitchison_Batch_p': float(row_c['Batch_p']), 'ComBat_Aitchison_Group_given_Batch_R2': float(row_c['Batch_then_Group_R2']), 'ComBat_Aitchison_Group_given_Batch_p': float(row_c['Batch_then_Group_p']), 'PC1_var_pct': float(v1), 'PC2_var_pct': float(v2)})
    summary.to_csv(TAB / 'key_stats_summary.tsv', sep='\t', header=['value'])
    summary.to_csv(SRC / 'key_stats_summary.tsv', sep='\t', header=['value'])
    summary.to_csv(TAB / 'key_stats_summary.csv', header=['value'])
    summary.to_csv(SRC / 'key_stats_summary.csv', header=['value'])
    legend = f"# Fig. HUMAnN ComBat-corrected PCoA (Fig.1C-style)\n\n**Source figure:** `figures/Fig_HUMAnN_PCoA_ComBat_C.jpg`\n**Plot data:** `source_data/Fig_HUMAnN_PCoA_ComBat_C_plotdata.tsv`\n**PERMANOVA:** `tables/Fig_HUMAnN_PCoA_ComBat_CD_adonis.tsv`\n\nCommunity (unstratified) HUMAnN KO abundances were prevalence-filtered (≥{PREV:.0%}),\nTSS-renormalized, CLR-transformed, and adjusted with parametric ComBat\n(batch = BioProject; protected covariates = disease Group + log sequencing depth).\nPCoA used Aitchison (Euclidean) distances on the ComBat-corrected CLR matrix.\nPERMANOVA: adonis, 999 permutations.\n\n- n samples = {len(samples)}; n KOs = {hum.shape[0]}\n- PCoA1 = {v1:.1f}%; PCoA2 = {v2:.1f}%\n- Group R² = {row_c['Group_R2']:.3f}, P = {row_c['Group_p']:.3f}\n- Cohort R² = {row_c['Batch_R2']:.3f}, P = {row_c['Batch_p']:.3f}\n- Group | Cohort R² = {row_c['Batch_then_Group_R2']:.3f}, P = {row_c['Batch_then_Group_p']:.3f}\n\n- Main C panel: `Fig_HUMAnN_PCoA_ComBat_C.{{pdf,svg,png,jpg}}`\n- Companion D: `Fig_HUMAnN_PCoA_ComBat_D.{{pdf,svg,png,jpg}}`\n- Before/after: `Fig_HUMAnN_PCoA_main_before_after.{{pdf,svg,png,jpg}}`\n- 2x2 overview: `Fig_HUMAnN_PCoA_before_after_ComBat.{{pdf,svg,png,jpg}}`\n- Variance bars: `Fig_HUMAnN_variance_Batch_vs_Group.{{pdf,svg,png,jpg}}`\n"
    (OUT / 'legend.md').write_text(legend)
    print('[DONE]', OUT)
if __name__ == '__main__':
    main()
for _d in (FIG, FIG_P, SRC, TAB):
    _d.mkdir(parents=True, exist_ok=True)
    path = Path(path)
    if str(path).endswith('.gz'):
        return gzip.open(path, 'rt')
    return open(path, 'r')
