from pathlib import Path
import os
PKG_ROOT = Path(__file__).resolve().parents[1]
SHARED = PKG_ROOT / 'shared_input'
RSCRIPT = os.environ.get('RSCRIPT', 'Rscript')
ROOT = PKG_ROOT / 'output' / 'Fig4_SGB_KO'
IN_AB = SHARED / 'sgb_bin_abundance.tsv'
IN_GRP = SHARED / 'metaphlan_sample_group.tsv'
IN_INFO = SHARED / 'sgb_genome_annotation.tsv'
IN_KO = SHARED / 'sgb_ko_presence_matrix.xls'
TAB = ROOT / 'tables'
PLOT = ROOT / 'plotdata'
FIG = ROOT / 'figures' / 'panels'
LOG = ROOT / 'logs'
ENR = ROOT / 'enrich' / 'ko_lists'
import json
import logging
import os
import subprocess
import sys
import warnings
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib import font_manager
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from skbio import DistanceMatrix
from skbio.stats.ordination import pcoa
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')
PROJ = ROOT.parents[1]
for d in (TAB, PLOT, FIG, LOG, ENR):
    d.mkdir(parents=True, exist_ok=True)
if os.path.exists(FONT):
    font_manager.fontManager.addfont(FONT)
mpl.rcParams.update({'font.family': 'Times New Roman', 'font.weight': 'normal', 'axes.titleweight': 'normal', 'axes.labelweight': 'normal', 'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.spines.right': False, 'axes.spines.top': False, 'axes.linewidth': 0.8, 'legend.frameon': False, 'figure.dpi': 300, 'savefig.dpi': 300, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
COL = {'PCOS': '#E64B35', 'Healthy': '#4DBBD5', 'PCOS_biased': '#E64B35', 'Healthy_biased': '#4DBBD5', 'Balanced': '#B0B0B0', 'Shared': '#00A087', 'PCOS-specific': '#E64B35', 'Healthy-specific': '#4DBBD5', 'Non-core': '#CCCCCC'}
BIAS_ORDER = ['Healthy_biased', 'Balanced', 'PCOS_biased']
CORE_ORDER = ['Shared', 'PCOS-specific', 'Healthy-specific', 'Non-core']
DELTA_CUT = 0.1
CORE_PREV = 0.7

def save_table(df: pd.DataFrame, path: Path, index: bool=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(f'{path}.tsv', sep='\t', index=index)
    df.to_csv(f'{path}.csv', index=index)

def save_fig(fig, base: Path, plot_df: pd.DataFrame | None=None):
    base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in ('pdf', 'svg', 'png', 'jpg'):
        fig.savefig(f'{base}.{fmt}', format=fmt, bbox_inches='tight', pad_inches=0.04)
    if plot_df is not None:
        save_table(plot_df, PLOT / f'{base.name}_plotdata')
    plt.close(fig)

def strip_ko(x: str) -> str:
    return str(x).split(':')[0].strip()

def residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xa = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(Xa, y, rcond=None)
    return y - Xa @ beta

def bias_class(d: float) -> str:
    if d >= DELTA_CUT:
        return 'PCOS_biased'
    if d <= -DELTA_CUT:
        return 'Healthy_biased'
    return 'Balanced'

def core_status(pp: float, ph: float) -> str:
    p_core = pp >= CORE_PREV
    h_core = ph >= CORE_PREV
    if p_core and h_core:
        return 'Shared'
    if p_core and (not h_core):
        return 'PCOS-specific'
    if h_core and (not p_core):
        return 'Healthy-specific'
    return 'Non-core'

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG / 'v2_run.log', mode='w')])
    logging.info('=== v2 group-aware length-adjusted SGB×KO ===')
    abund = pd.read_csv(IN_AB, sep='\t', index_col=0).clip(lower=0)
    ra = abund.div(abund.sum(0).replace(0, np.nan), axis=1) * 100.0
    ra = ra.fillna(0)
    grp = pd.read_csv(IN_GRP, sep='\t')
    grp.columns = ['Sample', 'Bioproject', 'Group']
    common_s = [s for s in ra.columns if s in set(grp.Sample)]
    ra = ra[common_s]
    gmap = grp.set_index('Sample').loc[common_s]
    pcos_s = gmap.index[gmap.Group == 'PCOS'].tolist()
    heal_s = gmap.index[gmap.Group == 'Healthy'].tolist()
    prev_p = (ra[pcos_s] > 0).sum(1) / len(pcos_s)
    prev_h = (ra[heal_s] > 0).sum(1) / len(heal_s)
    prev_all = (ra > 0).sum(1) / ra.shape[1]
    delta = prev_p - prev_h
    info = pd.read_csv(IN_INFO, sep='\t').set_index('ID')
    meta = pd.DataFrame({'Prev_PCOS': prev_p, 'Prev_Healthy': prev_h, 'Prev_overall': prev_all, 'Delta_Prev': delta})
    meta = meta.join(info[['P', 'G', 'S', 'length', 'completeness', 'contamination']], how='inner')
    meta['Phylum'] = meta['P'].fillna('unknown').astype(str).str.replace('p__', '', regex=False)
    meta['log_length'] = np.log10(meta['length'].clip(lower=1))
    meta['length_Mb'] = meta['length'] / 1000000.0
    meta['Bias_class'] = meta['Delta_Prev'].map(bias_class)
    meta['Core_status'] = [core_status(a, b) for a, b in zip(meta.Prev_PCOS, meta.Prev_Healthy)]
    meta['Bias_class'] = pd.Categorical(meta['Bias_class'], BIAS_ORDER, ordered=True)
    meta['Core_status'] = pd.Categorical(meta['Core_status'], CORE_ORDER, ordered=True)
    meta['Delta_Prev_resid'] = residualize(meta['Delta_Prev'].values, meta[['log_length', 'completeness']].values)
    ko_raw = pd.read_csv(IN_KO, sep='\t', index_col=0)
    mags = [c for c in ko_raw.columns if c in meta.index]
    ko_bin = (ko_raw[mags] > 0).astype(int)
    ko_bin.index = [strip_ko(i) for i in ko_bin.index]
    ko_bin = ko_bin.groupby(level=0).max().astype(int)
    meta = meta.loc[mags].copy()
    ko_bin = ko_bin[mags]
    meta['KO_richness'] = ko_bin.sum(0).values
    meta['KO_density'] = meta['KO_richness'] / meta['length_Mb']
    meta['KO_richness_resid'] = residualize(meta['KO_richness'].values, meta[['log_length', 'completeness']].values)
    frac = ko_bin.mean(1)
    ko_f = ko_bin.loc[(frac >= 0.02) & (frac <= 0.98)].copy()
    logging.info('SGB=%d | KO filtered=%d | PCOS n=%d Healthy n=%d', len(meta), ko_f.shape[0], len(pcos_s), len(heal_s))
    logging.info('Bias counts:\n%s', meta.Bias_class.value_counts())
    logging.info('Core counts:\n%s', meta.Core_status.value_counts())
    save_table(meta.reset_index().rename(columns={'index': 'SGB_ID'}), TAB / 'SGB_group_prevalence_metadata')
    logging.info('[PCoA / distance]')
    X = ko_f.T.values.astype(float)
    dm = squareform(pdist(X, metric='jaccard'))
    ids = list(ko_f.columns)
    dm_df = pd.DataFrame(dm, index=ids, columns=ids)
    dm_path = TAB / 'KO_jaccard_distance.tsv'
    dm_df.to_csv(dm_path, sep='\t')
    ord_res = pcoa(DistanceMatrix(dm, ids=ids), number_of_dimensions=3)
    scores = ord_res.samples.copy()
    scores.index = ids
    scores = scores.join(meta)
    var_exp = ord_res.proportion_explained.values[:3] * 100
    for pc in ('PC1', 'PC2'):
        scores[f'{pc}_length_resid'] = residualize(scores[pc].values, scores[['log_length', 'completeness']].values)
    save_table(scores.reset_index().rename(columns={'index': 'SGB_ID'}), TAB / 'PCoA_scores')
    meta_v = meta.reset_index().rename(columns={'index': 'SGB_ID'})
    meta_v_path = TAB / 'meta_for_vegan.tsv'
    meta_v.to_csv(meta_v_path, sep='\t', index=False)
    rscript = 'Rscript'
    r_helper = Path(__file__).with_name('02_vegan_length_conditioned.R')
    out_pref = str(TAB / 'vegan')
    cmd = [rscript, str(r_helper), f'dm={dm_path}', f'meta={meta_v_path}', f'out={out_pref}']
    logging.info('[vegan] %s', ' '.join(cmd))
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logging.info('[vegan] done')
    except subprocess.CalledProcessError as e:
        logging.error('vegan failed: %s\n%s', e.stderr[-2000:], e.stdout[-1000:])
        raise
    vegan_anova = pd.read_csv(f'{out_pref}_dbRDA_anova.tsv', sep='\t')
    vegan_adonis = pd.read_csv(f'{out_pref}_adonis_Bias_class.tsv', sep='\t')
    db_sites = pd.read_csv(f'{out_pref}_dbRDA_sites.tsv', sep='\t')

    def term_p(df, key):
        hit = df[df['Term'].astype(str).str.contains(key, case=False, na=False)]
        if len(hit) == 0:
            return (np.nan, np.nan)
        row = hit.iloc[0]
        pcol = 'Pr(>F)' if 'Pr(>F)' in df.columns else [c for c in df.columns if 'Pr' in c][0]
        fcol = 'F' if 'F' in df.columns else [c for c in df.columns if c.startswith('F')][0]
        return (float(row[fcol]), float(row[pcol]))
    bias_F, bias_P = term_p(vegan_adonis, 'Bias')
    dbrda_F, dbrda_P = term_p(vegan_anova, 'Model')
    logging.info('adonis Bias|length margin F=%.3f P=%.4g; dbRDA F=%.3f P=%.4g', bias_F, bias_P, dbrda_F, dbrda_P)
    logging.info('[Residual richness]')
    rows = []
    for label, col, order in [('Bias_class', 'Bias_class', BIAS_ORDER), ('Core_status', 'Core_status', CORE_ORDER)]:
        groups = [meta.loc[meta[col] == c, 'KO_richness_resid'].dropna().values for c in order if (meta[col] == c).any()]
        kw = stats.kruskal(*groups) if len(groups) > 1 else None
        if col == 'Bias_class':
            a = meta.loc[meta[col] == 'PCOS_biased', 'KO_richness_resid'].dropna()
            b = meta.loc[meta[col] == 'Healthy_biased', 'KO_richness_resid'].dropna()
            contrast = 'PCOS_biased_vs_Healthy_biased'
        else:
            a = meta.loc[meta[col] == 'PCOS-specific', 'KO_richness_resid'].dropna()
            b = meta.loc[meta[col] == 'Healthy-specific', 'KO_richness_resid'].dropna()
            contrast = 'PCOS-specific_vs_Healthy-specific'
        mw = stats.mannwhitneyu(a, b, alternative='two-sided') if len(a) and len(b) else None
        rows.append({'Grouping': label, 'KW_P': float(kw.pvalue) if kw else np.nan, 'Contrast': contrast, 'MW_P': float(mw.pvalue) if mw else np.nan, 'n_a': int(len(a)), 'n_b': int(len(b)), 'median_a': float(a.median()) if len(a) else np.nan, 'median_b': float(b.median()) if len(b) else np.nan})
    rich_sum = pd.DataFrame(rows)
    save_table(rich_sum, TAB / 'residual_richness_by_group')
    logging.info('[Logistic KO ~ ΔPrev + length]')
    y_delta = meta.loc[ko_f.columns, 'Delta_Prev'].values
    cov = meta.loc[ko_f.columns, ['log_length', 'completeness']].values
    y_resid = meta.loc[ko_f.columns, 'Delta_Prev_resid'].values
    name_map = {strip_ko(i): str(i) for i in ko_raw.index}
    logit_rows = []
    Y = ko_f.values.astype(float)
    for i, kid in enumerate(ko_f.index):
        x = Y[i]
        if x.std() == 0 or x.sum() < 5 or len(x) - x.sum() < 5:
            continue
        Xd = np.column_stack([y_delta, cov])
        Xd = sm.add_constant(Xd)
        try:
            fit = sm.GLM(x, Xd, family=sm.families.Binomial()).fit(disp=0, maxiter=100)
            beta = float(fit.params[1])
            pval = float(fit.pvalues[1])
            beta_L = float(fit.params[2])
            p_L = float(fit.pvalues[2])
        except Exception:
            continue
        rs, ps = stats.spearmanr(x, y_resid)
        logit_rows.append({'KO': kid, 'KO_annotation': name_map.get(kid, kid), 'KO_genome_frac': float(x.mean()), 'Logit_beta_DeltaPrev': beta, 'P_DeltaPrev': pval, 'Logit_beta_logLength': beta_L, 'P_logLength': p_L, 'Spearman_r_DeltaPrev_resid': rs, 'P_Spearman_resid': ps})
    logit = pd.DataFrame(logit_rows)
    logit['FDR_DeltaPrev'] = multipletests(logit['P_DeltaPrev'], method='fdr_bh')[1]
    logit['Direction'] = 'NS'
    logit.loc[(logit.FDR_DeltaPrev < 0.05) & (logit.Logit_beta_DeltaPrev > 0), 'Direction'] = 'PCOS_biased_assoc'
    logit.loc[(logit.FDR_DeltaPrev < 0.05) & (logit.Logit_beta_DeltaPrev < 0), 'Direction'] = 'Healthy_biased_assoc'
    logit = logit.sort_values('Logit_beta_DeltaPrev', ascending=False)
    save_table(logit, TAB / 'KO_logit_DeltaPrev_length_adjusted')
    top_pcos = logit.head(30)
    top_heal = logit.tail(30).sort_values('Logit_beta_DeltaPrev')
    save_table(top_pcos, TAB / 'KO_top30_PCOS_biased_assoc')
    save_table(top_heal, TAB / 'KO_top30_Healthy_biased_assoc')
    logging.info('Logit FDR: PCOS_assoc=%d Healthy_assoc=%d', (logit.Direction == 'PCOS_biased_assoc').sum(), (logit.Direction == 'Healthy_biased_assoc').sum())
    pos = logit[logit.Direction == 'PCOS_biased_assoc']
    neg = logit[logit.Direction == 'Healthy_biased_assoc']
    if len(pos) > 200:
        pos = pos.nlargest(200, 'Logit_beta_DeltaPrev')
    if len(neg) > 200:
        neg = neg.nsmallest(200, 'Logit_beta_DeltaPrev')
    (ENR / 'PCOS_biased_assoc.KO').write_text('\n'.join(pos.KO.tolist()) + '\n')
    (ENR / 'Healthy_biased_assoc.KO').write_text('\n'.join(neg.KO.tolist()) + '\n')
    logging.info('[Sensitivity] length tertiles')
    meta['length_tertile'] = pd.qcut(meta['length'], 3, labels=['L1_short', 'L2_mid', 'L3_long'])
    sens_rows = []
    for tert in ['L1_short', 'L2_mid', 'L3_long']:
        idx = meta.index[meta.length_tertile == tert].tolist()
        if len(idx) < 30:
            continue
        yd = meta.loc[idx, 'Delta_Prev'].values
        for kid in logit.nlargest(5, 'Logit_beta_DeltaPrev').KO.tolist() + logit.nsmallest(5, 'Logit_beta_DeltaPrev').KO.tolist():
            if kid not in ko_f.index:
                continue
            x = ko_f.loc[kid, idx].values.astype(float)
            if x.std() == 0:
                continue
            r, p = stats.spearmanr(x, yd)
            sens_rows.append({'length_tertile': tert, 'KO': kid, 'Spearman_r': r, 'P': p, 'n': len(idx)})
    sens = pd.DataFrame(sens_rows)
    save_table(sens, TAB / 'sensitivity_topKO_within_length_tertiles')
    top_phy = meta.Phylum.value_counts().head(6).index.tolist()
    meta['Phylum_plot'] = meta.Phylum.where(meta.Phylum.isin(top_phy), 'Other')
    phy_ct = pd.crosstab(meta.Bias_class, meta.Phylum_plot, normalize='index') * 100
    save_table(phy_ct.reset_index(), TAB / 'Bias_class_phylum_pct')
    logging.info('[Figures]')
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    ax = axes[0]
    for c in BIAS_ORDER:
        sub = scores[scores.Bias_class == c]
        ax.scatter(sub.PC1, sub.PC2, s=14, alpha=0.75, c=COL[c], edgecolors='white', linewidths=0.15, label=f"{c.replace('_', ' ')} (n={len(sub)})")
    ax.set_xlabel(f'PC1 ({var_exp[0]:.1f}%)')
    ax.set_ylabel(f'PC2 ({var_exp[1]:.1f}%)')
    ax.set_title(f'KO Jaccard PCoA\nadonis Bias | length P={bias_P:.3g}')
    ax.legend(fontsize=6, loc='best')
    ax = axes[1]
    cap = db_sites.merge(meta.reset_index().rename(columns={'index': 'SGB_ID'}), on='SGB_ID')
    xcol = [c for c in cap.columns if c.upper().startswith('CAP') or c == 'CAP1']
    if not xcol:
        xcol = [c for c in cap.columns if c not in meta.columns and c != 'SGB_ID'][:2]
    else:
        xcol = xcol[:2]
        if len(xcol) == 1:
            xcol = xcol + [c for c in cap.columns if c.startswith('MDS') or c.startswith('PC')][:1]
    if len(xcol) >= 2:
        for c in BIAS_ORDER:
            sub = cap[cap.Bias_class == c]
            ax.scatter(sub[xcol[0]], sub[xcol[1]], s=14, alpha=0.75, c=COL[c], edgecolors='white', linewidths=0.15, label=c.replace('_', ' '))
        ax.set_xlabel(xcol[0])
        ax.set_ylabel(xcol[1])
    ax.set_title(f'dbRDA Bias | Condition(length)\nANOVA P={dbrda_P:.3g}')
    ax.legend(fontsize=6)
    fig.suptitle('A  Length-adjusted KO composition by PCOS/Healthy prevalence bias', y=1.02, fontsize=9)
    save_fig(fig, FIG / 'A' / 'Fig_v2_PCoA_dbRDA_Bias', cap[['SGB_ID', xcol[0], xcol[1], 'Bias_class', 'Delta_Prev', 'length_Mb']] if len(xcol) >= 2 else scores.reset_index())
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    sns.boxplot(data=meta, x='Bias_class', y='KO_richness_resid', order=BIAS_ORDER, palette={c: COL[c] for c in BIAS_ORDER}, ax=axes[0], fliersize=1.5, linewidth=0.8)
    axes[0].axhline(0, ls='--', lw=0.6, c='grey')
    axes[0].set_xlabel('Prevalence bias class')
    axes[0].set_ylabel('KO richness residual\n(length + completeness)')
    p_bias = rich_sum.loc[rich_sum.Grouping == 'Bias_class', 'MW_P'].values[0]
    axes[0].set_title(f'By bias class\nPCOS vs Healthy biased MW P={p_bias:.3g}')
    axes[0].tick_params(axis='x', rotation=20)
    core_plot = meta[meta.Core_status != 'Non-core']
    sns.boxplot(data=core_plot, x='Core_status', y='KO_richness_resid', order=['Shared', 'PCOS-specific', 'Healthy-specific'], palette={c: COL[c] for c in ['Shared', 'PCOS-specific', 'Healthy-specific']}, ax=axes[1], fliersize=1.5, linewidth=0.8)
    axes[1].axhline(0, ls='--', lw=0.6, c='grey')
    axes[1].set_xlabel('Group-specific core (70%)')
    axes[1].set_ylabel('KO richness residual')
    p_core = rich_sum.loc[rich_sum.Grouping == 'Core_status', 'MW_P'].values[0]
    axes[1].set_title(f'Core SGBs only\nPCOS- vs Healthy-specific MW P={p_core:.3g}')
    axes[1].tick_params(axis='x', rotation=20)
    fig.suptitle('B  Genome-size–adjusted KO richness (not raw counts)', y=1.02, fontsize=9)
    save_fig(fig, FIG / 'B' / 'Fig_v2_residual_richness', meta.reset_index().rename(columns={'index': 'SGB_ID'})[['SGB_ID', 'Bias_class', 'Core_status', 'KO_richness', 'KO_richness_resid', 'length_Mb', 'Delta_Prev']])
    fig, ax = plt.subplots(figsize=(4.0, 3.6))
    v = logit.copy()
    v['neglog10FDR'] = -np.log10(v.FDR_DeltaPrev.clip(lower=1e-300))
    for dirc, col in [('NS', '#B0B0B0'), ('Healthy_biased_assoc', COL['Healthy']), ('PCOS_biased_assoc', COL['PCOS'])]:
        sub = v[v.Direction == dirc]
        ax.scatter(sub.Logit_beta_DeltaPrev, sub.neglog10FDR, s=8, c=col, alpha=0.7, linewidths=0, label=dirc.replace('_', ' '))
    ax.axhline(-np.log10(0.05), ls='--', lw=0.7, c='grey')
    ax.axvline(0, ls='--', lw=0.7, c='grey')
    ax.set_xlabel('Logistic $\\beta$ ($\\Delta$Prev | length + completeness)')
    ax.set_ylabel('$-\\log_{10}$(FDR)')
    ax.set_title('C  KO association with PCOS−Healthy prevalence bias')
    ax.legend(fontsize=6)
    save_fig(fig, FIG / 'C' / 'Fig_v2_logit_volcano', v[['KO', 'KO_annotation', 'Logit_beta_DeltaPrev', 'FDR_DeltaPrev', 'neglog10FDR', 'Direction']])
    show = pd.concat([logit.nlargest(12, 'Logit_beta_DeltaPrev'), logit.nsmallest(12, 'Logit_beta_DeltaPrev')]).sort_values('Logit_beta_DeltaPrev')
    fig, ax = plt.subplots(figsize=(4.6, 5.2))
    colors = [COL['PCOS'] if b > 0 else COL['Healthy'] for b in show.Logit_beta_DeltaPrev]
    y = np.arange(len(show))
    ax.barh(y, show.Logit_beta_DeltaPrev, color=colors, height=0.75, alpha=0.88)
    ax.set_yticks(y)
    ax.set_yticklabels(show.KO.tolist(), fontsize=7)
    ax.axvline(0, color='black', lw=0.7)
    ax.set_xlabel('Length-adjusted $\\beta$ (positive = PCOS-biased prevalence)')
    ax.set_title('D  Top KOs linked to group prevalence bias')
    save_fig(fig, FIG / 'D' / 'Fig_v2_top_KO_bars', show[['KO', 'KO_annotation', 'Logit_beta_DeltaPrev', 'FDR_DeltaPrev', 'Spearman_r_DeltaPrev_resid', 'Direction']])
    fig, ax = plt.subplots(figsize=(3.8, 3.5))
    for c in BIAS_ORDER:
        sub = meta[meta.Bias_class == c]
        ax.scatter(sub.Delta_Prev * 100, sub.KO_richness_resid, s=12, alpha=0.7, c=COL[c], edgecolors='white', linewidths=0.15, label=c.replace('_', ' '))
    ax.axvline(0, ls='--', lw=0.6, c='grey')
    ax.axhline(0, ls='--', lw=0.6, c='grey')
    ax.set_xlabel('$\\Delta$ prevalence (PCOS − Healthy, %)')
    ax.set_ylabel('KO richness residual')
    r, p = stats.spearmanr(meta.Delta_Prev, meta.KO_richness_resid)
    ax.set_title(f'E  Bias vs size-adjusted richness\nSpearman r={r:.2f} P={p:.2g}')
    ax.legend(fontsize=6)
    save_fig(fig, FIG / 'E' / 'Fig_v2_DeltaPrev_vs_resid_richness', meta.reset_index().rename(columns={'index': 'SGB_ID'})[['SGB_ID', 'Delta_Prev', 'Bias_class', 'KO_richness_resid', 'Phylum']])
    fig, ax = plt.subplots(figsize=(4.0, 3.4))
    cols = [c for c in top_phy if c in phy_ct.columns] + (['Other'] if 'Other' in phy_ct.columns else [])
    phy_ct.loc[BIAS_ORDER, cols].plot(kind='bar', stacked=True, ax=ax, colormap='Set2', width=0.75, edgecolor='white', linewidth=0.3)
    ax.set_ylabel('% of SGBs')
    ax.set_xlabel('Prevalence bias class')
    ax.set_title('F  Phylum composition (confounder check)')
    ax.legend(fontsize=5.5, bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.set_xticklabels([x.replace('_', '\n') for x in BIAS_ORDER], rotation=0)
    save_fig(fig, FIG / 'F' / 'Fig_v2_phylum_by_bias', phy_ct.loc[BIAS_ORDER, cols].reset_index().melt(id_vars='Bias_class', var_name='Phylum', value_name='Percent'))
    fig = plt.figure(figsize=(9.5, 7.8))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.32)
    ax = fig.add_subplot(gs[0, 0])
    for c in BIAS_ORDER:
        sub = scores[scores.Bias_class == c]
        ax.scatter(sub.PC1, sub.PC2, s=10, alpha=0.7, c=COL[c], linewidths=0, label=c.replace('_biased', '').replace('_', ' '))
    ax.set_xlabel(f'PC1 ({var_exp[0]:.1f}%)')
    ax.set_ylabel(f'PC2 ({var_exp[1]:.1f}%)')
    ax.set_title(f'A  PCoA | Bias adonis P={bias_P:.3g} (length cov.)')
    ax.legend(fontsize=6)
    ax = fig.add_subplot(gs[0, 1])
    sns.boxplot(data=meta, x='Bias_class', y='KO_richness_resid', order=BIAS_ORDER, palette={c: COL[c] for c in BIAS_ORDER}, ax=ax, fliersize=1, linewidth=0.7)
    ax.axhline(0, ls='--', lw=0.6, c='grey')
    ax.set_title(f'B  Residual richness | MW P={p_bias:.3g}')
    ax.tick_params(axis='x', rotation=25)
    ax.set_xlabel('')
    ax.set_ylabel('KO richness residual')
    ax = fig.add_subplot(gs[1, 0])
    for dirc, col in [('NS', '#B0B0B0'), ('Healthy_biased_assoc', COL['Healthy']), ('PCOS_biased_assoc', COL['PCOS'])]:
        sub = v[v.Direction == dirc]
        ax.scatter(sub.Logit_beta_DeltaPrev, sub.neglog10FDR, s=5, c=col, alpha=0.65, linewidths=0)
    ax.axhline(-np.log10(0.05), ls='--', lw=0.6, c='grey')
    ax.axvline(0, ls='--', lw=0.6, c='grey')
    ax.set_xlabel('$\\beta$($\\Delta$Prev | length)')
    ax.set_ylabel('$-\\log_{10}$ FDR')
    ax.set_title('C  Length-adjusted KO~bias volcano')
    ax = fig.add_subplot(gs[1, 1])
    ts = pd.concat([logit.nlargest(8, 'Logit_beta_DeltaPrev'), logit.nsmallest(8, 'Logit_beta_DeltaPrev')]).sort_values('Logit_beta_DeltaPrev')
    cols = [COL['PCOS'] if b > 0 else COL['Healthy'] for b in ts.Logit_beta_DeltaPrev]
    yy = np.arange(len(ts))
    ax.barh(yy, ts.Logit_beta_DeltaPrev, color=cols, height=0.75, alpha=0.88)
    ax.set_yticks(yy)
    ax.set_yticklabels(ts.KO, fontsize=6)
    ax.axvline(0, color='black', lw=0.6)
    ax.set_xlabel('Length-adjusted $\\beta$')
    ax.set_title('D  Top group-bias–associated KOs')
    fig.suptitle('Length-adjusted KO programs along PCOS vs Healthy prevalence bias', fontsize=10, y=0.995)
    save_fig(fig, FIG / 'composite' / 'Fig_v2_group_length_composite', None)
    summary = {'question': 'After length adjustment, do PCOS-biased vs Healthy-biased SGBs differ in KO programs?', 'n_SGB': int(len(meta)), 'n_KO': int(ko_f.shape[0]), 'bias_counts': meta.Bias_class.astype(str).value_counts().to_dict(), 'core_counts': meta.Core_status.astype(str).value_counts().to_dict(), 'adonis_Bias_F': bias_F, 'adonis_Bias_P': bias_P, 'dbRDA_P': dbrda_P, 'resid_richness_MW_bias_P': float(p_bias), 'resid_richness_MW_core_P': float(p_core), 'n_PCOS_assoc_FDR05': int((logit.Direction == 'PCOS_biased_assoc').sum()), 'n_Healthy_assoc_FDR05': int((logit.Direction == 'Healthy_biased_assoc').sum()), 'top5_PCOS': top_pcos.head(5)[['KO', 'KO_annotation', 'Logit_beta_DeltaPrev', 'FDR_DeltaPrev']].to_dict('records'), 'top5_Healthy': top_heal.head(5)[['KO', 'KO_annotation', 'Logit_beta_DeltaPrev', 'FDR_DeltaPrev']].to_dict('records'), 'spearman_Delta_vs_resid_richness': float(r)}
    (TAB / 'v2_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logging.info('Done.\n%s', json.dumps(summary, indent=2, ensure_ascii=False))
if __name__ == '__main__':
    main()
for _d in (TAB, PLOT, FIG, LOG, ENR):
    _d.mkdir(parents=True, exist_ok=True)
