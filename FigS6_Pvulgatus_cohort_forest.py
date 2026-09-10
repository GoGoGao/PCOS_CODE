from pathlib import Path
import os
PKG_ROOT = Path(__file__).resolve().parents[1]
SHARED = PKG_ROOT / 'shared_input'
RSCRIPT = os.environ.get('RSCRIPT', 'Rscript')
TSS_PATH = SHARED / 'metaphlan_species_filtered_tss.csv'
CLR_PATH = SHARED / 'metaphlan_species_filtered_clr.csv'
import warnings
from pathlib import Path
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats
from scipy.stats import binomtest, mannwhitneyu
from statsmodels.formula.api import mixedlm, ols
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')
BASE = Path(__file__).resolve().parent
META_PATH = PROJECT / 'metaphlan/microbiome_analysis/01_preprocessing/group_info_aligned.csv'
DIFF_PATH = PROJECT / 'metaphlan/microbiome_analysis/04_differential_analysis/differential_analysis_results.csv'
SPECIES = 'Phocaeicola_vulgatus'
SPECIES_ITALIC = '$\\it{Phocaeicola\\ vulgatus}$'
COHORTS = ['PRJNA530971', 'PRJNA549764', 'PRJNA791492']
PSEUDO = 0.001
RNG = np.random.default_rng(42)
COHORT_COLORS = {'PRJNA530971': '#00A087', 'PRJNA549764': '#3C5488', 'PRJNA791492': '#F39B7F'}
GROUP_COLORS = {'PCOS': '#E64B35', 'Healthy': '#4DBBD5'}
LOG_TICKS_RAW = [0, 0.01, 0.1, 1, 10, 100]
LOG_TICK_LABELS = ['0', '0.01', '0.1', '1', '10', '100']
matplotlib.rcParams.update({'font.family': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif', 'serif'], 'font.weight': 'regular', 'axes.labelweight': 'regular', 'axes.titleweight': 'regular', 'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10, 'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'axes.linewidth': 0.8, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'axes.unicode_minus': False})

def save_figure(fig, stem: str):
    for ext in ('pdf', 'svg', 'png', 'jpg'):
        fig.savefig(OUT / f'{stem}.{ext}', bbox_inches='tight', dpi=300)
    plt.close(fig)

def to_log(x, pseudo=PSEUDO):
    return np.log10(np.asarray(x, dtype=float) + pseudo)

def set_log_yaxis(ax, max_val: float, ylabel: bool=True, tick_fs: int=10, label_fs: int=11):
    ticks = [t for t in LOG_TICKS_RAW if t <= max_val * 1.5]
    if not ticks:
        ticks = [0, max_val]
    ax.set_yticks([to_log(t) for t in ticks])
    ax.set_yticklabels(LOG_TICK_LABELS[:len(ticks)], fontsize=tick_fs)
    ax.set_ylim(to_log(0) - 0.15, to_log(max_val) + 0.45)
    if ylabel:
        ax.set_ylabel('Relative Abundance (%)', fontsize=label_fs)

def cohens_d(x, y):
    x, y = (np.asarray(x, float), np.asarray(y, float))
    if len(x) < 2 or len(y) < 2:
        return np.nan
    nx, ny = (len(x), len(y))
    pooled = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    return 0.0 if pooled == 0 else (x.mean() - y.mean()) / pooled

def bootstrap_mean_diff_ci(a, b, n_boot=5000, alpha=0.05):
    diffs = []
    for _ in range(n_boot):
        ba = RNG.choice(a, size=len(a), replace=True)
        bb = RNG.choice(b, size=len(b), replace=True)
        diffs.append(ba.mean() - bb.mean())
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))

def fit_ols(sub: pd.DataFrame):
    model = ols("CLR ~ C(Group, Treatment(reference='Healthy'))", data=sub).fit()
    key = "C(Group, Treatment(reference='Healthy'))[T.PCOS]"
    return (float(model.params[key]), float(model.bse[key]), float(model.pvalues[key]), [float(x) for x in model.conf_int().loc[key].tolist()])

def fit_lmm(data: pd.DataFrame):
    md = data.copy()
    md['Group_binary'] = (md['Group'] == 'PCOS').astype(int)
    model = mixedlm('CLR ~ Group_binary', md, groups=md['Cohort'])
    result = model.fit(method='powell', disp=False)
    key = 'Group_binary'
    return (float(result.fe_params[key]), float(result.bse[key]), float(result.pvalues[key]), [float(x) for x in result.conf_int().loc[key].tolist()])

def random_effects_meta(effects, ses):
    effects = np.asarray(effects, float)
    ses = np.asarray(ses, float)
    w = 1.0 / ses ** 2
    fixed = np.sum(w * effects) / np.sum(w)
    q = np.sum(w * (effects - fixed) ** 2)
    df = len(effects) - 1
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
    rw = 1.0 / (ses ** 2 + tau2)
    pooled = np.sum(rw * effects) / np.sum(rw)
    pooled_se = np.sqrt(1.0 / np.sum(rw))
    z = pooled / pooled_se
    p = 2 * stats.norm.sf(abs(z))
    i2 = max(0.0, (q - df) / q) * 100 if q > 0 else 0.0
    return (pooled, pooled_se, p, (pooled - 1.96 * pooled_se, pooled + 1.96 * pooled_se), i2, tau2)

def load_data() -> pd.DataFrame:
    meta = pd.read_csv(META_PATH)
    tss = pd.read_csv(TSS_PATH, index_col=0)
    clr = pd.read_csv(CLR_PATH, index_col=0)
    if SPECIES not in tss.index or SPECIES not in clr.index:
        raise KeyError(f'{SPECIES} missing from abundance matrices')
    samples = [s for s in meta['Sample'] if s in tss.columns and s in clr.columns]
    meta = meta.set_index('Sample').loc[samples].copy()
    df = meta.rename(columns={'Bioproject': 'Cohort'}).reset_index()
    df['RA_pct'] = tss.loc[SPECIES, samples].values
    df['CLR'] = clr.loc[SPECIES, samples].values
    df['Detected'] = df['RA_pct'] > 0
    df['Group'] = pd.Categorical(df['Group'], categories=['Healthy', 'PCOS'], ordered=True)
    return df

def analyze_level(sub: pd.DataFrame, level: str) -> dict:
    pcos = sub[sub['Group'] == 'PCOS']
    healthy = sub[sub['Group'] == 'Healthy']
    ra_p = pcos['RA_pct'].values
    ra_h = healthy['RA_pct'].values
    clr_p = pcos['CLR'].values
    clr_h = healthy['CLR'].values
    mw_stat, mw_p = mannwhitneyu(clr_p, clr_h, alternative='two-sided')
    _, mw_ra_p = mannwhitneyu(ra_p, ra_h, alternative='two-sided')
    if level == 'Overall':
        beta, se, model_p, ci = fit_lmm(sub)
        method = 'LMM'
    else:
        beta, se, model_p, ci = fit_ols(sub)
        method = 'OLS'
    log2fc = np.log2((ra_p.mean() + PSEUDO) / (ra_h.mean() + PSEUDO))
    boot_lo, boot_hi = bootstrap_mean_diff_ci(clr_p, clr_h)
    return {'Level': level, 'Method': method, 'n_PCOS': int(len(pcos)), 'n_Healthy': int(len(healthy)), 'PCOS_prevalence': float(pcos['Detected'].mean()), 'Healthy_prevalence': float(healthy['Detected'].mean()), 'PCOS_mean_RA_pct': float(ra_p.mean()), 'Healthy_mean_RA_pct': float(ra_h.mean()), 'PCOS_median_RA_pct': float(np.median(ra_p)), 'Healthy_median_RA_pct': float(np.median(ra_h)), 'PCOS_IQR_low': float(np.percentile(ra_p, 25)), 'PCOS_IQR_high': float(np.percentile(ra_p, 75)), 'Healthy_IQR_low': float(np.percentile(ra_h, 25)), 'Healthy_IQR_high': float(np.percentile(ra_h, 75)), 'Log2FC_mean_RA': float(log2fc), 'Beta_CLR': beta, 'SE': se, 'CI95_low': ci[0], 'CI95_high': ci[1], 'Cohen_d_CLR': float(cohens_d(clr_p, clr_h)), 'CLR_mean_diff_boot_CI_low': boot_lo, 'CLR_mean_diff_boot_CI_high': boot_hi, 'MW_CLR_p': float(mw_p), 'MW_RA_p': float(mw_ra_p), 'Model_p': model_p, 'Direction_CLR': 'PCOS_enriched' if beta > 0 else 'Healthy_enriched', 'Direction_RA_median': 'PCOS_enriched' if np.median(ra_p) > np.median(ra_h) else 'Healthy_enriched' if np.median(ra_p) < np.median(ra_h) else 'Equal'}

def build_association_table(data: pd.DataFrame) -> pd.DataFrame:
    rows = [analyze_level(data, 'Overall')]
    for c in COHORTS:
        rows.append(analyze_level(data[data['Cohort'] == c].copy(), c))
    df = pd.DataFrame(rows)
    df['Model_FDR_within_table'] = multipletests(df['Model_p'], method='fdr_bh')[1]
    df['MW_CLR_FDR_within_table'] = multipletests(df['MW_CLR_p'], method='fdr_bh')[1]
    diff = pd.read_csv(DIFF_PATH)
    row = diff[diff['Species'] == SPECIES].iloc[0]
    df['GenomeWide_LMM_FDR'] = np.nan
    df['GenomeWide_Wilcoxon_FDR'] = np.nan
    df.loc[df['Level'] == 'Overall', 'GenomeWide_LMM_FDR'] = float(row['LMM_FDR'])
    df.loc[df['Level'] == 'Overall', 'GenomeWide_Wilcoxon_FDR'] = float(row['Wilcoxon_FDR'])
    df['Report_FDR'] = df['Model_FDR_within_table']
    df.loc[df['Level'] == 'Overall', 'Report_FDR'] = df.loc[df['Level'] == 'Overall', 'GenomeWide_LMM_FDR']
    return df

def build_descriptive(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    levels = [('Overall', data)] + [(c, data[data['Cohort'] == c]) for c in COHORTS]
    for level, sub in levels:
        for grp in ['Healthy', 'PCOS']:
            g = sub[sub['Group'] == grp]
            ra = g['RA_pct'].values
            rows.append({'Level': level, 'Group': grp, 'n': len(g), 'Prevalence': float(g['Detected'].mean()), 'Mean_RA_pct': float(ra.mean()), 'Median_RA_pct': float(np.median(ra)), 'IQR_low': float(np.percentile(ra, 25)), 'IQR_high': float(np.percentile(ra, 75)), 'Mean_CLR': float(g['CLR'].mean()), 'SD_CLR': float(g['CLR'].std(ddof=1))})
    return pd.DataFrame(rows)

def build_meta_consistency(assoc: pd.DataFrame):
    cohort = assoc[assoc['Level'] != 'Overall'].copy()
    pooled, se, p, ci, i2, tau2 = random_effects_meta(cohort['Beta_CLR'].values, cohort['SE'].values)
    meta = pd.DataFrame([{'Analysis': 'Random_effects_meta', 'k_cohorts': len(cohort), 'Beta_meta': pooled, 'SE_meta': se, 'CI95_low': ci[0], 'CI95_high': ci[1], 'P_meta': p, 'I2_pct': i2, 'Tau2': tau2}])
    n_clr = int((cohort['Direction_CLR'] == 'PCOS_enriched').sum())
    n_ra = int((cohort['Direction_RA_median'] == 'PCOS_enriched').sum())
    consistency = pd.DataFrame([{'Metric': 'CLR_beta_sign_concordance', 'Value': f'{n_clr}/{len(cohort)}', 'Detail': 'Cohorts with positive CLR beta (PCOS-enriched)', 'P_binomial_one_sided': binomtest(n_clr, n=len(cohort), p=0.5, alternative='greater').pvalue}, {'Metric': 'RA_median_sign_concordance', 'Value': f'{n_ra}/{len(cohort)}', 'Detail': 'Cohorts with higher PCOS median relative abundance', 'P_binomial_one_sided': binomtest(n_ra, n=len(cohort), p=0.5, alternative='greater').pvalue}, {'Metric': 'Cohorts_with_Model_FDR_lt_0.05', 'Value': int((cohort['Model_FDR_within_table'] < 0.05).sum()), 'Detail': 'Per-cohort OLS FDR (BH across Overall+3 cohorts)', 'P_binomial_one_sided': np.nan}, {'Metric': 'Magnitude_range_Beta_CLR', 'Value': f"{cohort['Beta_CLR'].min():.3f} to {cohort['Beta_CLR'].max():.3f}", 'Detail': 'Min–max CLR beta across cohorts', 'P_binomial_one_sided': np.nan}])
    return (meta, consistency)

def format_fdr(fdr: float) -> str:
    if not np.isfinite(fdr):
        return 'NA'
    if fdr < 0.001:
        return f'{fdr:.2e}'
    return f'{fdr:.3f}'

def draw_ref_style_boxplot(ax, plot_df, fdr_val, title_str, show_ylabel=True, show_sig=True, title_fs: int=11, tick_fs: int=10, label_fs: int=11, sig_fs: int=11, point_size: float=22):
    groups_order = ['PCOS', 'Healthy']
    positions = {g: i + 1 for i, g in enumerate(groups_order)}
    max_val = float(plot_df['Abundance'].max())
    box_data = [to_log(plot_df[plot_df['Group'] == g]['Abundance'].values) for g in groups_order]
    bp = ax.boxplot(box_data, positions=[1, 2], widths=0.45, patch_artist=True, showfliers=False, medianprops=dict(color='#E69F00', linewidth=2.0), whiskerprops=dict(color='#444444', linewidth=0.9), capprops=dict(color='#444444', linewidth=0.9), boxprops=dict(linewidth=0.9), zorder=2)
    for i, g in enumerate(groups_order):
        bp['boxes'][i].set_facecolor(GROUP_COLORS[g])
        bp['boxes'][i].set_alpha(0.35)
        bp['boxes'][i].set_edgecolor('#333333')
    scatter_rows = []
    np.random.seed(42)
    for g in groups_order:
        sub = plot_df[plot_df['Group'] == g].copy()
        jitter = np.random.normal(0, 0.07, len(sub))
        xvals = positions[g] + jitter
        yvals = to_log(sub['Abundance'].values)
        cvals = [COHORT_COLORS.get(b, '#888888') for b in sub['Cohort']]
        ax.scatter(xvals, yvals, c=cvals, s=point_size, alpha=0.85, edgecolors='white', linewidths=0.25, zorder=3)
        for xi, yi, ci, (_, row) in zip(xvals, yvals, sub['Cohort'], sub.iterrows()):
            scatter_rows.append({'Group': g, 'Cohort': ci, 'Abundance_raw': row['Abundance'], 'x_plot': xi, 'y_log10': yi})
    set_log_yaxis(ax, max_val, ylabel=show_ylabel, tick_fs=tick_fs, label_fs=label_fs)
    if show_sig and np.isfinite(fdr_val):
        sig_str = '***' if fdr_val < 0.001 else '**' if fdr_val < 0.01 else '*' if fdr_val < 0.05 else 'ns'
        y_top = to_log(max_val) + 0.18
        ax.annotate('', xy=(2, y_top), xytext=(1, y_top), arrowprops=dict(arrowstyle='-', color='#333333', lw=1.0))
        ax.text(1.5, y_top + 0.04, sig_str, ha='center', va='bottom', fontsize=sig_fs, color='#333333')
    ax.set_xticks([1, 2])
    ax.set_xticklabels(groups_order, fontsize=tick_fs)
    ax.set_xlim(0.4, 2.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(title_str, fontsize=title_fs, pad=6, linespacing=1.35)
    return pd.DataFrame(scatter_rows)

def plot_pooled_reference_style(data: pd.DataFrame, assoc: pd.DataFrame):
    pooled = assoc[assoc['Level'] == 'Overall'].iloc[0]
    fdr = float(pooled['Report_FDR'])
    sig = '***' if fdr < 0.001 else '**' if fdr < 0.01 else '*' if fdr < 0.05 else 'ns'
    title = f'{SPECIES_ITALIC}\nFDR = {fdr:.3f} {sig}'
    plot_df = pd.DataFrame({'Abundance': data['RA_pct'].values, 'Group': data['Group'].astype(str).values, 'Cohort': data['Cohort'].values})
    fig, ax = plt.subplots(figsize=(4.2, 4.8))
    scatter = draw_ref_style_boxplot(ax, plot_df, fdr, title)
    cohort_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=5.5, label=lbl, markeredgecolor='white', markeredgewidth=0.3) for lbl, c in COHORT_COLORS.items()]
    group_handles = [mpatches.Patch(facecolor=c, alpha=0.55, edgecolor='#333333', linewidth=0.6, label=g) for g, c in GROUP_COLORS.items()]
    leg1 = ax.legend(handles=cohort_handles, title='Cohort', title_fontsize=9, fontsize=9, loc='upper right', framealpha=0.85, handlelength=1.0, borderpad=0.4)
    ax.add_artist(leg1)
    ax.legend(handles=group_handles, title='Group', title_fontsize=9, fontsize=9, loc='lower right', framealpha=0.85, handlelength=1.0, borderpad=0.4)
    scatter['Species'] = SPECIES
    scatter.to_csv(OUT / 'Fig_Pvulgatus_pooled_boxplot_plotdata.tsv', sep='\t', index=False)
    save_figure(fig, 'Fig_Pvulgatus_pooled_boxplot')

def plot_cohort_panels(data: pd.DataFrame, assoc: pd.DataFrame):
    panels = list(COHORTS)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 5.0), sharey=False)
    all_rows = []
    for ax, level in zip(axes, panels):
        sub = data[data['Cohort'] == level]
        row = assoc[assoc['Level'] == level].iloc[0]
        fdr = float(row['Report_FDR'])
        es = float(row['Cohen_d_CLR'])
        prev_p = float(row['PCOS_prevalence']) * 100
        prev_h = float(row['Healthy_prevalence']) * 100
        direction = row['Direction_RA_median']
        arrow = 'PCOS↑' if direction == 'PCOS_enriched' else 'Healthy↑' if direction == 'Healthy_enriched' else 'Equal'
        title = f'{level}\nPrevalence: PCOS {prev_p:.1f}%, Healthy {prev_h:.1f}%\nEffect size = {es:.2f}; FDR = {format_fdr(fdr)}\nDirection: {arrow}'
        plot_df = pd.DataFrame({'Abundance': sub['RA_pct'].values, 'Group': sub['Group'].astype(str).values, 'Cohort': sub['Cohort'].values})
        sc = draw_ref_style_boxplot(ax, plot_df, fdr, title, show_ylabel=ax is axes[0], show_sig=True, title_fs=11, tick_fs=10, label_fs=11, sig_fs=12, point_size=26)
        sc['Panel'] = level
        all_rows.append(sc)
    cohort_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=7, label=lbl, markeredgecolor='white', markeredgewidth=0.3) for lbl, c in COHORT_COLORS.items()]
    group_handles = [mpatches.Patch(facecolor=c, alpha=0.55, edgecolor='#333333', linewidth=0.6, label=g) for g, c in GROUP_COLORS.items()]
    fig.legend(handles=cohort_handles + [mpatches.Patch(alpha=0, label='')] + group_handles, loc='lower center', ncol=6, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f'{SPECIES_ITALIC}: relative abundance by cohort (log-scale)', fontsize=13, y=1.03)
    plt.tight_layout()
    pd.concat(all_rows, ignore_index=True).to_csv(OUT / 'Fig_Pvulgatus_cohort_panels_plotdata.tsv', sep='\t', index=False)
    save_figure(fig, 'Fig_Pvulgatus_cohort_panels')

def write_stable(assoc: pd.DataFrame, meta: pd.DataFrame):
    rows = []
    for _, r in assoc.iterrows():
        rows.append({'Analysis': r['Level'], 'Method': r['Method'], 'n_PCOS': int(r['n_PCOS']), 'n_Healthy': int(r['n_Healthy']), 'Prevalence_PCOS': round(float(r['PCOS_prevalence']), 4), 'Prevalence_Healthy': round(float(r['Healthy_prevalence']), 4), 'Median_RA_pct_PCOS': round(float(r['PCOS_median_RA_pct']), 4), 'Median_RA_pct_Healthy': round(float(r['Healthy_median_RA_pct']), 4), 'Mean_RA_pct_PCOS': round(float(r['PCOS_mean_RA_pct']), 4), 'Mean_RA_pct_Healthy': round(float(r['Healthy_mean_RA_pct']), 4), 'Log2FC_mean_RA': round(float(r['Log2FC_mean_RA']), 4), 'Effect_size_Cohen_d_CLR': round(float(r['Cohen_d_CLR']), 4), 'CLR_beta_PCOS_vs_Healthy': round(float(r['Beta_CLR']), 4), 'CI95_low': round(float(r['CI95_low']), 4), 'CI95_high': round(float(r['CI95_high']), 4), 'Model_P': float(r['Model_p']), 'FDR': float(r['Report_FDR']), 'Direction_CLR': r['Direction_CLR'], 'Direction_RA_median': r['Direction_RA_median']})
    m = meta.iloc[0]
    rows.append({'Analysis': 'Random_effects_meta', 'Method': 'RE_meta', 'n_PCOS': '', 'n_Healthy': '', 'Prevalence_PCOS': '', 'Prevalence_Healthy': '', 'Median_RA_pct_PCOS': '', 'Median_RA_pct_Healthy': '', 'Mean_RA_pct_PCOS': '', 'Mean_RA_pct_Healthy': '', 'Log2FC_mean_RA': '', 'Effect_size_Cohen_d_CLR': '', 'CLR_beta_PCOS_vs_Healthy': round(float(m['Beta_meta']), 4), 'CI95_low': round(float(m['CI95_low']), 4), 'CI95_high': round(float(m['CI95_high']), 4), 'Model_P': float(m['P_meta']), 'FDR': '', 'Direction_CLR': 'PCOS_enriched' if m['Beta_meta'] > 0 else 'Healthy_enriched', 'Direction_RA_median': '', 'I2_pct': round(float(m['I2_pct']), 2)})
    stable = pd.DataFrame(rows)
    if 'I2_pct' not in stable.columns:
        stable['I2_pct'] = ''
    stable['I2_pct'] = stable['I2_pct'].fillna('')
    stable.to_csv(OUT / 'Table_S_Pvulgatus_association_stats.tsv', sep='\t', index=False)
    md_lines = ['# Table S. *Phocaeicola vulgatus* association statistics', '', 'Metrics requested by the reviewer: effect size, prevalence, relative abundance,', '95% confidence intervals, and corrected *P* (FDR), for pooled and per-cohort analyses.', '', "| Analysis | Method | n PCOS/Healthy | Prevalence PCOS/Healthy | Median RA% PCOS/Healthy | Mean RA% PCOS/Healthy | log2FC | Effect size (Cohen's d) | CLR beta (95% CI) | Model P | FDR | Direction (CLR / RA median) |", '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for _, r in assoc.iterrows():
        md_lines.append(f"| {r['Level']} | {r['Method']} | {int(r['n_PCOS'])}/{int(r['n_Healthy'])} | {r['PCOS_prevalence'] * 100:.1f}%/{r['Healthy_prevalence'] * 100:.1f}% | {r['PCOS_median_RA_pct']:.2f}/{r['Healthy_median_RA_pct']:.2f} | {r['PCOS_mean_RA_pct']:.2f}/{r['Healthy_mean_RA_pct']:.2f} | {r['Log2FC_mean_RA']:.3f} | {r['Cohen_d_CLR']:.3f} | {r['Beta_CLR']:.3f} ({r['CI95_low']:.3f}, {r['CI95_high']:.3f}) | {r['Model_p']:.2e} | {format_fdr(float(r['Report_FDR']))} | {r['Direction_CLR']} / {r['Direction_RA_median']} |")
    md_lines.append(f"| RE meta | RE_meta | — | — | — | — | — | — | {m['Beta_meta']:.3f} ({m['CI95_low']:.3f}, {m['CI95_high']:.3f}) | {m['P_meta']:.3g} | — | I²={m['I2_pct']:.1f}% |")
    md_lines += ['', 'Notes:', '- Overall: linear mixed model (CLR ~ Group, random intercept = cohort); FDR is genome-wide BH from the species-level differential analysis.', '- Per-cohort: OLS on CLR; FDR is BH-adjusted across Overall + three cohort tests in this table.', "- Effect size = Cohen's *d* on CLR abundance (PCOS − Healthy).", '- Source TSV: `Table_S_Pvulgatus_association_stats.tsv`.', '']
    (OUT / 'Table_S_Pvulgatus_association_stats.md').write_text('\n'.join(md_lines), encoding='utf-8')

def plot_forest(assoc: pd.DataFrame, meta: pd.DataFrame):
    rows = []
    for _, r in assoc[assoc['Level'] != 'Overall'].iterrows():
        rows.append({'Label': r['Level'], 'Beta': r['Beta_CLR'], 'CI_low': r['CI95_low'], 'CI_high': r['CI95_high'], 'FDR': r['Report_FDR'], 'Type': 'Cohort'})
    pooled = assoc[assoc['Level'] == 'Overall'].iloc[0]
    rows.append({'Label': 'Pooled LMM', 'Beta': pooled['Beta_CLR'], 'CI_low': pooled['CI95_low'], 'CI_high': pooled['CI95_high'], 'FDR': pooled['Report_FDR'], 'Type': 'Pooled'})
    rows.append({'Label': 'RE meta', 'Beta': meta['Beta_meta'].iloc[0], 'CI_low': meta['CI95_low'].iloc[0], 'CI_high': meta['CI95_high'].iloc[0], 'FDR': meta['P_meta'].iloc[0], 'Type': 'Meta'})
    pdf = pd.DataFrame(rows)
    pdf.to_csv(OUT / 'Fig_Pvulgatus_forest_plotdata.tsv', sep='\t', index=False)
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    colors = {'Cohort': '#3C5488', 'Pooled': '#E64B35', 'Meta': '#00A087'}
    y = np.arange(len(pdf))
    for i, r in pdf.iterrows():
        ax.errorbar(r['Beta'], i, xerr=[[r['Beta'] - r['CI_low']], [r['CI_high'] - r['Beta']]], fmt='o', color=colors[r['Type']], capsize=3.5, markersize=6.5)
        ax.text(r['CI_high'] + 0.15, i, f"β={r['Beta']:.2f}; FDR/P={r['FDR']:.3g}", va='center', fontsize=7.5, color='#333333')
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(pdf['Label'])
    ax.set_xlabel('Effect size (CLR beta, PCOS vs Healthy)')
    ax.set_title(f'Effect size and 95% CI: {SPECIES_ITALIC}')
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    xmax = max(pdf['CI_high'].max() + 2.8, 5)
    ax.set_xlim(min(-2.5, pdf['CI_low'].min() - 0.3), xmax)
    plt.tight_layout()
    save_figure(fig, 'Fig_Pvulgatus_forest')

def write_summary(assoc, desc, meta, consistency):
    pooled = assoc[assoc['Level'] == 'Overall'].iloc[0]
    lines = ['# Phocaeicola vulgatus — reviewer statistics summary', '', '## Pooled (LMM, n=169; genome-wide FDR from differential analysis)', f"- Effect size (CLR beta) = {pooled['Beta_CLR']:.3f} (95% CI {pooled['CI95_low']:.3f} to {pooled['CI95_high']:.3f})", f"- Model P = {pooled['Model_p']:.2e}; genome-wide LMM FDR = {pooled['GenomeWide_LMM_FDR']:.4f}", f"- Cohen's d (CLR) = {pooled['Cohen_d_CLR']:.3f}; log2FC (mean RA) = {pooled['Log2FC_mean_RA']:.3f}", f"- Prevalence: PCOS {pooled['PCOS_prevalence']:.1%}, Healthy {pooled['Healthy_prevalence']:.1%}", f"- Median RA%: PCOS {pooled['PCOS_median_RA_pct']:.2f}, Healthy {pooled['Healthy_median_RA_pct']:.2f}", f"- Mean RA%: PCOS {pooled['PCOS_mean_RA_pct']:.2f}, Healthy {pooled['Healthy_mean_RA_pct']:.2f}", '', '## Per-cohort (OLS on CLR)']
    for _, r in assoc[assoc['Level'] != 'Overall'].iterrows():
        lines.append(f"- {r['Level']}: β={r['Beta_CLR']:.3f} (95% CI {r['CI95_low']:.3f} to {r['CI95_high']:.3f}); P={r['Model_p']:.3g}; FDR={r['Report_FDR']:.3g}; prev PCOS/Healthy={r['PCOS_prevalence']:.1%}/{r['Healthy_prevalence']:.1%}; median RA%={r['PCOS_median_RA_pct']:.2f}/{r['Healthy_median_RA_pct']:.2f}; direction CLR={r['Direction_CLR']}, RA median={r['Direction_RA_median']}")
    lines += ['', '## Consistency / meta-analysis', f"- CLR beta concordance: {consistency.iloc[0]['Value']}", f"- RA median concordance: {consistency.iloc[1]['Value']}", f"- Random-effects meta β = {meta['Beta_meta'].iloc[0]:.3f} (95% CI {meta['CI95_low'].iloc[0]:.3f} to {meta['CI95_high'].iloc[0]:.3f}); P={meta['P_meta'].iloc[0]:.3g}; I²={meta['I2_pct'].iloc[0]:.1f}%", '', '## Output files', '- Pvulgatus_association_stats.tsv', '- Pvulgatus_descriptive_stats.tsv', '- Pvulgatus_meta_analysis.tsv', '- Pvulgatus_consistency.tsv', '- Fig_Pvulgatus_pooled_boxplot.{pdf,svg,png,jpg} + plotdata TSV', '- Fig_Pvulgatus_cohort_panels.{pdf,svg,png,jpg} + plotdata TSV', '- Fig_Pvulgatus_forest.{pdf,svg,png,jpg} + plotdata TSV', '- reviewer_response.md']
    (OUT / 'analysis_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

def write_reviewer_response(assoc, meta, consistency):
    pooled = assoc[assoc['Level'] == 'Overall'].iloc[0]
    cohort = assoc[assoc['Level'] != 'Overall']
    cohort_rows = []
    for _, r in cohort.iterrows():
        cohort_rows.append(f"| {r['Level']} | {r['n_PCOS']}/{r['n_Healthy']} | {r['PCOS_prevalence'] * 100:.1f}/{r['Healthy_prevalence'] * 100:.1f} | {r['PCOS_median_RA_pct']:.2f}/{r['Healthy_median_RA_pct']:.2f} | {r['PCOS_mean_RA_pct']:.2f}/{r['Healthy_mean_RA_pct']:.2f} | {r['Log2FC_mean_RA']:.2f} | {r['Beta_CLR']:.2f} ({r['CI95_low']:.2f}, {r['CI95_high']:.2f}) | {r['Cohen_d_CLR']:.2f} | {r['Model_p']:.2e} | {r['Report_FDR']:.3g} | {r['Direction_CLR']} |")
    text = f"# Point-by-point response (P. vulgatus statistics)\n\n- Decision type: revision response (statistics/reporting request)\n- Overall posture: provide complete pooled and per-cohort statistics; show direction and magnitude consistency with figures\n- Major risks: CLR beta is not positive in all three cohorts (2/3); RE meta is non-significant with high I² — must report transparently\n- Suggested ordering: pooled metrics → per-cohort table → consistency figures → limitation sentence\n\n| ID | Reviewer concern | Type | Severity | Proposed action | Missing author input |\n|---|---|---|---|---|---|\n| R.Pv.1 | Provide effect size, prevalence, relative abundance, CI, corrected P for *P. vulgatus* (pooled + per cohort) | Statistics/reporting | Major | Added tables + figures in `review_result/` | Confirm manuscript insertion location (Results / Supplementary Table) |\n| R.Pv.2 | Show whether association direction and magnitude are consistent across three cohorts | Robustness/consistency | Major | Cohort panels + forest plot; report concordance and I² | Confirm figure panel lettering in revised manuscript |\n\n\n**Reviewer comment.** The authors should provide the effect size, prevalence, relative abundance, confidence intervals and corrected *P* values for *Phocaeicola vulgatus*, including both the pooled analysis and each cohort separately. They should also show whether the direction and magnitude of the association are consistent across all three cohorts.\n\n**Response.** We thank the reviewer for this important request. We have now reported the full set of descriptive and association statistics for *P. vulgatus* at both the pooled and cohort-specific levels, and we have added figures that make the cross-cohort direction and magnitude of the association explicit.\n\n\n| Metric | Value |\n|---|---|\n| Prevalence (PCOS / Healthy) | {pooled['PCOS_prevalence'] * 100:.1f}% / {pooled['Healthy_prevalence'] * 100:.1f}% |\n| Median relative abundance (%) | {pooled['PCOS_median_RA_pct']:.2f} / {pooled['Healthy_median_RA_pct']:.2f} |\n| Mean relative abundance (%) | {pooled['PCOS_mean_RA_pct']:.2f} / {pooled['Healthy_mean_RA_pct']:.2f} |\n| log2 fold-change (mean RA) | {pooled['Log2FC_mean_RA']:.3f} |\n| Effect size (CLR beta, PCOS vs Healthy) | {pooled['Beta_CLR']:.3f} |\n| 95% CI of CLR beta | {pooled['CI95_low']:.3f} to {pooled['CI95_high']:.3f} |\n| Cohen's *d* (CLR) | {pooled['Cohen_d_CLR']:.3f} |\n| Model *P* | {pooled['Model_p']:.2e} |\n| Genome-wide BH FDR (LMM) | {pooled['GenomeWide_LMM_FDR']:.4f} |\n\nThese pooled estimates are consistent with the species-level differential abundance analysis in the manuscript (LMM FDR = {pooled['GenomeWide_LMM_FDR']:.3f}). The corresponding relative-abundance distribution is shown in **Fig. R1** (log-scale boxplot with cohort-coloured points; same visualisation style as the original top-species panels).\n\n\n| Cohort | n (PCOS/Healthy) | Prevalence (%) PCOS/Healthy | Median RA (%) PCOS/Healthy | Mean RA (%) PCOS/Healthy | log2FC | CLR beta (95% CI) | Cohen's d | Model P | BH FDR | Direction (CLR) |\n|---|---|---|---|---|---|---|---|---|---|---|\n{chr(10).join(cohort_rows)}\n\nPer-cohort FDR values are Benjamini–Hochberg adjusted across the Overall + three cohort tests reported in this table. **Fig. R2** shows the same log-scale relative-abundance boxplots separately for the pooled set and each cohort.\n\n\n- Relative-abundance median direction: **{consistency.iloc[1]['Value']}** cohorts show higher median *P. vulgatus* abundance in PCOS.\n- CLR beta sign concordance: **{consistency.iloc[0]['Value']}** cohorts show a positive (PCOS-enriched) CLR beta. Specifically, PRJNA530971 and PRJNA791492 are PCOS-enriched on CLR, whereas PRJNA549764 shows a near-null / slightly Healthy-directed CLR beta with wide confidence intervals (small sample size; prevalence 100% in both groups).\n- Random-effects meta-analysis of cohort CLR betas: β = {meta['Beta_meta'].iloc[0]:.3f} (95% CI {meta['CI95_low'].iloc[0]:.3f} to {meta['CI95_high'].iloc[0]:.3f}); *P* = {meta['P_meta'].iloc[0]:.3g}; I² = {meta['I2_pct'].iloc[0]:.1f}%.\n\n**Fig. R3** (forest plot) displays cohort-specific effect sizes with 95% CIs together with the pooled LMM and random-effects meta estimates. Together, these analyses show that the PCOS enrichment of *P. vulgatus* is strongest and significant in PRJNA530971, directionally concordant for relative-abundance medians in all three cohorts, and supported by the pooled LMM, while between-cohort heterogeneity in CLR effect magnitude is substantial (high I²). We now state this heterogeneity explicitly in the revised Results/Discussion.\n\nSource data for all panels are provided as TSV files in `review_result/`.\n\n- [ ] Insert pooled + per-cohort statistics table (or Supplementary Table) for *P. vulgatus*\n- [ ] Add Fig. R1–R3 (or renumber into main/supplementary figure panels)\n- [ ] Add one sentence on cross-cohort direction concordance and I² heterogeneity\n- [ ] AUTHOR_INPUT_NEEDED: exact manuscript section / figure panel IDs after layout\n\n- Confirm where the new table/figures will appear in the revised manuscript\n- Transparent reporting of 2/3 CLR concordance and high I² is required; do not overclaim uniform magnitude across cohorts\n\n- 已补齐汇总 LMM + 三队列 OLS：效应量(β)、流行率、相对丰度(均值/中位数)、95% CI、校正 P/FDR\n- 中位数相对丰度方向三队列一致（3/3 PCOS↑）；CLR β 符号 2/3 一致（PRJNA549764 接近零且 CI 跨零）\n- 随机效应 meta：β 为正但 P 不显著，I² 高——回复中已如实写出，建议正文加一句异质性说明\n- 需作者确认：新图表插入正文还是补充材料，以及最终图号\n"
    (OUT / 'reviewer_response.md').write_text(text, encoding='utf-8')

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_data()
    desc = build_descriptive(data)
    desc.to_csv(OUT / 'Pvulgatus_descriptive_stats.tsv', sep='\t', index=False)
    assoc = build_association_table(data)
    assoc.to_csv(OUT / 'Pvulgatus_association_stats.tsv', sep='\t', index=False)
    meta, consistency = build_meta_consistency(assoc)
    meta.to_csv(OUT / 'Pvulgatus_meta_analysis.tsv', sep='\t', index=False)
    consistency.to_csv(OUT / 'Pvulgatus_consistency.tsv', sep='\t', index=False)
    plot_pooled_reference_style(data, assoc)
    plot_cohort_panels(data, assoc)
    plot_forest(assoc, meta)
    write_stable(assoc, meta)
    write_summary(assoc, desc, meta, consistency)
    write_reviewer_response(assoc, meta, consistency)
    print('Done ->', OUT)
    print(assoc[['Level', 'Beta_CLR', 'CI95_low', 'CI95_high', 'Model_p', 'Report_FDR', 'PCOS_prevalence', 'Healthy_prevalence', 'PCOS_median_RA_pct', 'Healthy_median_RA_pct', 'Direction_CLR', 'Direction_RA_median']].to_string(index=False))
    print(meta.to_string(index=False))
    print(consistency.to_string(index=False))
if __name__ == '__main__':
    main()
