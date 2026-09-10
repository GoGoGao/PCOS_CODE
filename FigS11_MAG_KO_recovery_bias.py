from pathlib import Path
import os
PKG_ROOT = Path(__file__).resolve().parents[1]
SHARED = PKG_ROOT / 'shared_input'
RSCRIPT = os.environ.get('RSCRIPT', 'Rscript')
ROOT = PKG_ROOT / 'output' / 'FigS11_L1_bias'
import os
import warnings
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
warnings.filterwarnings('ignore')
BASE = Path(__file__).resolve().parent
META_MAG = PROJECT / 'sgb_kegg/kegg_stat/sgb.info.comb.tsv'
KO_MATRIX = PROJECT / 'sgb_kegg/kegg_stat/results/source_data/sample_KO_binary_matrix.tsv'
ENA_CACHE = BASE / 'cache' / 'ena_filereport_all.tsv'
TABLES = OUT / 'tables'
SOURCE = OUT / 'source_data'
FIG_PANELS = OUT / 'figures' / 'panels'
FIG_COMBINED = OUT / 'figures'
COHORTS = ['PRJNA530971', 'PRJNA549764', 'PRJNA791492']
METRICS = [('n_mag', 'MAGs per sample'), ('depth_gb', 'Sequencing depth (Gb)'), ('median_mag_n50', 'Median MAG N50 (bp)'), ('mean_completeness', 'Mean MAG completeness (%)'), ('mean_contamination', 'Mean MAG contamination (%)'), ('total_mag_length_mb', 'Total MAG length (Mb)'), ('n_ko', 'KOs detected per sample')]
matplotlib.rcParams.update({'font.family': 'Times New Roman', 'font.weight': 'regular', 'axes.labelweight': 'regular', 'axes.titleweight': 'regular', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'axes.unicode_minus': False, 'figure.dpi': 150, 'savefig.dpi': 300})
C_PCOS = '#D63B3B'
C_HEALTHY = '#3B7DD8'
PALETTE = {'PCOS': C_PCOS, 'Healthy': C_HEALTHY}

def ensure_dirs():
    for d in [TABLES, SOURCE, FIG_PANELS, FIG_COMBINED, ENA_CACHE.parent]:
        d.mkdir(parents=True, exist_ok=True)

def save_figure(fig, stem: str, panel: bool=False):
    folder = FIG_PANELS if panel else FIG_COMBINED
    for ext in ('pdf', 'svg', 'png', 'jpg'):
        fig.savefig(folder / f'{stem}.{ext}', bbox_inches='tight', dpi=300 if ext in ('png', 'jpg') else None)
    plt.close(fig)

def get_sample(mag_id: str) -> str:
    for sep in ('_bin.', '.bin.'):
        if sep in mag_id:
            return mag_id.split(sep)[0]
    parts = mag_id.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return mag_id

def rank_biserial(u_stat: float, n1: int, n2: int) -> float:
    return 2 * u_stat / (n1 * n2) - 1

def mann_whitney_table(df: pd.DataFrame, metric: str, group_col: str='Group') -> dict:
    g1 = df[df[group_col] == 'PCOS'][metric].dropna()
    g2 = df[df[group_col] == 'Healthy'][metric].dropna()
    if len(g1) < 2 or len(g2) < 2:
        return {'metric': metric, 'n_PCOS': len(g1), 'n_Healthy': len(g2), 'median_PCOS': np.nan, 'median_Healthy': np.nan, 'p_value': np.nan, 'rank_biserial': np.nan}
    u, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
    return {'metric': metric, 'n_PCOS': len(g1), 'n_Healthy': len(g2), 'median_PCOS': g1.median(), 'median_Healthy': g2.median(), 'IQR_PCOS': f'{g1.quantile(0.25):.4g}-{g1.quantile(0.75):.4g}', 'IQR_Healthy': f'{g2.quantile(0.25):.4g}-{g2.quantile(0.75):.4g}', 'p_value': p, 'rank_biserial': rank_biserial(u, len(g1), len(g2))}

def fetch_ena_filereport() -> pd.DataFrame:
    if ENA_CACHE.exists():
        ena = pd.read_csv(ENA_CACHE, sep='\t')
        if 'Sample' in ena.columns and 'sample' not in ena.columns:
            ena.rename(columns={'Sample': 'sample'}, inplace=True)
        return ena
    import urllib.request
    rows = []
    for acc in COHORTS:
        url = f'https://www.ebi.ac.uk/ena/portal/api/filereport?accession={acc}&result=read_run&fields=run_accession,base_count,read_count,fastq_bytes&format=tsv&limit=0'
        with urllib.request.urlopen(url, timeout=120) as resp:
            text = resp.read().decode()
        part = pd.read_csv(pd.io.common.StringIO(text), sep='\t')
        part['BIOPROJECT'] = acc
        rows.append(part)
    ena = pd.concat(rows, ignore_index=True)
    ena.rename(columns={'run_accession': 'sample'}, inplace=True)

    def parse_bytes(x):
        if pd.isna(x) or x == '':
            return np.nan
        parts = str(x).split(';')
        try:
            return sum((int(p) for p in parts))
        except ValueError:
            return np.nan
    ena['fastq_bytes_total'] = ena['fastq_bytes'].map(parse_bytes)
    ena['depth_gb'] = ena['base_count'] / 1000000000.0
    ena['fastq_gb'] = ena['fastq_bytes_total'] / 1000000000.0
    ena.to_csv(ENA_CACHE, sep='\t', index=False)
    return ena

def build_sample_metrics() -> pd.DataFrame:
    meta = pd.read_csv(META_MAG, sep='\t')
    meta['sample'] = meta['ID'].map(get_sample)
    mag_agg = meta.groupby(['sample', 'Group', 'BIOPROJECT']).agg(n_mag=('ID', 'count'), mean_completeness=('completeness', 'mean'), mean_contamination=('contamination', 'mean'), median_mag_n50=('N50', 'median'), mean_mag_n50=('N50', 'mean'), total_mag_length_mb=('length', lambda x: x.sum() / 1000000.0), median_genome_size_mb=('length', lambda x: x.median() / 1000000.0), high_quality_mag=('completeness', lambda x: (x >= 90).sum())).reset_index()
    sko = pd.read_csv(KO_MATRIX, sep='\t', index_col=0)
    ko_counts = sko.sum(axis=0)
    ko_df = pd.DataFrame({'sample': ko_counts.index, 'n_ko': ko_counts.values})
    groups = pd.read_csv(GROUP_INFO)
    groups.rename(columns={'Bioproject': 'BIOPROJECT', 'Sample': 'sample'}, inplace=True)
    ena = fetch_ena_filereport()
    depth = ena[['sample', 'BIOPROJECT', 'base_count', 'read_count', 'depth_gb', 'fastq_gb']]
    all_samples = groups.merge(depth, on=['sample', 'BIOPROJECT'], how='left')
    all_samples = all_samples.merge(ko_df, on='sample', how='left')
    all_samples = all_samples.merge(mag_agg, on=['sample', 'BIOPROJECT', 'Group'], how='left')
    all_samples['has_mag'] = all_samples['n_mag'].notna() & (all_samples['n_mag'] > 0)
    all_samples.loc[~all_samples['has_mag'], 'n_mag'] = 0
    all_samples.loc[~all_samples['has_mag'], 'n_ko'] = all_samples.loc[~all_samples['has_mag'], 'n_ko'].fillna(0)
    return all_samples

def comparison_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mag_pos = df[df['has_mag']].copy()
    overall_rows = [mann_whitney_table(mag_pos, m) for m, _ in METRICS]
    overall = pd.DataFrame(overall_rows)
    cohort_rows = []
    for cohort in COHORTS:
        sub = mag_pos[mag_pos['BIOPROJECT'] == cohort]
        for metric, label in METRICS:
            row = mann_whitney_table(sub, metric)
            row['BIOPROJECT'] = cohort
            row['metric_label'] = label
            cohort_rows.append(row)
    per_cohort = pd.DataFrame(cohort_rows)
    no_mag = df[~df['has_mag']]
    if len(no_mag) > 0:
        depth_row = mann_whitney_table(no_mag, 'depth_gb')
        depth_row['subset'] = 'no_MAG_samples'
        no_mag_stats = pd.DataFrame([depth_row])
    else:
        no_mag_stats = pd.DataFrame()
    recovery_rows = []
    for cohort in COHORTS + ['All']:
        sub = df if cohort == 'All' else df[df['BIOPROJECT'] == cohort]
        for grp in ['PCOS', 'Healthy']:
            s = sub[sub['Group'] == grp]
            recovery_rows.append({'BIOPROJECT': cohort, 'Group': grp, 'n_samples': len(s), 'n_with_MAG': int(s['has_mag'].sum()), 'MAG_recovery_rate_pct': 100 * s['has_mag'].mean(), 'median_depth_gb': s['depth_gb'].median()})
    recovery = pd.DataFrame(recovery_rows)
    return (overall, per_cohort, no_mag_stats, recovery, mag_pos)

def correlation_analysis(mag_pos: pd.DataFrame) -> pd.DataFrame:
    pairs = [('n_mag', 'n_ko'), ('depth_gb', 'n_mag'), ('depth_gb', 'n_ko'), ('total_mag_length_mb', 'n_ko'), ('median_mag_n50', 'n_ko'), ('read_count', 'n_mag')]
    rows = []
    for x, y in pairs:
        valid = mag_pos[[x, y]].dropna()
        if len(valid) < 5:
            continue
        r, p = stats.spearmanr(valid[x], valid[y])
        rows.append({'var_x': x, 'var_y': y, 'n': len(valid), 'spearman_r': r, 'p_value': p})
    return pd.DataFrame(rows)

def _boxplot_groups(ax, data: pd.DataFrame, metric: str, ylab: str):
    order = ['Healthy', 'PCOS']
    values = [data[data.Group == g][metric].dropna().values for g in order]
    bp = ax.boxplot(values, tick_labels=order, widths=0.55, patch_artist=True, showfliers=True, medianprops={'color': 'black', 'linewidth': 1})
    for patch, grp in zip(bp['boxes'], order):
        patch.set_facecolor(PALETTE[grp])
        patch.set_alpha(0.75)
    for i, grp in enumerate(order, start=1):
        y = data[data.Group == grp][metric].dropna().values
        x = np.random.default_rng(42).normal(i, 0.06, size=len(y))
        ax.scatter(x, y, c='black', alpha=0.35, s=12, zorder=3)
    ax.set_ylabel(ylab, fontsize=10)
    ax.set_xlabel('')
    return mann_whitney_table(data, metric)

def plot_group_boxplots(mag_pos: pd.DataFrame):
    key_metrics = [('n_mag', 'MAGs per sample'), ('depth_gb', 'Sequencing depth (Gb)'), ('median_mag_n50', 'Median MAG N50 (bp)'), ('n_ko', 'KOs detected per sample')]
    plot_rows = []
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))
    axes = axes.flatten()
    for ax, (metric, ylab) in zip(axes, key_metrics):
        sub = mag_pos[['Group', metric, 'BIOPROJECT']].dropna(subset=[metric])
        stat = _boxplot_groups(ax, sub, metric, ylab)
        p = stat['p_value']
        p_txt = 'P < 0.001' if p < 0.001 else f'P = {p:.3f}'
        ax.text(0.5, 0.97, p_txt, transform=ax.transAxes, ha='center', va='top', fontsize=9)
        for _, r in sub.iterrows():
            plot_rows.append({'sample_metric': metric, 'Group': r['Group'], 'BIOPROJECT': r['BIOPROJECT'], 'value': r[metric]})
    fig.suptitle('Technical covariates in MAG-positive samples (PCOS vs Healthy)', fontsize=11, y=1.02)
    fig.tight_layout()
    save_figure(fig, 'Fig_L1a_four_comparisons', panel=True)
    pd.DataFrame(plot_rows).to_csv(SOURCE / 'Fig_L1a_four_comparisons.tsv', sep='\t', index=False)

def plot_quality_boxplots(mag_pos: pd.DataFrame):
    metrics = [('mean_completeness', 'Mean MAG completeness (%)'), ('mean_contamination', 'Mean MAG contamination (%)'), ('total_mag_length_mb', 'Total MAG length (Mb)')]
    plot_rows = []
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.6))
    for ax, (metric, ylab) in zip(axes, metrics):
        sub = mag_pos[['Group', metric]].dropna()
        stat = _boxplot_groups(ax, sub, metric, ylab)
        p = stat['p_value']
        p_txt = 'P < 0.001' if p < 0.001 else f'P = {p:.3f}'
        ax.text(0.5, 0.97, p_txt, transform=ax.transAxes, ha='center', va='top', fontsize=8)
        for _, r in sub.iterrows():
            plot_rows.append({'metric': metric, 'Group': r['Group'], 'value': r[metric]})
    fig.suptitle('MAG quality and genome recovery proxies', fontsize=11, y=1.02)
    fig.tight_layout()
    save_figure(fig, 'Fig_L1b_mag_quality', panel=True)
    pd.DataFrame(plot_rows).to_csv(SOURCE / 'Fig_L1b_mag_quality.tsv', sep='\t', index=False)

def _boxplot_cohort(ax, data: pd.DataFrame, metric: str, ylab: str):
    from matplotlib.patches import Patch
    box_data, positions, colors = ([], [], [])
    pos = 1
    for cohort in COHORTS:
        for grp in ['Healthy', 'PCOS']:
            vals = data[(data.BIOPROJECT == cohort) & (data.Group == grp)][metric].dropna().values
            box_data.append(vals if len(vals) else [np.nan])
            positions.append(pos)
            colors.append(PALETTE[grp])
            pos += 1
        pos += 0.5
    bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax.set_xticks([1.5, 4.0, 6.5])
    ax.set_xticklabels(COHORTS, rotation=15, ha='right', fontsize=8)
    ax.set_ylabel(ylab, fontsize=9)
    ax.legend(handles=[Patch(facecolor=PALETTE[g], label=g) for g in ['Healthy', 'PCOS']], fontsize=8, loc='upper right')

def plot_cohort_facets(mag_pos: pd.DataFrame):
    key_metrics = [('n_mag', 'MAGs per sample'), ('depth_gb', 'Depth (Gb)'), ('n_ko', 'KOs per sample')]
    plot_rows = []
    fig, axes = plt.subplots(len(key_metrics), 1, figsize=(7.5, 8.5), sharex=True)
    for ax, (metric, ylab) in zip(axes, key_metrics):
        sub = mag_pos[['Group', metric, 'BIOPROJECT']].dropna(subset=[metric])
        _boxplot_cohort(ax, sub, metric, ylab)
        for _, r in sub.iterrows():
            plot_rows.append({'metric': metric, 'BIOPROJECT': r['BIOPROJECT'], 'Group': r['Group'], 'value': r[metric]})
    axes[-1].set_xlabel('Cohort', fontsize=10)
    fig.suptitle('Per-cohort technical covariate comparisons', fontsize=11, y=1.01)
    fig.tight_layout()
    save_figure(fig, 'Fig_L1c_cohort_facets', panel=True)
    pd.DataFrame(plot_rows).to_csv(SOURCE / 'Fig_L1c_cohort_facets.tsv', sep='\t', index=False)

def plot_scatter_correlations(mag_pos: pd.DataFrame, corr_df: pd.DataFrame):
    pairs = [('n_mag', 'n_ko', 'MAGs per sample', 'KOs per sample'), ('depth_gb', 'n_mag', 'Sequencing depth (Gb)', 'MAGs per sample'), ('depth_gb', 'n_ko', 'Sequencing depth (Gb)', 'KOs per sample'), ('total_mag_length_mb', 'n_ko', 'Total MAG length (Mb)', 'KOs per sample')]
    plot_rows = []
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))
    axes = axes.flatten()
    for ax, (x, y, xl, yl) in zip(axes, pairs):
        sub = mag_pos[['Group', x, y]].dropna()
        for grp, color in PALETTE.items():
            g = sub[sub['Group'] == grp]
            ax.scatter(g[x], g[y], c=color, alpha=0.65, s=22, label=grp, edgecolors='none')
            for _, r in g.iterrows():
                plot_rows.append({'x_var': x, 'y_var': y, 'Group': grp, 'x': r[x], 'y': r[y]})
        if len(sub) >= 5:
            slope, intercept, r_val, p_val, _ = stats.linregress(sub[x], sub[y])
            xx = np.linspace(sub[x].min(), sub[x].max(), 50)
            ax.plot(xx, slope * xx + intercept, color='#333333', lw=1.2, alpha=0.8)
            sp = corr_df[(corr_df['var_x'] == x) & (corr_df['var_y'] == y)]
            if len(sp):
                rho = sp.iloc[0]['spearman_r']
                p = sp.iloc[0]['p_value']
                ax.text(0.04, 0.96, f'Spearman rho = {rho:.2f}\nP = {p:.2e}', transform=ax.transAxes, va='top', fontsize=8)
        ax.set_xlabel(xl, fontsize=9)
        ax.set_ylabel(yl, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=9, bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle('Correlations among recovery metrics', fontsize=11, y=1.06)
    fig.tight_layout()
    save_figure(fig, 'Fig_L1d_correlations', panel=True)
    pd.DataFrame(plot_rows).to_csv(SOURCE / 'Fig_L1d_correlations.tsv', sep='\t', index=False)

def plot_mag_recovery(df: pd.DataFrame, recovery: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6))
    rec = recovery[recovery['BIOPROJECT'] != 'All'].copy()
    x = np.arange(len(COHORTS))
    w = 0.35
    for i, grp in enumerate(['Healthy', 'PCOS']):
        vals = [rec[(rec.BIOPROJECT == c) & (rec.Group == grp)]['MAG_recovery_rate_pct'].values[0] for c in COHORTS]
        axes[0].bar(x + (i - 0.5) * w, vals, width=w, label=grp, color=PALETTE[grp], alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(COHORTS, rotation=20, ha='right', fontsize=8)
    axes[0].set_ylabel('MAG recovery rate (%)', fontsize=9)
    axes[0].set_title('MAG recovery by cohort', fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0, 105)
    depth_plot = []
    for has_mag, label in [(True, 'With MAG'), (False, 'Without MAG')]:
        sub = df[df['has_mag'] == has_mag]
        for grp in ['PCOS', 'Healthy']:
            s = sub[sub.Group == grp]['depth_gb'].dropna()
            depth_plot.extend([{'subset': label, 'Group': grp, 'depth_gb': v} for v in s])
    depth_df = pd.DataFrame(depth_plot)
    subsets = ['With MAG', 'Without MAG']
    positions = []
    box_data = []
    colors = []
    pos = 1
    for subset in subsets:
        for grp in ['Healthy', 'PCOS']:
            vals = depth_df[(depth_df.subset == subset) & (depth_df.Group == grp)]['depth_gb'].dropna().values
            box_data.append(vals if len(vals) else [np.nan])
            positions.append(pos)
            colors.append(PALETTE[grp])
            pos += 1
        pos += 0.5
    bp = axes[1].boxplot(box_data, positions=positions, widths=0.55, patch_artist=True)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    axes[1].set_xticks([1.5, 4.0])
    axes[1].set_xticklabels(subsets, fontsize=8)
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(facecolor=PALETTE[g], label=g) for g in ['Healthy', 'PCOS']], fontsize=8, title='')
    axes[1].set_ylabel('Sequencing depth (Gb)', fontsize=9)
    axes[1].set_xlabel('')
    axes[1].set_title('Sequencing depth vs MAG recovery', fontsize=10)
    fig.tight_layout()
    save_figure(fig, 'Fig_L1e_recovery_depth', panel=True)
    recovery.to_csv(SOURCE / 'Fig_L1e_recovery_depth_summary.tsv', sep='\t', index=False)
    depth_df.to_csv(SOURCE / 'Fig_L1e_recovery_depth_points.tsv', sep='\t', index=False)

def plot_combined(mag_pos: pd.DataFrame):
    fig = plt.figure(figsize=(8.5, 7.0))
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.32)
    panels = [('n_mag', 'MAGs per sample', 'a'), ('depth_gb', 'Sequencing depth (Gb)', 'b'), ('median_mag_n50', 'Median MAG N50 (bp)', 'c'), ('n_ko', 'KOs per sample', 'd')]
    for i, (metric, ylab, label) in enumerate(panels):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        sub = mag_pos[['Group', metric]].dropna()
        stat = _boxplot_groups(ax, sub, metric, ylab)
        p = stat['p_value']
        p_txt = 'P < 0.001' if p < 0.001 else f'P = {p:.3f}'
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_xlabel('')
        ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=12, fontweight='regular')
        ax.text(0.5, 0.97, p_txt, transform=ax.transAxes, ha='center', va='top', fontsize=8)
    fig.suptitle('Layer 1 bias diagnostics: MAG recovery and functional detection', fontsize=11, y=1.01)
    save_figure(fig, 'Fig_L1_bias_diagnosis_combined')

def main():
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    ensure_dirs()
    print('Building sample-level metrics...')
    df = build_sample_metrics()
    df.to_csv(TABLES / 'TableSX_sample_technical_metrics.tsv', sep='\t', index=False)
    overall, per_cohort, no_mag_stats, recovery, mag_pos = comparison_tables(df)
    overall['metric_label'] = [lbl for _, lbl in METRICS]
    overall.to_csv(TABLES / 'TableSX_group_comparison_overall.tsv', sep='\t', index=False)
    per_cohort.to_csv(TABLES / 'TableSX_group_comparison_per_cohort.tsv', sep='\t', index=False)
    if len(no_mag_stats):
        no_mag_stats.to_csv(TABLES / 'TableSX_no_MAG_depth_comparison.tsv', sep='\t', index=False)
    recovery.to_csv(TABLES / 'TableSX_MAG_recovery_rate.tsv', sep='\t', index=False)
    no_mag_samples = df[~df['has_mag']][['sample', 'BIOPROJECT', 'Group', 'depth_gb', 'read_count', 'base_count']]
    no_mag_samples.to_csv(TABLES / 'TableSX_samples_without_MAG.tsv', sep='\t', index=False)
    corr_df = correlation_analysis(mag_pos)
    corr_df.to_csv(TABLES / 'TableSX_correlation_matrix.tsv', sep='\t', index=False)
    print('Overall comparisons (MAG-positive samples):')
    print(overall[['metric', 'median_PCOS', 'median_Healthy', 'p_value', 'rank_biserial']].to_string(index=False))
    print('\nGenerating figures...')
    plot_group_boxplots(mag_pos)
    plot_quality_boxplots(mag_pos)
    plot_cohort_facets(mag_pos)
    plot_scatter_correlations(mag_pos, corr_df)
    plot_mag_recovery(df, recovery)
    plot_combined(mag_pos)
    print(f'\nDone. Outputs in {OUT}')
if __name__ == '__main__':
    main()
