from pathlib import Path
import os
PKG_ROOT = Path(__file__).resolve().parents[1]
SHARED = PKG_ROOT / 'shared_input'
RSCRIPT = os.environ.get('RSCRIPT', 'Rscript')
INPUT_DIR = str(SHARED)
OUTPUT_DIR = str(PKG_ROOT / 'output' / 'Fig2A')
import argparse
import os
import sys
import re
import pickle
import warnings
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
matplotlib.rcParams.update({'font.family': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif', 'serif'], 'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'axes.linewidth': 0.8, 'lines.linewidth': 1.2, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.unicode_minus': False})
NPG_COLORS = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85']
COHORT_COLORS = {'PRJNA530971': '#3C5488', 'PRJNA549764': '#E64B35', 'PRJNA791492': '#00A087'}
COHORT_LABELS = {'PRJNA530971': 'PRJNA530971', 'PRJNA549764': 'PRJNA549764', 'PRJNA791492': 'PRJNA791492'}
GROUP_COLORS = {'PCOS': '#E64B35', 'Healthy': '#4DBBD5'}
PSEUDO = 0.001

def setup_logging(log_path: str, verbose: bool=False):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.FileHandler(log_path, 'w'), logging.StreamHandler(sys.stdout)])

def save_figure(fig, base_path: str, dpi: int=300):
    fig.savefig(f'{base_path}.pdf', format='pdf', bbox_inches='tight')
    fig.savefig(f'{base_path}.svg', format='svg', bbox_inches='tight')
    fig.savefig(f'{base_path}.jpg', format='jpg', dpi=dpi, bbox_inches='tight')
    logging.info(f'  [OK] Saved: {base_path} (.pdf / .svg / .jpg)')

def save_plotdata(df: pd.DataFrame, base_path: str):
    df.to_csv(f'{base_path}_plotdata.tsv', sep='\t', index=False)
    logging.info(f'  [OK] Saved: {base_path}_plotdata.tsv')
CODED_PATTERN = re.compile('GGB\\d+|SGB\\d+')
CODED_LEVEL = re.compile('^(F|OF|CF|C)GB\\d+$')

def _parse_tax_levels(tax_info: str) -> dict:
    levels = {}
    for part in tax_info.split('|'):
        if '__' in part:
            lvl, val = part.split('__', 1)
            levels[lvl.strip()] = val.strip()
    return levels

def build_fullname_lookup(fullname_tsv: str) -> dict:
    df = pd.read_csv(fullname_tsv, sep='\t')
    return dict(zip(df['TAX'], df['TAX_INFO']))

def get_display_label(species_name: str, fullname_lookup: dict):
    is_coded = bool(CODED_PATTERN.search(species_name))
    base = species_name.replace('_', ' ')
    if not is_coded:
        return (base, None)
    tax_info = fullname_lookup.get(species_name)
    if tax_info is None:
        return (base, None)
    levels = _parse_tax_levels(tax_info)
    annot = None
    for lvl in ['f', 'o', 'c', 'p']:
        val = levels.get(lvl, '')
        if val and (not CODED_LEVEL.match(val)):
            annot = f'{lvl}__{val}'
            break
    return (base, annot)

def format_title(species_name: str, fullname_lookup: dict, fdr_val: float, sig_str: str) -> str:
    title_name, annot = get_display_label(species_name, fullname_lookup)
    title_math = title_name.replace(' ', '\\ ')
    lines = [f'$\\it{{{title_math}}}$']
    if annot:
        lines.append(f'({annot})')
    lines.append(f'FDR = {fdr_val:.3f} {sig_str}')
    return '\n'.join(lines)
LOG_TICKS_RAW = [0, 0.01, 0.1, 1, 10, 100]
LOG_TICK_LABELS = ['0', '0.01', '0.1', '1', '10', '100']

def to_log(x, pseudo=PSEUDO):
    return np.log10(np.asarray(x, dtype=float) + pseudo)

def set_log_yaxis(ax, max_val: float):
    ticks = [t for t in LOG_TICKS_RAW if t <= max_val * 1.5]
    if not ticks:
        ticks = [0, max_val]
    ax.set_yticks([to_log(t) for t in ticks])
    ax.set_yticklabels(LOG_TICK_LABELS[:len(ticks)], fontsize=8)
    ax.set_ylim(to_log(0) - 0.15, to_log(max_val) + 0.3)
    ax.set_ylabel('Relative Abundance (%)', fontsize=9)

def draw_species_boxplot(ax, plot_df: pd.DataFrame, title_str: str, group_colors: dict, cohort_colors: dict, fdr_val: float):
    groups_order = ['PCOS', 'Healthy']
    positions = {g: i + 1 for i, g in enumerate(groups_order)}
    x_pos_map = positions
    max_val = plot_df['Abundance'].max()
    box_data = []
    for g in groups_order:
        vals = to_log(plot_df[plot_df['Group'] == g]['Abundance'].values)
        box_data.append(vals)
    bp = ax.boxplot(box_data, positions=[1, 2], widths=0.45, patch_artist=True, showfliers=False, medianprops=dict(color='#E69F00', linewidth=1.8), whiskerprops=dict(color='#444444', linewidth=0.8), capprops=dict(color='#444444', linewidth=0.8), boxprops=dict(linewidth=0.8), zorder=2)
    for i, g in enumerate(groups_order):
        bp['boxes'][i].set_facecolor(group_colors[g])
        bp['boxes'][i].set_alpha(0.35)
        bp['boxes'][i].set_edgecolor('#333333')
    np.random.seed(42)
    scatter_records = []
    for g in groups_order:
        sub = plot_df[plot_df['Group'] == g].copy()
        jitter = np.random.normal(0, 0.07, len(sub))
        xvals = x_pos_map[g] + jitter
        yvals = to_log(sub['Abundance'].values)
        cvals = [cohort_colors.get(b, '#888888') for b in sub['Cohort']]
        ax.scatter(xvals, yvals, c=cvals, s=16, alpha=0.85, edgecolors='white', linewidths=0.25, zorder=3)
        for xi, yi, ci, (_, row) in zip(xvals, yvals, sub['Cohort'], sub.iterrows()):
            scatter_records.append({'Group': g, 'Cohort': ci, 'Abundance_raw': row['Abundance'], 'x_plot': xi, 'y_log10': yi})
    set_log_yaxis(ax, max_val)
    sig_str = '***' if fdr_val < 0.001 else '**' if fdr_val < 0.01 else '*' if fdr_val < 0.05 else 'ns'
    y_top = to_log(max_val) + 0.15
    ax.annotate('', xy=(2, y_top), xytext=(1, y_top), arrowprops=dict(arrowstyle='-', color='#333333', lw=0.8))
    ax.text(1.5, y_top + 0.03, sig_str, ha='center', va='bottom', fontsize=9, color='#333333')
    ax.set_xticks([1, 2])
    ax.set_xticklabels(groups_order, fontsize=9)
    ax.set_xlim(0.4, 2.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(title_str, fontsize=8.5, fontweight='bold', pad=4, linespacing=1.4)
    return pd.DataFrame(scatter_records)

def build_cohort_legend_handles(cohort_colors: dict) -> list:
    return [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=5.5, label=lbl, markeredgecolor='white', markeredgewidth=0.3) for lbl, c in cohort_colors.items()]

def build_group_legend_handles(group_colors: dict) -> list:
    return [mpatches.Patch(facecolor=c, alpha=0.55, edgecolor='#333333', linewidth=0.6, label=grp) for grp, c in group_colors.items()]

def plot_top_species_composite(top_species_list: list, species_tss: pd.DataFrame, aligned_info: pd.DataFrame, results_df: pd.DataFrame, fullname_lookup: dict, output_dir: str, group_colors: dict, cohort_colors: dict, n_cols: int=4, top_n: int=8):
    n_sp = len(top_species_list)
    n_rows = (n_sp + n_cols - 1) // n_cols
    fig_w = 183 / 25.4
    fig_h = max(4.5, 3.2 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    axes_flat = np.array(axes).flatten()
    all_plot_records = []
    for idx, species in enumerate(top_species_list):
        ax = axes_flat[idx]
        plot_df = pd.DataFrame({'Abundance': species_tss.loc[species].values, 'Group': aligned_info['Group'].values, 'Cohort': aligned_info['Bioproject'].values})
        row = results_df[results_df['Species'] == species].iloc[0]
        fdr_val = row['LMM_FDR']
        sig_str = '***' if fdr_val < 0.001 else '**' if fdr_val < 0.01 else '*' if fdr_val < 0.05 else 'ns'
        title_str = format_title(species, fullname_lookup, fdr_val, sig_str)
        scatter_df = draw_species_boxplot(ax, plot_df, title_str, group_colors, cohort_colors, fdr_val)
        scatter_df['Species'] = species
        all_plot_records.append(scatter_df)
    for idx in range(n_sp, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    legend_ax = axes_flat[n_cols - 1] if n_sp < len(axes_flat) else fig.add_axes([0, 0, 0, 0])
    cohort_handles = build_cohort_legend_handles(cohort_colors)
    group_handles = build_group_legend_handles(group_colors)
    fig.legend(handles=cohort_handles + [mpatches.Patch(alpha=0, label=' ')] + group_handles, title='Cohort                   Group', title_fontsize=8, loc='lower right', bbox_to_anchor=(0.98, 0.01), framealpha=0.9, fontsize=7.5, ncol=2, columnspacing=1.0, handlelength=1.2)
    plt.tight_layout(rect=[0, 0.0, 1, 0.97])
    fig.suptitle('Top Differentially Abundant Species - Abundance Distribution\n(Linear Mixed Model, FDR < 0.05; y-axis: log-scale)', fontsize=10, fontweight='bold', y=0.995)
    base = os.path.join(output_dir, 'top_species_boxplots_composite')
    save_figure(fig, base)
    plt.close(fig)
    combo_df = pd.concat(all_plot_records, ignore_index=True)
    save_plotdata(combo_df, base)
    indiv_dir = os.path.join(output_dir, 'top_species_individual')
    os.makedirs(indiv_dir, exist_ok=True)
    for species, scatter_df in zip(top_species_list, all_plot_records):
        fig_i, ax_i = plt.subplots(figsize=(3.5, 4.0))
        plot_df = pd.DataFrame({'Abundance': species_tss.loc[species].values, 'Group': aligned_info['Group'].values, 'Cohort': aligned_info['Bioproject'].values})
        row = results_df[results_df['Species'] == species].iloc[0]
        fdr_val = row['LMM_FDR']
        sig_str = '***' if fdr_val < 0.001 else '**' if fdr_val < 0.01 else '*' if fdr_val < 0.05 else 'ns'
        title_str = format_title(species, fullname_lookup, fdr_val, sig_str)
        draw_species_boxplot(ax_i, plot_df, title_str, group_colors, cohort_colors, fdr_val)
        cohort_handles = build_cohort_legend_handles(cohort_colors)
        group_handles = build_group_legend_handles(group_colors)
        leg1 = ax_i.legend(handles=cohort_handles, title='Cohort', title_fontsize=7, fontsize=7, loc='upper right', framealpha=0.85, handlelength=1.0, borderpad=0.4)
        ax_i.add_artist(leg1)
        ax_i.legend(handles=group_handles, title='Group', title_fontsize=7, fontsize=7, loc='upper left', framealpha=0.85, handlelength=1.0, borderpad=0.4)
        plt.tight_layout()
        safe_name = re.sub('[^\\w]', '_', species)[:50]
        base_i = os.path.join(indiv_dir, f'species_{safe_name}')
        save_figure(fig_i, base_i)
        plt.close(fig_i)
        save_plotdata(scatter_df, base_i)
    logging.info(f'  [OK] Individual figures saved to: {indiv_dir}/')

def parse_args():
    p = argparse.ArgumentParser(description='Phase 4: Differential Species Analysis for PCOS Gut Microbiome', formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument('--work_dir', '-w', default=None, help='Project root directory')
    p.add_argument('--fullname_tsv', '-f', default=None, help='species_fullname.tsv path (default: auto-detected in work_dir)')
    p.add_argument('--top_n', type=int, default=4, help='Number of top species per group for boxplots (default: 4)')
    p.add_argument('--n_cols', type=int, default=4, help='Columns in composite boxplot figure (default: 4)')
    p.add_argument('--fdr', type=float, default=0.05, help='FDR threshold (default: 0.05)')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    p.add_argument('--verbose', '-v', action='store_true')
    return p.parse_args()

def main():
    args = parse_args()
    np.random.seed(args.seed)
    WORK_DIR = args.work_dir
    LOG_DIR = f'{WORK_DIR}/logs'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    setup_logging(f'{LOG_DIR}/04_differential_analysis_log.txt', args.verbose)
    fullname_tsv = args.fullname_tsv
    if fullname_tsv is None:
        for candidate in [f'{WORK_DIR}/species_fullname.tsv', f'{INPUT_DIR}/species_fullname.tsv', f'{WORK_DIR}/01_preprocessing/species_fullname.tsv']:
            if os.path.exists(candidate):
                fullname_tsv = candidate
                break
    if fullname_tsv is None or not os.path.exists(fullname_tsv):
        logging.warning('[WARN] species_fullname.tsv not found - GGB/SGB labels will not be annotated')
        fullname_lookup = {}
    else:
        fullname_lookup = build_fullname_lookup(fullname_tsv)
        logging.info(f'  Loaded species fullname lookup: {len(fullname_lookup)} entries')
    logging.info('=' * 70)
    logging.info('Differential Species Analysis Log')
    logging.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info('=' * 70)
    logging.info('[STEP 1] Loading preprocessed data')
    with open(f'{INPUT_DIR}/metaphlan_preprocessed.pkl', 'rb') as f:
        data = pickle.load(f)
    species_tss = data['species_tss']
    species_clr = data['species_clr']
    group_info = data['group_info']
    colors_group = data.get('colors_group', GROUP_COLORS)
    colors_batch = data.get('colors_batch', COHORT_COLORS)
    for k in COHORT_COLORS:
        if k not in colors_batch:
            colors_batch[k] = COHORT_COLORS[k]
    logging.info(f'  Species: {species_tss.shape[0]}')
    logging.info(f'  Samples: {species_tss.shape[1]}')
    species_tss_t = species_tss.T
    aligned_info = group_info.copy()
    aligned_info.index = aligned_info['Sample']
    aligned_info = aligned_info.loc[species_tss_t.index]
    pcos_samples = aligned_info[aligned_info['Group'] == 'PCOS'].index
    healthy_samples = aligned_info[aligned_info['Group'] == 'Healthy'].index
    logging.info('[STEP 2] Wilcoxon rank-sum test')
    wilcox_results = []
    for species in species_tss.index:
        pcos_vals = species_tss.loc[species, pcos_samples].values
        healthy_vals = species_tss.loc[species, healthy_samples].values
        stat, pval = mannwhitneyu(pcos_vals, healthy_vals, alternative='two-sided')
        pseudo = 0.001
        log2fc = np.log2((pcos_vals.mean() + pseudo) / (healthy_vals.mean() + pseudo))
        wilcox_results.append({'Species': species, 'PCOS_mean': pcos_vals.mean(), 'Healthy_mean': healthy_vals.mean(), 'PCOS_prevalence': np.sum(pcos_vals > 0) / len(pcos_vals), 'Healthy_prevalence': np.sum(healthy_vals > 0) / len(healthy_vals), 'Log2FC': log2fc, 'Wilcoxon_stat': stat, 'Wilcoxon_p': pval})
    wilcox_df = pd.DataFrame(wilcox_results)
    wilcox_df['Wilcoxon_FDR'] = multipletests(wilcox_df['Wilcoxon_p'], method='fdr_bh')[1]
    wilcox_df['Significant_Wilcoxon'] = wilcox_df['Wilcoxon_FDR'] < args.fdr
    logging.info(f"  Significant (FDR < {args.fdr}): {wilcox_df['Significant_Wilcoxon'].sum()}")
    logging.info('[STEP 3] Linear mixed-effects model (batch correction)')
    species_clr_t = species_clr.T
    lmm_results = []
    n_species = len(species_clr.index)
    for i, species in enumerate(species_clr.index):
        if (i + 1) % 50 == 0:
            logging.info(f'  Progress: {i + 1}/{n_species}')
        try:
            model_data = pd.DataFrame({'abundance': species_clr_t[species].values, 'Group': aligned_info['Group'].values, 'Batch': aligned_info['Bioproject'].values})
            model_data['Group_binary'] = (model_data['Group'] == 'PCOS').astype(int)
            model = smf.mixedlm('abundance ~ Group_binary', model_data, groups=model_data['Batch'])
            result = model.fit(method='powell', disp=False)
            lmm_results.append({'Species': species, 'LMM_coef': result.fe_params['Group_binary'], 'LMM_p': result.pvalues['Group_binary']})
        except Exception:
            lmm_results.append({'Species': species, 'LMM_coef': np.nan, 'LMM_p': np.nan})
    lmm_df = pd.DataFrame(lmm_results)
    valid = ~lmm_df['LMM_p'].isna()
    lmm_df.loc[valid, 'LMM_FDR'] = multipletests(lmm_df.loc[valid, 'LMM_p'], method='fdr_bh')[1]
    lmm_df['Significant_LMM'] = lmm_df['LMM_FDR'] < args.fdr
    logging.info(f"  Significant LMM (FDR < {args.fdr}): {lmm_df['Significant_LMM'].sum()}")
    logging.info('[STEP 4] Integrating results')
    results_df = wilcox_df.merge(lmm_df, on='Species')
    results_df['Consistent_Sig'] = results_df['Significant_Wilcoxon'] & results_df['Significant_LMM']
    results_df['Direction_Consistent'] = (results_df['Log2FC'] > 0) == (results_df['LMM_coef'] > 0)
    results_df['Direction'] = 'NS'
    results_df.loc[results_df['Significant_LMM'] & (results_df['LMM_coef'] > 0), 'Direction'] = 'PCOS_enriched'
    results_df.loc[results_df['Significant_LMM'] & (results_df['LMM_coef'] < 0), 'Direction'] = 'Healthy_enriched'
    pcos_enriched = results_df[results_df['Direction'] == 'PCOS_enriched'].sort_values('LMM_p')
    healthy_enriched = results_df[results_df['Direction'] == 'Healthy_enriched'].sort_values('LMM_p')
    logging.info(f'  PCOS-enriched: {len(pcos_enriched)}, Healthy-enriched: {len(healthy_enriched)}')
    logging.info('[STEP 5] Generating figures')
    fig, ax = plt.subplots(figsize=(10, 8))
    pt_colors = []
    for _, row in results_df.iterrows():
        if row['Direction'] == 'PCOS_enriched':
            pt_colors.append(colors_group['PCOS'])
        elif row['Direction'] == 'Healthy_enriched':
            pt_colors.append(colors_group['Healthy'])
        else:
            pt_colors.append('#CCCCCC')
    ax.scatter(results_df['Log2FC'], -np.log10(results_df['LMM_FDR'].fillna(1)), c=pt_colors, s=50, alpha=0.6, edgecolors='white', linewidths=0.3)
    ax.axhline(y=-np.log10(0.05), color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.axvline(x=1, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.axvline(x=-1, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    top_species_annot = pd.concat([pcos_enriched.head(8), healthy_enriched.head(8)])
    for _, row in top_species_annot.iterrows():
        label, _ = get_display_label(row['Species'], fullname_lookup)
        label = label[:25] + '...' if len(label) > 25 else label
        ax.annotate(label, xy=(row['Log2FC'], -np.log10(row['LMM_FDR'])), xytext=(5, 5), textcoords='offset points', fontsize=7, alpha=0.8, arrowprops=dict(arrowstyle='-', color='gray', alpha=0.3))
    ax.set_xlabel('Log$_2$ Fold Change (PCOS / Healthy)', fontsize=12)
    ax.set_ylabel('$-$Log$_{10}$(FDR)', fontsize=12)
    ax.set_title('Volcano Plot: Differential Species\n(Linear Mixed Model, batch-corrected)', fontsize=13, fontweight='bold')
    ax.legend(handles=[mpatches.Patch(color=colors_group['PCOS'], label=f'PCOS enriched (n={len(pcos_enriched)})', alpha=0.7), mpatches.Patch(color=colors_group['Healthy'], label=f'Healthy enriched (n={len(healthy_enriched)})', alpha=0.7), mpatches.Patch(color='#CCCCCC', label='Not significant', alpha=0.7)], loc='upper right', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save_figure(fig, f'{OUTPUT_DIR}/volcano_plot')
    plt.close()
    volcano_df = results_df[['Species', 'Log2FC', 'LMM_FDR', 'Direction']].copy()
    volcano_df['neg_log10_FDR'] = -np.log10(volcano_df['LMM_FDR'].fillna(1))
    save_plotdata(volcano_df, f'{OUTPUT_DIR}/volcano_plot')
    logging.info('  [OK] volcano_plot')
    sig_species_list = results_df[results_df['Significant_LMM']].sort_values('LMM_coef', ascending=False)
    if len(sig_species_list) > 0:
        n_top = min(40, len(sig_species_list))
        top_pcos = sig_species_list[sig_species_list['LMM_coef'] > 0].head(n_top // 2)
        top_hlthy = sig_species_list[sig_species_list['LMM_coef'] < 0].head(n_top // 2)
        top_sig = pd.concat([top_pcos, top_hlthy])
        heatmap_species = top_sig['Species'].tolist()
        heatmap_data = species_tss.loc[heatmap_species]
        heatmap_zscore = heatmap_data.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
        sample_order = list(pcos_samples) + list(healthy_samples)
        heatmap_zscore = heatmap_zscore[sample_order]
        fig, ax = plt.subplots(figsize=(14, max(8, len(heatmap_species) * 0.25)))
        col_colors = [colors_group['PCOS'] if s in pcos_samples else colors_group['Healthy'] for s in sample_order]
        sns.heatmap(heatmap_zscore, cmap='RdBu_r', center=0, xticklabels=False, yticklabels=True, cbar_kws={'label': 'Z-score', 'shrink': 0.5}, ax=ax, vmin=-3, vmax=3)
        for i, color in enumerate(col_colors):
            ax.axvline(x=i, color=color, linewidth=0.1, alpha=0.3)
        ylabels = []
        for sp in heatmap_species:
            name, annot = get_display_label(sp, fullname_lookup)
            lbl = name[:28] if not annot else f'{name[:20]} ({annot[:18]})'
            ylabels.append(lbl)
        ax.set_yticklabels(ylabels, fontsize=8)
        ax.axvline(x=len(pcos_samples), color='black', linewidth=2)
        ax.set_xlabel('Samples (PCOS -> Healthy)', fontsize=11)
        ax.set_title('Heatmap of Differentially Abundant Species (Z-score)', fontsize=13, fontweight='bold')
        ax.legend(handles=[mpatches.Patch(color=colors_group['PCOS'], label='PCOS'), mpatches.Patch(color=colors_group['Healthy'], label='Healthy')], loc='upper left', bbox_to_anchor=(1.15, 1))
        plt.tight_layout()
        save_figure(fig, f'{OUTPUT_DIR}/differential_heatmap')
        plt.close()
        logging.info('  [OK] differential_heatmap')
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    for ax_idx, (enriched_df, grp_key, ax_obj) in enumerate([(pcos_enriched, 'PCOS', axes[0]), (healthy_enriched, 'Healthy', axes[1])]):
        ax_obj.spines['top'].set_visible(False)
        ax_obj.spines['right'].set_visible(False)
        if len(enriched_df) == 0:
            ax_obj.text(0.5, 0.5, f'No {grp_key}-enriched species', ha='center', va='center', transform=ax_obj.transAxes)
            continue
        top_n_bar = min(15, len(enriched_df))
        plot_data = enriched_df.head(top_n_bar).sort_values('Log2FC', ascending=grp_key == 'Healthy')
        y_pos = np.arange(len(plot_data))
        fc_vals = plot_data['Log2FC'] if grp_key == 'PCOS' else -plot_data['Log2FC']
        ax_obj.barh(y_pos, fc_vals, color=colors_group[grp_key], alpha=0.7, edgecolor='white')
        for i, (_, row) in enumerate(plot_data.iterrows()):
            sig = '***' if row['LMM_FDR'] < 0.001 else '**' if row['LMM_FDR'] < 0.01 else '*'
            ax_obj.text(fc_vals.iloc[i] + 0.1, i, sig, va='center', fontsize=9)
        bar_labels = []
        for sp in plot_data['Species']:
            nm, ann = get_display_label(sp, fullname_lookup)
            bar_labels.append(nm[:30] if not ann else f'{nm[:22]} ({ann[:16]})')
        ax_obj.set_yticks(y_pos)
        ax_obj.set_yticklabels(bar_labels, fontsize=9)
        xlabel = 'Log$_2$ Fold Change' if grp_key == 'PCOS' else 'Log$_2$ Fold Change (absolute)'
        ax_obj.set_xlabel(xlabel, fontsize=11)
        ax_obj.set_title(f'{grp_key}-enriched Species (n={len(enriched_df)})', fontsize=12, fontweight='bold')
        ax_obj.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
    plt.suptitle('Top Differentially Abundant Species (FDR < 0.05, LMM)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, f'{OUTPUT_DIR}/top_differential_species')
    plt.close()
    logging.info('  [OK] top_differential_species')
    all_sig = pd.concat([pcos_enriched.head(15).assign(enriched_in='PCOS'), healthy_enriched.head(15).assign(enriched_in='Healthy')])
    if len(all_sig) > 0:
        all_sig = all_sig.sort_values('Log2FC', ascending=True)
        fig, ax = plt.subplots(figsize=(12, 10))
        bar_cols = [colors_group['PCOS'] if x == 'PCOS' else colors_group['Healthy'] for x in all_sig['enriched_in']]
        ax.barh(np.arange(len(all_sig)), all_sig['Log2FC'], color=bar_cols, alpha=0.8, edgecolor='white')
        lefse_labels = []
        for sp in all_sig['Species']:
            nm, ann = get_display_label(sp, fullname_lookup)
            lefse_labels.append(nm[:35] if not ann else f'{nm[:26]} ({ann[:18]})')
        ax.set_yticks(np.arange(len(all_sig)))
        ax.set_yticklabels(lefse_labels, fontsize=9)
        ax.set_xlabel('Log$_2$ Fold Change (PCOS / Healthy)', fontsize=12)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.legend(handles=[mpatches.Patch(color=colors_group['PCOS'], label='PCOS enriched', alpha=0.8), mpatches.Patch(color=colors_group['Healthy'], label='Healthy enriched', alpha=0.8)], loc='lower right', framealpha=0.9)
        ax.set_title('LEfSe-style Plot: Differentially Abundant Species\n(LMM, FDR < 0.05)', fontsize=13, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        save_figure(fig, f'{OUTPUT_DIR}/lefse_style_plot')
        plt.close()
        logging.info('  [OK] lefse_style_plot')
    logging.info('[STEP 5.5] Top-species boxplots (log-scale, cohort-colored)')
    top_n = args.top_n
    top_list = pd.concat([pcos_enriched.head(top_n), healthy_enriched.head(top_n)])['Species'].tolist()
    if len(top_list) == 0:
        logging.warning('  [WARN] No significant species found; skipping boxplots')
    else:
        plot_top_species_composite(top_species_list=top_list, species_tss=species_tss, aligned_info=aligned_info, results_df=results_df, fullname_lookup=fullname_lookup, output_dir=OUTPUT_DIR, group_colors=colors_group, cohort_colors=colors_batch, n_cols=args.n_cols, top_n=top_n)
        logging.info('  [OK] top_species_boxplots_composite + individual')
    logging.info('[STEP 6] Saving results')
    results_df.to_csv(f'{OUTPUT_DIR}/differential_analysis_results.csv', index=False)
    sig_results = results_df[results_df['Significant_LMM']].sort_values('LMM_p')
    sig_results.to_csv(f'{OUTPUT_DIR}/significant_species.csv', index=False)
    with open(f'{OUTPUT_DIR}/differential_data.pkl', 'wb') as f:
        pickle.dump({'results_df': results_df, 'pcos_enriched': pcos_enriched, 'healthy_enriched': healthy_enriched}, f)
    logging.info('  Saved: differential_analysis_results.csv')
    logging.info('  Saved: significant_species.csv')
    logging.info('  Saved: differential_data.pkl')
    logging.info('=' * 70)
    logging.info('Analysis Summary')
    logging.info(f'  Total species tested: {len(results_df)}')
    logging.info(f"  Wilcoxon significant: {results_df['Significant_Wilcoxon'].sum()}")
    logging.info(f"  LMM significant:      {results_df['Significant_LMM'].sum()}")
    logging.info(f'  PCOS-enriched:        {len(pcos_enriched)}')
    logging.info(f'  Healthy-enriched:     {len(healthy_enriched)}')
    logging.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info('=' * 70)
if __name__ == '__main__':
    main()
