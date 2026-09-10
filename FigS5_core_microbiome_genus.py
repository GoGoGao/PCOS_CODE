from pathlib import Path
import os
PKG_ROOT = Path(__file__).resolve().parents[1]
SHARED = PKG_ROOT / 'shared_input'
RSCRIPT = os.environ.get('RSCRIPT', 'Rscript')
INPUT_DIR = str(SHARED)
OUTPUT_DIR = str(PKG_ROOT / 'output' / 'FigS5')
import argparse
import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
mpl.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'axes.linewidth': 0.8, 'lines.linewidth': 1.2, 'pdf.fonttype': 42, 'ps.fonttype': 42})
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib_venn import venn2
warnings.filterwarnings('ignore')
NPG = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85']
LANCET = ['#00468B', '#ED0000', '#42B540', '#0099B4', '#925E9F', '#FDAF91', '#AD002A', '#ADB6B6', '#1B1919']
JCO = ['#0073C2', '#EFC000', '#868686', '#CD534C', '#7AA6DC', '#003C67', '#8F7700', '#3B3B3B', '#A73030', '#4A6990']
NEJM = ['#BC3C29', '#0072B5', '#E18727', '#20854E', '#7876B1', '#6F99AD', '#FFDC91', '#EE4C97']
PALETTES = {'npg': NPG, 'lancet': LANCET, 'jco': JCO, 'nejm': NEJM}

def parse_args():
    p = argparse.ArgumentParser(description='Module 5 (Genus): Core Microbiome Analysis at Genus Level', formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument('--genus', '-g', required=True, help='Genus relative abundance table (TSV/XLS, genus x samples)')
    p.add_argument('--group', '-m', required=True, help='Sample metadata TSV: columns Sample, Bioproject, Group')
    p.add_argument('--output', '-o', required=True, help='Output directory')
    p.add_argument('--prevalence', type=float, default=0.7, help='Primary prevalence threshold for core genus (default: 0.70)')
    p.add_argument('--min_prev', type=float, default=0.1, help='Pre-filter: remove genera present in < this fraction of samples (default: 0.10)')
    p.add_argument('--min_abund', type=float, default=0.01, help='Pre-filter: remove genera with mean abundance < this value (%%) (default: 0.01, mirrors species Module 1)')
    p.add_argument('--palette', default='npg', choices=['npg', 'lancet', 'jco', 'nejm'], help='Color palette (default: npg)')
    p.add_argument('--dpi', type=int, default=300, help='Figure DPI (default: 300)')
    p.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    p.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    return p.parse_args()

def save_figure(fig, plot_df, base_path, dpi=300):
    fig.savefig(f'{base_path}.pdf', format='pdf', bbox_inches='tight')
    fig.savefig(f'{base_path}.svg', format='svg', bbox_inches='tight')
    fig.savefig(f'{base_path}.jpg', format='jpg', dpi=dpi, bbox_inches='tight')
    if plot_df is not None:
        plot_df.to_csv(f'{base_path}_plotdata.tsv', sep='\t', index=False)
    logging.info('  Saved: %s (.pdf/.svg/.jpg%s)', base_path, '/_plotdata.tsv' if plot_df is not None else '')

def identify_core(abundance_df, samples, prevalence_threshold=0.7):
    subset = abundance_df[samples]
    prevalence = (subset > 0).sum(axis=1) / len(samples)
    mean_abund = subset.mean(axis=1)
    core_genera = prevalence[prevalence >= prevalence_threshold].index.tolist()
    return (core_genera, prevalence, mean_abund)

def main():
    args = parse_args()
    np.random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(args.output, 'M5_genus_core_log.txt'), mode='w')])
    pal = PALETTES[args.palette]
    colors_group = {'PCOS': pal[0], 'Healthy': pal[1]}
    colors_batch = {}
    logging.info('[1/7] Loading data')
    genus_df = pd.read_csv(args.genus, sep='\t', index_col=0)
    genus_df = genus_df.clip(lower=0)
    logging.info('  Raw genus table: %d genera x %d samples', *genus_df.shape)
    group_info = pd.read_csv(args.group, sep='\t')
    group_info.columns = ['Sample', 'Bioproject', 'Group']
    logging.info('  Group file: %d samples', len(group_info))
    common = sorted(set(genus_df.columns) & set(group_info['Sample']))
    genus_df = genus_df[common]
    group_info = group_info[group_info['Sample'].isin(common)].set_index('Sample').loc[common].reset_index()
    logging.info('  Aligned samples: %d', len(common))
    batches = sorted(group_info['Bioproject'].unique())
    batch_colors_list = pal[2:2 + len(batches)]
    colors_batch = dict(zip(batches, batch_colors_list))
    aligned_info = group_info.set_index('Sample').loc[common]
    pcos_samples = aligned_info[aligned_info['Group'] == 'PCOS'].index.tolist()
    healthy_samples = aligned_info[aligned_info['Group'] == 'Healthy'].index.tolist()
    logging.info('  PCOS: %d | Healthy: %d', len(pcos_samples), len(healthy_samples))
    logging.info('[2/7] Pre-filtering genera (prevalence >= %.0f%%, mean abund >= %.4f%%)', args.min_prev * 100, args.min_abund)
    global_prev = (genus_df > 0).sum(axis=1) / genus_df.shape[1]
    global_abund = genus_df.mean(axis=1)
    keep = (global_prev >= args.min_prev) & (global_abund >= args.min_abund)
    genus_filt = genus_df.loc[keep].copy()
    logging.info('  Retained: %d / %d genera (prev-only would keep %d)', keep.sum(), len(keep), (global_prev >= args.min_prev).sum())
    logging.info('[3/7] Identifying core microbiome')
    thresholds = [0.5, 0.7, 0.9]
    core_results = {}
    for thresh in thresholds:
        core_all, _, _ = identify_core(genus_filt, common, thresh)
        core_pcos, _, _ = identify_core(genus_filt, pcos_samples, thresh)
        core_healthy, _, _ = identify_core(genus_filt, healthy_samples, thresh)
        core_results[f'all_{int(thresh * 100)}'] = core_all
        core_results[f'pcos_{int(thresh * 100)}'] = core_pcos
        core_results[f'healthy_{int(thresh * 100)}'] = core_healthy
        logging.info('  Threshold %.0f%%: all=%d | PCOS=%d | Healthy=%d', thresh * 100, len(core_all), len(core_pcos), len(core_healthy))
    prim = int(args.prevalence * 100)
    core_pcos_key = f'pcos_{prim}'
    core_healthy_key = f'healthy_{prim}'
    core_pcos_set = set(core_results[core_pcos_key])
    core_healthy_set = set(core_results[core_healthy_key])
    shared_core = core_pcos_set & core_healthy_set
    pcos_specific = core_pcos_set - core_healthy_set
    healthy_specific = core_healthy_set - core_pcos_set
    logging.info('  Primary threshold %d%%: shared=%d | PCOS-specific=%d | Healthy-specific=%d', prim, len(shared_core), len(pcos_specific), len(healthy_specific))
    logging.info('[4/7] Computing genus statistics')
    _, prev_pcos, abund_pcos = identify_core(genus_filt, pcos_samples, 0)
    _, prev_healthy, abund_healthy = identify_core(genus_filt, healthy_samples, 0)
    genus_stats = pd.DataFrame({'Genus': genus_filt.index, 'PCOS_prevalence': prev_pcos.values, 'Healthy_prevalence': prev_healthy.values, 'PCOS_abundance': abund_pcos.values, 'Healthy_abundance': abund_healthy.values})
    genus_stats['Prevalence_diff'] = genus_stats['PCOS_prevalence'] - genus_stats['Healthy_prevalence']
    genus_stats['Core_status'] = 'Non-core'
    genus_stats.loc[genus_stats['Genus'].isin(shared_core), 'Core_status'] = 'Shared'
    genus_stats.loc[genus_stats['Genus'].isin(pcos_specific), 'Core_status'] = 'PCOS-specific'
    genus_stats.loc[genus_stats['Genus'].isin(healthy_specific), 'Core_status'] = 'Healthy-specific'
    logging.info('[5/7] Cross-batch consistency')
    batch_core = {}
    for batch in batches:
        b_info = aligned_info[aligned_info['Bioproject'] == batch]
        b_pcos = [s for s in b_info.index if s in pcos_samples]
        b_healthy = [s for s in b_info.index if s in healthy_samples]
        for grp_name, grp_samples in [('PCOS', b_pcos), ('Healthy', b_healthy)]:
            if len(grp_samples) >= 5:
                core_b, _, _ = identify_core(genus_filt, grp_samples, args.prevalence)
                batch_core[f'{batch}_{grp_name}'] = set(core_b)
                logging.info('  %s_%s: %d core genera', batch, grp_name, len(core_b))
    pcos_batch_keys = [k for k in batch_core if k.endswith('_PCOS')]
    healthy_batch_keys = [k for k in batch_core if k.endswith('_Healthy')]
    if pcos_batch_keys:
        consistent_pcos = set.intersection(*[batch_core[k] for k in pcos_batch_keys])
        logging.info('  Cross-batch consistent PCOS core: %d', len(consistent_pcos))
    if healthy_batch_keys:
        consistent_healthy = set.intersection(*[batch_core[k] for k in healthy_batch_keys])
        logging.info('  Cross-batch consistent Healthy core: %d', len(consistent_healthy))
    logging.info('[6/7] Generating figures')
    SINGLE_W_MM = 89 / 25.4
    DOUBLE_W_MM = 183 / 25.4
    fig_ab, axes = plt.subplots(1, 2, figsize=(DOUBLE_W_MM, DOUBLE_W_MM * 0.45))
    ax = axes[0]
    venn2([core_pcos_set, core_healthy_set], set_labels=('PCOS Core', 'Healthy Core'), set_colors=(colors_group['PCOS'], colors_group['Healthy']), alpha=0.65, ax=ax)
    ax.set_title(f'(A) Core Genus Overlap\n(Prevalence ≥ {prim}%)', fontsize=10, fontweight='bold')
    ax = axes[1]
    thresh_labels = ['50%', '70%', '90%']
    n_pcos = [len(core_results[f'pcos_{t}']) for t in [50, 70, 90]]
    n_healthy = [len(core_results[f'healthy_{t}']) for t in [50, 70, 90]]
    x = np.arange(3)
    w = 0.35
    b1 = ax.bar(x - w / 2, n_pcos, w, label='PCOS', color=colors_group['PCOS'], alpha=0.8)
    b2 = ax.bar(x + w / 2, n_healthy, w, label='Healthy', color=colors_group['Healthy'], alpha=0.8)
    ax.bar_label(b1, padding=2, fontsize=7)
    ax.bar_label(b2, padding=2, fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(thresh_labels)
    ax.set_xlabel('Prevalence Threshold', fontsize=9)
    ax.set_ylabel('Number of Core Genera', fontsize=9)
    ax.set_title('(B) Core Size by Threshold', fontsize=10, fontweight='bold')
    ax.legend(loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    thresh_plot_df = pd.DataFrame({'Threshold': thresh_labels * 2, 'Group': ['PCOS'] * 3 + ['Healthy'] * 3, 'N_core_genera': n_pcos + n_healthy})
    venn_plot_df = pd.DataFrame({'Category': ['PCOS_only', 'Shared', 'Healthy_only'], 'N': [len(pcos_specific), len(shared_core), len(healthy_specific)]})
    combined_ab = pd.concat([thresh_plot_df, venn_plot_df.rename(columns={'Category': 'Threshold', 'N': 'N_core_genera'}).assign(Group='Venn')], ignore_index=True)
    save_figure(fig_ab, combined_ab, os.path.join(args.output, 'M5G_Fig1_AB_venn_threshold'), dpi=args.dpi)
    plt.close(fig_ab)
    fig_a, ax = plt.subplots(figsize=(SINGLE_W_MM, SINGLE_W_MM))
    venn2([core_pcos_set, core_healthy_set], set_labels=('PCOS Core', 'Healthy Core'), set_colors=(colors_group['PCOS'], colors_group['Healthy']), alpha=0.65, ax=ax)
    ax.set_title(f'Core Genus Overlap (Prevalence ≥ {prim}%)', fontsize=10, fontweight='bold')
    save_figure(fig_a, venn_plot_df, os.path.join(args.output, 'M5G_Fig1_panelA_venn'), dpi=args.dpi)
    plt.close(fig_a)
    fig_b, ax = plt.subplots(figsize=(SINGLE_W_MM, SINGLE_W_MM))
    b1 = ax.bar(x - w / 2, n_pcos, w, label='PCOS', color=colors_group['PCOS'], alpha=0.8)
    b2 = ax.bar(x + w / 2, n_healthy, w, label='Healthy', color=colors_group['Healthy'], alpha=0.8)
    ax.bar_label(b1, padding=2, fontsize=7)
    ax.bar_label(b2, padding=2, fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(thresh_labels)
    ax.set_xlabel('Prevalence Threshold', fontsize=9)
    ax.set_ylabel('Number of Core Genera', fontsize=9)
    ax.set_title('Core Genera by Threshold', fontsize=10, fontweight='bold')
    ax.legend(loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    save_figure(fig_b, thresh_plot_df, os.path.join(args.output, 'M5G_Fig1_panelB_threshold'), dpi=args.dpi)
    plt.close(fig_b)
    fig_c, axes = plt.subplots(1, 2, figsize=(DOUBLE_W_MM, DOUBLE_W_MM * 0.45))
    for i, (grp, grp_core, grp_col) in enumerate([('PCOS', core_pcos_set, 'PCOS_prevalence'), ('Healthy', core_healthy_set, 'Healthy_prevalence')]):
        ax = axes[i]
        col_abund = 'PCOS_abundance' if grp == 'PCOS' else 'Healthy_abundance'
        in_core = genus_stats['Genus'].isin(grp_core)
        c_arr = [colors_group[grp] if v else '#CCCCCC' for v in in_core]
        s_arr = [55 if v else 18 for v in in_core]
        ax.scatter(genus_stats[grp_col] * 100, np.log10(genus_stats[col_abund] + 0.001), c=c_arr, s=s_arr, alpha=0.65, edgecolors='white', linewidths=0.2)
        ax.axvline(x=args.prevalence * 100, color='#333333', linestyle='--', linewidth=0.9, alpha=0.8, label=f'{prim}% threshold')
        ax.set_xlabel('Prevalence (%)', fontsize=9)
        ax.set_ylabel('log₁₀(Mean Abundance + 0.001)', fontsize=9)
        ax.set_title(f'(C{i + 1}) {grp} Group (core genera: {len(grp_core)})', fontsize=10, fontweight='bold')
        ax.legend(fontsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.tight_layout()
    prev_abund_df = genus_stats[['Genus', 'PCOS_prevalence', 'Healthy_prevalence', 'PCOS_abundance', 'Healthy_abundance', 'Core_status']].copy()
    save_figure(fig_c, prev_abund_df, os.path.join(args.output, 'M5G_Fig2_C_prevalence_abundance'), dpi=args.dpi)
    plt.close(fig_c)
    for i, (grp, grp_core, prev_col, abund_col) in enumerate([('PCOS', core_pcos_set, 'PCOS_prevalence', 'PCOS_abundance'), ('Healthy', core_healthy_set, 'Healthy_prevalence', 'Healthy_abundance')]):
        fig_ci, ax = plt.subplots(figsize=(SINGLE_W_MM, SINGLE_W_MM))
        in_core = genus_stats['Genus'].isin(grp_core)
        c_arr = [colors_group[grp] if v else '#CCCCCC' for v in in_core]
        s_arr = [55 if v else 18 for v in in_core]
        ax.scatter(genus_stats[prev_col] * 100, np.log10(genus_stats[abund_col] + 0.001), c=c_arr, s=s_arr, alpha=0.65, edgecolors='white', linewidths=0.2)
        ax.axvline(x=args.prevalence * 100, color='#333333', linestyle='--', linewidth=0.9, label=f'{prim}% threshold')
        ax.set_xlabel('Prevalence (%)', fontsize=9)
        ax.set_ylabel('log₁₀(Mean Abundance + 0.001)', fontsize=9)
        ax.set_title(f'{grp} Group (core genera: {len(grp_core)})', fontsize=10, fontweight='bold')
        ax.legend(fontsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        sub_df = genus_stats[['Genus', prev_col, abund_col, 'Core_status']].copy()
        save_figure(fig_ci, sub_df, os.path.join(args.output, f'M5G_Fig2_panelC{i + 1}_{grp.lower()}_scatter'), dpi=args.dpi)
        plt.close(fig_ci)
    shared_list = sorted(shared_core)
    shared_stats = genus_stats[genus_stats['Genus'].isin(shared_list)].copy()
    shared_stats['Total_abund'] = shared_stats['PCOS_abundance'] + shared_stats['Healthy_abundance']
    top_n = min(30, len(shared_stats))
    top_shared = shared_stats.nlargest(top_n, 'Total_abund')['Genus'].tolist()
    if top_shared:
        sample_order = pcos_samples + healthy_samples
        hm_data = genus_filt.loc[top_shared, sample_order]
        hm_zscore = hm_data.apply(lambda row: (row - row.mean()) / row.std() if row.std() > 0 else row * 0, axis=1)
        fig_d, ax = plt.subplots(figsize=(DOUBLE_W_MM, DOUBLE_W_MM * 0.72))
        sns.heatmap(hm_zscore, cmap='RdBu_r', center=0, xticklabels=False, yticklabels=True, cbar_kws={'label': 'Z-score', 'shrink': 0.5}, ax=ax, vmin=-2.5, vmax=2.5, linewidths=0)
        ylabels = [g.replace('_', ' ')[:30] for g in top_shared]
        ax.set_yticklabels(ylabels, fontsize=7)
        ax.axvline(x=len(pcos_samples), color='black', linewidth=1.5)
        ax.set_xlabel(f'Samples  (PCOS: n={len(pcos_samples)} | Healthy: n={len(healthy_samples)})', fontsize=9)
        ax.set_title(f'(D) Top {top_n} Shared Core Genera (Prevalence ≥ {prim}% in both groups)', fontsize=10, fontweight='bold')
        legend_elements = [mpatches.Patch(color=colors_group['PCOS'], label=f'PCOS (n={len(pcos_samples)})'), mpatches.Patch(color=colors_group['Healthy'], label=f'Healthy (n={len(healthy_samples)})')]
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.18, 0.5), fontsize=8, framealpha=0.9)
        plt.tight_layout()
        hm_plot_df = hm_zscore.copy()
        hm_plot_df.index.name = 'Genus'
        hm_plot_df = hm_plot_df.reset_index()
        hm_plot_df = pd.melt(hm_plot_df, id_vars='Genus', var_name='Sample', value_name='Zscore')
        hm_plot_df = hm_plot_df.merge(group_info[['Sample', 'Group', 'Bioproject']], on='Sample', how='left')
        save_figure(fig_d, hm_plot_df, os.path.join(args.output, 'M5G_Fig3_D_shared_core_heatmap'), dpi=args.dpi)
        plt.close(fig_d)
    gs_sorted = genus_stats.sort_values('Prevalence_diff', ascending=True)
    top_n_diff = 15
    top_diff = pd.concat([gs_sorted.head(top_n_diff), gs_sorted.tail(top_n_diff)])
    fig_e, ax = plt.subplots(figsize=(SINGLE_W_MM * 1.1, DOUBLE_W_MM * 0.55))
    bar_colors = [colors_group['PCOS'] if d > 0 else colors_group['Healthy'] for d in top_diff['Prevalence_diff']]
    y_pos = np.arange(len(top_diff))
    ax.barh(y_pos, top_diff['Prevalence_diff'] * 100, color=bar_colors, alpha=0.78, edgecolor='white', height=0.75)
    ylabels = [g.replace('_', ' ')[:28] for g in top_diff['Genus']]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('Prevalence Difference (PCOS − Healthy, %)', fontsize=9)
    ax.set_title(f'(E) Top Genera with Largest Prevalence Differences\n(PCOS vs Healthy)', fontsize=10, fontweight='bold')
    legend_elements = [mpatches.Patch(color=colors_group['PCOS'], label='Higher in PCOS', alpha=0.78), mpatches.Patch(color=colors_group['Healthy'], label='Higher in Healthy', alpha=0.78)]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save_figure(fig_e, top_diff[['Genus', 'PCOS_prevalence', 'Healthy_prevalence', 'Prevalence_diff', 'Core_status']].copy(), os.path.join(args.output, 'M5G_Fig4_E_prevalence_diff'), dpi=args.dpi)
    plt.close(fig_e)
    if batch_core:
        all_batch_genera = sorted(set.union(*batch_core.values()))
        presence_mat = pd.DataFrame(0, index=all_batch_genera, columns=sorted(batch_core.keys()))
        for k, gset in batch_core.items():
            for g in gset:
                presence_mat.loc[g, k] = 1
        presence_mat['N_batches'] = presence_mat.sum(axis=1)
        min_batches = max(2, len(batch_core) - 1)
        consistent = presence_mat[presence_mat['N_batches'] >= min_batches].copy()
        consistent = consistent.sort_values('N_batches', ascending=False).drop('N_batches', axis=1)
        if len(consistent) > 40:
            consistent = consistent.head(40)
        if len(consistent) > 0:
            fig_f, ax = plt.subplots(figsize=(DOUBLE_W_MM * 0.6, max(DOUBLE_W_MM * 0.5, len(consistent) * 0.22)))
            col_labels = [k.replace('PRJNA', 'P').replace('_', '\n') for k in consistent.columns]
            sns.heatmap(consistent, cmap='YlOrRd', cbar_kws={'label': 'Core (1=yes)', 'shrink': 0.5}, xticklabels=col_labels, yticklabels=True, linewidths=0.4, linecolor='white', ax=ax, vmin=0, vmax=1)
            ylabels = [g.replace('_', ' ')[:28] for g in consistent.index]
            ax.set_yticklabels(ylabels, fontsize=6)
            ax.set_xticklabels(col_labels, fontsize=7, rotation=45, ha='right')
            ax.set_title(f'(F) Cross-batch Core Genus Consistency\n(Present in ≥{min_batches} batch-group combinations)', fontsize=10, fontweight='bold')
            plt.tight_layout()
            consist_plot = consistent.copy()
            consist_plot.index.name = 'Genus'
            consist_plot = consist_plot.reset_index()
            save_figure(fig_f, consist_plot, os.path.join(args.output, 'M5G_Fig5_F_crossbatch_consistency'), dpi=args.dpi)
            plt.close(fig_f)
    logging.info('  Building composite figure')
    fig_comp = plt.figure(figsize=(DOUBLE_W_MM, DOUBLE_W_MM * 1.55))
    gs_outer = gridspec.GridSpec(3, 2, figure=fig_comp, hspace=0.42, wspace=0.38)
    ax0 = fig_comp.add_subplot(gs_outer[0, 0])
    venn2([core_pcos_set, core_healthy_set], set_labels=('PCOS Core', 'Healthy Core'), set_colors=(colors_group['PCOS'], colors_group['Healthy']), alpha=0.65, ax=ax0)
    ax0.set_title(f'(A) Core Genus Overlap (≥{prim}%)', fontsize=9, fontweight='bold')
    ax1 = fig_comp.add_subplot(gs_outer[0, 1])
    b1 = ax1.bar(x - w / 2, n_pcos, w, label='PCOS', color=colors_group['PCOS'], alpha=0.8)
    b2 = ax1.bar(x + w / 2, n_healthy, w, label='Healthy', color=colors_group['Healthy'], alpha=0.8)
    ax1.bar_label(b1, padding=1, fontsize=6)
    ax1.bar_label(b2, padding=1, fontsize=6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(thresh_labels, fontsize=7)
    ax1.set_ylabel('N Core Genera', fontsize=8)
    ax1.set_title('(B) Core Size by Threshold', fontsize=9, fontweight='bold')
    ax1.legend(fontsize=7)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2 = fig_comp.add_subplot(gs_outer[1, 0])
    in_core = genus_stats['Genus'].isin(core_pcos_set)
    c_arr = [colors_group['PCOS'] if v else '#CCCCCC' for v in in_core]
    s_arr = [45 if v else 12 for v in in_core]
    ax2.scatter(genus_stats['PCOS_prevalence'] * 100, np.log10(genus_stats['PCOS_abundance'] + 0.001), c=c_arr, s=s_arr, alpha=0.65, edgecolors='white', linewidths=0.15)
    ax2.axvline(x=args.prevalence * 100, color='#333', linestyle='--', linewidth=0.8)
    ax2.set_xlabel('Prevalence (%)', fontsize=8)
    ax2.set_ylabel('log₁₀(Abundance)', fontsize=8)
    ax2.set_title(f'(C1) PCOS (n core={len(core_pcos_set)})', fontsize=9, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax3 = fig_comp.add_subplot(gs_outer[1, 1])
    in_core = genus_stats['Genus'].isin(core_healthy_set)
    c_arr = [colors_group['Healthy'] if v else '#CCCCCC' for v in in_core]
    s_arr = [45 if v else 12 for v in in_core]
    ax3.scatter(genus_stats['Healthy_prevalence'] * 100, np.log10(genus_stats['Healthy_abundance'] + 0.001), c=c_arr, s=s_arr, alpha=0.65, edgecolors='white', linewidths=0.15)
    ax3.axvline(x=args.prevalence * 100, color='#333', linestyle='--', linewidth=0.8)
    ax3.set_xlabel('Prevalence (%)', fontsize=8)
    ax3.set_ylabel('log₁₀(Abundance)', fontsize=8)
    ax3.set_title(f'(C2) Healthy (n core={len(core_healthy_set)})', fontsize=9, fontweight='bold')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax4 = fig_comp.add_subplot(gs_outer[2, 0])
    bar_colors2 = [colors_group['PCOS'] if d > 0 else colors_group['Healthy'] for d in top_diff['Prevalence_diff']]
    y_pos2 = np.arange(len(top_diff))
    ax4.barh(y_pos2, top_diff['Prevalence_diff'] * 100, color=bar_colors2, alpha=0.78, edgecolor='white', height=0.72)
    ylabels2 = [g.replace('_', ' ')[:22] for g in top_diff['Genus']]
    ax4.set_yticks(y_pos2)
    ax4.set_yticklabels(ylabels2, fontsize=5.5)
    ax4.axvline(x=0, color='black', linewidth=0.7)
    ax4.set_xlabel('Prevalence Diff (%)', fontsize=8)
    ax4.set_title('(E) Prevalence Differences', fontsize=9, fontweight='bold')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax5 = fig_comp.add_subplot(gs_outer[2, 1])
    ax5.axis('off')
    summary_lines = ['Core Microbiome Summary', '', f'Threshold: {prim}% prevalence', f'Pre-filter: prev>={args.min_prev * 100:.0f}%, abund>={args.min_abund:.4f}%', '', f'Retained genera: {genus_filt.shape[0]}', f'PCOS core: {len(core_pcos_set)} genera', f'Healthy core: {len(core_healthy_set)} genera', '', f'Shared core: {len(shared_core)}', f'PCOS-specific: {len(pcos_specific)}', f'Healthy-specific: {len(healthy_specific)}', '', f'PCOS samples: n={len(pcos_samples)}', f'Healthy samples: n={len(healthy_samples)}', '', 'Cohorts: PRJNA530971', '         PRJNA549764', '         PRJNA791492']
    y_start = 0.97
    for line in summary_lines:
        weight = 'bold' if line in ('Core Microbiome Summary', 'Shared core:', 'PCOS-specific:', 'Healthy-specific:') or line.startswith('Threshold') else 'normal'
        ax5.text(0.05, y_start, line, transform=ax5.transAxes, fontsize=7.5, verticalalignment='top', fontweight=weight)
        y_start -= 0.053
    fig_comp.suptitle('Genus-level Core Microbiome Analysis\nPCOS vs Healthy Gut Microbiome (3 cohorts, MetaPhlAn4)', fontsize=10, fontweight='bold', y=1.01)
    save_figure(fig_comp, combined_ab, os.path.join(args.output, 'M5G_Fig_composite'), dpi=args.dpi)
    plt.close(fig_comp)
    logging.info('[7/7] Saving result tables')
    genus_stats.to_csv(os.path.join(args.output, 'M5G_genus_prevalence_stats.tsv'), sep='\t', index=False)
    core_list_rows = []
    for grp_name, gset in [('PCOS-specific', pcos_specific), ('Healthy-specific', healthy_specific), ('Shared', shared_core)]:
        for g in sorted(gset):
            row = genus_stats[genus_stats['Genus'] == g].iloc[0]
            core_list_rows.append({'Genus': g, 'Core_status': grp_name, 'PCOS_prevalence': f"{row['PCOS_prevalence']:.4f}", 'Healthy_prevalence': f"{row['Healthy_prevalence']:.4f}", 'PCOS_abundance': f"{row['PCOS_abundance']:.4f}", 'Healthy_abundance': f"{row['Healthy_abundance']:.4f}", 'Prevalence_diff': f"{row['Prevalence_diff']:.4f}"})
    core_df = pd.DataFrame(core_list_rows)
    core_df.to_csv(os.path.join(args.output, 'M5G_core_genus_list.tsv'), sep='\t', index=False)
    if batch_core:
        consist_out = consistent.copy() if len(consistent) > 0 else pd.DataFrame()
        if not consist_out.empty:
            consist_out.index.name = 'Genus'
            consist_out.to_csv(os.path.join(args.output, 'M5G_crossbatch_consistent_genera.tsv'), sep='\t')
    logging.info('=' * 60)
    logging.info('ANALYSIS SUMMARY')
    logging.info('=' * 60)
    logging.info('Genus table (filtered): %d genera x %d samples', genus_filt.shape[0], genus_filt.shape[1])
    logging.info('Primary threshold: %d%%', prim)
    logging.info('PCOS core genera:    %d', len(core_pcos_set))
    logging.info('Healthy core genera: %d', len(core_healthy_set))
    logging.info('Shared core:         %d', len(shared_core))
    logging.info('PCOS-specific:       %d', len(pcos_specific))
    logging.info('Healthy-specific:    %d', len(healthy_specific))
    if top_shared:
        logging.info('Top 5 shared core genera (by mean abundance):')
        for g in shared_stats.nlargest(5, 'Total_abund')['Genus'].tolist():
            row = genus_stats[genus_stats['Genus'] == g].iloc[0]
            logging.info('  %s  PCOS=%.2f%%  Healthy=%.2f%%', g, row['PCOS_abundance'], row['Healthy_abundance'])
    logging.info('Output directory: %s', args.output)
    logging.info('Done.')
if __name__ == '__main__':
    main()
