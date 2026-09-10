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
from itertools import combinations
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
from matplotlib_venn import venn3
warnings.filterwarnings('ignore')
NPG = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85']
LANCET = ['#00468B', '#ED0000', '#42B540', '#0099B4', '#925E9F', '#FDAF91', '#AD002A', '#ADB6B6', '#1B1919']
JCO = ['#0073C2', '#EFC000', '#868686', '#CD534C', '#7AA6DC', '#003C67', '#8F7700', '#3B3B3B', '#A73030', '#4A6990']
NEJM = ['#BC3C29', '#0072B5', '#E18727', '#20854E', '#7876B1', '#6F99AD', '#FFDC91', '#EE4C97']
PALETTES = {'npg': NPG, 'lancet': LANCET, 'jco': JCO, 'nejm': NEJM}

def parse_args():
    p = argparse.ArgumentParser(description='Module 5 Extension: Per-cohort Cross-level Validation', formatter_class=argparse.HelpFormatter, epilog='See module docstring for full usage.')
    p.add_argument('--genus', '-g', required=True, help='Genus abundance table (TSV/XLS, genus x samples, TSS pct)')
    p.add_argument('--species', '-s', required=True, help='Species abundance table (TSV, species x samples, TSS pct)')
    p.add_argument('--group', '-m', required=True, help='Metadata TSV: columns Sample, Bioproject, Group')
    p.add_argument('--output', '-o', required=True, help='Output directory')
    p.add_argument('--prevalence', type=float, default=0.7, help='Primary prevalence threshold (default: 0.70)')
    p.add_argument('--min_prev', type=float, default=0.1, help='Pre-filter: global prevalence >= this fraction (default: 0.10)')
    p.add_argument('--min_abund', type=float, default=0.01, help='Pre-filter: mean abundance >= this value pct (default: 0.01, matches Module 1 preprocessing)')
    p.add_argument('--min_n', type=int, default=5, help='Minimum samples per group per cohort to include (default: 5)')
    p.add_argument('--palette', default='npg', choices=['npg', 'lancet', 'jco', 'nejm'])
    p.add_argument('--dpi', type=int, default=300)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--verbose', '-v', action='store_true')
    return p.parse_args()

def save_fig(fig, df, base, dpi=300):
    fig.savefig(f'{base}.pdf', format='pdf', bbox_inches='tight')
    fig.savefig(f'{base}.svg', format='svg', bbox_inches='tight')
    fig.savefig(f'{base}.jpg', format='jpg', dpi=dpi, bbox_inches='tight')
    if df is not None:
        df.to_csv(f'{base}_plotdata.tsv', sep='\t', index=False)
    logging.info('  Saved: %s (.pdf/.svg/.jpg)', base)

def prefilter(df, min_prev=0.1, min_abund=0.01):
    df = df.clip(lower=0)
    gp = (df > 0).sum(axis=1) / df.shape[1]
    ga = df.mean(axis=1)
    keep = (gp >= min_prev) & (ga >= min_abund)
    return (df.loc[keep].copy(), gp, ga)

def identify_core(df, samples, thresh=0.7):
    sub = df[samples]
    prev = (sub > 0).sum(axis=1) / len(samples)
    return (set(prev[prev >= thresh].index), prev)

def jaccard(a, b):
    u = len(a | b)
    return len(a & b) / u if u > 0 else np.nan

def cohort_core_analysis(df, meta, batches, pcos_s, healthy_s, thresh, min_n, label):
    results = {}
    for batch in batches:
        b_meta = meta[meta['Bioproject'] == batch]
        b_pcos = [s for s in b_meta['Sample'] if s in pcos_s]
        b_healthy = [s for s in b_meta['Sample'] if s in healthy_s]
        n_p, n_h = (len(b_pcos), len(b_healthy))
        if n_p < min_n or n_h < min_n:
            logging.warning('  [%s] %s: PCOS n=%d, Healthy n=%d -- skipped (min_n=%d)', label, batch, n_p, n_h, min_n)
            continue
        c_pcos, prev_p = identify_core(df, b_pcos, thresh)
        c_healthy, prev_h = identify_core(df, b_healthy, thresh)
        shared = c_pcos & c_healthy
        pcos_sp = c_pcos - c_healthy
        hlt_sp = c_healthy - c_pcos
        results[batch] = {'pcos_core': c_pcos, 'healthy_core': c_healthy, 'shared': shared, 'pcos_specific': pcos_sp, 'healthy_specific': hlt_sp, 'prev_pcos': prev_p, 'prev_healthy': prev_h, 'n_pcos': n_p, 'n_healthy': n_h}
        logging.info('  [%s] %s: PCOS-core=%d | Healthy-core=%d | shared=%d | PCOS-only=%d | Healthy-only=%d', label, batch, len(c_pcos), len(c_healthy), len(shared), len(pcos_sp), len(hlt_sp))
    return results

def extract_parent_genus(species_name):
    return species_name.split('_')[0]

def main():
    args = parse_args()
    np.random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(args.output, 'M5_cohort_validation_log.txt'), mode='w')])
    pal = PALETTES[args.palette]
    colors_grp = {'PCOS': pal[0], 'Healthy': pal[1]}
    SINGLE = 89 / 25.4
    DOUBLE = 183 / 25.4
    logging.info('[1/8] Loading data')
    genus_raw = pd.read_csv(args.genus, sep='\t', index_col=0)
    species_raw = pd.read_csv(args.species, sep='\t', index_col=0)
    meta_all = pd.read_csv(args.group, sep='\t')
    meta_all.columns = ['Sample', 'Bioproject', 'Group']
    common = sorted(set(genus_raw.columns) & set(species_raw.columns) & set(meta_all['Sample']))
    genus_raw = genus_raw[common]
    species_raw = species_raw[common]
    meta_all = meta_all[meta_all['Sample'].isin(common)].reset_index(drop=True)
    logging.info('  Genus:   %d x %d | Species: %d x %d | Samples: %d', *genus_raw.shape, *species_raw.shape, len(common))
    batches = sorted(meta_all['Bioproject'].unique())
    pcos_s = meta_all[meta_all['Group'] == 'PCOS']['Sample'].tolist()
    healthy_s = meta_all[meta_all['Group'] == 'Healthy']['Sample'].tolist()
    n_batch = len(batches)
    batch_colors = dict(zip(batches, pal[2:2 + n_batch]))
    logging.info('[2/8] Pre-filtering')
    genus_filt, gp_g, ga_g = prefilter(genus_raw, args.min_prev, args.min_abund)
    species_filt, gp_s, ga_s = prefilter(species_raw, args.min_prev, args.min_abund)
    logging.info('  Genus retained:   %d / %d', *genus_filt.shape[:1], genus_raw.shape[0])
    logging.info('  Species retained: %d / %d', *species_filt.shape[:1], species_raw.shape[0])
    logging.info('[3/8] Per-cohort core analysis')
    g_cohort = cohort_core_analysis(genus_filt, meta_all, batches, pcos_s, healthy_s, args.prevalence, args.min_n, 'Genus')
    s_cohort = cohort_core_analysis(species_filt, meta_all, batches, pcos_s, healthy_s, args.prevalence, args.min_n, 'Species')
    valid_batches = sorted(set(g_cohort.keys()) & set(s_cohort.keys()))
    logging.info('  Valid cohorts (both levels): %s', valid_batches)
    g_global_pcos, _ = identify_core(genus_filt, pcos_s, args.prevalence)
    g_global_healthy, _ = identify_core(genus_filt, healthy_s, args.prevalence)
    s_global_pcos, _ = identify_core(species_filt, pcos_s, args.prevalence)
    s_global_healthy, _ = identify_core(species_filt, healthy_s, args.prevalence)
    g_global_shared = g_global_pcos & g_global_healthy
    s_global_shared = s_global_pcos & s_global_healthy
    logging.info('[4/8] Computing consistency metrics')

    def build_presence_matrix(cohort_res, taxa_universe):
        cols = []
        for b, res in cohort_res.items():
            cols.append((b, 'PCOS', res['pcos_core']))
            cols.append((b, 'Healthy', res['healthy_core']))
        col_names = [f'{b}_{g}' for b, g in [(c[0], c[1]) for c in cols]]
        mat = pd.DataFrame(0, index=sorted(taxa_universe), columns=col_names)
        for (b, grp, cset), cname in zip(cols, col_names):
            for t in cset:
                if t in mat.index:
                    mat.loc[t, cname] = 1
        return mat
    all_genera = set().union(*[r['pcos_core'] | r['healthy_core'] for r in g_cohort.values()])
    all_species = set().union(*[r['pcos_core'] | r['healthy_core'] for r in s_cohort.values()])
    g_presence = build_presence_matrix(g_cohort, all_genera)
    s_presence = build_presence_matrix(s_cohort, all_species)

    def cohort_jaccard_df(cohort_res, group_key='pcos_core'):
        bs = sorted(cohort_res.keys())
        mat = pd.DataFrame(np.nan, index=bs, columns=bs)
        for b1, b2 in combinations(bs, 2):
            j = jaccard(cohort_res[b1][group_key], cohort_res[b2][group_key])
            mat.loc[b1, b2] = j
            mat.loc[b2, b1] = j
        for b in bs:
            mat.loc[b, b] = 1.0
        return mat
    g_jacc_pcos = cohort_jaccard_df(g_cohort, 'pcos_core')
    g_jacc_healthy = cohort_jaccard_df(g_cohort, 'healthy_core')
    s_jacc_pcos = cohort_jaccard_df(s_cohort, 'pcos_core')
    s_jacc_healthy = cohort_jaccard_df(s_cohort, 'healthy_core')
    logging.info('  Genus PCOS Jaccard:\n%s', g_jacc_pcos.to_string())
    logging.info('  Species PCOS Jaccard:\n%s', s_jacc_pcos.to_string())
    logging.info('[5/8] Cross-level consistency (species -> parent genus)')
    xvalidation_rows = []
    for batch in valid_batches:
        s_res = s_cohort[batch]
        g_res = g_cohort[batch]
        for grp_key, grp_label in [('pcos_core', 'PCOS'), ('healthy_core', 'Healthy')]:
            sp_core = s_res[grp_key]
            g_core = g_res[grp_key]
            for sp in sp_core:
                parent = extract_parent_genus(sp)
                in_g = parent in g_core
                xvalidation_rows.append({'Batch': batch, 'Group': grp_label, 'Species': sp, 'Parent_genus': parent, 'Genus_in_core': int(in_g)})
    xval_df = pd.DataFrame(xvalidation_rows)
    if len(xval_df) > 0:
        agree_rate = xval_df.groupby(['Batch', 'Group'])['Genus_in_core'].mean().reset_index()
        agree_rate.columns = ['Batch', 'Group', 'Agreement_rate']
        logging.info('  Cross-level agreement rates:\n%s', agree_rate.to_string(index=False))
    else:
        agree_rate = pd.DataFrame()
    summary_rows = []
    for level, cohort_res in [('Genus', g_cohort), ('Species', s_cohort)]:
        for batch, res in cohort_res.items():
            summary_rows.append({'Level': level, 'Cohort': batch, 'N_PCOS': res['n_pcos'], 'N_Healthy': res['n_healthy'], 'PCOS_core': len(res['pcos_core']), 'Healthy_core': len(res['healthy_core']), 'Shared': len(res['shared']), 'PCOS_specific': len(res['pcos_specific']), 'Healthy_specific': len(res['healthy_specific'])})
    summary_df = pd.DataFrame(summary_rows)
    logging.info('\n  Core size summary:\n%s', summary_df.to_string(index=False))
    logging.info('[6/8] Generating figures')
    fig1 = plt.figure(figsize=(DOUBLE, DOUBLE * 0.88))
    gs1 = gridspec.GridSpec(2, 3, figure=fig1, hspace=0.52, wspace=0.42)

    def core_size_bar(ax, cohort_res, title, level_colors):
        bs = sorted(cohort_res.keys())
        cats = ['Shared', 'PCOS-specific', 'Healthy-specific']
        c_map = {'Shared': '#7570B3', 'PCOS-specific': pal[0], 'Healthy-specific': pal[1]}
        x = np.arange(len(bs))
        w = 0.22
        for i, cat in enumerate(cats):
            vals = []
            for b in bs:
                if cat == 'Shared':
                    vals.append(len(cohort_res[b]['shared']))
                elif cat == 'PCOS-specific':
                    vals.append(len(cohort_res[b]['pcos_specific']))
                else:
                    vals.append(len(cohort_res[b]['healthy_specific']))
            bars = ax.bar(x + (i - 1) * w, vals, w, label=cat, color=c_map[cat], alpha=0.82)
            ax.bar_label(bars, padding=1, fontsize=6)
        ax.set_xticks(x)
        ax.set_xticklabels([b.replace('PRJNA', 'P') for b in bs], fontsize=7)
        ax.set_ylabel('N taxa', fontsize=8)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.legend(fontsize=6, loc='upper right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    ax_a = fig1.add_subplot(gs1[0, 0])
    core_size_bar(ax_a, g_cohort, '(A) Genus: Per-cohort Core Size', batch_colors)
    ax_b = fig1.add_subplot(gs1[0, 1])
    core_size_bar(ax_b, s_cohort, '(B) Species: Per-cohort Core Size', batch_colors)

    def jacc_heatmap(ax, jacc_df, title):
        labels = [b.replace('PRJNA', 'P') for b in jacc_df.index]
        mask = np.triu(np.ones_like(jacc_df, dtype=bool), k=1)
        sns.heatmap(jacc_df, annot=True, fmt='.2f', cmap='YlGnBu', vmin=0, vmax=1, ax=ax, square=True, xticklabels=labels, yticklabels=labels, annot_kws={'size': 8}, linewidths=0.5, cbar_kws={'shrink': 0.7, 'label': 'Jaccard'})
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.tick_params(axis='y', rotation=0, labelsize=7)
    ax_c = fig1.add_subplot(gs1[0, 2])
    jacc_combined_g = g_jacc_pcos.copy()
    jacc_heatmap(ax_c, jacc_combined_g, '(C) Genus Jaccard\n(PCOS core, per-cohort)')
    ax_d = fig1.add_subplot(gs1[1, 0])
    jacc_heatmap(ax_d, s_jacc_pcos, '(D) Species Jaccard\n(PCOS core, per-cohort)')
    ax_e = fig1.add_subplot(gs1[1, 1])
    jacc_heatmap(ax_e, g_jacc_healthy, '(E) Genus Jaccard\n(Healthy core, per-cohort)')
    ax_f = fig1.add_subplot(gs1[1, 2])
    jacc_heatmap(ax_f, s_jacc_healthy, '(F) Species Jaccard\n(Healthy core, per-cohort)')
    fig1.suptitle(f'Per-cohort Core Microbiome: Size and Inter-cohort Jaccard Similarity\n(Prevalence threshold: {int(args.prevalence * 100)}%, PCOS/Healthy per cohort shown)', fontsize=10, fontweight='bold', y=1.01)
    plt.tight_layout()
    jacc_plot_df = pd.concat([g_jacc_pcos.stack().reset_index().assign(Level='Genus', Group='PCOS').rename(columns={'level_0': 'Cohort1', 'level_1': 'Cohort2', 0: 'Jaccard'}), s_jacc_pcos.stack().reset_index().assign(Level='Species', Group='PCOS').rename(columns={'level_0': 'Cohort1', 'level_1': 'Cohort2', 0: 'Jaccard'}), g_jacc_healthy.stack().reset_index().assign(Level='Genus', Group='Healthy').rename(columns={'level_0': 'Cohort1', 'level_1': 'Cohort2', 0: 'Jaccard'}), s_jacc_healthy.stack().reset_index().assign(Level='Species', Group='Healthy').rename(columns={'level_0': 'Cohort1', 'level_1': 'Cohort2', 0: 'Jaccard'})], ignore_index=True)
    save_fig(fig1, jacc_plot_df, os.path.join(args.output, 'M5CV_Fig1_cohort_core_jaccard'), dpi=args.dpi)
    plt.close(fig1)
    for panel_data, fname in [(g_cohort, 'M5CV_Fig1_panelA_genus_coresize'), (s_cohort, 'M5CV_Fig1_panelB_species_coresize')]:
        fig_p, ax_p = plt.subplots(figsize=(SINGLE * 1.1, SINGLE * 0.9))
        core_size_bar(ax_p, panel_data, 'Genus' if 'genus' in fname else 'Species', batch_colors)
        save_fig(fig_p, summary_df[summary_df['Level'] == ('Genus' if 'genus' in fname else 'Species')], os.path.join(args.output, fname), dpi=args.dpi)
        plt.close(fig_p)

    def upset_figure(presence_mat, cohort_keys, title, fname, dpi):
        cols = [c for c in presence_mat.columns if any((k in c for k in cohort_keys))]
        if len(cols) < 2:
            return
        labels = [c.replace('PRJNA', 'P').replace('_', '\n') for c in cols]
        n_sets = len(cols)
        rows = []
        for r in range(1, 2 ** n_sets):
            mask = [r >> i & 1 for i in range(n_sets)]
            sel = presence_mat[cols].copy()
            cond = np.ones(len(sel), dtype=bool)
            for i, m in enumerate(mask):
                cond &= sel[cols[i]] == m
            count = int(cond.sum())
            if count > 0:
                rows.append({'mask': mask, 'count': count, 'degree': sum(mask)})
        if not rows:
            return
        rows.sort(key=lambda x: -x['count'])
        n_inter = len(rows)
        fig_u = plt.figure(figsize=(max(DOUBLE * 0.8, n_inter * 0.55 + 1.5), DOUBLE * 0.55))
        gs_u = gridspec.GridSpec(2, 1, figure=fig_u, height_ratios=[2.5, 1.4], hspace=0.08)
        ax_bar = fig_u.add_subplot(gs_u[0])
        ax_mat = fig_u.add_subplot(gs_u[1], sharex=ax_bar)
        x = np.arange(n_inter)
        bar_colors_u = [pal[3] if r['degree'] > 1 else pal[5 % len(pal)] for r in rows]
        bars_u = ax_bar.bar(x, [r['count'] for r in rows], color=bar_colors_u, alpha=0.85, edgecolor='white', linewidth=0.5)
        for bar_obj, r in zip(bars_u, rows):
            ax_bar.text(bar_obj.get_x() + bar_obj.get_width() / 2, bar_obj.get_height() + 0.3, str(r['count']), ha='center', va='bottom', fontsize=7)
        ax_bar.set_ylabel('Intersection size', fontsize=8)
        ax_bar.set_title(title, fontsize=9, fontweight='bold')
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)
        ax_bar.tick_params(bottom=False, labelbottom=False)
        ax_bar.set_xlim(-0.5, n_inter - 0.5)
        ax_mat.set_ylim(-0.5, n_sets - 0.5)
        ax_mat.set_xlim(-0.5, n_inter - 0.5)
        ax_mat.set_yticks(range(n_sets))
        ax_mat.set_yticklabels(labels[::-1], fontsize=6.5)
        ax_mat.tick_params(bottom=False, labelbottom=False)
        ax_mat.spines['top'].set_visible(False)
        ax_mat.spines['right'].set_visible(False)
        ax_mat.spines['bottom'].set_visible(False)
        for xi, r in enumerate(rows):
            active_y = []
            for si, m in enumerate(r['mask']):
                yi = n_sets - 1 - si
                if m:
                    ax_mat.scatter(xi, yi, s=55, color=pal[3], zorder=3, linewidths=0)
                    active_y.append(yi)
                else:
                    ax_mat.scatter(xi, yi, s=35, color='#DDDDDD', zorder=2, linewidths=0)
            if len(active_y) > 1:
                ax_mat.plot([xi, xi], [min(active_y), max(active_y)], color=pal[3], linewidth=1.8, zorder=2)
        plt.tight_layout()
        pm_reset = presence_mat[cols].copy()
        pm_reset.index.name = 'Taxon'
        inter_df = pd.DataFrame([{'Combination': '|'.join([cols[i].replace('PRJNA', 'P') for i, m in enumerate(r['mask']) if m]), 'Count': r['count'], 'Degree': r['degree']} for r in rows])
        save_fig(fig_u, inter_df, os.path.join(args.output, fname), dpi=dpi)
        plt.close(fig_u)
    pcos_cols = [c for c in g_presence.columns if 'PCOS' in c]
    healthy_cols = [c for c in g_presence.columns if 'Healthy' in c]
    upset_figure(g_presence, [c for c in g_presence.columns], 'Genus Core: Cross-cohort Overlap (PCOS + Healthy)', 'M5CV_Fig2_genus_upset', args.dpi)
    upset_figure(s_presence, [c for c in s_presence.columns], 'Species Core: Cross-cohort Overlap (PCOS + Healthy)', 'M5CV_Fig3_species_upset', args.dpi)

    def consistency_heatmap(presence_mat, global_shared, title, fname, min_sum=None, max_taxa=40, dpi=300):
        if min_sum is None:
            min_sum = max(2, presence_mat.shape[1] - 1)
        row_sums = presence_mat.sum(axis=1)
        sel = presence_mat[row_sums >= min_sum].copy()
        sel['_sum'] = sel.sum(axis=1)
        sel = sel.sort_values('_sum', ascending=False).drop('_sum', axis=1)
        if len(sel) > max_taxa:
            sel = sel.head(max_taxa)
        if len(sel) == 0:
            return
        h = max(DOUBLE * 0.4, len(sel) * 0.22)
        fig_h, ax_h = plt.subplots(figsize=(DOUBLE * 0.65, h))
        row_colors = pd.Series([pal[4] if t in global_shared else '#CCCCCC' for t in sel.index], index=sel.index, name='Global shared')
        xlabels = [c.replace('PRJNA', 'P').replace('_', '\n') for c in sel.columns]
        sns.heatmap(sel, cmap='YlOrRd', ax=ax_h, xticklabels=xlabels, yticklabels=True, linewidths=0.3, linecolor='white', cbar_kws={'label': 'Core (1=yes)', 'shrink': 0.5}, vmin=0, vmax=1)
        ax_h.set_yticklabels([t.replace('_', ' ')[:30] for t in sel.index], fontsize=6)
        ax_h.set_xticklabels(xlabels, fontsize=7, rotation=45, ha='right')
        for i, taxon in enumerate(sel.index):
            if taxon in global_shared:
                ax_h.get_yticklabels()[i].set_color(pal[3])
                ax_h.get_yticklabels()[i].set_fontweight('bold')
        legend_els = [mpatches.Patch(color=pal[3], label='Also in global shared core'), mpatches.Patch(color='#CCCCCC', label='Not in global shared core')]
        ax_h.legend(handles=legend_els, loc='upper right', bbox_to_anchor=(1.0, -0.12), fontsize=7, ncol=2, framealpha=0.8)
        ax_h.set_title(title, fontsize=9, fontweight='bold')
        plt.tight_layout()
        sel_out = sel.copy()
        sel_out.index.name = 'Taxon'
        save_fig(fig_h, sel_out.reset_index(), os.path.join(args.output, fname), dpi=dpi)
        plt.close(fig_h)
    consistency_heatmap(g_presence, g_global_shared, f'Genus Core: Cross-cohort Consistency\n(Highlighted: present in >= {max(2, g_presence.shape[1] - 1)} cohort-group combos)', 'M5CV_Fig4_genus_consistency_heatmap', max_taxa=40, dpi=args.dpi)
    consistency_heatmap(s_presence, s_global_shared, f'Species Core: Cross-cohort Consistency\n(Highlighted: present in >= {max(2, s_presence.shape[1] - 1)} cohort-group combos)', 'M5CV_Fig5_species_consistency_heatmap', max_taxa=50, dpi=args.dpi)
    if len(xval_df) > 0:
        fig4 = plt.figure(figsize=(DOUBLE, DOUBLE * 0.45))
        gs4 = gridspec.GridSpec(1, 3, figure=fig4, wspace=0.38)
        ax4a = fig4.add_subplot(gs4[0, 0])
        agree_pivot = agree_rate.pivot(index='Batch', columns='Group', values='Agreement_rate').fillna(0) * 100
        x4 = np.arange(len(agree_pivot))
        w4 = 0.35
        b1 = ax4a.bar(x4 - w4 / 2, agree_pivot.get('PCOS', pd.Series(dtype=float)), w4, label='PCOS', color=colors_grp['PCOS'], alpha=0.82)
        b2 = ax4a.bar(x4 + w4 / 2, agree_pivot.get('Healthy', pd.Series(dtype=float)), w4, label='Healthy', color=colors_grp['Healthy'], alpha=0.82)
        ax4a.bar_label(b1, fmt='%.0f%%', padding=2, fontsize=7)
        ax4a.bar_label(b2, fmt='%.0f%%', padding=2, fontsize=7)
        ax4a.set_xticks(x4)
        ax4a.set_xticklabels([b.replace('PRJNA', 'P') for b in agree_pivot.index], fontsize=7)
        ax4a.set_ylabel('Species with core genus (%)', fontsize=8)
        ax4a.set_ylim(0, 115)
        ax4a.set_title('(A) Cross-level Agreement\n(Species -> Genus)', fontsize=9, fontweight='bold')
        ax4a.legend(fontsize=7)
        ax4a.spines['top'].set_visible(False)
        ax4a.spines['right'].set_visible(False)
        ax4b = fig4.add_subplot(gs4[0, 1])
        for batch in valid_batches:
            g_sh = len(g_cohort[batch]['shared'])
            s_sh = len(s_cohort[batch]['shared'])
            ax4b.scatter(g_sh, s_sh, s=90, color=batch_colors[batch], edgecolors='white', linewidths=0.5, zorder=3)
            ax4b.annotate(batch.replace('PRJNA', 'P'), (g_sh, s_sh), textcoords='offset points', xytext=(5, 4), fontsize=7)
        ax4b.scatter(len(g_global_shared), len(s_global_shared), s=140, color='#333333', marker='*', zorder=4, label='Global (all cohorts)')
        ax4b.set_xlabel('Genus shared core size', fontsize=8)
        ax4b.set_ylabel('Species shared core size', fontsize=8)
        ax4b.set_title('(B) Shared Core Size\nGenus vs Species', fontsize=9, fontweight='bold')
        ax4b.legend(fontsize=7)
        ax4b.spines['top'].set_visible(False)
        ax4b.spines['right'].set_visible(False)
        ax4c = fig4.add_subplot(gs4[0, 2])
        for batch in valid_batches:
            gp = len(g_cohort[batch]['pcos_specific'])
            sp = len(s_cohort[batch]['pcos_specific'])
            gh = len(g_cohort[batch]['healthy_specific'])
            sh = len(s_cohort[batch]['healthy_specific'])
            ax4c.scatter(gp, sp, s=80, color=colors_grp['PCOS'], marker='o', alpha=0.85, zorder=3)
            ax4c.scatter(gh, sh, s=80, color=colors_grp['Healthy'], marker='^', alpha=0.85, zorder=3)
            ax4c.annotate(batch.replace('PRJNA', 'P'), (gp, sp), textcoords='offset points', xytext=(4, 3), fontsize=6, color=colors_grp['PCOS'])
            ax4c.annotate(batch.replace('PRJNA', 'P'), (gh, sh), textcoords='offset points', xytext=(4, 3), fontsize=6, color=colors_grp['Healthy'])
        lim = max(ax4c.get_xlim()[1], ax4c.get_ylim()[1]) + 1
        ax4c.plot([0, lim], [0, lim], '--', color='#999', linewidth=0.8, alpha=0.6)
        ax4c.set_xlabel('Genus-specific count', fontsize=8)
        ax4c.set_ylabel('Species-specific count', fontsize=8)
        ax4c.set_title('(C) Group-specific Taxa\nGenus vs Species', fontsize=9, fontweight='bold')
        legend_els2 = [mpatches.Patch(color=colors_grp['PCOS'], label='PCOS-specific'), mpatches.Patch(color=colors_grp['Healthy'], label='Healthy-specific')]
        ax4c.legend(handles=legend_els2, fontsize=7)
        ax4c.spines['top'].set_visible(False)
        ax4c.spines['right'].set_visible(False)
        fig4.suptitle('Cross-level Consistency: Genus vs Species Core Microbiome', fontsize=10, fontweight='bold', y=1.02)
        plt.tight_layout()
        save_fig(fig4, xval_df, os.path.join(args.output, 'M5CV_Fig6_crosslevel_agreement'), dpi=args.dpi)
        plt.close(fig4)
    if len(valid_batches) == 3:
        for level_label, cohort_res, col in [('Genus', g_cohort, pal[0]), ('Species', s_cohort, pal[1])]:
            for grp_key, grp_label, grp_col in [('pcos_core', 'PCOS', colors_grp['PCOS']), ('healthy_core', 'Healthy', colors_grp['Healthy'])]:
                sets3 = [cohort_res[b][grp_key] for b in valid_batches]
                fig5, ax5 = plt.subplots(figsize=(SINGLE * 1.1, SINGLE * 1.1))
                venn3(sets3, set_labels=[b.replace('PRJNA', 'P') for b in valid_batches], set_colors=[batch_colors[b] for b in valid_batches], alpha=0.6, ax=ax5)
                ax5.set_title(f'{level_label} Core ({grp_label}): 3-Cohort Venn\n(Prevalence >= {int(args.prevalence * 100)}%)', fontsize=9, fontweight='bold')
                fname5 = f'M5CV_Fig7_{level_label.lower()}_{grp_label.lower()}_venn3'
                venn_df = pd.DataFrame({'Cohort': valid_batches, 'N_core': [len(cohort_res[b][grp_key]) for b in valid_batches], 'Group': grp_label, 'Level': level_label})
                save_fig(fig5, venn_df, os.path.join(args.output, fname5), dpi=args.dpi)
                plt.close(fig5)
    fig_comp = plt.figure(figsize=(DOUBLE, DOUBLE * 1.45))
    gsc = gridspec.GridSpec(3, 3, figure=fig_comp, hspace=0.52, wspace=0.42)
    ax_c0 = fig_comp.add_subplot(gsc[0, 0])
    core_size_bar(ax_c0, g_cohort, '(A) Genus Core Size per Cohort', batch_colors)
    ax_c1 = fig_comp.add_subplot(gsc[0, 1])
    core_size_bar(ax_c1, s_cohort, '(B) Species Core Size per Cohort', batch_colors)
    ax_c2 = fig_comp.add_subplot(gsc[0, 2])
    if len(agree_rate) > 0:
        agree_pivot2 = agree_rate.pivot(index='Batch', columns='Group', values='Agreement_rate').fillna(0) * 100
        x_c2 = np.arange(len(agree_pivot2))
        b1c = ax_c2.bar(x_c2 - 0.175, agree_pivot2.get('PCOS', pd.Series(dtype=float)), 0.35, label='PCOS', color=colors_grp['PCOS'], alpha=0.82)
        b2c = ax_c2.bar(x_c2 + 0.175, agree_pivot2.get('Healthy', pd.Series(dtype=float)), 0.35, label='Healthy', color=colors_grp['Healthy'], alpha=0.82)
        ax_c2.bar_label(b1c, fmt='%.0f%%', padding=1, fontsize=6)
        ax_c2.bar_label(b2c, fmt='%.0f%%', padding=1, fontsize=6)
        ax_c2.set_xticks(x_c2)
        ax_c2.set_xticklabels([b.replace('PRJNA', 'P') for b in agree_pivot2.index], fontsize=7)
        ax_c2.set_ylabel('Agreement (%)', fontsize=8)
        ax_c2.set_ylim(0, 115)
        ax_c2.set_title('(C) Cross-level Agreement\n(Species -> Genus)', fontsize=9, fontweight='bold')
        ax_c2.legend(fontsize=6)
        ax_c2.spines['top'].set_visible(False)
        ax_c2.spines['right'].set_visible(False)
    ax_c3 = fig_comp.add_subplot(gsc[1, 0])
    jacc_heatmap(ax_c3, g_jacc_pcos, '(D) Genus Jaccard\n(PCOS core)')
    ax_c4 = fig_comp.add_subplot(gsc[1, 1])
    jacc_heatmap(ax_c4, s_jacc_pcos, '(E) Species Jaccard\n(PCOS core)')
    ax_c5 = fig_comp.add_subplot(gsc[1, 2])
    jacc_heatmap(ax_c5, g_jacc_healthy, '(F) Genus Jaccard\n(Healthy core)')
    ax_c6 = fig_comp.add_subplot(gsc[2, 0])
    for batch in valid_batches:
        g_sh = len(g_cohort[batch]['shared'])
        s_sh = len(s_cohort[batch]['shared'])
        ax_c6.scatter(g_sh, s_sh, s=80, color=batch_colors[batch], edgecolors='white', linewidths=0.4, zorder=3)
        ax_c6.annotate(batch.replace('PRJNA', 'P'), (g_sh, s_sh), textcoords='offset points', xytext=(4, 3), fontsize=6.5)
    ax_c6.scatter(len(g_global_shared), len(s_global_shared), s=120, color='#333', marker='*', zorder=4, label='Global')
    ax_c6.set_xlabel('Genus shared core', fontsize=8)
    ax_c6.set_ylabel('Species shared core', fontsize=8)
    ax_c6.set_title('(G) Shared Core Size', fontsize=9, fontweight='bold')
    ax_c6.legend(fontsize=7)
    ax_c6.spines['top'].set_visible(False)
    ax_c6.spines['right'].set_visible(False)
    if len(valid_batches) == 3:
        ax_c7 = fig_comp.add_subplot(gsc[2, 1])
        sets_pcos = [g_cohort[b]['pcos_core'] for b in valid_batches]
        venn3(sets_pcos, set_labels=[b.replace('PRJNA', 'P') for b in valid_batches], set_colors=[batch_colors[b] for b in valid_batches], alpha=0.58, ax=ax_c7)
        ax_c7.set_title('(H) Genus PCOS Core\n3-Cohort Venn', fontsize=9, fontweight='bold')
        ax_c8 = fig_comp.add_subplot(gsc[2, 2])
        sets_hlt = [g_cohort[b]['healthy_core'] for b in valid_batches]
        venn3(sets_hlt, set_labels=[b.replace('PRJNA', 'P') for b in valid_batches], set_colors=[batch_colors[b] for b in valid_batches], alpha=0.58, ax=ax_c8)
        ax_c8.set_title('(I) Genus Healthy Core\n3-Cohort Venn', fontsize=9, fontweight='bold')
    fig_comp.suptitle(f'Per-cohort Cross-level Core Microbiome Validation (Genus & Species)\nPCOS Multi-cohort Metagenomics | Threshold: {int(args.prevalence * 100)}%', fontsize=10, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_fig(fig_comp, summary_df, os.path.join(args.output, 'M5CV_Fig_composite'), dpi=args.dpi)
    plt.close(fig_comp)
    logging.info('[7/8] Saving tables')
    summary_df.to_csv(os.path.join(args.output, 'M5CV_core_size_summary.tsv'), sep='\t', index=False)
    jacc_plot_df.to_csv(os.path.join(args.output, 'M5CV_jaccard_similarity.tsv'), sep='\t', index=False)
    if len(xval_df) > 0:
        xval_df.to_csv(os.path.join(args.output, 'M5CV_crosslevel_agreement.tsv'), sep='\t', index=False)
        agree_rate.to_csv(os.path.join(args.output, 'M5CV_crosslevel_agreement_rate.tsv'), sep='\t', index=False)

    def universal_core(cohort_res, key='shared'):
        if not cohort_res:
            return set()
        sets = [cohort_res[b][key] for b in cohort_res]
        return set.intersection(*sets)
    g_universal = universal_core(g_cohort, 'shared')
    s_universal = universal_core(s_cohort, 'shared')
    logging.info('  Universal genus core (shared, all cohorts): %d -> %s', len(g_universal), sorted(g_universal))
    logging.info('  Universal species core (shared, all cohorts): %d', len(s_universal))
    pd.DataFrame({'Genus': sorted(g_universal)}).to_csv(os.path.join(args.output, 'M5CV_universal_genus_core.tsv'), sep='\t', index=False)
    pd.DataFrame({'Species': sorted(s_universal)}).to_csv(os.path.join(args.output, 'M5CV_universal_species_core.tsv'), sep='\t', index=False)
    g_presence.index.name = 'Genus'
    g_presence.reset_index().to_csv(os.path.join(args.output, 'M5CV_genus_presence_matrix.tsv'), sep='\t', index=False)
    s_presence.index.name = 'Species'
    s_presence.reset_index().to_csv(os.path.join(args.output, 'M5CV_species_presence_matrix.tsv'), sep='\t', index=False)
    logging.info('[8/8] Summary')
    logging.info('=' * 65)
    logging.info('  Cohorts analyzed: %s', valid_batches)
    logging.info('  --- Genus level ---')
    for b in valid_batches:
        r = g_cohort[b]
        logging.info('    %s: shared=%d | PCOS-sp=%d | Healthy-sp=%d', b, len(r['shared']), len(r['pcos_specific']), len(r['healthy_specific']))
    logging.info('    Global shared core: %d', len(g_global_shared))
    logging.info('    Universal core (all cohorts): %d', len(g_universal))
    logging.info('  --- Species level ---')
    for b in valid_batches:
        r = s_cohort[b]
        logging.info('    %s: shared=%d | PCOS-sp=%d | Healthy-sp=%d', b, len(r['shared']), len(r['pcos_specific']), len(r['healthy_specific']))
    logging.info('    Global shared core: %d', len(s_global_shared))
    logging.info('    Universal core (all cohorts): %d', len(s_universal))
    if len(agree_rate) > 0:
        mean_agree = agree_rate['Agreement_rate'].mean() * 100
        logging.info('  Cross-level agreement mean: %.1f%%', mean_agree)
    logging.info('  Output: %s', args.output)
    logging.info('  Done.')
if __name__ == '__main__':
    main()
