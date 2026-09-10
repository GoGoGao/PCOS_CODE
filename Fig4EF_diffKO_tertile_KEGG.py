from pathlib import Path
import os
PKG_ROOT = Path(__file__).resolve().parents[1]
SHARED = PKG_ROOT / 'shared_input'
RSCRIPT = os.environ.get('RSCRIPT', 'Rscript')
import json
import logging
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import font_manager
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')
PROJ = ROOT.parents[1]
for d in (TAB, FIG, PLOT, LOG, ENR):
    d.mkdir(parents=True, exist_ok=True)
META_V2 = ROOT / 'tables' / 'SGB_group_prevalence_metadata.tsv'
Q_CUT = 0.01
FRAC_CUT = 0.1
TOP_ENRICH = 50
PATHWAY_Q = 0.01
if os.path.exists(FONT):
    font_manager.fontManager.addfont(FONT)
mpl.rcParams.update({'font.family': 'Times New Roman', 'font.weight': 'normal', 'axes.titleweight': 'normal', 'axes.labelweight': 'normal', 'font.size': 8, 'axes.spines.right': False, 'axes.spines.top': False, 'legend.frameon': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'figure.dpi': 300, 'savefig.dpi': 300})
COL = {'PCOS': '#E64B35', 'Healthy': '#4DBBD5', 'NS': '#B0B0B0'}

def save_table(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(f'{path}.tsv', sep='\t', index=False)
    df.to_csv(f'{path}.csv', index=False)

def save_fig(fig, base: Path, plot_df: pd.DataFrame | None=None):
    base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in ('pdf', 'svg', 'png', 'jpg'):
        fig.savefig(f'{base}.{fmt}', format=fmt, bbox_inches='tight', pad_inches=0.04)
    if plot_df is not None:
        save_table(plot_df, PLOT / f'{base.name}_plotdata')
    plt.close(fig)

def strip_ko(x: str) -> str:
    return str(x).split(':')[0].strip()

def is_global_overview(row: pd.Series) -> bool:
    sub = str(row.get('subcategory', ''))
    desc = str(row.get('Description', ''))
    kid = str(row.get('ID', ''))
    if 'Global and overview maps' in sub:
        return True
    if kid.startswith('ko011') or kid.startswith('map011'):
        return True
    if 'overview' in desc.lower() and 'cancer' in sub.lower():
        return False
    return False

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG / 'diffKO_tertile_focused.log', mode='w')])
    logging.info('=== Plan B: PCOS_tertile vs Healthy_tertile focused diffKO ===')
    meta = pd.read_csv(META_V2, sep='\t').set_index('SGB_ID')
    if 'Delta_tertile' not in meta.columns:
        meta['Delta_tertile'] = pd.qcut(meta['Delta_Prev'], 3, labels=['Healthy_tertile', 'Mid_tertile', 'PCOS_tertile'])
    if 'log_length' not in meta.columns:
        meta['log_length'] = np.log10(meta['length'].clip(lower=1))
    sub = meta[meta['Delta_tertile'].isin(['PCOS_tertile', 'Healthy_tertile'])].copy()
    sub['DiffGroup'] = np.where(sub['Delta_tertile'] == 'PCOS_tertile', 'PCOS_biased', 'Healthy_biased')
    n_p = int((sub.DiffGroup == 'PCOS_biased').sum())
    n_h = int((sub.DiffGroup == 'Healthy_biased').sum())
    cuts = meta['Delta_Prev'].quantile([1 / 3, 2 / 3])
    logging.info('PCOS_tertile=%d | Healthy_tertile=%d | cuts=%.4f / %.4f', n_p, n_h, cuts.iloc[0], cuts.iloc[1])
    save_table(sub.reset_index(), TAB / 'SGB_tertile_extreme_groups')
    ko_raw = pd.read_csv(IN_KO, sep='\t', index_col=0)
    mags = [m for m in sub.index if m in ko_raw.columns]
    sub = sub.loc[mags]
    ko_bin = (ko_raw[mags] > 0).astype(int)
    ko_bin.index = [strip_ko(i) for i in ko_bin.index]
    ko_bin = ko_bin.groupby(level=0).max().astype(int)
    frac = ko_bin.mean(1)
    ko_f = ko_bin.loc[(frac >= 0.02) & (frac <= 0.98)].copy()
    logging.info('KOs tested: %d', ko_f.shape[0])
    name_map = {strip_ko(i): str(i) for i in ko_raw.index}
    y_grp = (sub.loc[ko_f.columns, 'DiffGroup'] == 'PCOS_biased').astype(float).values
    cov = np.column_stack([sub.loc[ko_f.columns, 'log_length'].values, sub.loc[ko_f.columns, 'completeness'].values])
    rows = []
    Y = ko_f.values.astype(float)
    pcos_ids = sub.index[sub.DiffGroup == 'PCOS_biased']
    heal_ids = sub.index[sub.DiffGroup == 'Healthy_biased']
    for i, kid in enumerate(ko_f.index):
        x = Y[i]
        if x.sum() < 5 or len(x) - x.sum() < 5 or x.std() == 0:
            continue
        Xd = sm.add_constant(np.column_stack([y_grp, cov]))
        try:
            fit = sm.GLM(x, Xd, family=sm.families.Binomial()).fit(disp=0, maxiter=100)
            beta = float(fit.params[1])
            pval = float(fit.pvalues[1])
            beta_L = float(fit.params[2])
            p_L = float(fit.pvalues[2])
        except Exception:
            continue
        fp = float(ko_f.loc[kid, pcos_ids].mean())
        fh = float(ko_f.loc[kid, heal_ids].mean())
        rows.append({'KO': kid, 'KO_annotation': name_map.get(kid, kid), 'PCOS_tertile_frac': fp, 'Healthy_tertile_frac': fh, 'frac_diff_PCOS_minus_Healthy': fp - fh, 'Logit_beta_PCOS_vs_Healthy': beta, 'P': pval, 'Logit_beta_logLength': beta_L, 'P_logLength': p_L})
    res = pd.DataFrame(rows)
    res['qvalue'] = multipletests(res['P'], method='fdr_bh')[1]
    res['abs_frac_diff'] = res['frac_diff_PCOS_minus_Healthy'].abs()
    res['Direction'] = 'NS'
    hit = (res.qvalue < Q_CUT) & (res.abs_frac_diff >= FRAC_CUT)
    res.loc[hit & (res.Logit_beta_PCOS_vs_Healthy > 0), 'Direction'] = 'PCOS_enriched'
    res.loc[hit & (res.Logit_beta_PCOS_vs_Healthy < 0), 'Direction'] = 'Healthy_enriched'
    res['Direction_q_only'] = 'NS'
    res.loc[(res.qvalue < Q_CUT) & (res.Logit_beta_PCOS_vs_Healthy > 0), 'Direction_q_only'] = 'PCOS_enriched'
    res.loc[(res.qvalue < Q_CUT) & (res.Logit_beta_PCOS_vs_Healthy < 0), 'Direction_q_only'] = 'Healthy_enriched'
    res = res.sort_values('qvalue')
    save_table(res, TAB / 'diffKO_tertile_focused_length_adjusted')
    n_pko = int((res.Direction == 'PCOS_enriched').sum())
    n_hko = int((res.Direction == 'Healthy_enriched').sum())
    n_pko_q = int((res.Direction_q_only == 'PCOS_enriched').sum())
    n_hko_q = int((res.Direction_q_only == 'Healthy_enriched').sum())
    logging.info('Focused q<%.2f & |Δfrac|≥%.2f: PCOS=%d Healthy=%d | q-only: PCOS=%d Healthy=%d', Q_CUT, FRAC_CUT, n_pko, n_hko, n_pko_q, n_hko_q)
    save_table(res[res.Direction == 'PCOS_enriched'], TAB / 'diffKO_PCOS_enriched_focused')
    save_table(res[res.Direction == 'Healthy_enriched'], TAB / 'diffKO_Healthy_enriched_focused')
    pcos_list = res[res.Direction == 'PCOS_enriched'].nsmallest(TOP_ENRICH, 'qvalue')['KO'].tolist()
    heal_list = res[res.Direction == 'Healthy_enriched'].nsmallest(TOP_ENRICH, 'qvalue')['KO'].tolist()
    (ENR / 'PCOS_enriched_tertile.KO').write_text('\n'.join(pcos_list) + ('\n' if pcos_list else ''))
    (ENR / 'Healthy_enriched_tertile.KO').write_text('\n'.join(heal_list) + ('\n' if heal_list else ''))
    logging.info('Enrich lists: PCOS Top%d=%d | Healthy Top%d=%d', TOP_ENRICH, len(pcos_list), TOP_ENRICH, len(heal_list))
    import re
    from matplotlib.lines import Line2D

    def _gene_name(ann: str) -> str:
        s = str(ann)
        if ':' in s:
            s = s.split(':', 1)[1]
        parts = [p.strip() for p in re.split('[,;]', s) if p.strip()]
        genes = []
        for p in parts:
            if re.match('^E\\d', p) and '.' in p:
                continue
            if re.fullmatch('[A-Za-z][A-Za-z0-9_.-]{1,20}', p):
                genes.append(p)
        out = []
        for g in genes:
            if g not in out:
                out.append(g)
            if len(out) == 2:
                break
        return '/'.join(out) if out else parts[0] if parts else str(ann)
    v = res.copy()
    v['neglog10q'] = -np.log10(v['qvalue'].clip(lower=1e-300))
    v['gene'] = v['KO_annotation'].map(_gene_name)
    n_ns = int((res.Direction == 'NS').sum())
    beta_hard = 6.0
    artifact = (v.Direction == 'NS') & (v.Logit_beta_PCOS_vs_Healthy.abs() > beta_hard)
    v_plot = v.loc[~artifact].copy()
    xmax = float(np.nanmax(np.abs(v_plot['Logit_beta_PCOS_vs_Healthy'])))
    xlim = (-xmax - 0.08, xmax + 0.08)
    ymax = float(v_plot['neglog10q'].max()) + 0.18
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    for dirc, col, z in [('NS', COL['NS'], 1), ('Healthy_enriched', COL['Healthy'], 2), ('PCOS_enriched', COL['PCOS'], 3)]:
        s = v_plot[v_plot.Direction == dirc]
        ax.scatter(s.Logit_beta_PCOS_vs_Healthy, s.neglog10q, s=15, c=col, alpha=0.85, linewidths=0, zorder=z, rasterized=True)
    top_p = v_plot[v_plot.Direction == 'PCOS_enriched'].nsmallest(5, 'qvalue').reset_index(drop=True)
    top_h = v_plot[v_plot.Direction == 'Healthy_enriched'].nsmallest(5, 'qvalue').reset_index(drop=True)
    ax.scatter(top_p.Logit_beta_PCOS_vs_Healthy, top_p.neglog10q, s=32, c=COL['PCOS'], edgecolors='black', linewidths=0.6, zorder=4)
    ax.scatter(top_h.Logit_beta_PCOS_vs_Healthy, top_h.neglog10q, s=32, c=COL['Healthy'], edgecolors='black', linewidths=0.6, zorder=4)
    ax.axhline(-np.log10(Q_CUT), ls='--', lw=0.75, c='#666666', zorder=0)
    ax.axvline(0, ls='--', lw=0.75, c='#666666', zorder=0)
    ax.set_xlim(xlim)
    ax.set_ylim(-0.02, ymax)
    ax.set_xlabel('Logistic $\\beta$ (PCOS vs Healthy tertile | length)')
    ax.set_ylabel('$-\\log_{10}$(q-value)')
    off_p = {'K00526': (0.05, 0.07), 'K08483': (0.06, 0.05), 'K01223': (0.07, 0.16), 'K00525': (0.09, -0.13), 'K02770': (0.11, 0.02)}
    off_h = {'K19294': (-0.05, 0.07), 'K00176': (-0.06, 0.05), 'K06295': (-0.07, 0.14), 'K06387': (-0.06, -0.13), 'K00177': (-0.09, 0.01)}
    for _, r in top_p.iterrows():
        dx, dy = off_p.get(r['KO'], (0.06, 0.06))
        ax.text(r['Logit_beta_PCOS_vs_Healthy'] + dx, r['neglog10q'] + dy, r['gene'], fontsize=10.5, color=COL['PCOS'], ha='left', va='center', zorder=6)
    for _, r in top_h.iterrows():
        dx, dy = off_h.get(r['KO'], (-0.06, 0.06))
        ax.text(r['Logit_beta_PCOS_vs_Healthy'] + dx, r['neglog10q'] + dy, r['gene'], fontsize=10.5, color=COL['Healthy'], ha='right', va='center', zorder=6)
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=COL['PCOS'], markersize=7, label=f'PCOS enriched (n={n_pko})'), Line2D([0], [0], marker='o', color='w', markerfacecolor=COL['Healthy'], markersize=7, label=f'Healthy enriched (n={n_hko})'), Line2D([0], [0], marker='o', color='w', markerfacecolor=COL['NS'], markersize=7, label=f'NS (n={n_ns})')]
    ax.legend(handles=handles, loc='lower left', fontsize=10.5, frameon=False, handletextpad=0.25, labelspacing=0.22, borderaxespad=0.2)
    fig.tight_layout(pad=0.1)
    save_fig(fig, FIG / 'A' / 'Fig_diffKO_tertile_volcano', v_plot[['KO', 'KO_annotation', 'gene', 'Logit_beta_PCOS_vs_Healthy', 'qvalue', 'neglog10q', 'Direction', 'frac_diff_PCOS_minus_Healthy']])
    show = pd.concat([res[res.Direction == 'PCOS_enriched'].nsmallest(15, 'qvalue'), res[res.Direction == 'Healthy_enriched'].nsmallest(15, 'qvalue')])
    show = show.sort_values('Logit_beta_PCOS_vs_Healthy')
    fig, ax = plt.subplots(figsize=(4.8, 5.4))
    colors = [COL['PCOS'] if b > 0 else COL['Healthy'] for b in show.Logit_beta_PCOS_vs_Healthy]
    y = np.arange(len(show))
    ax.barh(y, show.Logit_beta_PCOS_vs_Healthy, color=colors, height=0.75, alpha=0.88)
    ax.set_yticks(y)
    ax.set_yticklabels(show.KO.tolist(), fontsize=7)
    ax.axvline(0, color='black', lw=0.7)
    ax.set_xlabel('Length-adjusted $\\beta$ (positive = PCOS tertile enriched)')
    ax.set_title(f'Top differential KOs (q<{Q_CUT} & |Δfrac|≥{FRAC_CUT})')
    save_fig(fig, FIG / 'B' / 'Fig_diffKO_tertile_top_bars', show[['KO', 'KO_annotation', 'Logit_beta_PCOS_vs_Healthy', 'qvalue', 'PCOS_tertile_frac', 'Healthy_tertile_frac', 'Direction']])
    rscript = 'Rscript'
    enr_skill = ''
    enr_out = OUT / 'enrich' / 'kegg'
    enr_out.mkdir(parents=True, exist_ok=True)
    if len(pcos_list) == 0 and len(heal_list) == 0:
        logging.warning('No focused KOs for enrichment; skip KEGG')
        pw_sig_q = pd.DataFrame()
    else:
        ko_files = []
        if pcos_list:
            ko_files.append(str(ENR / 'PCOS_enriched_tertile.KO'))
        if heal_list:
            ko_files.append(str(ENR / 'Healthy_enriched_tertile.KO'))
        cmd = [rscript, enr_skill, f"ko_files={','.join(ko_files)}", f'out_dir={enr_out}', 'run_pathway=TRUE', 'run_module=TRUE', 'pval_cutoff=0.05']
        logging.info('Running KEGG: %s', ' '.join(map(str, cmd)))
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logging.error('KEGG failed:\n%s\n%s', e.stderr[-3000:], e.stdout[-1000:])
            raise
        pw_all = enr_out / 'pathway' / 'pathway_all.tsv'
        pw_sig_q = pd.DataFrame()
        if pw_all.exists():
            pw = pd.read_csv(pw_all, sep='\t')
            if 'qvalue' not in pw.columns and 'p.adjust' in pw.columns:
                pw['qvalue'] = pw['p.adjust']
            mask_g = pw.apply(is_global_overview, axis=1)
            pw_f = pw.loc[~mask_g].copy()
            save_table(pw, TAB / 'kegg_pathway_all_raw')
            save_table(pw_f, TAB / 'kegg_pathway_all_noGlobal')
            pw_sig_q = pw_f[pw_f['qvalue'] < PATHWAY_Q].copy()
            save_table(pw_sig_q, TAB / 'kegg_pathway_qvalue_significant_focused')
            logging.info('Pathway rows: raw=%d noGlobal=%d q<%.2f=%d', len(pw), len(pw_f), PATHWAY_Q, len(pw_sig_q))
            if len(pw_sig_q) > 0:
                plot = pw_sig_q.copy()
                if 'GeneRatio_num' not in plot.columns and 'GeneRatio' in plot.columns:
                    plot['GeneRatio_num'] = plot['GeneRatio'].map(lambda s: float(str(s).split('/')[0]) / float(str(s).split('/')[1]))
                plot['neglog10q'] = -np.log10(plot['qvalue'].clip(lower=1e-300))
                parts = []
                for g in plot['Group'].unique():
                    parts.append(plot[plot.Group == g].nsmallest(12, 'qvalue'))
                top = pd.concat(parts).sort_values(['Group', 'qvalue']).reset_index(drop=True)
                top['ylab'] = [f"{d[:42]} ({('PCOS' if 'PCOS' in g else 'Healthy')})" for d, g in zip(top['Description'], top['Group'])]
                colors = [COL['PCOS'] if 'PCOS' in g else COL['Healthy'] for g in top['Group']]
                fig, ax = plt.subplots(figsize=(6.2, max(3.5, 0.28 * len(top) + 1.2)))
                ax.barh(np.arange(len(top)), top['neglog10q'], color=colors, alpha=0.85, height=0.75)
                ax.set_yticks(np.arange(len(top)))
                ax.set_yticklabels(top['ylab'], fontsize=6.5)
                ax.axvline(-np.log10(PATHWAY_Q), ls='--', lw=0.7, c='grey')
                ax.set_xlabel('$-\\log_{10}$(q-value)')
                ax.set_title(f'KEGG pathways q<{PATHWAY_Q} (no Global maps)\nTop{TOP_ENRICH} focused KOs / tertile extreme')
                save_fig(fig, FIG / 'C' / 'Fig_diffKO_tertile_kegg_pathway_qsig', top[['Group', 'ID', 'Description', 'GeneRatio', 'qvalue', 'neglog10q', 'Count']])
        mod_all = enr_out / 'module' / 'module_all.tsv'
        if mod_all.exists():
            md = pd.read_csv(mod_all, sep='\t')
            if 'qvalue' not in md.columns and 'p.adjust' in md.columns:
                md['qvalue'] = md['p.adjust']
            md_sig = md[md['qvalue'] < PATHWAY_Q].copy() if 'qvalue' in md.columns else md.iloc[0:0]
            save_table(md_sig, TAB / 'kegg_module_qvalue_significant_focused')
            save_table(md, TAB / 'kegg_module_all')
        bubble = enr_out / 'pathway' / 'pathway_bubble.png'
        if bubble.exists():
            for ext in ('.png', '.pdf', '.svg'):
                src = enr_out / 'pathway' / f'pathway_bubble{ext}'
                if src.exists():
                    shutil.copy(src, FIG / 'C' / f'Fig_diffKO_tertile_kegg_bubble{ext}')
    summary = {'design': 'Plan B: PCOS_tertile vs Healthy_tertile', 'n_PCOS_tertile_SGB': n_p, 'n_Healthy_tertile_SGB': n_h, 'Delta_Prev_cut_lo': float(cuts.iloc[0]), 'Delta_Prev_cut_hi': float(cuts.iloc[1]), 'n_KO_tested': int(len(res)), 'focus_rule': f'q<{Q_CUT} AND |frac_diff|>={FRAC_CUT}', 'n_PCOS_enriched_focused': n_pko, 'n_Healthy_enriched_focused': n_hko, 'n_PCOS_enriched_q_only': n_pko_q, 'n_Healthy_enriched_q_only': n_hko_q, 'enrich_topN': TOP_ENRICH, 'n_PCOS_in_enrich_list': len(pcos_list), 'n_Healthy_in_enrich_list': len(heal_list), 'pathway_q_cut': PATHWAY_Q, 'drop_Global_overview': True, 'n_pathway_q_focused': int(len(pw_sig_q)) if isinstance(pw_sig_q, pd.DataFrame) else 0, 'model': 'GLM binomial: KO_presence ~ group + log10(length) + completeness; q=BH-FDR', 'top5_PCOS': res[res.Direction == 'PCOS_enriched'].nsmallest(5, 'qvalue')[['KO', 'KO_annotation', 'Logit_beta_PCOS_vs_Healthy', 'qvalue', 'frac_diff_PCOS_minus_Healthy']].to_dict('records'), 'top5_Healthy': res[res.Direction == 'Healthy_enriched'].nsmallest(5, 'qvalue')[['KO', 'KO_annotation', 'Logit_beta_PCOS_vs_Healthy', 'qvalue', 'frac_diff_PCOS_minus_Healthy']].to_dict('records')}
    (TAB / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    lines = ['# Diff KO Plan B: ΔPrev tertile extremes (focused)', '', f'- Contrast: **PCOS_tertile** (n={n_p}) vs **Healthy_tertile** (n={n_h})', f'- ΔPrev cuts: **{cuts.iloc[0] * 100:.2f}%** / **{cuts.iloc[1] * 100:.2f}%**', f'- Model: binomial GLM `presence ~ group + log10(length) + completeness`; **q = BH-FDR**', f'- Focused call: **q<{Q_CUT}** and **|Δfrac|≥{FRAC_CUT}** → PCOS **{n_pko}**, Healthy **{n_hko}**', f'- q-only (no |Δfrac|): PCOS **{n_pko_q}**, Healthy **{n_hko_q}**', f'- Enrichment gene lists: Top{TOP_ENRICH}/side (PCOS {len(pcos_list)}, Healthy {len(heal_list)})', f"- KEGG pathways **q<{PATHWAY_Q}**, Global/overview maps dropped: **{summary['n_pathway_q_focused']}**", '', '## Top PCOS-enriched KOs (focused)']
    for r in summary['top5_PCOS']:
        lines.append(f"- {r['KO']} ({r['KO_annotation']}): β={r['Logit_beta_PCOS_vs_Healthy']:.2f}, Δfrac={r['frac_diff_PCOS_minus_Healthy']:.2f}, q={r['qvalue']:.2e}")
    lines += ['', '## Top Healthy-enriched KOs (focused)']
    for r in summary['top5_Healthy']:
        lines.append(f"- {r['KO']} ({r['KO_annotation']}): β={r['Logit_beta_PCOS_vs_Healthy']:.2f}, Δfrac={r['frac_diff_PCOS_minus_Healthy']:.2f}, q={r['qvalue']:.2e}")
    (OUT / 'analysis_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    logging.info('Done: %s', json.dumps(summary, indent=2, ensure_ascii=False)[:2000])
if __name__ == '__main__':
    main()
