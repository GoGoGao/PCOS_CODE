from pathlib import Path
import os
PKG_ROOT = Path(__file__).resolve().parents[1]
SHARED = PKG_ROOT / 'shared_input'
RSCRIPT = os.environ.get('RSCRIPT', 'Rscript')
ROOT = PKG_ROOT / 'output' / 'Fig2B_cMD3'
CMD3 = SHARED
RA_DIR = SHARED / 'cMD3_relative_abundance'
META_FP = SHARED / 'cMD3_sample_metadata_minimal.tsv'
OUT = ROOT
OUT_TAB = OUT / 'tables'
OUT_FIG = OUT / 'figures'
OUT_PD = OUT / 'plotdata'
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
for d in (OUT_TAB, OUT_FIG, OUT_PD, OUT_FIG / 'panels', OUT_FIG / 'panels' / 'A', OUT_FIG / 'panels' / 'B', OUT_FIG / 'panels' / 'C', OUT_FIG / 'composite'):
    d.mkdir(parents=True, exist_ok=True)
SPECIES = ['Phocaeicola vulgatus', 'Bacteroides uniformis']
MIN_N = 20
PSEUDO = 1e-06
ALPHA = 0.05
CLASS_MAP = {'T2D': 'metabolic', 'IGT': 'metabolic', 'IBD': 'inflammatory', 'CRC': 'neoplastic', 'ACVD': 'cardiovascular', 'cirrhosis': 'hepatic', 'ME/CFS': 'other', 'schizophrenia': 'neuropsychiatric', 'migraine': 'other', 'STH': 'infectious'}
CLASS_COLOR = {'metabolic': '#2A9D8F', 'inflammatory': '#457B9D', 'neoplastic': '#6D597A', 'cardiovascular': '#E9C46A', 'hepatic': '#B56576', 'neuropsychiatric': '#264653', 'infectious': '#8D99AE', 'other': '#ADB5BD'}
mpl.rcParams.update({'font.family': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif', 'serif'], 'font.weight': 'normal', 'axes.titleweight': 'normal', 'axes.labelweight': 'normal', 'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.5, 'figure.dpi': 150, 'savefig.dpi': 600, 'savefig.bbox': 'tight', 'axes.linewidth': 0.7, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'axes.unicode_minus': False})

def dump_both(df: pd.DataFrame, stem: Path):
    df.to_csv(f'{stem}.tsv', sep='\t', index=False)
    df.to_csv(f'{stem}.csv', index=False)

def save_all(fig, stem: Path):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'svg', 'png', 'jpg'):
        kw = {'bbox_inches': 'tight', 'pad_inches': 0.02}
        if ext in ('png', 'jpg'):
            kw['dpi'] = 600
        fig.savefig(f'{stem}.{ext}', **kw)

def resolve_cols(mdf: pd.DataFrame, mat: pd.DataFrame) -> list[str]:
    for fld in ('sample_id', 'NCBI_accession', 'subject_id'):
        if fld not in mdf.columns:
            continue
        ids = mdf[fld].dropna().astype(str).unique().tolist()
        hit = [c for c in ids if c in mat.columns]
        if hit:
            return hit
    return []

def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    nt = 0
    lt = 0
    for x in a:
        nt += np.sum(x > b)
        lt += np.sum(x < b)
    n = len(a) * len(b)
    return float((nt - lt) / n) if n else np.nan

def discover_eligible(base: pd.DataFrame) -> pd.DataFrame:
    avail = {p.name.replace('_wide.tsv', '') for p in RA_DIR.glob('*_wide.tsv')}
    rows = []
    for st, g in base.groupby('study_name'):
        if st not in avail:
            continue
        mat = pd.read_csv(RA_DIR / f'{st}_wide.tsv', sep='\t', index_col=0)
        mat.columns = mat.columns.astype(str)
        ctrl_md = g.loc[g['study_condition'].astype(str) == 'control']
        ctrl_cols = resolve_cols(ctrl_md, mat)
        if len(ctrl_cols) < MIN_N:
            continue
        for cond, gg in g.groupby(g['study_condition'].astype(str)):
            if cond in ('control', 'nan', 'NA', 'None'):
                continue
            cl = cond.lower()
            if 'obes' in cl or cond.upper() == 'BMI':
                continue
            case_cols = resolve_cols(gg, mat)
            if len(case_cols) < MIN_N:
                continue
            rows.append({'study': st, 'condition': cond, 'class': CLASS_MAP.get(cond, 'other'), 'n_case': len(case_cols), 'n_ctrl': len(ctrl_cols), 'has_Pv': 'Phocaeicola vulgatus' in mat.index, 'has_Bu': 'Bacteroides uniformis' in mat.index})
    return pd.DataFrame(rows).sort_values(['condition', 'study']).reset_index(drop=True)

def run_one(mat: pd.DataFrame, sp: str, case_cols: list[str], ctrl_cols: list[str]):
    if sp not in mat.index:
        return None
    a = mat.loc[sp, case_cols].astype(float).to_numpy()
    b = mat.loc[sp, ctrl_cols].astype(float).to_numpy()
    try:
        u, p = mannwhitneyu(a, b, alternative='two-sided')
    except ValueError:
        u, p = (np.nan, np.nan)
    med_c, med_k = (float(np.median(a)), float(np.median(b)))
    mean_c, mean_k = (float(np.mean(a)), float(np.mean(b)))
    log10_med_diff = np.log10(med_c + PSEUDO) - np.log10(med_k + PSEUDO)
    delta = cliffs_delta(a, b)
    if med_c > med_k:
        direction = 'case_higher'
    elif med_c < med_k:
        direction = 'case_lower'
    else:
        direction = 'case_higher' if mean_c >= mean_k else 'case_lower'
    return {'species': sp, 'n_case': len(case_cols), 'n_ctrl': len(ctrl_cols), 'median_case': med_c, 'median_ctrl': med_k, 'mean_case': mean_c, 'mean_ctrl': mean_k, 'log10_median_diff': log10_med_diff, 'cliffs_delta': delta, 'U': u, 'p_mannwhitney': p, 'direction': direction, 'sig_nominal': bool(p < ALPHA) if pd.notna(p) else False}

def collect_long_abundance(mat: pd.DataFrame, sp: str, case_cols, ctrl_cols, study, condition):
    rows = []
    for sid in case_cols:
        rows.append({'species': sp, 'study': study, 'condition': condition, 'group': 'Case', 'sample_id': sid, 'abundance': float(mat.loc[sp, sid])})
    for sid in ctrl_cols:
        rows.append({'species': sp, 'study': study, 'condition': condition, 'group': 'Control', 'sample_id': sid, 'abundance': float(mat.loc[sp, sid])})
    return rows

def analyze(base: pd.DataFrame, eligible: pd.DataFrame):
    stats_rows = []
    long_rows = []
    for _, r in eligible.iterrows():
        st, cond = (r['study'], r['condition'])
        mat = pd.read_csv(RA_DIR / f'{st}_wide.tsv', sep='\t', index_col=0)
        mat.columns = mat.columns.astype(str)
        md = base.loc[base['study_name'] == st]
        case_md = md.loc[md['study_condition'].astype(str) == cond]
        ctrl_md = md.loc[md['study_condition'].astype(str) == 'control']
        case_cols = resolve_cols(case_md, mat)
        ctrl_cols = resolve_cols(ctrl_md, mat)
        assert len(case_cols) >= MIN_N and len(ctrl_cols) >= MIN_N
        for sp in SPECIES:
            if sp not in mat.index:
                continue
            res = run_one(mat, sp, case_cols, ctrl_cols)
            if res is None:
                continue
            res.update({'study': st, 'condition': cond, 'class': r['class'], 'panel_id': f'{cond}|{st}'})
            stats_rows.append(res)
            long_rows.extend(collect_long_abundance(mat, sp, case_cols, ctrl_cols, st, cond))
    stats = pd.DataFrame(stats_rows)
    stats['p_fdr_within_species'] = np.nan
    for sp, idx in stats.groupby('species').groups.items():
        p = stats.loc[idx, 'p_mannwhitney'].to_numpy()
        mask = np.isfinite(p)
        adj = np.full(len(p), np.nan)
        if mask.sum():
            adj[mask] = multipletests(p[mask], method='fdr_bh')[1]
        stats.loc[idx, 'p_fdr_within_species'] = adj
    stats['sig_fdr'] = stats['p_fdr_within_species'] < ALPHA
    long_df = pd.DataFrame(long_rows)
    return (stats, long_df)

def summarize(stats: pd.DataFrame) -> pd.DataFrame:
    out = []
    for sp, g in stats.groupby('species'):
        met = g[g['class'] == 'metabolic']
        out.append({'species': sp, 'n_tests': len(g), 'n_sig_nominal': int(g['sig_nominal'].sum()), 'n_sig_fdr': int(g['sig_fdr'].sum()), 'n_sig_case_higher': int((g['sig_nominal'] & (g['direction'] == 'case_higher')).sum()), 'n_sig_case_lower': int((g['sig_nominal'] & (g['direction'] == 'case_lower')).sum()), 'n_metabolic_tests': len(met), 'n_metabolic_sig_nominal': int(met['sig_nominal'].sum()), 'n_metabolic_sig_case_higher': int((met['sig_nominal'] & (met['direction'] == 'case_higher')).sum()), 'median_cliffs_delta_all': float(g['cliffs_delta'].median()), 'median_cliffs_delta_metabolic': float(met['cliffs_delta'].median()) if len(met) else np.nan})
    return pd.DataFrame(out)

def panel_label(study: str, condition: str, n_case: int, n_ctrl: int) -> str:
    return f'{condition}\n{study}\n(n={n_case}/{n_ctrl})'

def _draw_forest_panel(ax, g: pd.DataFrame, letter: str, species: str, show_legend: bool, show_ylabels: bool=True):
    g = g.sort_values(['class', 'condition', 'study']).reset_index(drop=True)
    y = np.arange(len(g))
    colors = [CLASS_COLOR.get(c, '#888') for c in g['class']]
    sig = g['sig_nominal'].astype(bool).to_numpy()
    ax.axvline(0, color='#BBBBBB', ls='--', lw=1.0, zorder=0)
    if (~sig).any():
        ax.scatter(g.loc[~sig, 'cliffs_delta'], y[~sig], c=[colors[i] for i in np.where(~sig)[0]], s=70, edgecolors='#555555', linewidths=0.6, zorder=3)
    if sig.any():
        ax.scatter(g.loc[sig, 'cliffs_delta'], y[sig], c=[colors[i] for i in np.where(sig)[0]], s=160, edgecolors='#B71C1C', linewidths=2.0, zorder=4)
    for yi, r in g.iterrows():
        x = float(r['cliffs_delta'])
        p = float(r['p_mannwhitney'])
        is_sig = bool(r['sig_nominal'])
        label = f'* P={p:.2g}' if is_sig else f'P={p:.2g}'
        ax.text(x + 0.04, yi, label, va='center', ha='left', fontsize=14 if is_sig else 12, color='#B71C1C' if is_sig else '#777777')
    ax.set_yticks(y)
    if show_ylabels:
        labels = [f'{r.condition} | {r.study} ({r.n_case}/{r.n_ctrl})' for r in g.itertuples()]
        ax.set_yticklabels(labels, fontsize=14)
        ax.tick_params(axis='y', length=3)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis='y', length=0)
    ax.invert_yaxis()
    ax.set_xlim(-1.05, 1.25)
    ax.set_ylim(len(g) - 0.55, -0.55)
    ax.set_xlabel("Cliff's delta", fontsize=16)
    ax.set_title('')
    ax.text(0.0, 1.02, f'{letter}  ', transform=ax.transAxes, ha='left', va='bottom', fontsize=18, fontstyle='normal', clip_on=False)
    ax.text(0.06, 1.02, species, transform=ax.transAxes, ha='left', va='bottom', fontsize=18, fontstyle='italic', clip_on=False)
    ax.tick_params(axis='x', labelsize=14)
    if show_legend:
        present = [c for c in CLASS_COLOR if c in set(g['class'])]
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=CLASS_COLOR[c], markeredgecolor='#333', markersize=10, label=c) for c in present]
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#B71C1C', markeredgecolor='#B71C1C', markersize=10, label='* P<0.05'))
        ax.legend(handles=handles, loc='lower right', fontsize=12, title='class', title_fontsize=12, frameon=False, borderpad=0.2, labelspacing=0.25, handletextpad=0.35)

def plot_forest(stats: pd.DataFrame):
    figs = {}
    for i, sp in enumerate(SPECIES):
        g = stats.loc[stats['species'] == sp].copy()
        n = len(g)
        fig, ax = plt.subplots(figsize=(7.2, max(4.0, 0.42 * n + 0.6)))
        _draw_forest_panel(ax, g, chr(65 + i), sp, show_legend=True)
        fig.subplots_adjust(left=0.42, right=0.98, top=0.93, bottom=0.1)
        letter = chr(65 + i)
        save_all(fig, OUT_FIG / 'panels' / letter / f'Fig_perStudy_{letter}')
        pd_df = g.sort_values(['class', 'condition', 'study'])[['species', 'study', 'condition', 'class', 'n_case', 'n_ctrl', 'cliffs_delta', 'log10_median_diff', 'p_mannwhitney', 'p_fdr_within_species', 'direction', 'sig_nominal', 'sig_fdr']].copy()
        dump_both(pd_df, OUT_PD / f'Fig_perStudy_{letter}_forest_plotdata')
        dump_both(pd_df, OUT_FIG / 'panels' / letter / f'Fig_perStudy_{letter}_plotdata')
        figs[sp] = (fig, ax, g)
        plt.close(fig)
    return figs

def plot_heatmap(stats: pd.DataFrame):
    g = stats.copy()
    g['row'] = g['condition'] + ' | ' + g['study']
    order = g[['row', 'class', 'condition', 'study']].drop_duplicates().sort_values(['class', 'condition', 'study'])
    mat = g.pivot_table(index='row', columns='species', values='cliffs_delta', aggfunc='first').reindex(order['row']).reindex(columns=SPECIES)
    dump_both(g[['species', 'study', 'condition', 'class', 'cliffs_delta', 'p_mannwhitney', 'sig_nominal', 'direction']], OUT_PD / 'Fig_perStudy_C_heatmap_plotdata')
    return mat

def plot_composite(stats: pd.DataFrame):
    n_row = int(stats['species'].value_counts().max())
    fig_h = max(5.0, 0.46 * n_row + 0.8)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, fig_h))
    for i, sp in enumerate(SPECIES):
        g = stats.loc[stats['species'] == sp].copy()
        _draw_forest_panel(axes[i], g, chr(65 + i), sp, show_legend=i == 1, show_ylabels=i == 0)
    fig.subplots_adjust(left=0.3, right=0.995, top=0.94, bottom=0.08, wspace=0.08)
    save_all(fig, OUT_FIG / 'composite' / 'Fig_perStudy_twoSpecies')
    dump_both(stats, OUT_PD / 'Fig_perStudy_twoSpecies_composite_plotdata')
    dump_both(stats, OUT_FIG / 'composite' / 'Fig_perStudy_twoSpecies_plotdata')
    plt.close(fig)

def main():
    print('Loading metadata...')
    meta = pd.read_csv(META_FP, sep='\t', low_memory=False)
    base = meta.loc[(meta['body_site'] == 'stool') & (meta['gender'] == 'female') & (meta['age_category'] == 'adult')].copy()
    eligible = discover_eligible(base)
    dump_both(eligible, OUT_TAB / 'Q0_eligible_study_condition')
    print('Eligible study×condition pairs:')
    print(eligible.to_string(index=False))
    stats, long_df = analyze(base, eligible)
    dump_both(stats, OUT_TAB / 'per_study_species_stats')
    dump_both(long_df, OUT_TAB / 'per_study_abundance_long')
    dump_both(long_df, OUT_PD / 'per_study_abundance_long')
    overview = summarize(stats)
    dump_both(overview, OUT_TAB / 'per_study_overview')
    met = stats.loc[stats['class'] == 'metabolic'].copy()
    dump_both(met, OUT_TAB / 'per_study_metabolic_subset')
    plot_forest(stats)
    plot_heatmap(stats)
    plot_composite(stats)
    lines = ['# Per-study official-disease species benchmark', '', 'Rules: adult female stool; official study_condition; case & control n≥20;', 'no batch correction; no obesity/BMI stratification.', '', '## Eligible contrasts', eligible.to_string(index=False), '', '## Overview', overview.to_string(index=False), '', '## All tests (sorted)', stats.sort_values(['species', 'p_mannwhitney'])[['species', 'condition', 'study', 'n_case', 'n_ctrl', 'median_case', 'median_ctrl', 'cliffs_delta', 'p_mannwhitney', 'p_fdr_within_species', 'direction', 'sig_nominal']].to_string(index=False), '']
    (OUT / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    print('Wrote', OUT)
if __name__ == '__main__':
    main()
for _d in (OUT_TAB, OUT_FIG, OUT_PD, OUT_FIG / 'panels', OUT_FIG / 'panels' / 'A', OUT_FIG / 'panels' / 'B', OUT_FIG / 'panels' / 'C', OUT_FIG / 'composite'):
    _d.mkdir(parents=True, exist_ok=True)
