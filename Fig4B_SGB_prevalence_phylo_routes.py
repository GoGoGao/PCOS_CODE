from pathlib import Path
import os
PKG_ROOT = Path(__file__).resolve().parents[1]
SHARED = PKG_ROOT / 'shared_input'
RSCRIPT = os.environ.get('RSCRIPT', 'Rscript')
import json
import logging
import os
import subprocess
import warnings
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')
IN = ROOT / 'input'
TMP = ROOT / 'logs' / 'tmp'
for d in [TAB, PLOT, FIG, LOG, TMP]:
    d.mkdir(parents=True, exist_ok=True)
for sub in 'ABCDEFGH':
    (FIG / sub).mkdir(parents=True, exist_ok=True)
(FIG / 'composite').mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', handlers=[logging.FileHandler(LOG / 'run.log', mode='w'), logging.StreamHandler()])
log = logging.getLogger('jinhua')
if os.path.exists(FONT):
    font_manager.fontManager.addfont(FONT)
mpl.rcParams.update({'font.family': 'Times New Roman', 'font.weight': 'normal', 'axes.titleweight': 'normal', 'axes.labelweight': 'normal', 'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7.5, 'figure.dpi': 300, 'savefig.dpi': 300, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
NPG = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85']

def save_table(df: pd.DataFrame, path: Path, index: bool=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(f'{path}.tsv', sep='\t', index=index)
    df.to_csv(f'{path}.csv', index=index)

def save_fig(fig, base: Path, plot_df: pd.DataFrame | None=None):
    base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in ('pdf', 'svg', 'png', 'jpg'):
        fig.savefig(f'{base}.{fmt}', format=fmt, bbox_inches='tight', pad_inches=0.05)
    if plot_df is not None:
        save_table(plot_df, PLOT / f'{base.name}_plotdata')
    plt.close(fig)

def strip_ko(name: str) -> str:
    return str(name).split(':')[0].strip()

def load_tips_from_tree(tree_path: Path) -> list[str]:
    tip_file = TMP / 'tips.txt'
    subprocess.check_call(['Rscript', '-e', f'library(ape); t=read.tree("{tree_path}"); writeLines(t$tip.label, "{tip_file}")'])
    return tip_file.read_text().strip().splitlines()

def export_cophenetic(tree_path: Path, tips: list[str]) -> pd.DataFrame:
    out = TMP / 'cophenetic.tsv'
    rcode = f'''\nlibrary(ape)\nlibrary(phangorn)\nt <- read.tree("{tree_path}")\nif (!is.rooted(t)) t <- midpoint(t)\nd <- cophenetic(t)\ntips <- readLines("{TMP / 'tips.txt'}")\nd <- d[tips, tips, drop=FALSE]\nwrite.table(d, "{out}", sep="\\t", quote=FALSE)\n'''
    subprocess.check_call(['Rscript', '-e', rcode])
    dist = pd.read_csv(out, sep='\t', index_col=0)
    dist.index = dist.index.astype(str)
    dist.columns = dist.columns.astype(str)
    return dist.loc[tips, tips]

def load_ko_binary(tips: list[str]) -> pd.DataFrame:
    ko = pd.read_csv(IN / 'total.KO.xls', sep='\t', index_col=0)
    ko.index = [strip_ko(i) for i in ko.index]
    cols = [c for c in tips if c in ko.columns]
    mat = (ko[cols] > 0).astype(np.uint8)
    prev = mat.mean(axis=1)
    mat = mat.loc[(prev >= 0.02) & (prev <= 0.98)]
    return mat.T

def build_tip_meta(tips: list[str]) -> pd.DataFrame:
    info = pd.read_csv(IN / 'sgb_info_comb.tsv', sep='\t').set_index('ID')
    diff = pd.read_csv(IN / 'diffMAG_all_results.tsv', sep='\t').set_index('MAG')
    ab = pd.read_csv(IN / 'bin_abundance_table_tab.tsv', sep='\t', index_col=0)
    group = pd.read_csv(IN / 'group.tsv', sep='\t').set_index('Sample')
    core = pd.read_csv(IN / 'SGB_core_list.tsv', sep='\t')
    xb = pd.read_csv(IN / 'SGB_crossbatch_consistent.tsv', sep='\t')
    hub = pd.read_csv(IN / 'top10_hubs_annotated.tsv', sep='\t')
    common = [c for c in ab.columns if c in group.index]
    ab = ab[common]
    group = group.loc[common]
    pcos = group.index[group['Group'] == 'PCOS']
    heal = group.index[group['Group'] == 'Healthy']
    core_map = core.set_index('SGB_ID')['Core_status'].to_dict()
    xb_p = set(xb.loc[xb['In_PCOS_all_cohorts'] == 1, 'SGB_ID'])
    xb_h = set(xb.loc[xb['In_Healthy_all_cohorts'] == 1, 'SGB_ID'])
    hub_nodes = set(hub['node'])
    hub_recurrent = set(hub.loc[hub['recurrence_class'].str.contains('Recurrent', na=False), 'node'])
    rows = []
    for tip in tips:
        if tip in info.index:
            r = info.loc[tip]
            bioproject = r['BIOPROJECT']
            origin = r['Group']
            phylum = str(r['P']).replace('p__', '')
            family = str(r['F']).replace('f__', '')
            genus = str(r['G']).replace('g__', '')
            species = str(r['S']).replace('s__', '') if pd.notna(r['S']) else ''
            completeness = float(r['completeness'])
            contamination = float(r['contamination'])
            length = float(r['length'])
            n50 = float(r['N50'])
            unknown = species.strip() in ('', 'nan', 'None')
        else:
            bioproject = origin = phylum = family = genus = species = 'Unknown'
            completeness = contamination = length = n50 = np.nan
            unknown = True
        if tip in diff.index:
            direction = diff.loc[tip, 'Direction']
            lmm_coef = float(diff.loc[tip, 'LMM_coef'])
            lmm_fdr = float(diff.loc[tip, 'LMM_FDR'])
            sig_lmm = bool(diff.loc[tip, 'Significant_LMM'])
            dir_consistent = bool(diff.loc[tip, 'Direction_Consistent']) if 'Direction_Consistent' in diff.columns else False
        else:
            direction, lmm_coef, lmm_fdr, sig_lmm, dir_consistent = ('NS', np.nan, np.nan, False, False)
        if tip in ab.index:
            prev = float((ab.loc[tip] > 0).mean())
            prev_p = float((ab.loc[tip, pcos] > 0).mean())
            prev_h = float((ab.loc[tip, heal] > 0).mean())
            mean_ab = float(np.log1p(ab.loc[tip]).mean())
        else:
            prev = prev_p = prev_h = mean_ab = np.nan
        rows.append({'SGB_ID': tip, 'BioProject': bioproject, 'Origin_Group': origin, 'Phylum': phylum, 'Family': family, 'Genus': genus, 'Species': species if not unknown else '', 'Species_status': 'Unknown_species' if unknown else 'Known_species', 'completeness': completeness, 'contamination': contamination, 'length': length, 'length_Mb': length / 1000000.0 if pd.notna(length) else np.nan, 'N50': n50, 'HQ': int(pd.notna(completeness) and completeness >= 90 and (contamination <= 5)), 'Enrichment': direction, 'LMM_coef': lmm_coef, 'LMM_FDR': lmm_fdr, 'Significant_LMM': int(sig_lmm), 'Direction_Consistent': int(dir_consistent), 'Prevalence': prev, 'PCOS_prevalence': prev_p, 'Healthy_prevalence': prev_h, 'Delta_prev': prev_p - prev_h if pd.notna(prev_p) else np.nan, 'mean_log1p_ab': mean_ab, 'Core_status': core_map.get(tip, 'Non-core'), 'Is_core': int(tip in core_map), 'Crossbatch_PCOS': int(tip in xb_p), 'Crossbatch_Healthy': int(tip in xb_h), 'Crossbatch_any': int(tip in xb_p or tip in xb_h), 'Is_hub': int(tip in hub_nodes), 'Is_recurrent_hub': int(tip in hub_recurrent)})
    meta = pd.DataFrame(rows).set_index('SGB_ID')
    return (meta, ab, group)

def add_ko_features(meta: pd.DataFrame, ko_bin: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tips = [t for t in meta.index if t in ko_bin.index]
    X = ko_bin.loc[tips]
    richness = X.sum(axis=1).astype(float)
    length_Mb = meta.loc[tips, 'length_Mb']
    completeness = meta.loc[tips, 'completeness']
    density = richness / length_Mb.replace(0, np.nan)
    df_reg = pd.DataFrame({'richness': richness, 'length_Mb': length_Mb, 'completeness': completeness}).dropna()
    from numpy.linalg import lstsq
    A = np.column_stack([np.ones(len(df_reg)), df_reg['length_Mb'].values, df_reg['completeness'].values])
    coef, _, _, _ = lstsq(A, df_reg['richness'].values, rcond=None)
    pred = A @ coef
    resid = pd.Series(df_reg['richness'].values - pred, index=df_reg.index)
    Xn = X.loc[tips]
    Xn = Xn.loc[:, Xn.std(axis=0) > 0]
    pca = PCA(n_components=5, random_state=0)
    scores = pca.fit_transform(Xn.values.astype(float))
    pc_df = pd.DataFrame(scores, index=tips, columns=[f'KO_PC{i + 1}' for i in range(5)])
    var_exp = pd.Series(pca.explained_variance_ratio_, index=pc_df.columns)
    for col in pc_df.columns:
        y = pc_df[col]
        ok = y.index.intersection(df_reg.index)
        A2 = np.column_stack([np.ones(len(ok)), meta.loc[ok, 'length_Mb'].values, meta.loc[ok, 'completeness'].values])
        c2, _, _, _ = lstsq(A2, y.loc[ok].values, rcond=None)
        pc_df.loc[ok, f'{col}_resid'] = y.loc[ok].values - A2 @ c2
    meta = meta.copy()
    meta.loc[tips, 'KO_richness'] = richness
    meta.loc[tips, 'KO_density_perMb'] = density
    meta.loc[tips, 'KO_richness_resid'] = resid.reindex(tips)
    for c in pc_df.columns:
        meta.loc[tips, c] = pc_df[c]
    save_table(var_exp.rename('variance_explained').reset_index().rename(columns={'index': 'PC'}), TAB / 'KO_PCA_variance')
    return (meta, X)

def vcv_from_dist(D: np.ndarray) -> np.ndarray:
    n = D.shape[0]
    raise NotImplementedError

def export_vcv(tree_path: Path, tips: list[str]) -> pd.DataFrame:
    out = TMP / 'vcv.tsv'
    rcode = f'''\nlibrary(ape)\nlibrary(phangorn)\nt <- read.tree("{tree_path}")\nif (!is.rooted(t)) t <- midpoint(t)\ntips <- readLines("{TMP / 'tips.txt'}")\nt <- keep.tip(t, tips)\nV <- vcv(t)\nV <- V[tips, tips, drop=FALSE]\nwrite.table(V, "{out}", sep="\\t", quote=FALSE)\n'''
    subprocess.check_call(['Rscript', '-e', rcode])
    V = pd.read_csv(out, sep='\t', index_col=0)
    V.index = V.index.astype(str)
    V.columns = V.columns.astype(str)
    return V.loc[tips, tips]

def pagel_lambda(x: pd.Series, V: pd.DataFrame) -> dict:
    tips = [t for t in V.index if t in x.index and pd.notna(x.loc[t])]
    if len(tips) < 20:
        return {'trait': x.name, 'n': len(tips), 'lambda': np.nan, 'logL': np.nan, 'logL0': np.nan, 'logL1': np.nan, 'p_vs0': np.nan}
    y = x.loc[tips].values.astype(float)
    C = V.loc[tips, tips].values.astype(float)
    n = len(y)

    def nll(lam):
        lam = float(np.clip(lam, 1e-08, 1.0))
        Cv = C.copy()
        eye = np.eye(n)
        Cv = lam * Cv + (1 - lam) * np.diag(np.diag(Cv))
        Cv = Cv + np.eye(n) * 1e-08 * np.mean(np.diag(Cv))
        try:
            L = np.linalg.cholesky(Cv)
        except np.linalg.LinAlgError:
            return 1000000000000.0
        ones = np.ones(n)
        Li = np.linalg.solve(L, eye)
        inv = Li.T @ Li
        a = float(ones @ inv @ y / (ones @ inv @ ones))
        r = y - a
        sig2 = float(r @ inv @ r) / n
        if sig2 <= 0:
            return 1000000000000.0
        logdet = 2 * np.sum(np.log(np.diag(L)))
        return 0.5 * (n * np.log(2 * np.pi * sig2) + logdet + n)
    res = minimize_scalar(nll, bounds=(1e-08, 1.0), method='bounded', options={'xatol': 1e-05})
    lam_hat = float(res.x)
    nll_hat = float(res.fun)
    nll0 = float(nll(1e-08))
    nll1 = float(nll(1.0))
    lr = 2 * (nll0 - nll_hat)
    p = float(stats.chi2.sf(max(lr, 0), df=1))
    return {'trait': x.name, 'n': n, 'lambda': lam_hat, 'logL': -nll_hat, 'logL0': -nll0, 'logL1': -nll1, 'LR_vs0': lr, 'p_vs0': p}

def bloomberg_K_via_R(trait_tsv: Path, tree_path: Path, out_tsv: Path) -> pd.DataFrame:
    rcode = f'\nlibrary(ape)\nlibrary(phangorn)\nlibrary(picante)\nt <- read.tree("{tree_path}")\nif (!is.rooted(t)) t <- midpoint(t)\ntr <- read.table("{trait_tsv}", sep="\\t", header=TRUE, row.names=1, check.names=FALSE)\ntips <- intersect(rownames(tr), t$tip.label)\nt <- keep.tip(t, tips)\ntr <- tr[tips, , drop=FALSE]\nres <- data.frame(trait=character(), K=double(), pic.variance.obs=double(),\n                  pic.variance.rnd.mean=double(), Z=double(), P=double(), n=integer(),\n                  stringsAsFactors=FALSE)\nfor (col in colnames(tr)) {{\n  x <- tr[, col]\n  names(x) <- rownames(tr)\n  ok <- is.finite(x)\n  if (sum(ok) < 20) next\n  tt <- keep.tip(t, names(x)[ok])\n  xx <- x[ok]\n  xx <- xx[tt$tip.label]\n  ps <- try(phylosignal(xx, tt, reps=999), silent=TRUE)\n  if (inherits(ps, "try-error")) next\n  res <- rbind(res, data.frame(trait=col, K=ps$K, pic.variance.obs=ps$PIC.variance.obs,\n                               pic.variance.rnd.mean=ps$PIC.variance.rnd.mean,\n                               Z=ps$PIC.variance.Z, P=ps$PIC.variance.P, n=sum(ok)))\n}}\nwrite.table(res, "{out_tsv}", sep="\\t", quote=FALSE, row.names=FALSE)\n'
    subprocess.check_call(['Rscript', '-e', rcode])
    return pd.read_csv(out_tsv, sep='\t')

def binary_mpd_ses(positive_tips: list[str], D: pd.DataFrame, nperm: int=999, label: str='') -> dict:
    tips = list(D.index)
    pos = [t for t in positive_tips if t in tips]
    k = len(pos)
    if k < 3:
        return {'trait': label, 'n_positive': k, 'mpd_obs': np.nan, 'mpd_null_mean': np.nan, 'mpd_null_sd': np.nan, 'SES': np.nan, 'p_lower': np.nan, 'p_upper': np.nan}
    sub = D.loc[pos, pos].values
    iu = np.triu_indices(k, 1)
    obs = float(sub[iu].mean())
    null = np.empty(nperm)
    for i in range(nperm):
        rnd = np.random.choice(tips, size=k, replace=False)
        s = D.loc[rnd, rnd].values
        null[i] = s[iu].mean()
    mu, sd = (float(null.mean()), float(null.std(ddof=1)))
    ses = (obs - mu) / sd if sd > 0 else np.nan
    p_lower = (np.sum(null <= obs) + 1) / (nperm + 1)
    p_upper = (np.sum(null >= obs) + 1) / (nperm + 1)
    return {'trait': label, 'n_positive': k, 'mpd_obs': obs, 'mpd_null_mean': mu, 'mpd_null_sd': sd, 'SES': ses, 'p_lower': p_lower, 'p_upper': p_upper}

def mantel_test(A: np.ndarray, B: np.ndarray, nperm: int=999) -> dict:
    n = A.shape[0]
    au = A[np.triu_indices(n, 1)]
    bu = B[np.triu_indices(n, 1)]
    r_obs = float(stats.spearmanr(au, bu).correlation)
    count = 0
    idx = np.arange(n)
    for _ in range(nperm):
        np.random.shuffle(idx)
        Bp = B[idx][:, idx]
        bu_p = Bp[np.triu_indices(n, 1)]
        r = float(stats.spearmanr(au, bu_p).correlation)
        if abs(r) >= abs(r_obs):
            count += 1
    p = (count + 1) / (nperm + 1)
    return {'r': r_obs, 'p': p, 'n': n, 'nperm': nperm}

def partial_mantel(A: np.ndarray, B: np.ndarray, C: np.ndarray, nperm: int=999) -> dict:
    n = A.shape[0]
    iu = np.triu_indices(n, 1)
    a, b, c = (A[iu], B[iu], C[iu])

    def resid(y, x):
        slope, intercept, *_ = stats.linregress(x, y)
        return y - (intercept + slope * x)
    ar, br, cr = (stats.rankdata(a), stats.rankdata(b), stats.rankdata(c))
    a_c = resid(ar, cr)
    b_c = resid(br, cr)
    r_obs = float(np.corrcoef(a_c, b_c)[0, 1])
    count = 0
    idx = np.arange(n)
    for _ in range(nperm):
        np.random.shuffle(idx)
        Bp = B[idx][:, idx]
        b2 = stats.rankdata(Bp[iu])
        b2_c = resid(b2, cr)
        r = float(np.corrcoef(a_c, b2_c)[0, 1])
        if abs(r) >= abs(r_obs):
            count += 1
    p = (count + 1) / (nperm + 1)
    return {'r': r_obs, 'p': p, 'n': n, 'nperm': nperm}

def correlation_distance_from_abundance(ab: pd.DataFrame, tips: list[str]) -> pd.DataFrame:
    X = ab.loc[[t for t in tips if t in ab.index]].astype(float)
    X = X.loc[X.sum(axis=1) > 0]
    tips2 = list(X.index)
    R = X.rank(axis=1).values
    R = R - R.mean(axis=1, keepdims=True)
    denom = np.sqrt((R ** 2).sum(axis=1, keepdims=True))
    denom[denom == 0] = 1
    Rn = R / denom
    corr = Rn @ Rn.T
    corr = np.clip(corr, -1, 1)
    dist = 1 - np.abs(corr)
    np.fill_diagonal(dist, 0)
    return pd.DataFrame(dist, index=tips2, columns=tips2)

def jaccard_distance(ko_bin: pd.DataFrame, tips: list[str]) -> pd.DataFrame:
    X = ko_bin.loc[[t for t in tips if t in ko_bin.index]].astype(bool).values
    tips2 = [t for t in tips if t in ko_bin.index]
    d = squareform(pdist(X, metric='jaccard'))
    return pd.DataFrame(d, index=tips2, columns=tips2)

def length_distance(meta: pd.DataFrame, tips: list[str]) -> pd.DataFrame:
    x = meta.loc[tips, 'length_Mb'].astype(float)
    v = x.values.reshape(-1, 1)
    d = np.abs(v - v.T)
    return pd.DataFrame(d, index=tips, columns=tips)

def genus_collapse_enrichment(meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g, sub in meta.groupby('Genus'):
        if g in ('', 'nan', 'Unclassified', 'Unknown') or pd.isna(g):
            continue
        n = len(sub)
        n_p = int((sub['Enrichment'] == 'PCOS_enriched').sum())
        n_h = int((sub['Enrichment'] == 'Healthy_enriched').sum())
        n_sig = n_p + n_h
        n_xb = int(sub['Crossbatch_any'].sum())
        n_hub = int(sub['Is_hub'].sum())
        n_unk = int((sub['Species_status'] == 'Unknown_species').sum())
        rows.append({'Genus': g, 'Phylum': sub['Phylum'].mode().iloc[0] if len(sub) else '', 'n_SGB': n, 'n_PCOS_enriched': n_p, 'n_Healthy_enriched': n_h, 'n_sig': n_sig, 'frac_sig': n_sig / n, 'n_crossbatch': n_xb, 'n_hub': n_hub, 'n_unknown': n_unk, 'mean_Delta_prev': float(sub['Delta_prev'].mean()), 'mean_KO_PC1_resid': float(sub['KO_PC1_resid'].mean()) if 'KO_PC1_resid' in sub else np.nan, 'mean_completeness': float(sub['completeness'].mean()), 'mean_length_Mb': float(sub['length_Mb'].mean())})
    out = pd.DataFrame(rows).sort_values(['n_sig', 'n_SGB'], ascending=False)
    return out

def pgls_like_delta_vs_ko(meta: pd.DataFrame, V: pd.DataFrame) -> pd.DataFrame:
    cols = ['Delta_prev', 'KO_PC1_resid', 'length_Mb', 'completeness']
    tips = [t for t in V.index if all((pd.notna(meta.loc[t, c]) for c in cols))]
    y = meta.loc[tips, 'Delta_prev'].values.astype(float)
    X = np.column_stack([np.ones(len(tips)), meta.loc[tips, 'KO_PC1_resid'].values, meta.loc[tips, 'length_Mb'].values, meta.loc[tips, 'completeness'].values])
    names = ['Intercept', 'KO_PC1_resid', 'length_Mb', 'completeness']
    C = V.loc[tips, tips].values.astype(float)
    C = C + np.eye(len(tips)) * 1e-06 * np.mean(np.diag(C))

    def gls(Cmat):
        try:
            L = np.linalg.cholesky(Cmat)
        except np.linalg.LinAlgError:
            return None
        y2 = np.linalg.solve(L, y)
        X2 = np.linalg.solve(L, X)
        beta, *_ = np.linalg.lstsq(X2, y2, rcond=None)
        resid = y2 - X2 @ beta
        dof = max(len(y) - X.shape[1], 1)
        sig2 = float(resid @ resid) / dof
        XtX_inv = np.linalg.inv(X2.T @ X2)
        se = np.sqrt(np.diag(XtX_inv) * sig2)
        tstat = beta / se
        pval = 2 * stats.t.sf(np.abs(tstat), df=dof)
        return (beta, se, tstat, pval, sig2)
    rows = []
    for model, Cmat in [('BM_lambda1', C), ('star_lambda0', np.diag(np.diag(C)))]:
        out = gls(Cmat)
        if out is None:
            continue
        beta, se, tstat, pval, sig2 = out
        for i, nm in enumerate(names):
            rows.append({'model': model, 'term': nm, 'beta': float(beta[i]), 'se': float(se[i]), 't': float(tstat[i]), 'p': float(pval[i]), 'n': len(tips), 'sigma2': sig2})
    return pd.DataFrame(rows)

def hub_phylogenetic_uniqueness(meta: pd.DataFrame, D: pd.DataFrame) -> pd.DataFrame:
    tips = list(D.index)
    mean_dist = D.mean(axis=1)
    rows = []
    for tip in tips:
        rows.append({'SGB_ID': tip, 'mean_phylo_dist': float(mean_dist.loc[tip]), 'Is_hub': int(meta.loc[tip, 'Is_hub']) if tip in meta.index else 0, 'Is_recurrent_hub': int(meta.loc[tip, 'Is_recurrent_hub']) if tip in meta.index else 0, 'Prevalence': meta.loc[tip, 'Prevalence'] if tip in meta.index else np.nan, 'length_Mb': meta.loc[tip, 'length_Mb'] if tip in meta.index else np.nan, 'KO_PC1_resid': meta.loc[tip, 'KO_PC1_resid'] if tip in meta.index else np.nan, 'Species_status': meta.loc[tip, 'Species_status'] if tip in meta.index else ''})
    return pd.DataFrame(rows)

def fig_A_lambda_bar(lambda_df: pd.DataFrame):
    df = lambda_df.copy().sort_values('lambda', ascending=True)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    colors = [NPG[0] if p < 0.05 else NPG[5] for p in df['p_vs0']]
    ax.barh(df['trait'], df['lambda'], color=colors, edgecolor='none')
    ax.axvline(0, color='grey', lw=0.5)
    ax.set_xlabel("Pagel's λ")
    ax.set_ylabel('')
    ax.set_xlim(0, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    save_fig(fig, FIG / 'A' / 'Fig_phylo_signal_lambda', df)

def fig_B_K_bar(K_df: pd.DataFrame):
    df = K_df.copy().sort_values('K', ascending=True)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    colors = [NPG[2] if p < 0.05 else NPG[5] for p in df['P']]
    ax.barh(df['trait'], df['K'], color=colors, edgecolor='none')
    ax.set_xlabel("Blomberg's K")
    ax.set_ylabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    save_fig(fig, FIG / 'B' / 'Fig_phylo_signal_K', df)

def fig_C_binary_ses(bin_df: pd.DataFrame):
    df = bin_df.copy().sort_values('SES')
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    colors = [NPG[3] if min(p, q) < 0.05 else NPG[5] for p, q in zip(df['p_lower'], df['p_upper'])]
    ax.barh(df['trait'], df['SES'], color=colors, edgecolor='none')
    ax.axvline(0, color='black', lw=0.6)
    ax.set_xlabel('MPD SES (negative = clustered)')
    ax.set_ylabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    save_fig(fig, FIG / 'C' / 'Fig_binary_MPD_SES', df)

def fig_D_mantel_heatmap(mantel_df: pd.DataFrame):
    labels = ['Phylo', 'KO', 'Cooccur', 'Length']
    M = pd.DataFrame(np.nan, index=labels, columns=labels)
    P = pd.DataFrame(np.nan, index=labels, columns=labels)
    key = {('Phylo', 'KO'): ('Phylo', 'KO'), ('Phylo', 'Cooccur'): ('Phylo', 'Cooccur'), ('Phylo', 'Length'): ('Phylo', 'Length'), ('KO', 'Cooccur'): ('KO', 'Cooccur'), ('KO', 'Length'): ('KO', 'Length'), ('Cooccur', 'Length'): ('Cooccur', 'Length')}
    for _, r in mantel_df.iterrows():
        if r['type'] != 'simple':
            continue
        a, b = (r['matrix_A'], r['matrix_B'])
        if (a, b) in [(x, y) for x, y in key]:
            M.loc[a, b] = r['r']
            M.loc[b, a] = r['r']
            P.loc[a, b] = r['p']
            P.loc[b, a] = r['p']
    for lab in labels:
        M.loc[lab, lab] = 1.0
        P.loc[lab, lab] = 0.0
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    sns.heatmap(M, annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-0.2, vmax=0.8, square=True, ax=ax, cbar_kws={'label': 'Spearman Mantel r', 'shrink': 0.8})
    ax.set_title('Distance Mantel')
    save_fig(fig, FIG / 'D' / 'Fig_Mantel_heatmap', M.reset_index().rename(columns={'index': 'matrix'}))

def fig_E_partial_mantel(pm: pd.DataFrame):
    df = pm.copy()
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    y = np.arange(len(df))
    colors = [NPG[0] if p < 0.05 else NPG[5] for p in df['p']]
    ax.barh(y, df['r'], color=colors, edgecolor='none')
    ax.set_yticks(y)
    ax.set_yticklabels(df['test'])
    ax.axvline(0, color='black', lw=0.6)
    ax.set_xlabel('Partial Mantel r (Spearman)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    save_fig(fig, FIG / 'E' / 'Fig_partial_Mantel', df)

def fig_F_hub_uniqueness(hub_df: pd.DataFrame):
    plot = hub_df.copy()
    plot['hub_class'] = np.where(plot['Is_recurrent_hub'] == 1, 'Recurrent hub', np.where(plot['Is_hub'] == 1, 'Hub', 'Non-hub'))
    order = ['Non-hub', 'Hub', 'Recurrent hub']
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    sns.boxplot(data=plot, x='hub_class', y='mean_phylo_dist', order=order, palette=[NPG[5], NPG[1], NPG[0]], ax=ax, fliersize=1, linewidth=0.8)
    ax.set_xlabel('')
    ax.set_ylabel('Mean phylogenetic distance')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for label in ax.get_xticklabels():
        label.set_rotation(15)
    save_fig(fig, FIG / 'F' / 'Fig_hub_phylo_uniqueness', plot)

def fig_G_genus_collapse(genus_df: pd.DataFrame):
    df = genus_df.head(15).copy()
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    ax.barh(df['Genus'][::-1], df['n_PCOS_enriched'][::-1], color=NPG[0], label='PCOS_enriched')
    ax.barh(df['Genus'][::-1], df['n_Healthy_enriched'][::-1], left=df['n_PCOS_enriched'][::-1], color=NPG[2], label='Healthy_enriched')
    ax.set_xlabel('Number of differentially enriched SGBs')
    ax.legend(frameon=False, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    save_fig(fig, FIG / 'G' / 'Fig_genus_collapse_enrichment', df)

def fig_H_delta_vs_ko(meta: pd.DataFrame):
    df = meta[['Delta_prev', 'KO_PC1_resid', 'HQ', 'length_Mb']].dropna().copy()
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.scatter(df['KO_PC1_resid'], df['Delta_prev'], c=df['length_Mb'], cmap='viridis', s=8, alpha=0.7, linewidths=0)
    r, p = stats.spearmanr(df['KO_PC1_resid'], df['Delta_prev'])
    ax.set_xlabel('KO PC1 residual (size-adjusted)')
    ax.set_ylabel('Δ prevalence (PCOS − Healthy)')
    ax.text(0.05, 0.95, f'Spearman r={r:.2f}\nP={p:.1e}', transform=ax.transAxes, va='top', ha='left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    cb = fig.colorbar(ax.collections[0], ax=ax, shrink=0.8)
    cb.set_label('Genome length (Mb)')
    save_fig(fig, FIG / 'H' / 'Fig_DeltaPrev_vs_KOPC1resid', df)

def make_composite():
    from PIL import Image
    panels = [FIG / 'A' / 'Fig_phylo_signal_lambda.jpg', FIG / 'B' / 'Fig_phylo_signal_K.jpg', FIG / 'C' / 'Fig_binary_MPD_SES.jpg', FIG / 'D' / 'Fig_Mantel_heatmap.jpg', FIG / 'E' / 'Fig_partial_Mantel.jpg', FIG / 'F' / 'Fig_hub_phylo_uniqueness.jpg', FIG / 'G' / 'Fig_genus_collapse_enrichment.jpg', FIG / 'H' / 'Fig_DeltaPrev_vs_KOPC1resid.jpg']
    imgs = [Image.open(p) for p in panels if p.exists()]
    if len(imgs) < 4:
        return
    w = max((im.width for im in imgs))
    h = max((im.height for im in imgs))
    imgs = [im.resize((w, h)) for im in imgs]
    while len(imgs) < 8:
        imgs.append(Image.new('RGB', (w, h), (255, 255, 255)))
    canvas = Image.new('RGB', (w * 4, h * 2), (255, 255, 255))
    for i, im in enumerate(imgs[:8]):
        r, c = divmod(i, 4)
        canvas.paste(im, (c * w, r * h))
    base = FIG / 'composite' / 'Fig_jinhuaSGB_composite'
    canvas.save(f'{base}.jpg', quality=95)
    canvas.save(f'{base}.png')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(canvas)
    ax.axis('off')
    fig.savefig(f'{base}.pdf', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(f'{base}.svg', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)

def main():
    np.random.seed(42)
    tree = IN / 'SGB.unrooted.tree.nwk'
    log.info('Loading tips')
    tips = load_tips_from_tree(tree)
    log.info('n tips=%d', len(tips))
    log.info('Building tip metadata')
    meta, ab, group = build_tip_meta(tips)
    log.info('Loading KO matrix')
    ko_bin = load_ko_binary(tips)
    log.info('KO matrix tips=%d KOs=%d', ko_bin.shape[0], ko_bin.shape[1])
    meta, ko_bin = add_ko_features(meta, ko_bin)
    save_table(meta.reset_index(), TAB / 'tip_metadata_full')
    log.info('Exporting cophenetic + VCV')
    D = export_cophenetic(tree, tips)
    V = export_vcv(tree, tips)
    save_table(D, TAB / 'cophenetic_distance', index=True)
    log.info('Route1: continuous phylogenetic signal')
    cont_traits = ['Prevalence', 'Delta_prev', 'LMM_coef', 'KO_richness', 'KO_richness_resid', 'KO_density_perMb', 'KO_PC1', 'KO_PC2', 'KO_PC1_resid', 'KO_PC2_resid', 'length_Mb', 'completeness']
    trait_export = meta[cont_traits].copy()
    trait_path = TMP / 'cont_traits.tsv'
    trait_export.to_csv(trait_path, sep='\t')
    K_df = bloomberg_K_via_R(trait_path, tree, TMP / 'blomberg_K.tsv')
    save_table(K_df, TAB / 'route1_Blomberg_K')
    lambda_rows = []
    for col in cont_traits:
        res = pagel_lambda(meta[col].rename(col), V)
        lambda_rows.append(res)
    lambda_df = pd.DataFrame(lambda_rows)
    ok = lambda_df['p_vs0'].notna()
    lambda_df.loc[ok, 'p_vs0_BH'] = multipletests(lambda_df.loc[ok, 'p_vs0'], method='fdr_bh')[1]
    save_table(lambda_df, TAB / 'route1_Pagel_lambda')
    hq_tips = meta.index[meta['HQ'] == 1].tolist()
    V_hq = V.loc[[t for t in hq_tips if t in V.index], [t for t in hq_tips if t in V.index]]
    hq_rows = []
    for col in ['Prevalence', 'Delta_prev', 'KO_PC1_resid', 'KO_richness_resid']:
        hq_rows.append(pagel_lambda(meta.loc[V_hq.index, col].rename(col + '_HQ'), V_hq))
    save_table(pd.DataFrame(hq_rows), TAB / 'route1_Pagel_lambda_HQ')
    log.info('Route1: binary MPD SES')
    binary_defs = {'PCOS_enriched': meta.index[meta['Enrichment'] == 'PCOS_enriched'].tolist(), 'Healthy_enriched': meta.index[meta['Enrichment'] == 'Healthy_enriched'].tolist(), 'Unknown_species': meta.index[meta['Species_status'] == 'Unknown_species'].tolist(), 'Is_core': meta.index[meta['Is_core'] == 1].tolist(), 'Crossbatch_any': meta.index[meta['Crossbatch_any'] == 1].tolist(), 'Is_hub': meta.index[meta['Is_hub'] == 1].tolist(), 'Is_recurrent_hub': meta.index[meta['Is_recurrent_hub'] == 1].tolist(), 'HQ': meta.index[meta['HQ'] == 1].tolist()}
    bin_rows = [binary_mpd_ses(v, D, nperm=999, label=k) for k, v in binary_defs.items()]
    bin_df = pd.DataFrame(bin_rows)
    save_table(bin_df, TAB / 'route1_binary_MPD_SES')
    bp_rows = []
    for bp, sub in meta.groupby('BioProject'):
        tips_bp = [t for t in sub.index if t in V.index]
        if len(tips_bp) < 30:
            continue
        Vbp = V.loc[tips_bp, tips_bp]
        for col in ['Prevalence', 'Delta_prev', 'KO_PC1_resid']:
            r = pagel_lambda(meta.loc[tips_bp, col].rename(f'{col}|{bp}'), Vbp)
            r['BioProject'] = bp
            r['trait_base'] = col
            bp_rows.append(r)
    save_table(pd.DataFrame(bp_rows), TAB / 'route1_Pagel_lambda_by_BioProject')
    log.info('Route2: distance matrices + Mantel')
    ko_dist = jaccard_distance(ko_bin, tips)
    co_dist = correlation_distance_from_abundance(ab, tips)
    common = sorted(set(D.index) & set(ko_dist.index) & set(co_dist.index) & set(meta.index))
    Dp = D.loc[common, common].values
    KOd = ko_dist.loc[common, common].values
    COd = co_dist.loc[common, common].values
    Ld = length_distance(meta, common).values
    mantel_rows = []
    pairs = [('Phylo', 'KO', Dp, KOd), ('Phylo', 'Cooccur', Dp, COd), ('Phylo', 'Length', Dp, Ld), ('KO', 'Cooccur', KOd, COd), ('KO', 'Length', KOd, Ld), ('Cooccur', 'Length', COd, Ld)]
    for a, b, A, B in pairs:
        log.info('Mantel %s vs %s', a, b)
        res = mantel_test(A, B, nperm=999)
        mantel_rows.append({'type': 'simple', 'matrix_A': a, 'matrix_B': b, 'control': '', **res})
    partial_defs = [('Phylo vs KO | Length', Dp, KOd, Ld), ('Phylo vs Cooccur | Length', Dp, COd, Ld), ('Phylo vs Cooccur | KO', Dp, COd, KOd), ('KO vs Cooccur | Phylo', KOd, COd, Dp), ('KO vs Cooccur | Length', KOd, COd, Ld), ('KO vs Cooccur | Phylo+Length_proxy', KOd, COd, (Dp + Ld) / 2)]
    partial_defs = partial_defs[:5]
    pm_rows = []
    for name, A, B, C in partial_defs:
        log.info('Partial Mantel %s', name)
        res = partial_mantel(A, B, C, nperm=999)
        mantel_rows.append({'type': 'partial', 'matrix_A': name.split(' vs ')[0], 'matrix_B': name.split(' vs ')[1].split(' | ')[0], 'control': name.split('| ')[-1], **res})
        pm_rows.append({'test': name, **res})
    mantel_df = pd.DataFrame(mantel_rows)
    save_table(mantel_df, TAB / 'route2_Mantel_all')
    pm_df = pd.DataFrame(pm_rows)
    save_table(pm_df, TAB / 'route2_partial_Mantel')
    hq_common = [t for t in common if meta.loc[t, 'HQ'] == 1]
    log.info('HQ Mantel n=%d', len(hq_common))
    hq_m = []
    Dp_h = D.loc[hq_common, hq_common].values
    KOd_h = ko_dist.loc[hq_common, hq_common].values
    COd_h = co_dist.loc[hq_common, hq_common].values
    Ld_h = length_distance(meta, hq_common).values
    for a, b, A, B in [('Phylo', 'KO', Dp_h, KOd_h), ('Phylo', 'Cooccur', Dp_h, COd_h), ('KO', 'Cooccur', KOd_h, COd_h)]:
        res = mantel_test(A, B, nperm=999)
        hq_m.append({'type': 'simple_HQ', 'matrix_A': a, 'matrix_B': b, **res})
    res = partial_mantel(KOd_h, COd_h, Dp_h, nperm=999)
    hq_m.append({'type': 'partial_HQ', 'matrix_A': 'KO', 'matrix_B': 'Cooccur', 'control': 'Phylo', **res})
    res = partial_mantel(Dp_h, KOd_h, Ld_h, nperm=999)
    hq_m.append({'type': 'partial_HQ', 'matrix_A': 'Phylo', 'matrix_B': 'KO', 'control': 'Length', **res})
    save_table(pd.DataFrame(hq_m), TAB / 'route2_Mantel_HQ')
    hub_df = hub_phylogenetic_uniqueness(meta, D)
    save_table(hub_df, TAB / 'route2_hub_phylo_uniqueness')
    a = hub_df.loc[hub_df['Is_hub'] == 1, 'mean_phylo_dist']
    b = hub_df.loc[hub_df['Is_hub'] == 0, 'mean_phylo_dist']
    hub_stat = {'hub_n': int(len(a)), 'nonhub_n': int(len(b)), 'hub_mean': float(a.mean()), 'nonhub_mean': float(b.mean()), 'MWU_p': float(stats.mannwhitneyu(a, b, alternative='two-sided').pvalue) if len(a) and len(b) else np.nan}
    save_table(pd.DataFrame([hub_stat]), TAB / 'route2_hub_vs_nonhub_stats')
    log.info('Route3: genus collapse + PGLS-like')
    genus_df = genus_collapse_enrichment(meta)
    save_table(genus_df, TAB / 'route3_genus_collapse')
    n_tip_p = int((meta['Enrichment'] == 'PCOS_enriched').sum())
    n_tip_h = int((meta['Enrichment'] == 'Healthy_enriched').sum())
    n_gen_p = int((genus_df['n_PCOS_enriched'] > 0).sum())
    n_gen_h = int((genus_df['n_Healthy_enriched'] > 0).sum())
    collapse_summary = pd.DataFrame([{'n_PCOS_enriched_tips': n_tip_p, 'n_Healthy_enriched_tips': n_tip_h, 'n_genera_with_PCOS_enriched': n_gen_p, 'n_genera_with_Healthy_enriched': n_gen_h, 'n_genera_total': int(len(genus_df)), 'max_PCOS_in_one_genus': int(genus_df['n_PCOS_enriched'].max()) if len(genus_df) else 0, 'max_Healthy_in_one_genus': int(genus_df['n_Healthy_enriched'].max()) if len(genus_df) else 0}])
    save_table(collapse_summary, TAB / 'route3_enrichment_tip_vs_genus')
    pgls_df = pgls_like_delta_vs_ko(meta, V)
    save_table(pgls_df, TAB / 'route3_PGLS_DeltaPrev_vs_KOPC1')
    xb_tips = meta.index[meta['Crossbatch_any'] == 1].tolist()
    xb_ses = binary_mpd_ses(xb_tips, D, nperm=999, label='Crossbatch_any')
    pair_rows = []
    for g, sub in meta.groupby('Genus'):
        unk = sub.index[sub['Species_status'] == 'Unknown_species'].tolist()
        kn = sub.index[sub['Species_status'] == 'Known_species'].tolist()
        if not unk or not kn:
            continue
        for u in unk:
            dvec = D.loc[u, kn] if u in D.index else None
            if dvec is None:
                continue
            nearest = dvec.idxmin()
            pair_rows.append({'Unknown': u, 'Nearest_known': nearest, 'Genus': g, 'phylo_dist': float(dvec.loc[nearest]), 'Unknown_prev': meta.loc[u, 'Prevalence'], 'Known_prev': meta.loc[nearest, 'Prevalence'], 'Unknown_Delta_prev': meta.loc[u, 'Delta_prev'], 'Known_Delta_prev': meta.loc[nearest, 'Delta_prev'], 'Unknown_KO_PC1_resid': meta.loc[u, 'KO_PC1_resid'], 'Known_KO_PC1_resid': meta.loc[nearest, 'KO_PC1_resid'], 'Unknown_completeness': meta.loc[u, 'completeness'], 'Known_completeness': meta.loc[nearest, 'completeness'], 'Unknown_length_Mb': meta.loc[u, 'length_Mb'], 'Known_length_Mb': meta.loc[nearest, 'length_Mb']})
    pair_df = pd.DataFrame(pair_rows)
    save_table(pair_df, TAB / 'route3_Unknown_vs_nearestKnown')
    if len(pair_df):
        pair_stats = []
        for col_u, col_k, name in [('Unknown_prev', 'Known_prev', 'Prevalence'), ('Unknown_Delta_prev', 'Known_Delta_prev', 'Delta_prev'), ('Unknown_KO_PC1_resid', 'Known_KO_PC1_resid', 'KO_PC1_resid'), ('Unknown_completeness', 'Known_completeness', 'completeness'), ('Unknown_length_Mb', 'Known_length_Mb', 'length_Mb')]:
            d = (pair_df[col_u] - pair_df[col_k]).dropna()
            if len(d) >= 3:
                w = stats.wilcoxon(d)
                pair_stats.append({'metric': name, 'n_pairs': len(d), 'median_diff_U_minus_K': float(d.median()), 'wilcoxon_p': float(w.pvalue)})
        save_table(pd.DataFrame(pair_stats), TAB / 'route3_Unknown_vs_Known_paired_stats')
    save_table(pd.DataFrame([xb_ses]), TAB / 'route3_crossbatch_MPD_SES')
    clade_ann = genus_df.loc[genus_df['n_crossbatch'] > 0].head(20).copy()
    save_table(clade_ann, TAB / 'route3_crossbatch_genera_KO_annotation')
    log.info('Plotting')
    rename = {'Prevalence': 'Prevalence', 'Delta_prev': 'Δ prevalence', 'LMM_coef': 'LMM coefficient', 'KO_richness': 'KO richness', 'KO_richness_resid': 'KO richness (size-adj.)', 'KO_density_perMb': 'KO density / Mb', 'KO_PC1': 'KO PC1', 'KO_PC2': 'KO PC2', 'KO_PC1_resid': 'KO PC1 (size-adj.)', 'KO_PC2_resid': 'KO PC2 (size-adj.)', 'length_Mb': 'Genome length', 'completeness': 'Completeness'}
    lam_plot = lambda_df.copy()
    lam_plot['trait'] = lam_plot['trait'].map(lambda x: rename.get(x, x))
    fig_A_lambda_bar(lam_plot)
    K_plot = K_df.copy()
    K_plot['trait'] = K_plot['trait'].map(lambda x: rename.get(x, x))
    fig_B_K_bar(K_plot)
    fig_C_binary_ses(bin_df)
    fig_D_mantel_heatmap(mantel_df)
    fig_E_partial_mantel(pm_df)
    fig_F_hub_uniqueness(hub_df)
    fig_G_genus_collapse(genus_df)
    fig_H_delta_vs_ko(meta)
    try:
        make_composite()
    except Exception as e:
        log.warning('composite failed: %s', e)
    summary = {'n_tips': len(tips), 'n_KO_filtered': int(ko_bin.shape[1]), 'route1_lambda_sig_BH05': lambda_df.loc[lambda_df['p_vs0_BH'] < 0.05, 'trait'].tolist() if 'p_vs0_BH' in lambda_df else [], 'route1_K_sig_P05': K_df.loc[K_df['P'] < 0.05, 'trait'].tolist() if len(K_df) else [], 'route1_binary_clustered_SES_neg_p05': bin_df.loc[(bin_df['SES'] < 0) & (bin_df['p_lower'] < 0.05), 'trait'].tolist(), 'route2_mantel': mantel_df.to_dict(orient='records'), 'route3_tip_vs_genus': collapse_summary.to_dict(orient='records')[0], 'hub_stat': hub_stat}
    (TAB / 'analysis_summary.json').write_text(json.dumps(summary, indent=2, default=str))
    log.info('Done')
    return summary
if __name__ == '__main__':
    main()
