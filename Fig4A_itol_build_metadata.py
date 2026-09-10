from pathlib import Path
import os
PKG_ROOT = Path(__file__).resolve().parents[1]
SHARED = PKG_ROOT / 'shared_input'
RSCRIPT = os.environ.get('RSCRIPT', 'Rscript')
from pathlib import Path
import pandas as pd
TREE = Path('.')
INFO = Path('.')
DIFF = ROOT / '15-diffMAG/tables/diffMAG_all_results.tsv'
ABUND = ROOT / 'sci-rev/sci_code/figure5/input/bin_abundance_table_tab.tsv'
GROUP = ROOT / 'sci-rev/sci_code/figure2/input/group.tsv'

def tip_labels_from_newick(path: Path) -> list[str]:
    import subprocess
    tmp = OUT / 'logs' / 'tips.txt'
    tmp.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(['Rscript', '-e', f'library(ape); t=read.tree("{path}"); writeLines(t$tip.label, "{tmp}")'])
    return tmp.read_text().strip().splitlines()

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'tables').mkdir(exist_ok=True)
    tips = tip_labels_from_newick(TREE)
    info = pd.read_csv(INFO, sep='\t')
    diff = pd.read_csv(DIFF, sep='\t')
    ab = pd.read_csv(ABUND, sep='\t', index_col=0)
    group = pd.read_csv(GROUP, sep='\t').set_index('Sample')
    common = [c for c in ab.columns if c in group.index]
    ab = ab[common]
    group = group.loc[common]
    pcos = group.index[group['Group'] == 'PCOS']
    heal = group.index[group['Group'] == 'Healthy']
    emap = diff.set_index('MAG')['Direction'].to_dict()
    info_map = info.set_index('ID')
    rows = []
    for tip in tips:
        if tip in info_map.index:
            bioproject = info_map.loc[tip, 'BIOPROJECT']
            origin_group = info_map.loc[tip, 'Group']
            phylum = str(info_map.loc[tip, 'P']).replace('p__', '')
            genus = str(info_map.loc[tip, 'G']).replace('g__', '').strip()
            if genus in ('', 'nan', 'None'):
                genus = 'Unknown'
            s_raw = str(info_map.loc[tip, 'S']) if 'S' in info_map.columns else ''
            species_status = 'Unknown_species' if str(s_raw).strip() in ('', 's__', 'nan', 'None') else 'Known_species'
        else:
            bioproject, origin_group, phylum, genus = ('Unknown', 'Unknown', 'Unknown', 'Unknown')
            species_status = 'Unknown_species'
        direction = emap.get(tip, 'NS')
        if tip in ab.index:
            overall = float((ab.loc[tip] > 0).mean())
            pcos_prev = float((ab.loc[tip, pcos] > 0).mean())
            heal_prev = float((ab.loc[tip, heal] > 0).mean())
        else:
            overall = pcos_prev = heal_prev = float('nan')
        rows.append({'ID': tip, 'BioProject': bioproject, 'Phylum': phylum, 'Genus': genus, 'Species_status': species_status, 'Enrichment': direction, 'Prevalence': overall, 'PCOS_prevalence': pcos_prev, 'Healthy_prevalence': heal_prev, 'Origin_sample_Group': origin_group})
    meta = pd.DataFrame(rows)
    meta.to_csv(OUT / 'tables/itol_metadata.tsv', sep='\t', index=False)
    meta.to_csv(OUT / 'tables/itol_metadata.csv', index=False)
    print(meta['Enrichment'].value_counts().to_dict())
    print('n=', len(meta))
if __name__ == '__main__':
    main()
