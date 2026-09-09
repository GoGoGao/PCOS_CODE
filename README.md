# PCOS metagenome analysis 

data.zip includes inputdata and scripts for the multi-cohort PCOS gut microbiome study (MetaPhlAn4, SGB, HUMAnN).

## `input/`

| File | Description |
|------|-------------|
| `metaphlan_species_abundance.tsv` | Species relative abundance (MetaPhlAn4) |
| `metaphlan_sample_group.tsv` | Sample ID, BioProject, PCOS/Healthy label |
| `metaphlan_genus_abundance.xls` | Genus relative abundance |
| `metaphlan_species_rename.tsv` | Species display names |
| `sgb_bin_abundance.tsv` | SGB abundance matrix |
| `sgb_genome_annotation.tsv` | SGB taxonomy and genome quality metrics |
| `sgb_ko_presence_matrix.xls` | SGB × KEGG Orthology presence/absence |
| `sgb_phylogeny_unrooted.nwk` | Midpoint-unrooted SGB phylogeny |
| `humann_ko_cpm_named.tsv.gz` | Community-level HUMAnN KO abundances (gzip) |

Raw reads: NCBI SRA `PRJNA530971`, `PRJNA549764`, `PRJNA791492`.

## `code/`

Analysis scripts named by manuscript figure (e.g. `Fig1B_S2_alpha_diversity.py`, `Fig3_species_network_ComBat.R`).
