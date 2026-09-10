from pathlib import Path
import os
PKG_ROOT = Path(__file__).resolve().parents[1]
SHARED = PKG_ROOT / "shared_input"
RSCRIPT = os.environ.get("RSCRIPT", "Rscript")

PATHS = {
  "species_abundance": SHARED / "metaphlan_species_abundance.tsv",
  "sample_group": SHARED / "metaphlan_sample_group.tsv",
  "preprocessed": SHARED / "metaphlan_preprocessed.pkl",
  "genus_abundance": SHARED / "metaphlan_genus_abundance.xls",
  "sgb_abundance": SHARED / "sgb_bin_abundance.tsv",
  "sgb_annotation": SHARED / "sgb_genome_annotation.tsv",
  "sgb_ko": SHARED / "sgb_ko_presence_matrix.xls",
  "humann_ko": SHARED / "humann_ko_cpm_named.tsv.gz",
}
