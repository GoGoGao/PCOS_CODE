pkg_root <- Sys.getenv('GITHUB_DATA_ROOT', unset=NA)
if (is.na(pkg_root)) {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep('^--file=', args, value = TRUE)
  if (length(file_arg)) pkg_root <- normalizePath(file.path(dirname(sub('^--file=', '', file_arg)), '..'))
  else pkg_root <- normalizePath('..')
}
shared <- file.path(pkg_root, 'shared_input')
out_root <- file.path(pkg_root, 'output')
suppressPackageStartupMessages({
  library(ape)
  library(dplyr)
})
root <- ""
tree_in <- ""
meta_f <- file.path(root, "tables/itol_metadata.tsv")
itol_dir <- file.path(root, "itol")
dir.create(itol_dir, recursive = TRUE, showWarnings = FALSE)
tree <- read.tree(tree_in)
meta <- read.delim(meta_f, stringsAsFactors = FALSE, check.names = FALSE)
meta <- meta %>% filter(ID %in% tree$tip.label)
stopifnot(nrow(meta) == Ntip(tree))
message("Tips: ", Ntip(tree), "; metadata rows: ", nrow(meta))
write.tree(tree, file.path(itol_dir, "SGB.unrooted.tree.nwk"))
file.copy(tree_in, file.path(itol_dir, "SGB.unrooted.tree"), overwrite = TRUE)
bp_colors <- c(
  PRJNA530971 = "#3C5488",
  PRJNA549764 = "#E64B35",
  PRJNA791492 = "#00A087"
)
enr_colors <- c(
  PCOS_enriched = "#E64B35",
  Healthy_enriched = "#4DBBD5",
  NS = "#D9D9D9"
)
write_colorstrip <- function(df, id_col, group_col, colors, outfile,
                             dataset_label, legend_title, strip_width = 25) {
  con <- file(outfile, "w")
  on.exit(close(con), add = TRUE)
  writeLines("DATASET_COLORSTRIP", con)
  writeLines("SEPARATOR TAB", con)
  writeLines(paste0("DATASET_LABEL\t", dataset_label), con)
  writeLines("COLOR\t#000000", con)
  writeLines(paste0("STRIP_WIDTH\t", strip_width), con)
  writeLines("MARGIN\t5", con)
  writeLines("BORDER_WIDTH\t0", con)
  writeLines("SHOW_INTERNAL\t0", con)
  labs <- names(colors)
  writeLines(paste0("LEGEND_TITLE\t", legend_title), con)
  writeLines(paste0("LEGEND_SHAPES\t", paste(rep(1, length(labs)), collapse = "\t")), con)
  writeLines(paste0("LEGEND_COLORS\t", paste(unname(colors[labs]), collapse = "\t")), con)
  writeLines(paste0("LEGEND_LABELS\t", paste(labs, collapse = "\t")), con)
  writeLines("DATA", con)
  for (i in seq_len(nrow(df))) {
    id <- df[[id_col]][i]
    g <- as.character(df[[group_col]][i])
    col <- if (g %in% names(colors)) colors[[g]] else "#999999"
    writeLines(paste(id, col, g, sep = "\t"), con)
  }
  message("Wrote ", outfile)
}
write_gradient <- function(df, id_col, value_col, outfile, dataset_label,
                           color_min = "#F7FBFF", color_mid = "#6BAED6",
                           color_max = "#08306B") {
  vals <- df[[value_col]]
  con <- file(outfile, "w")
  on.exit(close(con), add = TRUE)
  writeLines("DATASET_GRADIENT", con)
  writeLines("SEPARATOR TAB", con)
  writeLines(paste0("DATASET_LABEL\t", dataset_label), con)
  writeLines("COLOR\t#000000", con)
  writeLines(paste0("COLOR_MIN\t", color_min), con)
  writeLines(paste0("COLOR_MID\t", color_mid), con)
  writeLines(paste0("COLOR_MAX\t", color_max), con)
  writeLines(sprintf("USER_MIN_VALUE\t%.6f", min(vals, na.rm = TRUE)), con)
  writeLines(sprintf("USER_MID_VALUE\t%.6f", stats::median(vals, na.rm = TRUE)), con)
  writeLines(sprintf("USER_MAX_VALUE\t%.6f", max(vals, na.rm = TRUE)), con)
  writeLines("MARGIN\t5", con)
  writeLines("STRIP_WIDTH\t30", con)
  writeLines("SHOW_VALUES\t0", con)
  writeLines("DATA", con)
  for (i in seq_len(nrow(df))) {
    v <- df[[value_col]][i]
    if (is.na(v)) next
    writeLines(sprintf("%s\t%.8f", df[[id_col]][i], v), con)
  }
  message("Wrote ", outfile)
}
phylum_levels <- c(
  "Bacillota_A", "Bacteroidota", "Actinomycetota", "Bacillota_I",
  "Bacillota_C", "Pseudomonadota", "Bacillota", "Desulfobacterota",
  "Verrucomicrobiota", "Cyanobacteriota", "Fusobacteriota", "Patescibacteria"
)
phylum_palette <- c(
  "#E64B35", "#4DBBD5", "#00A087", "#3C5488",
  "#F39B7F", "#8491B4", "#91D1C2", "#DC0000",
  "#7E6148", "#B09C85", "#F4A261", "#2A9D8F"
)
obs_phyla <- unique(as.character(meta$Phylum))
extra <- setdiff(obs_phyla, phylum_levels)
if (length(extra) > 0) {
  phylum_levels <- c(phylum_levels, sort(extra))
  extra_cols <- grDevices::hcl.colors(max(length(extra), 1), "Dark 3")
  phylum_palette <- c(phylum_palette, extra_cols[seq_along(extra)])
}
phylum_colors <- setNames(phylum_palette[seq_along(phylum_levels)], phylum_levels)
phylum_colors <- phylum_colors[names(phylum_colors) %in% obs_phyla]
write_colorstrip(
  meta, "ID", "Phylum", phylum_colors,
  file.path(itol_dir, "itol_Phylum_COLORSTRIP.txt"),
  "Phylum", "Phylum"
)
write_colorstrip(
  meta, "ID", "BioProject", bp_colors,
  file.path(itol_dir, "itol_BioProject_COLORSTRIP.txt"),
  "BioProject", "BioProject"
)
write_colorstrip(
  meta, "ID", "Enrichment", enr_colors,
  file.path(itol_dir, "itol_Enrichment_COLORSTRIP.txt"),
  "Enrichment", "Enrichment_LMM_FDR0.05"
)
enr_cn_map <- c(
  PCOS_enriched = "PCOS富集",
  Healthy_enriched = "Healthy富集",
  NS = "无显著性差异"
)
meta$Enrichment_CN <- unname(enr_cn_map[meta$Enrichment])
enr_cn_colors <- c(
  `PCOS富集` = "#E64B35",
  `Healthy富集` = "#4DBBD5",
  `无显著性差异` = "#D9D9D9"
)
write_colorstrip(
  meta, "ID", "Enrichment_CN", enr_cn_colors,
  file.path(itol_dir, "itol_Enrichment_CN_COLORSTRIP.txt"),
  "Enrichment_CN", "富集方向_LMM_FDR0.05"
)
write_gradient(
  meta, "ID", "Prevalence",
  file.path(itol_dir, "itol_Prevalence_GRADIENT.txt"),
  "Prevalence"
)
write_gradient(
  meta, "ID", "PCOS_prevalence",
  file.path(itol_dir, "itol_PCOS_prevalence_GRADIENT.txt"),
  "PCOS_prevalence",
  color_min = "#FFF5F0", color_mid = "#FB6A4A", color_max = "#A50F15"
)
write_gradient(
  meta, "ID", "Healthy_prevalence",
  file.path(itol_dir, "itol_Healthy_prevalence_GRADIENT.txt"),
  "Healthy_prevalence",
  color_min = "#F7FBFF", color_mid = "#6BAED6", color_max = "#08306B"
)
meta$Prevalence_diff <- meta$PCOS_prevalence - meta$Healthy_prevalence
meta$DeltaPrev_pct <- meta$PCOS_prevalence * 100 - meta$Healthy_prevalence * 100
dp <- meta$DeltaPrev_pct
scale_dp <- max(30, ceiling(max(abs(dp), na.rm = TRUE) / 5) * 5)
write_gradient_fixed <- function(df, id_col, value_col, outfile, dataset_label,
                                 color_min, color_mid, color_max, vmin, vmid, vmax) {
  con <- file(outfile, "w"); on.exit(close(con), add = TRUE)
  writeLines("DATASET_GRADIENT", con)
  writeLines("SEPARATOR TAB", con)
  writeLines(paste0("DATASET_LABEL\t", dataset_label), con)
  writeLines("COLOR\t#000000", con)
  writeLines(paste0("COLOR_MIN\t", color_min), con)
  writeLines(paste0("COLOR_MID\t", color_mid), con)
  writeLines(paste0("COLOR_MAX\t", color_max), con)
  writeLines(sprintf("USER_MIN_VALUE\t%.6f", vmin), con)
  writeLines(sprintf("USER_MID_VALUE\t%.6f", vmid), con)
  writeLines(sprintf("USER_MAX_VALUE\t%.6f", vmax), con)
  writeLines("MARGIN\t5", con)
  writeLines("STRIP_WIDTH\t30", con)
  writeLines("SHOW_VALUES\t0", con)
  writeLines("DATA", con)
  for (i in seq_len(nrow(df))) {
    v <- df[[value_col]][i]
    if (is.na(v)) next
    writeLines(sprintf("%s\t%.8f", df[[id_col]][i], v), con)
  }
  message("Wrote ", outfile)
}
write_gradient_fixed(
  meta, "ID", "DeltaPrev_pct",
  file.path(itol_dir, "itol_Prevalence_diff_GRADIENT.txt"),
  "DeltaPrev_pct",
  "#053061", "#F7F7F7", "#67001F",
  -scale_dp, 0, scale_dp
)
file.copy(file.path(itol_dir, "itol_Prevalence_diff_GRADIENT.txt"),
          file.path(itol_dir, "itol_DeltaPrev_GRADIENT.txt"), overwrite = TRUE)
if (requireNamespace("itol.toolkit", quietly = TRUE)) {
  try({
    unit_bp <- itol.toolkit::create_unit(
      data = meta[, c("ID", "BioProject")],
      key = "BioProject_toolkit",
      type = "DATASET_COLORSTRIP",
      tree = tree
    )
    itol.toolkit::write_unit(unit_bp, file.path(itol_dir, "itol_BioProject_COLORSTRIP_toolkit.txt"))
  }, silent = TRUE)
  try({
    unit_en <- itol.toolkit::create_unit(
      data = meta[, c("ID", "Enrichment")],
      key = "Enrichment_toolkit",
      type = "DATASET_COLORSTRIP",
      tree = tree
    )
    itol.toolkit::write_unit(unit_en, file.path(itol_dir, "itol_Enrichment_COLORSTRIP_toolkit.txt"))
  }, silent = TRUE)
  try({
    unit_pr <- itol.toolkit::create_unit(
      data = meta[, c("ID", "Prevalence")],
      key = "Prevalence_toolkit",
      type = "DATASET_GRADIENT",
      tree = tree
    )
    itol.toolkit::write_unit(unit_pr, file.path(itol_dir, "itol_Prevalence_GRADIENT_toolkit.txt"))
  }, silent = TRUE)
}
write.table(meta, file.path(root, "tables/itol_metadata_used.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
write.csv(meta, file.path(root, "tables/itol_metadata_used.csv"), row.names = FALSE)
message("Done. Files:")
message(paste(list.files(itol_dir), collapse = "\n"))
