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
  library(tidyverse)
  library(sva)
})
get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- "--file="
  script_path <- sub(file_arg, "", args[grepl(file_arg, args)])
  if (length(script_path) == 0) normalizePath(".") else normalizePath(dirname(script_path))
}
clr_transform <- function(mat) {
  mat <- as.matrix(mat)
  storage.mode(mat) <- "double"
  pos <- mat[mat > 0]
  if (length(pos) == 0) stop("No positive abundance values found.")
  pseudocount <- min(pos) / 2
  log_mat <- log(mat + pseudocount)
  sweep(log_mat, 2, colMeans(log_mat), "-")
}
main <- function() {
  script_dir <- get_script_dir()
  root_dir <- normalizePath(file.path(script_dir, ".."), mustWork = TRUE)
  sgb_dir <- normalizePath(
    file.path(root_dir, "..", "..", "04-ml-perf", "sgb_machine_learning"),
    mustWork = TRUE
  )
  out_data <- file.path(root_dir, "data")
  dir.create(out_data, recursive = TRUE, showWarnings = FALSE)
  group_file <- file.path(sgb_dir, "group.tsv")
  abundance_file <- file.path(sgb_dir, "bin_abundance_table_tab.tsv")
  message("SGB data dir    : ", sgb_dir)
  message("Output data dir : ", out_data)
  group_df <- readr::read_tsv(group_file, show_col_types = FALSE) %>%
    dplyr::rename(SampleID = Sample, Group = Group, Cohort = Bioproject) %>%
    dplyr::mutate(
      SampleID = as.character(SampleID),
      Group = factor(Group, levels = c("Healthy", "PCOS")),
      Cohort = as.character(Cohort)
    )
  abundance_df <- readr::read_tsv(abundance_file, show_col_types = FALSE)
  tax_col <- colnames(abundance_df)[1]
  abundance_df <- abundance_df %>%
    dplyr::rename(TAX = !!sym(tax_col)) %>%
    dplyr::mutate(TAX = as.character(TAX))
  sample_ids <- intersect(colnames(abundance_df)[-1], group_df$SampleID)
  if (length(sample_ids) == 0) stop("No overlapping samples between abundance and group tables.")
  abundance_df <- abundance_df %>%
    dplyr::select(TAX, all_of(sample_ids))
  group_df <- group_df %>%
    dplyr::filter(SampleID %in% sample_ids) %>%
    dplyr::arrange(match(SampleID, sample_ids))
  mat <- abundance_df %>%
    tibble::column_to_rownames("TAX") %>%
    as.matrix()
  storage.mode(mat) <- "double"
  mat[mat < 0] <- 0
  zero_taxa <- rowSums(mat, na.rm = TRUE) == 0
  if (any(zero_taxa)) {
    message("Removing ", sum(zero_taxa), " all-zero taxa before CLR/ComBat.")
    mat <- mat[!zero_taxa, , drop = FALSE]
  }
  clr_mat <- clr_transform(mat)
  batch <- group_df$Cohort
  mod <- model.matrix(~ Group, data = group_df)
  message("Running ComBat: ", nrow(clr_mat), " SGBs x ", ncol(clr_mat), " samples")
  message("Batches: ", paste(sort(unique(batch)), collapse = ", "))
  combat_mat <- ComBat(
    dat = clr_mat,
    batch = batch,
    mod = mod,
    par.prior = TRUE,
    prior.plots = FALSE
  )
  combat_mat[!is.finite(combat_mat)] <- 0
  combat_df <- combat_mat %>%
    as.data.frame() %>%
    tibble::rownames_to_column("TAX")
  out_clr <- file.path(out_data, "sgb_input_clr.tsv")
  out_combat <- file.path(out_data, "sgb_input_combat_clr.tsv")
  out_meta <- file.path(out_data, "combat_metadata.tsv")
  readr::write_tsv(combat_df, out_combat)
  readr::write_tsv(
    clr_mat %>% as.data.frame() %>% tibble::rownames_to_column("TAX"),
    out_clr
  )
  readr::write_tsv(group_df, out_meta)
  message("Saved CLR matrix : ", out_clr)
  message("Saved ComBat CLR : ", out_combat)
  message("Done.")
}
main()
