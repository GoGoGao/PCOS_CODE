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
  library(phyloseq)
  library(ggClusterNet)
  library(igraph)
})
get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- "--file="
  script_path <- sub(file_arg, "", args[grepl(file_arg, args)])
  if (length(script_path) == 0) normalizePath(".") else normalizePath(dirname(script_path))
}
build_single_network <- function(ps_obj, out_dir, method = "spearman",
                                 r_threshold = 0.5, p_threshold = 0.05,
                                 N = 250) {
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  ntaxa_ps <- phyloseq::ntaxa(ps_obj)
  N_use <- min(N, ntaxa_ps)
  message("  - taxa: ", ntaxa_ps, ", using top N = ", N_use)
  res <- ggClusterNet::network.2(
    ps = ps_obj,
    N = N_use,
    big = FALSE,
    select_layout = TRUE,
    layout_net = "model_Gephi.2",
    r.threshold = r_threshold,
    p.threshold = p_threshold,
    label = FALSE,
    path = out_dir,
    zipi = TRUE,
    ncol = 1,
    nrow = 1,
    method = method,
    fill = "Phylum",
    size = "igraph.degree"
  )
  edges_file <- file.path(out_dir, "co_occurrence_edges.csv")
  if (!is.null(res[[2]])) readr::write_csv(res[[2]], edges_file)
  invisible(res)
}
build_phyloseq <- function(root_dir, sgb_dir) {
  group_file <- file.path(sgb_dir, "group.tsv")
  abundance_file <- file.path(root_dir, "data", "sgb_input_combat_clr.tsv")
  tax_file <- file.path(sgb_dir, "sgb_info_comb.tsv")
  group_df <- readr::read_tsv(group_file, show_col_types = FALSE) %>%
    dplyr::rename(SampleID = Sample, Group = Group, Cohort = Bioproject) %>%
    dplyr::mutate(
      SampleID = as.character(SampleID),
      Group = as.factor(Group),
      Cohort = as.character(Cohort)
    )
  abundance_df <- readr::read_tsv(abundance_file, show_col_types = FALSE) %>%
    dplyr::mutate(TAX = as.character(TAX))
  sample_ids <- intersect(colnames(abundance_df)[-1], group_df$SampleID)
  abundance_df <- abundance_df %>% dplyr::select(TAX, all_of(sample_ids))
  group_df <- group_df %>%
    dplyr::filter(SampleID %in% sample_ids) %>%
    dplyr::distinct(SampleID, .keep_all = TRUE) %>%
    dplyr::arrange(SampleID)
  tax_full_df <- readr::read_tsv(tax_file, show_col_types = FALSE) %>%
    dplyr::mutate(TAX = as.character(ID)) %>%
    dplyr::select(TAX, Kingdom = K, Phylum = P, Class = C, Order = O,
                  Family = F, Genus = G, Species = S) %>%
    dplyr::distinct(TAX, .keep_all = TRUE)
  tax_split <- abundance_df %>%
    dplyr::select(TAX) %>%
    dplyr::left_join(tax_full_df, by = "TAX") %>%
    dplyr::mutate(
      Kingdom = ifelse(is.na(Kingdom) | Kingdom == "", "Unassigned", Kingdom),
      Phylum = ifelse(is.na(Phylum) | Phylum == "", "Unassigned", Phylum),
      Class = ifelse(is.na(Class) | Class == "", "Unassigned", Class),
      Order = ifelse(is.na(Order) | Order == "", "Unassigned", Order),
      Family = ifelse(is.na(Family) | Family == "", "Unassigned", Family),
      Genus = ifelse(is.na(Genus) | Genus == "", "Unassigned", Genus),
      Species = ifelse(is.na(Species) | Species == "", TAX, Species)
    )
  otu_mat <- abundance_df %>%
    tibble::column_to_rownames("TAX") %>%
    as.matrix()
  storage.mode(otu_mat) <- "double"
  otu_mat[!is.finite(otu_mat)] <- 0
  zero_taxa <- rowSums(otu_mat, na.rm = TRUE) == 0
  if (any(zero_taxa)) {
    otu_mat <- otu_mat[!zero_taxa, , drop = FALSE]
    tax_split <- tax_split %>% dplyr::filter(TAX %in% rownames(otu_mat))
  }
  tax_mat <- tax_split %>% tibble::column_to_rownames("TAX") %>% as.matrix()
  group_df_ps <- group_df %>% tibble::column_to_rownames("SampleID")
  phyloseq(
    otu_table(otu_mat, taxa_are_rows = TRUE),
    sample_data(group_df_ps),
    tax_table(tax_mat)
  )
}
main <- function() {
  script_dir <- get_script_dir()
  root_dir <- normalizePath(file.path(script_dir, ".."), mustWork = TRUE)
  sgb_dir <- normalizePath(
    file.path(root_dir, "..", "..", "04-ml-perf", "sgb_machine_learning"),
    mustWork = TRUE
  )
  result_root <- file.path(root_dir, "result")
  dir.create(result_root, recursive = TRUE, showWarnings = FALSE)
  message("Building phyloseq from ComBat-corrected SGB abundances ...")
  ps_all <- build_phyloseq(root_dir, sgb_dir)
  saveRDS(ps_all, file.path(result_root, "ps_all_combat_sgb.rds"))
  sd <- sample_data(ps_all) %>% as.data.frame()
  groups <- sort(unique(as.character(sd$Group)))
  cohorts <- sort(unique(as.character(sd$Cohort)))
  args_scope <- commandArgs(trailingOnly = TRUE)
  run_overall <- TRUE
  cohorts_to_run <- cohorts
  if (length(args_scope) >= 1) {
    scope <- args_scope[1]
    if (scope == "overall") {
      run_overall <- TRUE
      cohorts_to_run <- character(0)
    } else if (grepl("^cohort:", scope)) {
      target <- sub("^cohort:", "", scope)
      run_overall <- FALSE
      cohorts_to_run <- target
    }
  }
  if (run_overall) {
    message("Building overall ComBat SGB networks ...")
    for (g in groups) {
      sample_ids <- rownames(sd)[sd$Group == g]
      if (length(sample_ids) < 5) next
      ps_sub <- prune_taxa(taxa_sums(prune_samples(sample_ids, ps_all)) > 0,
                           prune_samples(sample_ids, ps_all))
      build_single_network(ps_sub, file.path(result_root, "overall", g))
    }
  }
  if (length(cohorts_to_run) > 0) {
    message("Building cohort-specific ComBat SGB networks ...")
    for (coh in cohorts_to_run) {
      for (g in groups) {
        sample_ids <- rownames(sd)[sd$Cohort == coh & sd$Group == g]
        if (length(sample_ids) < 5) next
        ps_sub <- prune_taxa(taxa_sums(prune_samples(sample_ids, ps_all)) > 0,
                             prune_samples(sample_ids, ps_all))
        build_single_network(ps_sub, file.path(result_root, coh, g))
      }
    }
  }
  message("ComBat SGB network construction finished.")
}
main()
