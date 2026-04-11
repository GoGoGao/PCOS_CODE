#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(tidyverse)
  library(phyloseq)
  library(igraph)
  library(ggraph)
  library(ggrepel)
  library(ggpubr)
  library(scales)
})

GROUP_COLS <- c(PCOS = "#E64B35", Healthy = "#4DBBD5")
COHORT_COLS <- c(PRJNA530971 = "#00A087", PRJNA549764 = "#3C5488",
                 PRJNA791492 = "#F39B7F")
PHY_PALETTE <- c(
  Firmicutes = "#E64B35", Bacteroidota = "#4DBBD5",
  Actinobacteria = "#00A087", Proteobacteria = "#3C5488",
  Verrucomicrobia = "#F39B7F", Bacillota = "#B09C85",
  Candidatus_Melainabacteria = "#91D1C2",
  Bacteria_unclassified = "#AAAAAA", Unassigned = "#CCCCCC"
)


# ============================================================
# Module 1: Build Phyloseq Object
# ============================================================

build_phyloseq <- function(group_file, abundance_file, tax_full_file) {
  group_df <- readr::read_tsv(group_file, col_types = cols()) %>%
    dplyr::rename(SampleID = Sample, Cohort = Bioproject) %>%
    dplyr::mutate(SampleID = as.character(SampleID),
                  Group = as.factor(Group),
                  Cohort = as.character(Cohort))

  abundance_df <- readr::read_tsv(abundance_file, col_types = cols())
  tax_col <- colnames(abundance_df)[1]
  abundance_df <- abundance_df %>% dplyr::rename(TAX = !!sym(tax_col))

  sample_ids_abund <- setdiff(colnames(abundance_df), "TAX")
  intersect_samples <- intersect(sample_ids_abund, unique(group_df$SampleID))

  abundance_df <- abundance_df %>%
    dplyr::select(TAX, all_of(intersect_samples))
  group_df <- group_df %>%
    dplyr::filter(SampleID %in% intersect_samples) %>%
    dplyr::distinct(SampleID, .keep_all = TRUE)

  tax_full_df <- readr::read_tsv(tax_full_file, col_types = cols()) %>%
    dplyr::mutate(TAX = as.character(TAX))

  tax_merged <- abundance_df %>%
    dplyr::select(TAX) %>%
    dplyr::left_join(tax_full_df, by = "TAX")

  tax_split <- tidyr::separate_wider_delim(
    tax_merged, TAX_INFO, delim = "|",
    names = c("Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"),
    too_few = "align_start", cols_remove = TRUE
  ) %>%
    dplyr::mutate(across(Kingdom:Genus, ~ ifelse(is.na(.x), "Unassigned", .x)),
                  Species = ifelse(is.na(Species), TAX, Species))

  otu_mat <- abundance_df %>%
    tibble::column_to_rownames("TAX") %>%
    dplyr::select(all_of(intersect_samples)) %>%
    as.matrix()

  zero_taxa <- rowSums(otu_mat) == 0
  if (any(zero_taxa)) {
    otu_mat <- otu_mat[!zero_taxa, , drop = FALSE]
    tax_split <- tax_split %>% dplyr::filter(TAX %in% rownames(otu_mat))
  }

  tax_mat <- tax_split %>%
    tibble::column_to_rownames("TAX") %>% as.matrix()

  group_df_ps <- group_df %>% tibble::column_to_rownames("SampleID")

  phyloseq(
    otu_table(otu_mat, taxa_are_rows = TRUE),
    sample_data(group_df_ps),
    tax_table(tax_mat)
  )
}


# ============================================================
# Module 2: Co-occurrence Network Construction
# ============================================================

build_single_network <- function(ps_obj, out_dir, method = "spearman",
                                 r_threshold = 0.5, p_threshold = 0.05,
                                 N = 250) {
  if (!dir.exists(out_dir))
    dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  N_use <- min(N, phyloseq::ntaxa(ps_obj))

  res <- ggClusterNet::network.2(
    ps = ps_obj, N = N_use, big = FALSE, select_layout = TRUE,
    layout_net = "model_Gephi.2", r.threshold = r_threshold,
    p.threshold = p_threshold, label = FALSE, path = out_dir,
    zipi = TRUE, ncol = 1, nrow = 1, method = method,
    fill = "Phylum", size = "igraph.degree"
  )

  if (!is.null(res[[2]])) {
    readr::write_csv(res[[2]], file.path(out_dir, "co_occurrence_edges.csv"))
  }
  invisible(res)
}

run_networks_by_group_cohort <- function(ps_all, result_root) {
  sd <- phyloseq::sample_data(ps_all) %>% as.data.frame()
  groups <- sort(unique(as.character(sd$Group)))
  cohorts <- sort(unique(as.character(sd$Cohort)))

  for (g in groups) {
    sample_ids <- rownames(sd)[sd$Group == g]
    if (length(sample_ids) < 5) next
    ps_sub <- prune_samples(sample_ids, ps_all)
    ps_sub <- prune_taxa(taxa_sums(ps_sub) > 0, ps_sub)
    build_single_network(ps_sub, file.path(result_root, "overall", g))
  }

  for (coh in cohorts) {
    for (g in groups) {
      sample_ids <- rownames(sd)[sd$Cohort == coh & sd$Group == g]
      if (length(sample_ids) < 5) next
      ps_sub <- prune_samples(sample_ids, ps_all)
      ps_sub <- prune_taxa(taxa_sums(ps_sub) > 0, ps_sub)
      build_single_network(ps_sub, file.path(result_root, coh, g))
    }
  }
}


# ============================================================
# Module 3: Network Property Summary
# ============================================================

parse_path_info <- function(file_path, result_root) {
  rel <- fs::path_rel(file_path, start = result_root)
  parts <- strsplit(as.character(rel), .Platform$file.sep)[[1]]
  if (length(parts) < 2) return(tibble(level = NA, cohort = NA, group = NA))
  if (parts[1] == "overall") {
    tibble(level = "overall", cohort = NA_character_, group = parts[2])
  } else {
    tibble(level = "cohort", cohort = parts[1], group = parts[2])
  }
}

summarize_network_properties <- function(result_root) {
  node_files <- list.files(result_root, pattern = "_node_properties\\.csv$",
                           full.names = TRUE, recursive = TRUE)

  all_nodes <- purrr::map_dfr(node_files, function(f) {
    info <- parse_path_info(f, result_root)
    df <- readr::read_csv(f, show_col_types = FALSE)
    colnames(df)[1] <- "node"
    df %>%
      tidyr::pivot_longer(-node, names_to = "property", values_to = "value") %>%
      dplyr::mutate(node = as.character(node),
                    level = info$level, cohort = info$cohort, group = info$group)
  }) %>%
    dplyr::mutate(property = gsub("^igraph\\.", "", property))

  network_summary <- all_nodes %>%
    dplyr::group_by(level, cohort, group, property) %>%
    dplyr::summarise(
      n_nodes = dplyr::n_distinct(node),
      mean_value = mean(value, na.rm = TRUE),
      median_value = median(value, na.rm = TRUE),
      sd_value = sd(value, na.rm = TRUE),
      .groups = "drop")

  out_dir <- file.path(result_root, "summary")
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  readr::write_tsv(all_nodes, file.path(out_dir, "node_properties_long.tsv"))
  readr::write_tsv(network_summary, file.path(out_dir, "network_property_summary.tsv"))
  list(nodes = all_nodes, summary = network_summary)
}


# ============================================================
# Module 4: Network Property Visualization
# ============================================================

PROP_LABELS <- c(degree = "Degree", betweenness = "Betweenness",
                 closeness = "Closeness")

compute_signif <- function(df, facet_vars) {
  df %>%
    group_by(across(all_of(facet_vars))) %>%
    summarise(
      p = tryCatch(
        wilcox.test(value[group == "Healthy"], value[group == "PCOS"])$p.value,
        error = function(e) NA_real_),
      y.position = max(value, na.rm = TRUE) * 1.08,
      .groups = "drop") %>%
    filter(!is.na(p), p < 0.05) %>%
    mutate(label = case_when(p < 0.001 ~ "***", p < 0.01 ~ "**", p < 0.05 ~ "*"),
           group1 = "Healthy", group2 = "PCOS")
}

plot_node_props <- function(df, title, facet_type = "wrap") {
  facet_vars <- if (facet_type == "wrap") "property_label"
                else c("property_label", "cohort")
  sig_df <- compute_signif(df, facet_vars)

  p <- ggplot(df, aes(x = group, y = value, fill = group)) +
    geom_violin(trim = FALSE, alpha = 0.25, color = NA, scale = "width") +
    geom_boxplot(width = 0.15, outlier.shape = NA, alpha = 0.85) +
    geom_jitter(aes(color = group), width = 0.08, alpha = 0.15, size = 0.25,
                show.legend = FALSE) +
    scale_fill_manual(values = GROUP_COLS) +
    scale_color_manual(values = GROUP_COLS, guide = "none") +
    labs(y = "Value", title = title) +
    theme_bw(base_size = 10) +
    theme(axis.title.x = element_blank(),
          strip.background = element_blank(),
          strip.text = element_text(face = "bold"))

  if (facet_type == "wrap") {
    p <- p + facet_wrap(~ property_label, scales = "free_y", nrow = 1)
  } else {
    p <- p + facet_grid(property_label ~ cohort, scales = "free_y")
  }

  if (nrow(sig_df) > 0) {
    p <- p + stat_pvalue_manual(sig_df, label = "label",
                                y.position = "y.position",
                                bracket.size = 0.3, size = 3.5)
  }
  p
}


# ============================================================
# Module 5: Network Plot (Single)
# ============================================================

build_network_plot <- function(edge_df, node_df, group_label, cohort_label) {
  if (all(c("source", "target") %in% names(edge_df)))
    edge_df <- rename(edge_df, from = source, to = target)
  if ("ID" %in% names(node_df))
    node_df <- rename(node_df, node = ID)

  cor_col <- intersect(c("correlation", "weight"), names(edge_df))[1]

  node_df <- node_df %>%
    mutate(Phylum = str_remove(Phylum, "^p__") %>% replace_na("Unassigned"),
           Species_clean = str_remove(Species, "^s__"),
           Genus_clean = str_remove(Genus, "^g__"),
           display_name = case_when(
             !is.na(Species_clean) & Species_clean != "" ~
               str_replace(Species_clean, "_", " "),
             !is.na(Genus_clean) & Genus_clean != "" ~
               paste0(str_replace(Genus_clean, "_", " "), " sp."),
             TRUE ~ node))

  sel_cols <- c("from", "to", na.omit(cor_col))
  g <- graph_from_data_frame(edge_df[, sel_cols, drop = FALSE],
                             directed = FALSE, vertices = node_df)

  if (!is.na(cor_col)) {
    e_cor <- edge_attr(g, cor_col)
    if (length(E(g)) > 200) {
      thr <- quantile(abs(e_cor), 0.25, na.rm = TRUE)
      keep <- which(abs(e_cor) >= thr)
      if (length(keep) > 0 && length(keep) < length(E(g))) {
        g <- subgraph.edges(g, keep, delete.vertices = FALSE)
        e_cor <- edge_attr(g, cor_col)
      }
    }
    E(g)$sign <- ifelse(e_cor >= 0, "positive", "negative")
    E(g)$cor_abs <- abs(e_cor)
  }

  iso <- which(degree(g) == 0)
  if (length(iso) > 0) g <- delete_vertices(g, iso)
  V(g)$deg <- degree(g)

  deg_thr <- quantile(V(g)$deg, 0.75, na.rm = TRUE)
  V(g)$hub_label <- ifelse(V(g)$deg >= max(deg_thr, 2),
                           vertex_attr(g, "display_name"), NA_character_)

  phy_in_graph <- sort(unique(vertex_attr(g, "Phylum")))
  phy_cols <- PHY_PALETTE[intersect(phy_in_graph, names(PHY_PALETTE))]
  missing <- setdiff(phy_in_graph, names(PHY_PALETTE))
  if (length(missing))
    phy_cols <- c(phy_cols, setNames(rep("#CCCCCC", length(missing)), missing))

  set.seed(42)
  lay <- create_layout(g, layout = "fr")

  p <- ggraph(lay) +
    geom_edge_link(aes(color = sign, edge_alpha = cor_abs,
                       edge_width = cor_abs), lineend = "round") +
    scale_edge_color_manual(values = c(positive = "#3C5488",
                                       negative = "#E64B35"),
                            name = "Correlation", drop = TRUE) +
    scale_edge_alpha(range = c(0.15, 0.7), guide = "none") +
    scale_edge_width(range = c(0.15, 1.0), guide = "none") +
    geom_node_point(aes(size = deg, fill = Phylum),
                    shape = 21, color = "grey30", stroke = 0.3, alpha = 0.9) +
    scale_size_continuous(range = c(2, 8), name = "Degree") +
    scale_fill_manual(values = phy_cols, name = "Phylum") +
    geom_text_repel(
      data = function(d) filter(d, !is.na(hub_label)),
      aes(x = x, y = y, label = hub_label), inherit.aes = FALSE,
      size = 2.2, fontface = "italic", max.overlaps = 30,
      segment.alpha = 0.3, segment.size = 0.2) +
    labs(title = paste0("Co-occurrence Network \u2014 ", group_label),
         subtitle = cohort_label) +
    theme_void() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 13),
          plot.subtitle = element_text(hjust = 0.5, size = 10),
          legend.position = "right")
  p
}


# ============================================================
# Module 6: Hub Species Extraction
# ============================================================

add_hub_score <- function(df) {
  df %>%
    mutate(rank_deg = rank(-degree),
           rank_bet = rank(-betweenness),
           rank_clo = rank(-closeness),
           hub_score = (rank_deg + rank_bet + rank_clo) / 3) %>%
    arrange(hub_score)
}

extract_hubs <- function(node_csv, topN = 20) {
  df <- readr::read_csv(node_csv, show_col_types = FALSE)
  colnames(df)[1] <- "node"
  df <- df %>% rename_with(~ gsub("^igraph\\.", "", .x))
  add_hub_score(df) %>% slice_head(n = topN)
}


# ============================================================
# Module 7: Venn / UpSet Comparisons
# ============================================================

read_hub_set <- function(path, set_name) {
  if (!file.exists(path)) return(tibble(set = character(0), node = character(0)))
  df <- readr::read_tsv(path, col_types = cols())
  colnames(df)[1] <- "node"
  tibble(set = set_name, node = as.character(df$node))
}

intersection_summary <- function(hub_list) {
  sets <- names(hub_list)
  combs <- map(2:min(length(sets), 4),
               ~ combn(sets, .x, simplify = FALSE)) %>% flatten()
  map_dfr(combs, function(s) {
    inter <- Reduce(intersect, hub_list[s])
    tibble(comparison = paste(s, collapse = " \u2229 "),
           n_intersect = length(inter),
           species = paste(sort(inter), collapse = "; "))
  }) %>% arrange(desc(n_intersect))
}

make_membership_df <- function(hub_list) {
  all_nodes <- sort(unique(unlist(hub_list)))
  mat <- map_dfc(hub_list, ~ as.integer(all_nodes %in% .x))
  bind_cols(tibble(Species = all_nodes), mat)
}

plot_venn_sets <- function(hub_list, fill_colors, title) {
  if (!requireNamespace("ggvenn", quietly = TRUE)) {
    message("ggvenn not installed")
    return(NULL)
  }
  ggvenn::ggvenn(hub_list, fill_color = unname(fill_colors),
                 fill_alpha = 0.45, stroke_color = "grey40",
                 stroke_size = 0.4, set_name_size = 5, text_size = 4) +
    labs(title = title) +
    theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 13))
}


# ============================================================
# Module 8: Hub-Differential-Core Integration
# ============================================================

integrate_hub_diff_core <- function(hub_long, diff_file, core_file,
                                    out_dir) {
  hub_annot <- hub_long %>%
    separate_wider_delim(set, delim = "_",
                         names = c("cohort", "group"),
                         too_few = "align_start")

  if (file.exists(diff_file)) {
    diff_df <- readr::read_csv(diff_file, show_col_types = FALSE) %>%
      transmute(Species, Direction, Wilcoxon_FDR, LMM_FDR,
                Significant_Wilcoxon, Significant_LMM, Consistent_Sig)
    hub_annot <- left_join(hub_annot, diff_df, by = c("node" = "Species"))
  }

  if (file.exists(core_file)) {
    core_df <- readr::read_tsv(core_file, col_types = cols()) %>%
      transmute(Species, PCOS_prevalence, Healthy_prevalence,
                PCOS_abundance, Healthy_abundance, Core_status)
    hub_annot <- left_join(hub_annot, core_df, by = c("node" = "Species"))
  }

  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  readr::write_tsv(hub_annot,
                    file.path(out_dir, "hub_species_with_diff_and_core.tsv"))
  hub_annot
}


# ============================================================
# Utility: Save plots in multiple formats
# ============================================================

save_plot_multi <- function(p, path, w = 7, h = 6) {
  ggsave(paste0(path, ".pdf"), p, width = w, height = h, device = cairo_pdf)
  ggsave(paste0(path, ".jpg"), p, width = w, height = h, dpi = 300)
}


message("PCOS multi-cohort metagenome analysis pipeline (R)")
message("Source this file or call individual functions.")
