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
  library(dplyr)
  library(tidyr)
  library(readr)
  library(ggplot2)
  library(patchwork)
  library(ComplexHeatmap)
  library(circlize)
  library(grid)
})
TOP_N <- 20L
STEM <- "Fig_hub_top20"
SCRIPT_DIR <- tryCatch(
  dirname(sys.frame(1)$ofile),
  error = function(e) {
    args <- commandArgs(trailingOnly = FALSE)
    f <- grep("^--file=", args, value = TRUE)
    if (length(f)) dirname(sub("^--file=", "", f[1])) else getwd()
  }
)
ROOT <- normalizePath(file.path(SCRIPT_DIR, ".."))
COMBAT_RES <- normalizePath(file.path(ROOT, "..", "ComBat_network_check", "result"))
out_fig <- file.path(ROOT, "figures", "hub_top20")
out_src <- file.path(ROOT, "source_data")
out_tab <- file.path(ROOT, "tables", "hub_top20")
for (d in c(out_fig, file.path(out_fig, "panels"), file.path(out_fig, "composite"),
            out_src, out_tab)) {
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
}
if (requireNamespace("systemfonts", quietly = TRUE)) {
  tnr_path <- tryCatch(
    systemfonts::match_fonts("Times New Roman")$path[[1]],
    error = function(e) "/usr/share/fonts/msttcore/times.ttf"
  )
  if (is.null(tnr_path) || !nzchar(tnr_path) || !file.exists(tnr_path)) {
    tnr_path <- "/usr/share/fonts/msttcore/times.ttf"
  }
  if (file.exists(tnr_path)) {
    tryCatch(
      systemfonts::register_font(
        name = "Times New Roman",
        plain = tnr_path,
        italic = sub("times\\.ttf$", "timesi.ttf", tnr_path, ignore.case = TRUE),
        bold = sub("times\\.ttf$", "timesbd.ttf", tnr_path, ignore.case = TRUE),
        bolditalic = sub("times\\.ttf$", "timesbi.ttf", tnr_path, ignore.case = TRUE)
      ),
      error = function(e) invisible(NULL)
    )
  }
}
FONT_FAMILY <- "Times New Roman"
theme_nature <- function(base_size = 8.5, base_family = FONT_FAMILY) {
  theme_classic(base_size = base_size, base_family = base_family) +
    theme(
      axis.line = element_line(linewidth = 0.3, colour = "black"),
      axis.ticks = element_line(linewidth = 0.3, colour = "black"),
      axis.title = element_text(size = base_size, family = base_family,
                               colour = "black", face = "plain"),
      axis.text = element_text(size = base_size - 0.3, colour = "black",
                              family = base_family, face = "plain"),
      legend.title = element_text(size = base_size - 0.1, family = base_family,
                                  colour = "black", face = "plain"),
      legend.text = element_text(size = base_size - 0.4, family = base_family,
                                 colour = "black", face = "plain"),
      strip.text = element_text(size = base_size, family = base_family,
                               colour = "black", face = "plain"),
      strip.background = element_blank(),
      plot.title = element_text(size = base_size + 1, hjust = 0,
                               family = base_family, colour = "black",
                               face = "plain"),
      plot.subtitle = element_text(size = base_size - 0.2, colour = "black",
                                  hjust = 0, family = base_family,
                                  face = "plain"),
      plot.tag = element_text(size = 10, family = base_family, colour = "black",
                             face = "plain"),
      panel.grid = element_blank(),
      plot.background = element_rect(fill = "white", colour = NA),
      panel.background = element_rect(fill = "white", colour = NA),
      legend.key.size = unit(3, "mm"),
      legend.background = element_blank()
    )
}
save_pub <- function(plot, filename, width_mm = 183, height_mm = 140, dpi = 600) {
  w <- width_mm / 25.4
  h <- height_mm / 25.4
  pdf.options(useDingbats = FALSE)
  svglite::svglite(
    paste0(filename, ".svg"), width = w, height = h,
    system_fonts = list(sans = FONT_FAMILY, serif = FONT_FAMILY, mono = FONT_FAMILY)
  )
  print(plot); grDevices::dev.off()
  grDevices::cairo_pdf(paste0(filename, ".pdf"), width = w, height = h,
                       family = FONT_FAMILY)
  print(plot); grDevices::dev.off()
  ragg::agg_png(paste0(filename, ".png"), width = w, height = h, units = "in",
                res = dpi)
  print(plot); grDevices::dev.off()
  ragg::agg_jpeg(paste0(filename, ".jpg"), width = w, height = h, units = "in",
                 res = dpi, quality = 95)
  print(plot); grDevices::dev.off()
}
write_plotdata <- function(df, stem) {
  write_tsv(df, file.path(out_src, paste0(stem, "_plotdata.tsv")))
  write_csv(df, file.path(out_src, paste0(stem, "_plotdata.csv")))
}
export_figure <- function(plot, tsv_df, stem, width_mm, height_mm) {
  path_stem <- file.path(out_fig, "panels", stem)
  write_plotdata(tsv_df, stem)
  save_pub(plot, path_stem, width_mm = width_mm, height_mm = height_mm)
  for (ext in c("pdf", "svg", "png", "jpg")) {
    file.copy(paste0(path_stem, ".", ext),
              file.path(out_fig, "composite", paste0(stem, ".", ext)),
              overwrite = TRUE)
  }
  message("Saved: ", stem)
}
save_heatmap_pub <- function(draw_fun, filename, width_mm, height_mm, dpi = 600) {
  w <- width_mm / 25.4
  h <- height_mm / 25.4
  pdf.options(useDingbats = FALSE)
  svglite::svglite(
    paste0(filename, ".svg"), width = w, height = h,
    system_fonts = list(sans = FONT_FAMILY, serif = FONT_FAMILY, mono = FONT_FAMILY)
  )
  draw_fun(); grDevices::dev.off()
  grDevices::cairo_pdf(paste0(filename, ".pdf"), width = w, height = h,
                       family = FONT_FAMILY)
  draw_fun(); grDevices::dev.off()
  ragg::agg_png(paste0(filename, ".png"), width = w, height = h, units = "in",
                res = dpi)
  draw_fun(); grDevices::dev.off()
  ragg::agg_jpeg(paste0(filename, ".jpg"), width = w, height = h, units = "in",
                 res = dpi, quality = 95)
  draw_fun(); grDevices::dev.off()
}
np <- read_tsv(file.path(ROOT, "tables", "node_properties_long_r0.6.tsv"),
               show_col_types = FALSE)
cohorts <- c("PRJNA549764", "PRJNA530971", "PRJNA791492")
groups <- c("Healthy", "PCOS")
set_levels <- c(
  "549764 Healthy", "549764 PCOS",
  "530971 Healthy", "530971 PCOS",
  "791492 Healthy", "791492 PCOS"
)
wide <- np %>%
  filter(level == "cohort", cohort %in% cohorts, group %in% groups,
         property %in% c("degree", "betweenness", "closeness")) %>%
  select(node, cohort, group, property, value) %>%
  pivot_wider(names_from = property, values_from = value)
topn <- wide %>%
  group_by(cohort, group) %>%
  arrange(desc(degree), desc(betweenness), desc(closeness), node, .by_group = TRUE) %>%
  mutate(
    rank = row_number(),
    n_tied_max = sum(degree == max(degree)),
    max_degree = max(degree),
    n_nodes = n()
  ) %>%
  filter(rank <= TOP_N) %>%
  ungroup() %>%
  mutate(
    net = paste(cohort, group, sep = "_"),
    net_lab = paste(cohort, group, sep = " / "),
    set_lab = paste0(sub("^PRJNA", "", cohort), " ", group)
  )
topn$set_lab <- factor(topn$set_lab, levels = set_levels)
topn$net <- factor(
  topn$net,
  levels = c(
    "PRJNA549764_Healthy", "PRJNA549764_PCOS",
    "PRJNA530971_Healthy", "PRJNA530971_PCOS",
    "PRJNA791492_Healthy", "PRJNA791492_PCOS"
  )
)
write_tsv(topn, file.path(out_tab, "top20_degree_hubs_cohort.tsv"))
write_csv(topn, file.path(out_tab, "top20_degree_hubs_cohort.csv"))
tax <- {
  files <- list.files(COMBAT_RES, pattern = "_Gephi_allnode\\.csv$",
                      recursive = TRUE, full.names = TRUE)
  bind_rows(lapply(files, function(f) {
    read_csv(f, show_col_types = FALSE) %>%
      select(any_of(c("ID", "Kingdom", "Phylum", "Class", "Order",
                      "Family", "Genus", "Species")))
  })) %>%
    distinct(ID, .keep_all = TRUE) %>%
    mutate(
      Phylum = gsub("^p__", "", Phylum),
      Genus = gsub("^g__", "", Genus),
      Species = gsub("^s__", "", Species),
      Phylum = dplyr::recode(
        Phylum,
        Bacteroidota = "Bacteroidetes",
        Actinomycetota = "Actinobacteria",
        Bacillota = "Firmicutes",
        Pseudomonadota = "Proteobacteria",
        Verrucomicrobiota = "Verrucomicrobia",
        .default = Phylum
      )
    )
}
assign_feature <- function(sp, genus, phylum) {
  oral_g <- c("Actinomyces", "Actinobaculum", "Streptococcus", "Veillonella",
              "Acinetobacter", "Rothia", "Prevotella", "Neisseria")
  patho <- c("Escherichia_coli", "Ruminococcus_gnavus", "Flavonifractor_plautii",
             "Bacteroides_fragilis", "Enterocloster_clostridioformis",
             "Clostridium_innocuum")
  if (!is.na(genus) && genus %in% oral_g) return("Oral / upper-GI associated")
  if (sp %in% patho) return("Pathobiont / inflammation-linked")
  if (!is.na(genus) && genus %in% c("Blautia", "Bacteroides", "Phocaeicola",
      "Faecalibacterium", "Parabacteroides", "Dorea", "Anaerostipes",
      "Agathobaculum", "Bifidobacterium", "Roseburia", "Eubacterium",
      "Coprococcus", "Ruminococcus")) {
    return("Gut commensal / SCFA-linked")
  }
  if (grepl("^GGB|^SGB|bacterium_|_sp_", sp)) return("Unclassified / poorly characterized")
  "Other gut taxa"
}
hub_annot <- topn %>%
  left_join(tax, by = c("node" = "ID")) %>%
  mutate(
    display = gsub("_", " ", node),
    Genus = ifelse(is.na(Genus) | Genus == "", sub("_.*$", "", node), Genus),
    Phylum = ifelse(is.na(Phylum) | Phylum == "", "Unknown", Phylum),
    feature = mapply(assign_feature, node, Genus, Phylum),
    role_within_cohort = NA_character_
  )
for (co in cohorts) {
  h <- hub_annot$node[hub_annot$cohort == co & hub_annot$group == "Healthy"]
  p <- hub_annot$node[hub_annot$cohort == co & hub_annot$group == "PCOS"]
  shared <- intersect(h, p)
  hub_annot$role_within_cohort[hub_annot$cohort == co & hub_annot$node %in% shared] <-
    "Shared Healthy-PCOS hub"
  hub_annot$role_within_cohort[hub_annot$cohort == co & hub_annot$group == "Healthy" &
                                 hub_annot$node %in% setdiff(h, p)] <-
    "Healthy-only hub"
  hub_annot$role_within_cohort[hub_annot$cohort == co & hub_annot$group == "PCOS" &
                                 hub_annot$node %in% setdiff(p, h)] <-
    "PCOS-only hub"
}
recurrence <- hub_annot %>%
  distinct(node, net) %>%
  count(node, name = "n_networks") %>%
  mutate(
    recurrence_class = case_when(
      n_networks >= 3 ~ "Recurrent (>=3 networks)",
      n_networks == 2 ~ "Shared by 2 networks",
      TRUE ~ "Network-private"
    )
  )
hub_annot <- hub_annot %>% left_join(recurrence, by = "node")
write_tsv(hub_annot, file.path(out_tab, "top20_hubs_annotated.tsv"))
write_csv(hub_annot, file.path(out_tab, "top20_hubs_annotated.csv"))
lists <- split(as.character(hub_annot$node), hub_annot$set_lab)
lists <- lists[set_levels]
m <- make_comb_mat(lists)
cs <- comb_size(m)
set_cols <- c(
  "549764 Healthy" = "#7AA9C9",
  "549764 PCOS"    = "#D08A90",
  "530971 Healthy" = "#5B8FA8",
  "530971 PCOS"    = "#C07078",
  "791492 Healthy" = "#4A7A95",
  "791492 PCOS"    = "#A85860"
)
upset_stem <- file.path(out_fig, "panels", paste0(STEM, "_UpSet"))
n_comb <- length(cs)
upset_w_mm <- if (n_comb > 25) 220 else 183
upset_h_mm <- if (n_comb > 25) 130 else 115
draw_upset <- function() {
  ht <- draw(
    UpSet(
      m,
      set_order = set_levels,
      comb_order = order(comb_degree(m), -cs),
      bg_col = c("#F5F5F5", "#EAEAEA"),
      bg_pt_col = "#BDBDBD",
      pt_size = unit(2.6, "mm"),
      lwd = 1.1,
      row_names_gp = gpar(fontsize = 8, fontfamily = FONT_FAMILY, fontface = "plain"),
      column_title = NULL,
      top_annotation = upset_top_annotation(
        m,
        height = unit(3.4, "cm"),
        bar_width = 0.55,
        gp = gpar(fill = "#4A6FA5", col = NA),
        annotation_name_gp = gpar(fontsize = 8, fontfamily = FONT_FAMILY),
        axis_param = list(gp = gpar(fontsize = 7, fontfamily = FONT_FAMILY))
      ),
      right_annotation = upset_right_annotation(
        m,
        width = unit(2.4, "cm"),
        gp = gpar(fill = set_cols[set_levels], col = NA),
        annotation_name_gp = gpar(fontsize = 8, fontfamily = FONT_FAMILY),
        axis_param = list(gp = gpar(fontsize = 7, fontfamily = FONT_FAMILY))
      )
    ),
    padding = unit(c(2, 2, 2, 2), "mm")
  )
  od <- column_order(ht)
  decorate_annotation("intersection_size", {
    grid.text(
      cs[od],
      x = seq_along(cs),
      y = unit(cs[od], "native") + unit(2, "pt"),
      default.units = "native",
      just = "bottom",
      gp = gpar(fontsize = 6.5, fontfamily = FONT_FAMILY, fontface = "plain")
    )
  })
  invisible(ht)
}
save_heatmap_pub(draw_upset, upset_stem, width_mm = upset_w_mm, height_mm = upset_h_mm)
for (ext in c("pdf", "svg", "png", "jpg")) {
  file.copy(paste0(upset_stem, ".", ext),
            file.path(out_fig, "composite", paste0(STEM, "_UpSet.", ext)),
            overwrite = TRUE)
}
comb_size_df <- tibble(
  combination = names(cs),
  intersection_size = as.integer(cs),
  degree = as.integer(comb_degree(m))
) %>% arrange(desc(intersection_size), desc(degree))
write_tsv(comb_size_df, file.path(out_tab, "upset_combination_sizes.tsv"))
write_csv(comb_size_df, file.path(out_tab, "upset_combination_sizes.csv"))
write_plotdata(
  hub_annot %>%
    select(node, display, cohort, group, set_lab, rank, degree, betweenness,
           closeness, Phylum, Genus, feature, role_within_cohort,
           n_networks, recurrence_class, n_tied_max),
  paste0(STEM, "_UpSet")
)
union_sp <- hub_annot %>%
  distinct(node, display, Phylum, Genus, feature, n_networks) %>%
  arrange(desc(n_networks), Phylum, Genus, node)
heat_df <- expand.grid(
  display = union_sp$display,
  set_lab = set_levels,
  stringsAsFactors = FALSE
) %>%
  as_tibble() %>%
  left_join(
    hub_annot %>% select(display, set_lab, rank, degree, feature, Phylum),
    by = c("display", "set_lab")
  ) %>%
  left_join(union_sp %>% select(display, n_networks, Genus), by = "display") %>%
  mutate(
    set_lab = factor(set_lab, levels = set_levels),
    display = factor(display, levels = rev(union_sp$display))
  )
p_heat <- ggplot(heat_df, aes(x = set_lab, y = display)) +
  geom_tile(aes(fill = rank), colour = "white", linewidth = 0.2) +
  scale_fill_gradientn(
    colours = c("#08306B", "#2171B5", "#6BAED6", "#C6DBEF", "#F7FBFF"),
    na.value = "#F0F0F0",
    limits = c(1, TOP_N),
    breaks = c(1, 10, TOP_N),
    name = "Degree\nrank"
  ) +
  labs(
    x = NULL, y = NULL,
    title = paste0("Top-", TOP_N, " degree species across cohort networks"),
    subtitle = "Fill = rank within network (1 = highest degree); grey = not in top-20"
  ) +
  theme_nature(base_size = 7.5) +
  theme(
    axis.text.x = element_text(size = 6.5, angle = 35, hjust = 1, vjust = 1),
    axis.text.y = element_text(size = 5.2, face = "italic"),
    legend.position = "right",
    plot.margin = margin(4, 6, 4, 4)
  )
export_figure(
  p_heat, heat_df, paste0(STEM, "_presence_heatmap"),
  width_mm = 150,
  height_mm = max(160, 10 + nrow(union_sp) * 2.6)
)
feat_levels <- c(
  "Gut commensal / SCFA-linked",
  "Pathobiont / inflammation-linked",
  "Oral / upper-GI associated",
  "Unclassified / poorly characterized",
  "Other gut taxa"
)
hub_annot <- hub_annot %>%
  mutate(
    feature = factor(feature, levels = feat_levels),
    role_within_cohort = factor(
      role_within_cohort,
      levels = c("Shared Healthy-PCOS hub", "Healthy-only hub", "PCOS-only hub")
    )
  )
feat_counts <- hub_annot %>%
  count(set_lab, feature, name = "n") %>%
  mutate(set_lab = factor(set_lab, levels = set_levels))
role_counts <- hub_annot %>%
  count(cohort, role_within_cohort, name = "n") %>%
  mutate(cohort = factor(cohort, levels = cohorts))
col_feat <- c(
  "Gut commensal / SCFA-linked" = "#44BB99",
  "Pathobiont / inflammation-linked" = "#CC6677",
  "Oral / upper-GI associated" = "#E28E2C",
  "Unclassified / poorly characterized" = "#BBBBBB",
  "Other gut taxa" = "#77AADD"
)
col_role <- c(
  "Shared Healthy-PCOS hub" = "#7A9AB8",
  "Healthy-only hub" = "#7AA9C9",
  "PCOS-only hub" = "#D08A90"
)
p_feat <- ggplot(feat_counts, aes(x = set_lab, y = n, fill = feature)) +
  geom_col(width = 0.75, colour = NA) +
  scale_fill_manual(values = col_feat, name = "Ecological feature") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.05)),
                     breaks = seq(0, TOP_N, 5)) +
  labs(x = NULL, y = paste0("Number of top-", TOP_N, " hubs"),
       title = paste0("Ecological feature composition of top-", TOP_N, " hubs")) +
  theme_nature(base_size = 8.5) +
  theme(axis.text.x = element_text(angle = 35, hjust = 1, vjust = 1, size = 7),
        legend.position = "right")
p_role <- ggplot(role_counts, aes(x = cohort, y = n, fill = role_within_cohort)) +
  geom_col(width = 0.65, colour = NA) +
  scale_fill_manual(values = col_role, name = "Within-cohort role") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
  labs(x = NULL, y = paste0("Hub slots (n = ", 2 * TOP_N, " / cohort)"),
       title = "Healthy vs PCOS hub overlap within each cohort",
       subtitle = paste0("Each cohort contributes ", TOP_N, " Healthy + ", TOP_N, " PCOS top hubs")) +
  theme_nature(base_size = 8.5) +
  theme(legend.position = "right")
rec_sp <- hub_annot %>%
  filter(n_networks >= 2) %>%
  distinct(node, display, n_networks, feature, Phylum, Genus) %>%
  arrange(desc(n_networks), display)
rec_presence <- hub_annot %>%
  filter(node %in% rec_sp$node) %>%
  mutate(
    display = factor(display, levels = rev(rec_sp$display)),
    set_lab = factor(set_lab, levels = set_levels)
  )
p_rec <- ggplot(rec_presence, aes(x = set_lab, y = display, fill = feature)) +
  geom_tile(colour = "white", linewidth = 0.25) +
  scale_fill_manual(values = col_feat, name = "Feature") +
  labs(x = NULL, y = NULL,
       title = paste0("Top-", TOP_N, " hubs appearing in >=2 networks"),
       subtitle = "Colour = ecological feature class") +
  theme_nature(base_size = 8) +
  theme(
    axis.text.x = element_text(angle = 35, hjust = 1, vjust = 1, size = 7),
    axis.text.y = element_text(size = 6.5, face = "italic"),
    legend.position = "right"
  )
export_figure(p_feat, feat_counts, paste0(STEM, "_feature_bars"),
              width_mm = 160, height_mm = 95)
export_figure(p_role, role_counts, paste0(STEM, "_within_cohort_roles"),
              width_mm = 130, height_mm = 85)
export_figure(
  p_rec, rec_presence, paste0(STEM, "_recurrent_species"),
  width_mm = 155,
  height_mm = max(80, 18 + nrow(rec_sp) * 3.8)
)
fig_comp <- (p_role | p_feat) / p_rec +
  plot_layout(heights = c(1, 1.35)) +
  plot_annotation(
    tag_levels = "A",
    theme = theme(
      plot.tag = element_text(size = 10, family = FONT_FAMILY, face = "plain")
    )
  )
write_plotdata(
  bind_rows(
    mutate(role_counts, panel = "A_within_cohort_roles"),
    mutate(feat_counts, panel = "B_feature")
  ),
  paste0(STEM, "_features_composite")
)
save_pub(fig_comp, file.path(out_fig, "panels", paste0(STEM, "_features_composite")),
         width_mm = 183, height_mm = 200)
for (ext in c("pdf", "svg", "png", "jpg")) {
  file.copy(
    file.path(out_fig, "panels", paste0(STEM, "_features_composite.", ext)),
    file.path(out_fig, "composite", paste0(STEM, "_features_composite.", ext)),
    overwrite = TRUE
  )
}
n_union <- n_distinct(hub_annot$node)
n_private <- sum(recurrence$recurrence_class == "Network-private")
n_rec2 <- sum(recurrence$recurrence_class == "Shared by 2 networks")
n_rec3 <- sum(recurrence$recurrence_class == "Recurrent (>=3 networks)")
shared_by_cohort <- sapply(cohorts, function(co) {
  h <- hub_annot$node[hub_annot$cohort == co & hub_annot$group == "Healthy"]
  p <- hub_annot$node[hub_annot$cohort == co & hub_annot$group == "PCOS"]
  length(intersect(h, p))
})
rec3 <- recurrence %>%
  filter(n_networks >= 3) %>%
  arrange(desc(n_networks), node) %>%
  left_join(hub_annot %>% distinct(node, display, feature), by = "node")
md <- c(
  paste0("# Top-", TOP_N, " degree species UpSet (Fig3_network_combined_ComBat)"),
  "",
  "## Scope",
  "",
  paste0("- Networks: 6 cohort graphs; hub = top ", TOP_N, " by degree (ties: betweenness → closeness → name)."),
  "- Edge filter: CLR–ComBat; Spearman |r| ≥ 0.6, P < 0.05.",
  "",
  "## Core conclusion",
  "",
  sprintf(
    paste0(
      "Expanding from top-10 to top-%d, the union grows to **%d** species, ",
      "of which **%d** remain network-private. Recurrent hubs (≥3 networks): **%d**; ",
      "two-network shares: **%d**. Within-cohort Healthy∩PCOS overlap: ",
      "PRJNA549764=%d, PRJNA530971=%d, PRJNA791492=%d (of %d)."
    ),
    TOP_N, n_union, n_private, n_rec3, n_rec2,
    shared_by_cohort["PRJNA549764"], shared_by_cohort["PRJNA530971"],
    shared_by_cohort["PRJNA791492"], TOP_N
  ),
  "",
  "### Recurrent (≥3 networks)",
  ""
)
if (nrow(rec3)) {
  for (i in seq_len(nrow(rec3))) {
    md <- c(md, sprintf("- *%s* (n=%d; %s)",
                        rec3$display[i], rec3$n_networks[i], rec3$feature[i]))
  }
} else {
  md <- c(md, "- None.")
}
md <- c(
  md, "",
  "## Largest UpSet intersections",
  ""
)
for (i in seq_len(min(12, nrow(comb_size_df)))) {
  md <- c(md, sprintf("- size %d (sets=%d): `%s`",
                      comb_size_df$intersection_size[i],
                      comb_size_df$degree[i],
                      comb_size_df$combination[i]))
}
md <- c(
  md, "",
  "## Files",
  "",
  "| Output | Path |",
  "|--------|------|",
  paste0("| UpSet | `figures/hub_top20/composite/", STEM, "_UpSet.*` |"),
  paste0("| Presence heatmap | `figures/hub_top20/composite/", STEM, "_presence_heatmap.*` |"),
  paste0("| Feature composite | `figures/hub_top20/composite/", STEM, "_features_composite.*` |"),
  "| Annotated hubs | `tables/hub_top20/top20_hubs_annotated.tsv` |",
  paste0("| Source data | `source_data/", STEM, "_*_plotdata.tsv|csv` |"),
  ""
)
writeLines(md, file.path(out_fig, "hub_top20_upset_summary.md"))
writeLines(md, file.path(out_tab, "hub_top20_upset_summary.md"))
message("Done. Top-", TOP_N, " union=", n_union,
        "; private=", n_private,
        "; recurrent>=3=", n_rec3,
        "; UpSet combinations=", n_comb)
