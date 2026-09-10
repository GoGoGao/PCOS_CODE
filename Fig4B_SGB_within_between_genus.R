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
  library(phangorn)
  library(ggplot2)
  library(patchwork)
  library(dplyr)
  library(tidyr)
  library(ggtree)
  library(scales)
})
ROOT <- ""
IN <- file.path(ROOT, "input")
TAB <- file.path(ROOT, "tables")
FIG <- file.path(ROOT, "figures", "prevalence_phylo")
PLOT <- file.path(ROOT, "plotdata", "prevalence_phylo")
dir.create(FIG, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(FIG, "panels"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(FIG, "composite"), recursive = TRUE, showWarnings = FALSE)
dir.create(PLOT, recursive = TRUE, showWarnings = FALSE)
FONT <- "Times New Roman"
PAL <- list(
  neutral_dark = "#272727",
  neutral_mid  = "#767676",
  signal       = "#3C5488",
  accent       = "#E64B35",
  within       = "#00A087",
  between      = "#8491B4",
  loess        = "#E64B35"
)
theme_nat <- function(base_size = 8) {
  theme_classic(base_size = base_size, base_family = FONT) +
    theme(
      text = element_text(family = FONT, face = "plain", colour = "black"),
      axis.title = element_text(size = base_size, face = "plain"),
      axis.text = element_text(size = base_size - 0.5, face = "plain", colour = "black"),
      legend.title = element_text(size = base_size - 0.5, face = "plain"),
      legend.text = element_text(size = base_size - 1, face = "plain"),
      plot.title = element_text(size = base_size, face = "plain", hjust = 0),
      plot.tag = element_text(size = base_size + 1, face = "plain"),
      axis.line = element_line(linewidth = 0.35, colour = "black"),
      axis.ticks = element_line(linewidth = 0.35, colour = "black"),
      panel.grid = element_blank(),
      strip.background = element_blank(),
      strip.text = element_text(size = base_size, face = "plain"),
      legend.key.size = unit(3.2, "mm")
    )
}
theme_set(theme_nat())
save_table <- function(df, stem) {
  write.table(df, file.path(PLOT, paste0(stem, ".tsv")), sep = "\t",
              quote = FALSE, row.names = FALSE)
  write.csv(df, file.path(PLOT, paste0(stem, ".csv")), row.names = FALSE)
}
save_panel <- function(plot, stem, width_mm, height_mm, dpi = 600) {
  w <- width_mm / 25.4
  h <- height_mm / 25.4
  base <- file.path(FIG, "panels", stem)
  svglite::svglite(paste0(base, ".svg"), width = w, height = h)
  print(plot); dev.off()
  grDevices::cairo_pdf(paste0(base, ".pdf"), width = w, height = h, family = FONT)
  print(plot); dev.off()
  ragg::agg_png(paste0(base, ".png"), width = w, height = h, units = "in", res = dpi)
  print(plot); dev.off()
  ragg::agg_jpeg(paste0(base, ".jpg"), width = w, height = h, units = "in", res = dpi, quality = 95)
  print(plot); dev.off()
  ragg::agg_tiff(paste0(base, ".tiff"), width = w, height = h, units = "in", res = dpi, compression = "lzw")
  print(plot); dev.off()
}
save_composite <- function(plot, stem, width_mm, height_mm, dpi = 600) {
  w <- width_mm / 25.4
  h <- height_mm / 25.4
  base <- file.path(FIG, "composite", stem)
  svglite::svglite(paste0(base, ".svg"), width = w, height = h)
  print(plot); dev.off()
  grDevices::cairo_pdf(paste0(base, ".pdf"), width = w, height = h, family = FONT)
  print(plot); dev.off()
  ragg::agg_png(paste0(base, ".png"), width = w, height = h, units = "in", res = dpi)
  print(plot); dev.off()
  ragg::agg_jpeg(paste0(base, ".jpg"), width = w, height = h, units = "in", res = dpi, quality = 95)
  print(plot); dev.off()
  ragg::agg_tiff(paste0(base, ".tiff"), width = w, height = h, units = "in", res = dpi, compression = "lzw")
  print(plot); dev.off()
}
tree <- read.tree(file.path(IN, "SGB.unrooted.tree.nwk"))
if (!is.rooted(tree)) tree <- midpoint(tree)
meta <- read.delim(file.path(TAB, "tip_metadata_full.tsv"), check.names = FALSE)
lam <- read.delim(file.path(TAB, "route1_Pagel_lambda.tsv"), check.names = FALSE)
Kdf <- read.delim(file.path(TAB, "route1_Blomberg_K.tsv"), check.names = FALSE)
tips <- intersect(tree$tip.label, meta$SGB_ID)
tree <- keep.tip(tree, tips)
meta <- meta[match(tips, meta$SGB_ID), ]
rownames(meta) <- meta$SGB_ID
stopifnot(all(tree$tip.label == meta$SGB_ID))
D <- cophenetic(tree)
prev <- setNames(meta$Prevalence, meta$SGB_ID)
n <- length(tips)
idx <- which(upper.tri(D), arr.ind = TRUE)
phy <- D[idx]
dp <- abs(prev[tips[idx[, 1]]] - prev[tips[idx[, 2]]])
same_genus <- meta$Genus[idx[, 1]] == meta$Genus[idx[, 2]] &
  meta$Genus[idx[, 1]] != "" & !is.na(meta$Genus[idx[, 1]])
pair_df <- data.frame(
  tip_i = tips[idx[, 1]],
  tip_j = tips[idx[, 2]],
  phylo_dist = as.numeric(phy),
  abs_delta_prev = as.numeric(dp),
  same_genus = same_genus,
  stringsAsFactors = FALSE
)
set.seed(42)
mantel_r <- suppressWarnings(cor(pair_df$phylo_dist, pair_df$abs_delta_prev, method = "spearman"))
nperm <- 999
cnt <- 0L
prev_num <- as.numeric(prev)
for (i in seq_len(nperm)) {
  pr <- sample(prev_num)
  dpi <- abs(pr[idx[, 1]] - pr[idx[, 2]])
  ri <- suppressWarnings(cor(pair_df$phylo_dist, dpi, method = "spearman"))
  if (abs(ri) >= abs(mantel_r)) cnt <- cnt + 1L
}
mantel_p <- (cnt + 1) / (nperm + 1)
lam_prev <- lam$lambda[lam$trait == "Prevalence"]
lam_p <- lam$p_vs0[lam$trait == "Prevalence"]
K_prev <- Kdf$K[Kdf$trait == "Prevalence"]
K_p <- Kdf$P[Kdf$trait == "Prevalence"]
nbins <- 12
pair_df$bin <- cut_number(pair_df$phylo_dist, n = nbins)
bin_sum <- pair_df %>%
  group_by(bin) %>%
  summarise(
    n_pairs = n(),
    phylo_mid = median(phylo_dist),
    mean_abs_dp = mean(abs_delta_prev),
    se_abs_dp = sd(abs_delta_prev) / sqrt(n()),
    .groups = "drop"
  )
pair_gen <- pair_df %>%
  mutate(class = ifelse(same_genus, "Within genus", "Between genera"))
wt <- wilcox.test(abs_delta_prev ~ class, data = pair_gen)
med_tab <- pair_gen %>%
  group_by(class) %>%
  summarise(n = n(), median = median(abs_delta_prev), mean = mean(abs_delta_prev), .groups = "drop")
set.seed(1)
if (nrow(pair_df) > 80000) {
  pair_plot <- pair_df[sample.int(nrow(pair_df), 80000), ]
} else {
  pair_plot <- pair_df
}
stats_df <- data.frame(
  metric = c("Pagel_lambda", "Blomberg_K", "Mantel_Spearman_phylo_vs_absDeltaPrev",
             "Wilcoxon_within_vs_between_genus"),
  estimate = c(lam_prev, K_prev, mantel_r, as.numeric(wt$statistic)),
  p_value = c(lam_p, K_p, mantel_p, wt$p.value),
  n = c(n, n, nrow(pair_df), nrow(pair_gen)),
  note = c(
    "continuous phylogenetic signal",
    "PIC variance randomization (from route1)",
    sprintf("upper-triangle pairs=%s; permutations=%s", nrow(pair_df), nperm),
    sprintf("within median=%.3f; between median=%.3f",
            med_tab$median[med_tab$class == "Within genus"],
            med_tab$median[med_tab$class == "Between genera"])
  )
)
save_table(stats_df, "Fig_Prevalence_phylo_signal_stats")
save_table(bin_sum, "Fig_Prevalence_distance_decay_bins")
save_table(med_tab, "Fig_Prevalence_within_between_genus_summary")
save_table(pair_plot[, c("tip_i", "tip_j", "phylo_dist", "abs_delta_prev", "same_genus")],
           "Fig_A_distance_decay_pairs_plotdata")
save_table(pair_gen %>%
             group_by(class) %>%
             group_modify(~ {
               k <- min(nrow(.x), 20000L)
               dplyr::slice_sample(.x, n = k)
             }) %>%
             ungroup() %>%
             select(tip_i, tip_j, phylo_dist, abs_delta_prev, class),
           "Fig_B_within_between_genus_plotdata")
tree_ann <- data.frame(
  SGB_ID = meta$SGB_ID,
  Prevalence = meta$Prevalence,
  Phylum = meta$Phylum,
  Genus = meta$Genus,
  stringsAsFactors = FALSE
)
save_table(tree_ann, "Fig_C_tree_prevalence_plotdata")
ann_a <- sprintf(
  "Mantel r = %.2f, P = %.3f\nPagel's \u03bb = %.2f (P = %.1e)",
  mantel_r, mantel_p, lam_prev, lam_p
)
pA <- ggplot(pair_plot, aes(phylo_dist, abs_delta_prev)) +
  geom_hex(bins = 40, linewidth = 0) +
  scale_fill_gradient(
    name = "Pair\ncount",
    low = "#F7F7F7", high = PAL$signal,
    trans = "log10",
    breaks = c(1, 10, 100, 1000),
    labels = c("1", "10", "100", "1000")
  ) +
  geom_line(data = bin_sum, aes(phylo_mid, mean_abs_dp),
            inherit.aes = FALSE, colour = PAL$accent, linewidth = 0.7) +
  geom_point(data = bin_sum, aes(phylo_mid, mean_abs_dp),
             inherit.aes = FALSE, colour = PAL$accent, size = 1.4) +
  annotate("text", x = Inf, y = -Inf, label = ann_a,
           hjust = 1.05, vjust = -0.4, size = 2.4, family = FONT,
           colour = PAL$neutral_dark, lineheight = 1.05) +
  labs(
    x = "Phylogenetic distance",
    y = "|Δ prevalence|"
  ) +
  coord_cartesian(ylim = c(0, 1)) +
  theme_nat(8) +
  theme(legend.position = c(0.02, 0.98),
        legend.justification = c(0, 1),
        legend.background = element_blank())
pair_gen$class <- factor(pair_gen$class, levels = c("Within genus", "Between genera"))
set.seed(2)
pair_gen_draw <- bind_rows(
  pair_gen %>% filter(class == "Within genus"),
  pair_gen %>% filter(class == "Between genera") %>% slice_sample(n = 30000)
)
ann_b <- sprintf("Wilcoxon P = %.1e\nn(within) = %s",
                 wt$p.value,
                 format(med_tab$n[med_tab$class == "Within genus"], big.mark = ","))
pB <- ggplot(pair_gen_draw, aes(class, abs_delta_prev, fill = class)) +
  geom_violin(scale = "width", colour = NA, alpha = 0.85, linewidth = 0) +
  geom_boxplot(width = 0.18, outlier.size = 0.25, outlier.alpha = 0.25,
               colour = PAL$neutral_dark, fill = "white", linewidth = 0.3) +
  scale_fill_manual(values = c("Within genus" = PAL$within, "Between genera" = PAL$between),
                    guide = "none") +
  annotate("text", x = 1.5, y = 0.98, label = ann_b,
           size = 2.3, family = FONT, colour = PAL$neutral_dark, lineheight = 1.05) +
  labs(x = NULL, y = "|Δ prevalence|") +
  coord_cartesian(ylim = c(0, 1)) +
  theme_nat(8) +
  theme(axis.text.x = element_text(size = 7))
phy_keep <- names(sort(table(meta$Phylum), decreasing = TRUE))[1:6]
meta$Phylum_plot <- ifelse(meta$Phylum %in% phy_keep, meta$Phylum, "Other")
td <- data.frame(label = meta$SGB_ID, Prevalence = meta$Prevalence,
                 Phylum_plot = meta$Phylum_plot)
pC_base <- ggtree(tree, layout = "fan", open.angle = 12, linewidth = 0.12) +
  layout_circular()
pC_base <- ggtree(tree, layout = "fan", open.angle = 12) +
  geom_tree(linewidth = 0.12, colour = "#B0B0B0")
pC <- pC_base %<+% td +
  geom_tippoint(aes(colour = Prevalence), size = 0.55, stroke = 0) +
  scale_colour_gradient(
    name = "Prevalence",
    low = "#F4E6D0", high = "#3C5488",
    limits = c(0, 1),
    breaks = c(0, 0.5, 1)
  ) +
  theme_nat(8) +
  theme(
    legend.position = "right",
    legend.background = element_blank(),
    plot.margin = margin(2, 2, 2, 2)
  ) +
  annotate("text", x = -Inf, y = Inf,
           label = sprintf("n = %d SGBs", n),
           hjust = -0.05, vjust = 1.5,
           size = 2.3, family = FONT, colour = PAL$neutral_mid)
right <- (pB / pC) + plot_layout(heights = c(1, 1.15))
fig <- (pA | right) +
  plot_layout(widths = c(1.35, 1)) +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(family = FONT, face = "plain", size = 9))
message("Exporting panels...")
save_panel(pA, "Fig_Prevalence_phylo_A_distance_decay", width_mm = 95, height_mm = 78)
save_panel(pB, "Fig_Prevalence_phylo_B_within_between", width_mm = 70, height_mm = 55)
save_panel(pC, "Fig_Prevalence_phylo_C_tree", width_mm = 70, height_mm = 70)
message("Exporting composite...")
save_composite(fig, "Fig_Prevalence_phylo_signal", width_mm = 180, height_mm = 95)
legend_txt <- paste(
  "Phylogenetic structuring of SGB prevalence.",
  sprintf("(a) Pairwise phylogenetic distance versus absolute prevalence difference (|Δ prevalence|) across %d SGB pairs (hexbin density).", nrow(pair_df)),
  sprintf("Orange points/line show mean |Δ prevalence| in %d equal-count distance bins.", nbins),
  sprintf("Mantel Spearman r = %.3f (P = %.3f, %d permutations); Pagel's λ = %.3f for prevalence.", mantel_r, mantel_p, nperm, lam_prev),
  "(b) |Δ prevalence| is lower for within-genus pairs than between-genera pairs (two-sided Wilcoxon rank-sum test).",
  sprintf("(c) Midpoint-rooted SGB tree with tip colour indicating prevalence (n = %d).", n),
  "Prevalence is the fraction of samples with relative abundance > 0."
)
writeLines(legend_txt, file.path(FIG, "figure_legend.txt"))
message("Done.")
message(sprintf("Mantel r=%.4f P=%.4f; lambda=%.4f; within median=%.3f between=%.3f",
                mantel_r, mantel_p, lam_prev,
                med_tab$median[med_tab$class == "Within genus"],
                med_tab$median[med_tab$class == "Between genera"]))
