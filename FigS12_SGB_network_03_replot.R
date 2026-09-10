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
  library(igraph)
  library(tidygraph)
  library(ggraph)
  library(ggplot2)
  library(patchwork)
  library(ggpubr)
  library(scales)
  library(svglite)
  library(ragg)
  library(systemfonts)
  library(cowplot)
})
pdf.options(useDingbats = FALSE)
get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- "--file="
  sp <- sub(file_arg, "", args[grepl(file_arg, args)])
  if (length(sp) == 0) normalizePath(".") else normalizePath(dirname(sp))
}
script_dir <- get_script_dir()
root_dir <- normalizePath(file.path(script_dir, ".."))
combat_root <- root_dir
combat_result <- file.path(combat_root, "result")
out_fig <- file.path(root_dir, "figures")
out_src <- file.path(root_dir, "source_data")
out_tab <- file.path(root_dir, "tables")
for (d in c(out_fig, out_src, out_tab,
            file.path(out_fig, c("A", "B", "C", "D", "E", "F", "panels", "composite")))) {
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
}
tnr_path <- tryCatch(
  systemfonts::match_fonts("Times New Roman")$path[[1]],
  error = function(e) "/usr/share/fonts/msttcore/times.ttf"
)
if (is.null(tnr_path) || !nzchar(tnr_path) || !file.exists(tnr_path)) {
  tnr_path <- "/usr/share/fonts/msttcore/times.ttf"
}
FONT_FAMILY <- "Times New Roman"
if (requireNamespace("systemfonts", quietly = TRUE) && file.exists(tnr_path)) {
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
R_THRESHOLD <- 0.6
P_THRESHOLD <- 0.05
SUBTITLE_NET <- sprintf(
  "SGB CLR–ComBat; Spearman |r|>=%.1f, P<%.2f; top-250 bins; Louvain modules; size∝abundance",
  R_THRESHOLD, P_THRESHOLD
)
EDGE_POS <- "#C47A7A"
EDGE_NEG <- "#7A9AB8"
GROUP_FILL <- c(Healthy = "#7AA9C9", PCOS = "#D08A90")
GROUP_COL  <- c(Healthy = "#0072B2", PCOS = "#B2182B")
MODULE_COLS <- c(
  "#77AADD", "#EE8866", "#EEDD88", "#FFAABB", "#99DDFF",
  "#44BB99", "#BBCC33", "#AAAA00", "#DDDDDD", "#BBBBBB",
  "#CC6677", "#882255", "#6699CC", "#117733", "#88CCAA",
  "#AA4499", "#DDCC77", "#88AADD", "#44AA99", "#BB6677"
)
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
      panel.border = element_blank(),
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
export_figure <- function(plot, tsv_df, stem, width_mm, height_mm, also_dirs = NULL) {
  path_stem <- file.path(out_fig, "panels", stem)
  write_plotdata(tsv_df, stem)
  save_pub(plot, path_stem, width_mm = width_mm, height_mm = height_mm)
  if (!is.null(also_dirs)) {
    for (d in also_dirs) {
      for (ext in c("pdf", "svg", "png", "jpg")) {
        src <- paste0(path_stem, ".", ext)
        dst <- file.path(out_fig, d, paste0(stem, ".", ext))
        if (file.exists(src) && normalizePath(src) != normalizePath(dst, mustWork = FALSE)) {
          file.copy(src, dst, overwrite = TRUE)
        }
      }
    }
  }
  message("Saved: ", stem)
}
NETS <- tribble(
  ~panel, ~cohort,        ~group,    ~tag,
  "A",    "overall",      "Healthy", "a",
  "A",    "overall",      "PCOS",    "a",
  "B",    "PRJNA549764",  "Healthy", "b",
  "B",    "PRJNA549764",  "PCOS",    "b",
  "C",    "PRJNA530971",  "Healthy", "c",
  "C",    "PRJNA530971",  "PCOS",    "c",
  "D",    "PRJNA791492",  "Healthy", "d",
  "D",    "PRJNA791492",  "PCOS",    "d"
) %>%
  mutate(
    net_dir = file.path(combat_result, cohort, group),
    edge_csv = file.path(net_dir, paste0(group, "_Gephi_edge.csv")),
    node_csv = file.path(net_dir, paste0(group, "_Gephi_allnode.csv")),
    prop_csv = file.path(net_dir, paste0(group, "_node_properties.csv")),
    label = paste(cohort, group, sep = " | ")
  )
missing <- NETS %>% filter(!file.exists(edge_csv) | !file.exists(node_csv))
if (nrow(missing) > 0) {
  stop("Missing ComBat network files:\n",
       paste(missing$edge_csv, collapse = "\n"))
}
layout_modular_sunflower <- function(g, memb, seed = 1) {
  set.seed(seed)
  n <- igraph::vcount(g)
  xy <- matrix(0, nrow = n, ncol = 2)
  mods <- sort(unique(memb))
  n_mod <- length(mods)
  sizes <- as.numeric(table(factor(memb, levels = mods)))
  rad <- 0.12 + 0.28 * sqrt(sizes / max(sizes))
  golden <- pi * (3 - sqrt(5))
  centres <- matrix(0, n_mod, 2)
  for (i in seq_len(n_mod)) {
    r <- sqrt(i / max(n_mod, 1)) * 0.78
    th <- i * golden
    centres[i, ] <- c(r * cos(th), r * sin(th))
  }
  for (i in seq_along(mods)) {
    idx <- which(memb == mods[i])
    sub <- igraph::induced_subgraph(g, idx)
    nv <- igraph::vcount(sub)
    if (nv == 1L) {
      ly <- cbind(0, 0)
    } else if (nv == 2L) {
      ly <- cbind(c(-0.15, 0.15), c(0, 0))
    } else {
      ly <- igraph::layout_with_graphopt(
        sub, niter = 450, charge = 0.05, spring.constant = 0.8
      )
      ly <- scale(ly, center = TRUE, scale = FALSE)
      rmax <- max(sqrt(rowSums(ly^2)), 1e-6)
      ly <- ly / rmax * rad[i]
    }
    xy[idx, 1] <- ly[, 1] + centres[i, 1]
    xy[idx, 2] <- ly[, 2] + centres[i, 2]
  }
  xy <- scale(xy, center = TRUE, scale = FALSE)
  rmax <- max(sqrt(rowSums(xy^2)), 1e-6)
  xy <- xy / rmax
  colnames(xy) <- c("x", "y")
  xy
}
fmt_num <- function(x) {
  ifelse(abs(x) >= 100, formatC(x, format = "f", digits = 0),
         ifelse(abs(x) >= 10, formatC(x, format = "f", digits = 1),
                formatC(x, format = "f", digits = 3)))
}
build_group_graph <- function(edge_csv, node_csv, prop_csv, seed = 2026,
                              r_threshold = R_THRESHOLD) {
  nodes <- read.csv(node_csv, stringsAsFactors = FALSE, check.names = FALSE)
  edges <- read.csv(edge_csv, stringsAsFactors = FALSE, check.names = FALSE)
  props <- if (file.exists(prop_csv)) {
    read.csv(prop_csv, stringsAsFactors = FALSE, check.names = FALSE)
  } else {
    NULL
  }
  if (!is.null(props)) {
    colnames(props)[1] <- "ID"
    if ("igraph.degree" %in% names(props)) {
      props <- dplyr::rename(props, degree = igraph.degree)
    }
  }
  edges <- edges %>%
    mutate(
      correlation = as.numeric(correlation),
      cor_sign = ifelse(correlation > 0 | cor == "+", "Positive", "Negative"),
      weight_raw = abs(correlation)
    ) %>%
    filter(is.finite(correlation), abs(correlation) >= r_threshold)
  if (nrow(edges) == 0) {
    stop("No edges retained at |r|>=", r_threshold, " for ", edge_csv)
  }
  if (!is.null(props) && "degree" %in% names(props)) {
    nodes <- nodes %>%
      left_join(props %>% select(ID, any_of(c("degree", "igraph.closeness",
                                              "igraph.betweenness",
                                              "igraph.cen.degree"))),
                by = "ID")
  } else {
    nodes$degree <- NA_real_
  }
  nodes <- nodes %>%
    mutate(
      mean = as.numeric(mean),
      degree = as.numeric(degree)
    )
  keep <- unique(c(edges$source, edges$target))
  nodes <- nodes %>% filter(ID %in% keep)
  g0 <- graph_from_data_frame(
    d = edges[, c("source", "target", "cor_sign", "correlation", "weight_raw")],
    vertices = nodes %>%
      select(name = ID, mean, degree, any_of(c("Kingdom", "Phylum", "Genus", "Species"))),
    directed = FALSE
  )
  g0 <- simplify(
    g0, remove.multiple = TRUE, remove.loops = TRUE,
    edge.attr.comb = list(
      cor_sign = "first", correlation = "first", weight_raw = "first", "ignore"
    )
  )
  deg <- igraph::degree(g0)
  V(g0)$degree <- as.numeric(deg)
  mean_v <- V(g0)$mean
  if (all(is.finite(mean_v)) && all(mean_v > 0, na.rm = TRUE) &&
      max(mean_v, na.rm = TRUE) > min(mean_v, na.rm = TRUE)) {
    V(g0)$size_var <- scales::rescale(log10(mean_v + 1e-8), to = c(0.7, 3.2))
  } else {
    V(g0)$size_var <- scales::rescale(as.numeric(deg), to = c(0.7, 3.2))
  }
  set.seed(seed)
  cl <- cluster_louvain(g0)
  memb <- as.integer(membership(cl))
  V(g0)$module <- memb
  V(g0)$module_lab <- paste0("M", memb)
  el <- as_edgelist(g0, names = FALSE)
  intra <- memb[el[, 1]] == memb[el[, 2]]
  n_e <- ecount(g0)
  base_lo <- ifelse(n_e > 3500, 0.008, ifelse(n_e > 2200, 0.012, 0.018))
  base_hi <- ifelse(n_e > 3500, 0.12,  ifelse(n_e > 2200, 0.16,  0.22))
  E(g0)$edge_alpha <- ifelse(intra, base_hi, base_lo)
  E(g0)$edge_width <- ifelse(intra, 0.28, 0.05)
  E(g0)$intra_module <- ifelse(intra, "Intra-module", "Inter-module")
  xy <- layout_modular_sunflower(g0, memb, seed = seed)
  tg <- as_tbl_graph(g0)
  lay <- create_layout(tg, layout = "manual", x = xy[, 1], y = xy[, 2])
  n_nodes <- vcount(g0)
  n_edges <- ecount(g0)
  n_pos <- sum(E(g0)$cor_sign == "Positive")
  n_neg <- sum(E(g0)$cor_sign == "Negative")
  avg_k <- mean(degree(g0))
  dens <- edge_density(g0)
  list(
    layout = lay, graph = g0, membership = memb,
    n_modules = length(unique(memb)),
    stats = list(
      n_nodes = n_nodes, n_edges = n_edges, n_pos = n_pos, n_neg = n_neg,
      avg_degree = avg_k, connectance = dens
    )
  )
}
make_annot_label <- function(st) {
  paste0(
    "n=", fmt_num(st$n_nodes),
    "  L=", fmt_num(st$n_edges),
    "\nK=", fmt_num(st$avg_degree),
    "  C=", fmt_num(st$connectance),
    "\n+/−=", fmt_num(st$n_pos), "/", fmt_num(st$n_neg)
  )
}
plot_one_network <- function(obj, title, show_legend = FALSE, tag = NULL,
                             show_stats_annot = TRUE, compact = FALSE) {
  lay <- obj$layout
  lab <- make_annot_label(obj$stats)
  n_mod <- obj$n_modules
  mod_ids <- sort(unique(obj$membership))
  mod_cols <- setNames(
    MODULE_COLS[((seq_along(mod_ids) - 1) %% length(MODULE_COLS)) + 1],
    paste0("M", mod_ids)
  )
  lay$module_lab <- factor(paste0("M", lay$module), levels = names(mod_cols))
  title_size <- if (compact) 8.5 else 9.5
  tag_size <- if (compact) 10 else 11
  expand_mult <- if (compact) 0.03 else 0.06
  marg <- if (compact) margin(1, 1, 1, 1) else margin(2, 2, 2, 2)
  p <- ggraph(lay) +
    geom_edge_link(
      aes(colour = cor_sign, alpha = edge_alpha, width = edge_width,
          filter = intra_module == "Inter-module"),
      show.legend = FALSE
    ) +
    geom_edge_link(
      aes(colour = cor_sign, alpha = edge_alpha, width = edge_width,
          filter = intra_module == "Intra-module"),
      show.legend = show_legend
    ) +
    geom_node_point(
      aes(size = size_var, fill = module_lab),
      shape = 21, colour = "black", stroke = 0.12, alpha = 0.95
    ) +
    scale_edge_colour_manual(
      name = "Edge",
      values = c(Positive = EDGE_POS, Negative = EDGE_NEG),
      guide = if (show_legend) {
        guide_legend(order = 1, override.aes = list(alpha = 1, width = 0.7))
      } else {
        "none"
      }
    ) +
    scale_edge_alpha_identity(guide = "none") +
    scale_edge_width_identity(guide = "none") +
    scale_fill_manual(name = "Module", values = mod_cols, guide = "none") +
    scale_size_identity() +
    labs(title = title, tag = tag) +
    theme_void(base_size = 8.5, base_family = FONT_FAMILY) +
    theme(
      plot.title = element_text(
        size = title_size, hjust = 0.5, colour = "black",
        family = FONT_FAMILY, face = "plain",
        margin = margin(b = 0, t = 0)
      ),
      plot.tag = element_text(size = tag_size, family = FONT_FAMILY, colour = "black",
                             face = "plain"),
      legend.position = if (show_legend) "bottom" else "none",
      legend.box = "horizontal",
      legend.title = element_text(size = 8, family = FONT_FAMILY, colour = "black",
                                  face = "plain"),
      legend.text = element_text(size = 7.5, family = FONT_FAMILY, colour = "black",
                                 face = "plain"),
      plot.margin = marg,
      panel.border = element_blank(),
      plot.background = element_rect(fill = "white", colour = NA),
      panel.background = element_rect(fill = "white", colour = NA)
    ) +
    scale_x_continuous(expand = expansion(mult = expand_mult)) +
    scale_y_continuous(expand = expansion(mult = expand_mult))
  if (isTRUE(compact)) {
    p <- p + theme(aspect.ratio = 1) + coord_cartesian(clip = "off")
  } else {
    p <- p + coord_equal(clip = "off")
  }
  if (isTRUE(show_stats_annot)) {
    p <- p + annotate(
      "text", x = -Inf, y = Inf, label = lab,
      hjust = -0.05, vjust = 1.2, size = 2.6, lineheight = 0.95,
      colour = "black", family = FONT_FAMILY
    )
  }
  p
}
make_topology_table_plot <- function(stats_df, tag = "G",
                                     panel_map = NULL) {
  if (is.null(panel_map)) {
    panel_map <- c(
      `PRJNA549764__Healthy` = "A",
      `PRJNA549764__PCOS` = "B",
      `PRJNA530971__Healthy` = "C",
      `PRJNA530971__PCOS` = "D",
      `PRJNA791492__Healthy` = "E",
      `PRJNA791492__PCOS` = "F"
    )
  }
  tab <- stats_df %>%
    mutate(key = paste(cohort, group, sep = "__")) %>%
    filter(key %in% names(panel_map)) %>%
    mutate(
      Panel = unname(panel_map[key]),
      Cohort = as.character(cohort),
      Group = as.character(group),
      Nodes = as.character(as.integer(n_nodes)),
      Edges = as.character(as.integer(n_edges)),
      `Mean degree` = as.character(fmt_num(avg_degree)),
      Connectance = as.character(fmt_num(connectance)),
      `Positive/Negative` = paste0(as.integer(n_pos), "/", as.integer(n_neg)),
      Modules = as.character(as.integer(n_modules))
    ) %>%
    arrange(Panel) %>%
    select(Panel, Cohort, Group, Nodes, Edges, `Mean degree`, Connectance,
           `Positive/Negative`, Modules)
  cols <- names(tab)
  col_w <- c(Panel = 0.65, Cohort = 1.45, Group = 0.95, Nodes = 0.75,
             Edges = 0.8, `Mean degree` = 1.15, Connectance = 1.15,
             `Positive/Negative` = 1.45, Modules = 0.85)
  col_w <- col_w[cols]
  x_right <- cumsum(as.numeric(col_w))
  x_left <- c(0, x_right[-length(x_right)])
  x_mid <- (x_left + x_right) / 2
  names(x_mid) <- cols
  x_max <- max(x_right)
  nrows <- nrow(tab)
  ncols <- length(cols)
  body <- tab %>%
    mutate(row_id = dplyr::row_number()) %>%
    tidyr::pivot_longer(
      cols = -row_id, names_to = "col_name", values_to = "label"
    ) %>%
    mutate(
      col = match(col_name, cols),
      row = row_id,
      label = as.character(label),
      is_header = FALSE,
      x = unname(x_mid[col_name]),
      xmin = unname(x_left[col]),
      xmax = unname(x_right[col])
    )
  header <- tibble(
    row = 0L,
    col = seq_len(ncols),
    col_name = cols,
    label = cols,
    is_header = TRUE,
    x = unname(x_mid[cols]),
    xmin = unname(x_left),
    xmax = unname(x_right)
  )
  cells <- bind_rows(header, body) %>%
    mutate(y = as.numeric(nrows + 1L - row))
  row_bands <- tibble(
    row = 0:nrows,
    y = as.numeric(nrows + 1L - row),
    is_header = row == 0L,
    fill = dplyr::case_when(
      row == 0L ~ "#F0F0F0",
      row %% 2L == 0L ~ "#FAFAFA",
      TRUE ~ "#FFFFFF"
    )
  )
  p <- ggplot() +
    geom_rect(
      data = row_bands,
      aes(xmin = 0, xmax = x_max, ymin = y - 0.5, ymax = y + 0.5, fill = fill),
      colour = NA, show.legend = FALSE
    ) +
    scale_fill_identity() +
    geom_text(
      data = dplyr::filter(cells, is_header),
      aes(x = x, y = y, label = label),
      family = FONT_FAMILY, colour = "black", size = 2.45,
      fontface = "plain"
    ) +
    geom_text(
      data = dplyr::filter(cells, !is_header),
      aes(x = x, y = y, label = label),
      family = FONT_FAMILY, colour = "black", size = 2.85,
      fontface = "plain"
    ) +
    geom_hline(
      yintercept = seq(0.5, nrows + 1.5, by = 1),
      colour = "black", linewidth = 0.3
    ) +
    geom_vline(
      xintercept = c(0, x_right),
      colour = "black", linewidth = 0.3
    ) +
    scale_x_continuous(limits = c(0, x_max), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0.5, nrows + 1.5), expand = c(0, 0)) +
    coord_cartesian(clip = "off") +
    labs(tag = tag) +
    theme_void(base_family = FONT_FAMILY) +
    theme(
      plot.tag = element_text(size = 11, family = FONT_FAMILY, colour = "black",
                             face = "plain", margin = margin(0, 2, 0, 0)),
      plot.tag.position = c(0, 1),
      plot.margin = margin(2, 2, 2, 2),
      plot.background = element_rect(fill = "white", colour = NA)
    )
  list(plot = p, table = tab)
}
obj_to_tsv <- function(obj, cohort, group) {
  nodes_out <- as_tibble(as_data_frame(obj$graph, what = "vertices")) %>%
    transmute(
      cohort = cohort, group = group, layer = "node", ID = name,
      module = module, mean = as.numeric(mean),
      degree = as.numeric(degree), size_var = as.numeric(size_var),
      cor_sign = NA_character_, intra_module = NA_character_,
      correlation = NA_real_
    )
  edges_out <- as_tibble(as_data_frame(obj$graph, what = "edges")) %>%
    transmute(
      cohort = cohort, group = group, layer = "edge",
      ID = paste(from, to, sep = "|"),
      module = NA_integer_, mean = NA_real_, degree = NA_real_,
      size_var = as.numeric(edge_alpha),
      cor_sign, intra_module,
      correlation = as.numeric(correlation)
    )
  bind_rows(nodes_out, edges_out)
}
message("Building ComBat co-occurrence graphs ...")
graph_objs <- list()
all_tsv <- list()
stats_rows <- list()
for (i in seq_len(nrow(NETS))) {
  row <- NETS[i, ]
  key <- paste(row$cohort, row$group, sep = "__")
  message("  ", key)
  obj <- build_group_graph(row$edge_csv, row$node_csv, row$prop_csv, seed = 2026 + i)
  graph_objs[[key]] <- obj
  all_tsv[[key]] <- obj_to_tsv(obj, row$cohort, row$group)
  stats_rows[[key]] <- tibble(
    cohort = row$cohort, group = row$group, panel = row$panel,
    n_nodes = obj$stats$n_nodes, n_edges = obj$stats$n_edges,
    n_pos = obj$stats$n_pos, n_neg = obj$stats$n_neg,
    avg_degree = obj$stats$avg_degree, connectance = obj$stats$connectance,
    n_modules = obj$n_modules
  )
}
stats_df <- bind_rows(stats_rows)
write_tsv(stats_df, file.path(out_tab, "network_topology_summary.tsv"))
write_csv(stats_df, file.path(out_tab, "network_topology_summary.csv"))
pair_map <- list(
  A = list(cohort = "overall",     title = "Overall SGB (ComBat)"),
  B = list(cohort = "PRJNA549764", title = "PRJNA549764 SGB (ComBat)"),
  C = list(cohort = "PRJNA530971", title = "PRJNA530971 SGB (ComBat)"),
  D = list(cohort = "PRJNA791492", title = "PRJNA791492 SGB (ComBat)")
)
pair_plots <- list()
for (pn in names(pair_map)) {
  coh <- pair_map[[pn]]$cohort
  ttl <- pair_map[[pn]]$title
  keys <- c(paste0(coh, "__Healthy"), paste0(coh, "__PCOS"))
  p_h <- plot_one_network(graph_objs[[keys[1]]], "Healthy", show_legend = FALSE)
  p_p <- plot_one_network(graph_objs[[keys[2]]], "PCOS", show_legend = FALSE)
  p_pair <- (p_h | p_p) +
    plot_annotation(
      title = ttl,
      subtitle = SUBTITLE_NET,
      theme = theme(
        plot.title = element_text(size = 10.5, family = FONT_FAMILY, colour = "black",
                                  face = "plain"),
        plot.subtitle = element_text(size = 7.5, family = FONT_FAMILY, colour = "black",
                                     face = "plain"),
        plot.background = element_rect(fill = "white", colour = NA)
      )
    )
  tsv_pair <- bind_rows(all_tsv[[keys[1]]], all_tsv[[keys[2]]])
  stem <- paste0("Fig3", pn, "_network_", coh)
  export_figure(p_pair, tsv_pair, stem, width_mm = 180, height_mm = 95,
                also_dirs = pn)
  for (g in c("Healthy", "PCOS")) {
    k <- paste0(coh, "__", g)
    p1 <- plot_one_network(graph_objs[[k]], g, show_legend = TRUE)
    stem1 <- paste0("Fig3", pn, "_", coh, "_", g)
    export_figure(p1, all_tsv[[k]], stem1, width_mm = 90, height_mm = 88,
                  also_dirs = pn)
  }
  pair_plots[[pn]] <- p_pair
}
message("Building combined Figure 3 network composite ...")
cohort_order <- c("PRJNA549764", "PRJNA530971", "PRJNA791492")
panel_map_combined <- c(
  `PRJNA549764__Healthy` = "A",
  `PRJNA549764__PCOS` = "B",
  `PRJNA530971__Healthy` = "C",
  `PRJNA530971__PCOS` = "D",
  `PRJNA791492__Healthy` = "E",
  `PRJNA791492__PCOS` = "F"
)
MARGIN_LR <- margin(1, 4, 1, 4)
cohort_cols <- list()
for (coh in cohort_order) {
  plots_g <- list()
  for (g in c("Healthy", "PCOS")) {
    k <- paste0(coh, "__", g)
    tag <- unname(panel_map_combined[k])
    plots_g[[g]] <- plot_one_network(
      graph_objs[[k]], title = g, show_legend = FALSE, tag = tag,
      show_stats_annot = FALSE, compact = TRUE
    ) + theme(plot.margin = MARGIN_LR)
  }
  cohort_cols[[coh]] <- (plots_g$Healthy / plots_g$PCOS) +
    plot_layout(heights = c(1, 1)) +
    plot_annotation(
      title = coh,
      theme = theme(
        plot.title = element_text(
          size = 9, hjust = 0.5, family = FONT_FAMILY, colour = "black",
          face = "plain", margin = margin(b = 1, t = 0)
        ),
        plot.margin = MARGIN_LR
      )
    )
}
net_grid <- (cohort_cols[[1]] | cohort_cols[[2]] | cohort_cols[[3]]) +
  plot_layout(widths = c(1, 1, 1))
p_leg_src <- ggplot(
  data.frame(
    Edge = factor(c("Positive", "Negative"), levels = c("Positive", "Negative")),
    x = c(1, 2), y = 1
  )
) +
  geom_segment(aes(x = x - 0.35, xend = x + 0.35, y = y, yend = y, colour = Edge),
               linewidth = 1.1) +
  scale_colour_manual(values = c(Positive = EDGE_POS, Negative = EDGE_NEG)) +
  labs(colour = "Edge") +
  theme_void(base_family = FONT_FAMILY) +
  theme(
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.title = element_text(size = 8.5, family = FONT_FAMILY, colour = "black",
                                face = "plain"),
    legend.text = element_text(size = 8, family = FONT_FAMILY, colour = "black",
                               face = "plain"),
    legend.margin = margin(0, 0, 0, 0),
    legend.box.margin = margin(0, 0, 0, 0)
  ) +
  guides(colour = guide_legend(title.position = "left", nrow = 1,
                               override.aes = list(linewidth = 1.3)))
leg_strip <- patchwork::wrap_elements(
  full = cowplot::ggdraw() +
    cowplot::draw_grob(cowplot::get_legend(p_leg_src), x = 0.28, y = 0.0,
                       width = 0.44, height = 1.0)
) & theme(plot.margin = MARGIN_LR)
tab_G <- make_topology_table_plot(
  stats_df %>% dplyr::filter(cohort != "overall"),
  tag = "G",
  panel_map = panel_map_combined
)
tab_G$plot <- tab_G$plot + theme(plot.margin = MARGIN_LR)
write_tsv(tab_G$table, file.path(out_tab, "Fig3G_network_topology_table.tsv"))
write_csv(tab_G$table, file.path(out_tab, "Fig3G_network_topology_table.csv"))
write_plotdata(tab_G$table, "Fig3G_network_topology_table")
write_tsv(tab_G$table, file.path(out_tab, "Fig3I_network_topology_table.tsv"))
write_csv(tab_G$table, file.path(out_tab, "Fig3I_network_topology_table.csv"))
write_plotdata(tab_G$table, "Fig3I_network_topology_table")
fig_combined <- (leg_strip / net_grid / tab_G$plot) +
  plot_layout(heights = c(0.04, 1, 0.30)) +
  plot_annotation(
    title = "SGB cohort co-occurrence networks after CLR–ComBat batch correction",
    theme = theme(
      plot.title = element_text(size = 11, family = FONT_FAMILY, colour = "black",
                                face = "plain", hjust = 0.5, margin = margin(b = 4)),
      plot.background = element_rect(fill = "white", colour = NA),
      plot.margin = margin(2, 2, 2, 2)
    )
  )
tsv_combined <- bind_rows(all_tsv[names(all_tsv)[grepl("^PRJNA", names(all_tsv))]])
export_figure(fig_combined, tsv_combined, "Fig3_network_combined_ComBat",
              width_mm = 183, height_mm = 200, also_dirs = "composite")
export_figure(tab_G$plot, tab_G$table, "Fig3G_network_topology_table",
              width_mm = 183, height_mm = 48, also_dirs = "composite")
export_figure(tab_G$plot, tab_G$table, "Fig3I_network_topology_table",
              width_mm = 183, height_mm = 48, also_dirs = "composite")
tsv_all <- bind_rows(all_tsv)
message("Recomputing node properties at |r|>=", R_THRESHOLD, " ...")
node_props_from_graph <- function(g, cohort, group, level) {
  comp <- igraph::components(g)
  lcc <- which(comp$membership == which.max(comp$csize))
  g_lcc <- igraph::induced_subgraph(g, lcc)
  deg_all <- igraph::degree(g)
  clo_lcc <- igraph::closeness(g_lcc, normalized = TRUE)
  bet_lcc <- igraph::betweenness(g_lcc, normalized = FALSE)
  cen_all <- igraph::degree(g, normalized = TRUE)
  nm <- igraph::V(g)$name
  clo <- setNames(rep(NA_real_, length(nm)), nm)
  bet <- setNames(rep(NA_real_, length(nm)), nm)
  clo[igraph::V(g_lcc)$name] <- as.numeric(clo_lcc)
  bet[igraph::V(g_lcc)$name] <- as.numeric(bet_lcc)
  cohort_out <- if (identical(as.character(level), "overall")) {
    NA_character_
  } else {
    as.character(cohort)
  }
  level_out <- as.character(level)
  group_out <- as.character(group)
  tibble(
    node = nm,
    degree = as.numeric(deg_all),
    closeness = as.numeric(clo[nm]),
    betweenness = as.numeric(bet[nm]),
    cen.degree = as.numeric(cen_all),
    level = level_out,
    cohort = cohort_out,
    group = group_out
  ) %>%
    filter(is.finite(closeness), is.finite(betweenness)) %>%
    pivot_longer(
      cols = c(degree, closeness, betweenness, cen.degree),
      names_to = "property", values_to = "value"
    )
}
node_rows <- list()
for (key in names(graph_objs)) {
  parts <- strsplit(key, "__", fixed = TRUE)[[1]]
  coh <- parts[1]; grp <- parts[2]
  level <- if (coh == "overall") "overall" else "cohort"
  node_rows[[key]] <- node_props_from_graph(graph_objs[[key]]$graph, coh, grp, level)
}
node_df <- bind_rows(node_rows) %>%
  mutate(
    group = factor(group, levels = c("Healthy", "PCOS")),
    property = factor(property, levels = c("degree", "closeness", "betweenness", "cen.degree")),
    property_lab = recode(
      as.character(property),
      degree = "Degree",
      closeness = "Closeness",
      betweenness = "Betweenness",
      cen.degree = "Centralized degree"
    ),
    property_lab = factor(
      property_lab,
      levels = c("Degree", "Closeness", "Betweenness", "Centralized degree")
    )
  )
write_tsv(node_df, file.path(out_tab, "node_properties_long_r0.6.tsv"))
write_csv(node_df, file.path(out_tab, "node_properties_long_r0.6.csv"))
plot_node_props <- function(df, title) {
  ggplot(df, aes(x = group, y = value)) +
    geom_jitter(
      aes(colour = group), width = 0.16, height = 0,
      size = 0.5, alpha = 0.22, stroke = 0, show.legend = FALSE
    ) +
    geom_violin(aes(fill = group), trim = TRUE, alpha = 0.35,
                colour = NA, width = 0.85) +
    geom_boxplot(
      aes(fill = group), width = 0.22, outlier.shape = NA,
      colour = "black", linewidth = 0.3, alpha = 0.9
    ) +
    facet_wrap(~ property_lab, scales = "free_y", nrow = 1) +
    stat_compare_means(
      comparisons = list(c("Healthy", "PCOS")),
      method = "wilcox.test",
      label = "p.format",
      hide.ns = FALSE,
      vjust = -0.2,
      tip.length = 0.01,
      size = 2.8,
      family = FONT_FAMILY,
      color = "black"
    ) +
    scale_fill_manual(values = GROUP_FILL) +
    scale_colour_manual(values = GROUP_FILL) +
    scale_y_continuous(expand = expansion(mult = c(0.02, 0.22))) +
    labs(title = title, x = NULL, y = "Node property", fill = "Group") +
    theme_nature(base_size = 8.5) +
    theme(
      legend.position = "right",
      panel.spacing = unit(2.5, "mm")
    )
}
df_e <- node_df %>% filter(level == "overall")
p_e <- plot_node_props(
  df_e,
  sprintf("Overall SGB node properties (ComBat; |r|≥%.1f)", R_THRESHOLD)
)
export_figure(p_e, df_e, "Fig3E_overall_node_properties",
              width_mm = 183, height_mm = 70, also_dirs = "E")
df_f <- node_df %>% filter(level == "cohort") %>%
  mutate(cohort = factor(cohort, levels = c("PRJNA530971", "PRJNA549764", "PRJNA791492")))
p_f <- ggplot(df_f, aes(x = group, y = value)) +
  geom_jitter(
    aes(colour = group), width = 0.14, height = 0,
    size = 0.4, alpha = 0.18, stroke = 0, show.legend = FALSE
  ) +
  geom_violin(aes(fill = group), trim = TRUE, alpha = 0.30,
              colour = NA, width = 0.85) +
  geom_boxplot(
    aes(fill = group), width = 0.20, outlier.shape = NA,
    colour = "black", linewidth = 0.28, alpha = 0.9
  ) +
  facet_grid(cohort ~ property_lab, scales = "free_y") +
  stat_compare_means(
    comparisons = list(c("Healthy", "PCOS")),
    method = "wilcox.test",
    label = "p.format",
    hide.ns = FALSE,
    vjust = -0.15,
    tip.length = 0.008,
    size = 2.4,
    family = FONT_FAMILY,
    color = "black"
  ) +
  scale_fill_manual(values = GROUP_FILL) +
  scale_colour_manual(values = GROUP_FILL) +
  scale_y_continuous(expand = expansion(mult = c(0.02, 0.28))) +
  labs(
    title = sprintf(
      "Cohort-stratified SGB node properties (ComBat; |r|≥%.1f)", R_THRESHOLD
    ),
    subtitle = "Primary inference level; closeness on largest connected component; PRJNA549764 small-n caution",
    x = NULL, y = "Node property", fill = "Group"
  ) +
  theme_nature(base_size = 8.5) +
  theme(
    legend.position = "right",
    panel.spacing = unit(2.0, "mm"),
    strip.text.y = element_text(size = 7, angle = 0, colour = "black", face = "plain")
  )
export_figure(p_f, df_f, "Fig3F_cohort_node_properties",
              width_mm = 200, height_mm = 150, also_dirs = "F")
mean_deg <- stats_df %>%
  mutate(
    cohort = factor(cohort, levels = c("overall", "PRJNA530971", "PRJNA549764", "PRJNA791492")),
    group = factor(group, levels = c("Healthy", "PCOS"))
  )
p_deg <- ggplot(mean_deg, aes(x = cohort, y = avg_degree, fill = group)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.62, colour = "black",
           linewidth = 0.25) +
  geom_text(
    aes(label = fmt_num(avg_degree)),
    position = position_dodge(width = 0.7), vjust = -0.4, size = 2.8,
    family = FONT_FAMILY, colour = "black"
  ) +
  scale_fill_manual(values = GROUP_FILL) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(
    title = sprintf(
      "SGB mean degree after CLR–ComBat (|r|≥%.1f)", R_THRESHOLD
    ),
    x = NULL, y = "Average degree", fill = "Group"
  ) +
  theme_nature(base_size = 8.5) +
  theme(axis.text.x = element_text(angle = 20, hjust = 1, colour = "black"))
export_figure(p_deg, mean_deg, "Fig_mean_degree_ComBat_summary",
              width_mm = 140, height_mm = 70, also_dirs = "composite")
message("Assembling full Figure 3 composite ...")
fig3_full <- (
  (pair_plots$A / pair_plots$B / pair_plots$C / pair_plots$D) |
  (p_e / p_f)
) +
  plot_layout(widths = c(1.05, 0.95)) +
  plot_annotation(
    title = sprintf(
      "Figure 3. SGB co-occurrence networks (CLR–ComBat; |r|≥%.1f)",
      R_THRESHOLD
    ),
    subtitle = "Batch-corrected abundances; cohort-stratified networks are primary; pooled overall is exploratory",
    theme = theme(
      plot.title = element_text(size = 11.5, family = FONT_FAMILY, colour = "black",
                                face = "plain"),
      plot.subtitle = element_text(size = 8, family = FONT_FAMILY, colour = "black",
                                   face = "plain"),
      plot.background = element_rect(fill = "white", colour = NA)
    )
  )
save_pub(fig3_full, file.path(out_fig, "composite", "Fig3_full_ComBat"),
         width_mm = 240, height_mm = 320, dpi = 400)
write_plotdata(tsv_all, "Fig3_full_ComBat_networks")
write_plotdata(bind_rows(df_e, df_f), "Fig3_full_ComBat_node_props")
message("Done. Outputs under: ", root_dir)
message("Display threshold: |r|>=", R_THRESHOLD)
