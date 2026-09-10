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
  library(MMUPHin)
  library(dplyr)
})
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 7)
abd_path   <- args[[1]]
meta_path  <- args[[2]]
train_path <- args[[3]]
test_path  <- args[[4]]
out_train  <- args[[5]]
out_test   <- args[[6]]
out_feats  <- args[[7]]
pseudo <- 1e-5
prev_thr <- 0.10
abd_thr  <- 1e-5
to_proportion <- function(mat) {
  mat[mat < 0] <- 0
  col_s <- colSums(mat)
  col_s[col_s == 0] <- 1
  sweep(mat, 2, col_s, "/")
}
clr_transform <- function(mat, pseudo = 1e-5) {
  mat[mat < 0] <- 0
  mat_p <- mat + pseudo
  apply(mat_p, 2, function(x) log(x) - mean(log(x)))
}
prevalence_filter_train <- function(mat, prev_thr = 0.10, abd_thr = 1e-5) {
  prev <- apply(mat, 1, function(x) mean(x > abd_thr))
  keep <- prev >= prev_thr
  filtered <- mat[keep, , drop = FALSE]
  col_s <- colSums(filtered)
  col_s[col_s == 0] <- 1
  list(mat = sweep(filtered, 2, col_s, "/"), keep = keep, features = rownames(filtered))
}
apply_batch_from_train <- function(test_mat, train_raw, train_adj, batch_train, batch_test) {
  stopifnot(identical(rownames(test_mat), rownames(train_raw)))
  stopifnot(identical(rownames(train_raw), rownames(train_adj)))
  out <- test_mat
  for (b in unique(as.character(batch_test))) {
    idx_tr <- which(as.character(batch_train) == b)
    idx_te <- which(as.character(batch_test) == b)
    if (length(idx_te) == 0) next
    if (length(idx_tr) < 2) {
      next
    }
    mu_raw <- rowMeans(train_raw[, idx_tr, drop = FALSE])
    mu_adj <- rowMeans(train_adj[, idx_tr, drop = FALSE])
    shift <- mu_adj - mu_raw
    sd_raw <- apply(train_raw[, idx_tr, drop = FALSE], 1, sd)
    sd_adj <- apply(train_adj[, idx_tr, drop = FALSE], 1, sd)
    scale <- ifelse(sd_raw > 1e-12, sd_adj / sd_raw, 1)
    scale[!is.finite(scale)] <- 1
    scale <- pmin(pmax(scale, 0.1), 10)
    te <- test_mat[, idx_te, drop = FALSE]
    te <- sweep(te, 1, mu_raw, "-")
    te <- sweep(te, 1, scale, "*")
    te <- sweep(te, 1, mu_adj, "+")
    te[te < 0] <- 0
    out[, idx_te] <- te
  }
  col_s <- colSums(out)
  col_s[col_s == 0] <- 1
  sweep(out, 2, col_s, "/")
}
run_mmuphin_train <- function(abd_mat, meta_ord) {
  abd_mat[abd_mat < 0] <- 0
  fit <- adjust_batch(
    feature_abd = abd_mat,
    batch       = "Cohort",
    covariates  = NULL,
    data        = meta_ord,
    control     = list(verbose = FALSE)
  )
  corrected <- fit$feature_abd_adj
  corrected[corrected < 0] <- 0
  col_s <- colSums(corrected)
  col_s[col_s == 0] <- 1
  sweep(corrected, 2, col_s, "/")
}
abd <- as.matrix(read.table(abd_path, sep = "\t", header = TRUE,
                            row.names = 1, check.names = FALSE))
meta <- read.csv(meta_path, stringsAsFactors = FALSE)
rownames(meta) <- meta$SampleID
train_ids <- readLines(train_path)
test_ids  <- readLines(test_path)
train_ids <- train_ids[train_ids != ""]
test_ids  <- test_ids[test_ids != ""]
stopifnot(all(train_ids %in% colnames(abd)))
stopifnot(all(test_ids %in% colnames(abd)))
stopifnot(all(train_ids %in% meta$SampleID))
stopifnot(all(test_ids %in% meta$SampleID))
abd <- to_proportion(abd)
tr_raw0 <- abd[, train_ids, drop = FALSE]
filt <- prevalence_filter_train(tr_raw0, prev_thr, abd_thr)
feats <- filt$features
if (length(feats) < 5) stop("Too few features after prevalence filter")
tr_raw <- filt$mat
te_raw <- abd[feats, test_ids, drop = FALSE]
col_s <- colSums(te_raw)
col_s[col_s == 0] <- 1
te_raw <- sweep(te_raw, 2, col_s, "/")
meta_tr <- meta[train_ids, , drop = FALSE]
meta_tr$Cohort <- factor(meta_tr$Cohort)
meta_te <- meta[test_ids, , drop = FALSE]
tr_adj <- run_mmuphin_train(tr_raw, meta_tr)
n_batch_tr <- length(unique(meta_tr$Cohort))
if (n_batch_tr >= 2) {
  te_adj <- apply_batch_from_train(
    te_raw, tr_raw, tr_adj,
    batch_train = meta_tr$Cohort,
    batch_test  = meta_te$Cohort
  )
} else {
  te_adj <- te_raw
}
tr_clr <- clr_transform(tr_adj, pseudo)
te_clr <- clr_transform(te_adj, pseudo)
write_clr <- function(clr_mat, ids, path) {
  df <- as.data.frame(t(clr_mat), check.names = FALSE)
  df <- cbind(SampleID = ids, df)
  write.csv(df, path, row.names = FALSE, quote = TRUE)
}
write_clr(tr_clr, train_ids, out_train)
write_clr(te_clr, test_ids, out_test)
writeLines(feats, out_feats)
cat(sprintf("OK feats=%d train=%d test=%d\n",
            length(feats), length(train_ids), length(test_ids)))
