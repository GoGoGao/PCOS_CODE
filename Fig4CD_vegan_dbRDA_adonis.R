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
  library(vegan)
})
args <- commandArgs(trailingOnly = TRUE)
get <- function(n, d = NULL) {
  hit <- grep(paste0("^", n, "="), args, value = TRUE)
  if (!length(hit)) return(d)
  sub(paste0("^", n, "="), "", hit)
}
dm_file <- get("dm")
meta_file <- get("meta")
out_prefix <- get("out")
set.seed(42)
meta <- read.delim(meta_file, sep = "\t", stringsAsFactors = FALSE, check.names = FALSE)
rownames(meta) <- meta$SGB_ID
dm <- as.matrix(read.delim(dm_file, sep = "\t", row.names = 1, check.names = FALSE))
ids <- intersect(rownames(dm), rownames(meta))
dm <- dm[ids, ids]
meta <- meta[ids, , drop = FALSE]
meta$Bias_class <- factor(meta$Bias_class)
meta$Core_status <- factor(meta$Core_status)
meta$log_length <- log10(pmax(meta$length, 1))
ok <- complete.cases(meta[, c("Bias_class", "Core_status", "log_length", "completeness")])
if (any(!ok)) {
  message(sprintf("[WARN] dropping %d rows with NA covariates", sum(!ok)))
}
meta <- meta[ok, , drop = FALSE]
dm <- dm[rownames(meta), rownames(meta)]
d <- as.dist(dm)
adonis_bias <- adonis2(d ~ Bias_class + log_length + completeness,
                       data = meta, permutations = 999, by = "margin")
adonis_core <- adonis2(d ~ Core_status + log_length + completeness,
                       data = meta, permutations = 999, by = "margin")
cap <- capscale(d ~ Bias_class + Condition(log_length + completeness), data = meta)
cap_anova <- anova(cap, permutations = 999)
sink(paste0(out_prefix, "_adonis_Bias_class.txt"))
print(adonis_bias)
sink()
sink(paste0(out_prefix, "_adonis_Core_status.txt"))
print(adonis_core)
sink()
sink(paste0(out_prefix, "_dbRDA_Bias_ConditionLength.txt"))
print(cap)
print(cap_anova)
sink()
tidy_adonis <- function(a, term_focus) {
  df <- as.data.frame(a)
  df$Term <- rownames(df)
  df
}
write.table(tidy_adonis(adonis_bias), paste0(out_prefix, "_adonis_Bias_class.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
write.table(tidy_adonis(adonis_core), paste0(out_prefix, "_adonis_Core_status.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
cap_df <- as.data.frame(cap_anova)
cap_df$Term <- rownames(cap_df)
write.table(cap_df, paste0(out_prefix, "_dbRDA_anova.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
sc <- scores(cap, display = "sites", choices = 1:2)
sc <- as.data.frame(sc)
sc$SGB_ID <- rownames(sc)
write.table(sc, paste0(out_prefix, "_dbRDA_sites.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
message("[OK] vegan length-conditioned tests written to ", out_prefix)
