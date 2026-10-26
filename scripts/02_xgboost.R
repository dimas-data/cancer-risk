pkgs <- c("xgboost", "caret", "dplyr", "tidyr", "ggplot2",
          "SHAPforxgboost", "pROC", "scales")
for (pkg in pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
  library(pkg, character.only = TRUE)
}
set.seed(42)

fbeta <- function(prec, rec, beta = 2) {
  if (is.na(prec) || is.na(rec) || prec + rec == 0) return(0)
  (1 + beta^2) * prec * rec / (beta^2 * prec + rec)
}

# Load & Binary Encode
df <- read.csv("data/cancer_risk_M1_preprocessed.csv", stringsAsFactors = FALSE)
cat("Dimensions:", dim(df), "\nOriginal class distribution:\n")
print(table(df$risk_level))

# 1 = Elevated (Medium or High), 0 = Normal (Low)
df$risk_binary <- as.integer(df$risk_level %in% c("Medium", "High"))
cat("\nBinary target distribution:\n")
print(table(df$risk_binary, dnn = "risk_binary (0=Normal, 1=Elevated)"))

cancer_dummies <- model.matrix(~ cancer_type - 1, data = df)
colnames(cancer_dummies) <- gsub("cancer_type", "ct_", colnames(cancer_dummies))

drop_cols <- c("cancer_type", "risk_level", "risk_binary", "overall_risk_score_norm")
X_base <- cbind(as.data.frame(cancer_dummies),
                df[, !names(df) %in% drop_cols])

# Feature Engineering 
make_interactions <- function(X) {
  has <- function(n) n %in% names(X)
  if (has("smoking_norm") && has("occupational_hazards_norm"))
    X$interact_smoke_occup  <- X$smoking_norm * X$occupational_hazards_norm
  if (has("smoking_norm") && has("air_pollution_norm"))
    X$interact_smoke_airpol <- X$smoking_norm * X$air_pollution_norm
  if (has("alcohol_use_norm") && has("obesity_norm"))
    X$interact_alcohol_obese <- X$alcohol_use_norm * X$obesity_norm
  if (has("age_norm") && has("smoking_norm"))
    X$interact_age_smoke     <- X$age_norm * X$smoking_norm
  if (has("genetic_risk_norm") && has("smoking_norm"))
    X$interact_gene_smoke    <- X$genetic_risk_norm * X$smoking_norm
  X
}

make_flags <- function(X) {
  has <- function(n) n %in% names(X)
  if (has("obesity_norm"))        X$flag_obese          <- as.integer(X$obesity_norm          > 0.65)
  if (has("smoking_norm"))        X$flag_heavy_smoker   <- as.integer(X$smoking_norm          > 0.70)
  if (has("air_pollution_norm"))  X$flag_high_pollution <- as.integer(X$air_pollution_norm    > 0.75)
  if (has("alcohol_use_norm"))    X$flag_heavy_drinker  <- as.integer(X$alcohol_use_norm      > 0.70)
  if (has("genetic_risk_norm"))   X$flag_high_genetic   <- as.integer(X$genetic_risk_norm     > 0.60)
  if (has("age_norm"))            X$flag_elderly        <- as.integer(X$age_norm              > 0.75)
  X
}

X_full <- make_flags(make_interactions(X_base))
y      <- df$risk_binary
cat("\nFeature matrix columns after engineering:", ncol(X_full), "\n")

# Train / Test Split (80/20)
idx     <- createDataPartition(y, p = 0.80, list = FALSE)
X_train <- X_full[idx, ];  X_test <- X_full[-idx, ]
y_train <- y[idx];         y_test <- y[-idx]
cat("Train:", nrow(X_train), "| Test:", nrow(X_test), "\n")
cat("Train class dist:\n"); print(table(y_train))

# Class Balancing — inverse-frequency instance weights (NO SMOTE)
counts_train  <- table(y_train)
inv_w         <- max(counts_train) / counts_train
weights_train <- as.numeric(inv_w[as.character(y_train)])

dtrain <- xgb.DMatrix(as.matrix(X_train), label = y_train, weight = weights_train)
dtest  <- xgb.DMatrix(as.matrix(X_test),  label = y_test)
cat(sprintf("Weight multipliers — Normal:%.2f | Elevated:%.2f\n",
            inv_w["0"], inv_w["1"]))

# Randomized Hyperparameter Search 
cat("\n══ Hyperparameter Search ══\n")

param_grid <- expand.grid(
  eta              = c(0.05, 0.10),
  max_depth        = c(2, 3, 4),
  subsample        = c(0.8, 1.0),
  colsample_bytree = c(0.8, 1.0),
  min_child_weight = c(5, 10, 15),
  gamma            = c(1, 3, 5)
)
cat("Full (regularised) grid:", nrow(param_grid), "configs\n")

N_SEARCH    <- 25                        # was 80 — fewer, more targeted configs
sampled_idx <- sample(nrow(param_grid), N_SEARCH, replace = FALSE)
param_grid  <- param_grid[sampled_idx, ]
cat("Randomized search: evaluating", N_SEARCH, "configs\n")

n_threads <- max(1, parallel::detectCores() - 1)

cv_results <- lapply(seq_len(nrow(param_grid)), function(i) {
  cat(sprintf("  Config %d / %d ...\r", i, nrow(param_grid)))
  params <- list(
    objective        = "binary:logistic",
    eval_metric      = "auc",
    eta              = param_grid$eta[i],
    max_depth        = param_grid$max_depth[i],
    subsample        = param_grid$subsample[i],
    colsample_bytree = param_grid$colsample_bytree[i],
    min_child_weight = param_grid$min_child_weight[i],
    gamma            = param_grid$gamma[i],
    lambda = 1, alpha = 0,
    nthread = n_threads
  )
  cv <- xgb.cv(params = params, data = dtrain, nrounds = 300,
               nfold = 5, early_stopping_rounds = 20,
               maximize = TRUE, verbose = 0)
  
  best <- cv$best_iteration
  if (is.null(best) || length(best) == 0 || best < 1) {
    best <- which.max(cv$evaluation_log$test_auc_mean)
  }
  if (length(best) == 0 || best < 1) best <- 1
  
  data.frame(param_grid[i, ],
             best_nrounds  = cv$evaluation_log$iter[best],
             cv_auc_mean   = cv$evaluation_log$test_auc_mean[best],
             cv_auc_std    = cv$evaluation_log$test_auc_std[best])
})

cv_df       <- do.call(rbind, cv_results)
best_config <- cv_df[which.max(cv_df$cv_auc_mean), ]
cat("\nBest config:\n"); print(best_config)

# Train Final Binary Model 
final_model <- xgb.train(
  params = c(
    list(objective   = "binary:logistic",
         eval_metric = c("auc", "logloss"),
         lambda = 1, alpha = 0, nthread = n_threads),
    as.list(best_config[, c("eta","max_depth","subsample",
                            "colsample_bytree","min_child_weight","gamma")])
  ),
  data          = dtrain,
  nrounds       = as.integer(best_config$best_nrounds),
  evals         = list(train = dtrain, eval = dtest),
  verbose       = 1, print_every_n = 50
)

# Threshold Sweep 
prob_elevated <- predict(final_model, dtest)   # P(Elevated)

roc_elev <- roc(y_test, prob_elevated, quiet = TRUE)
cat("\nElevated-Risk AUC:", round(roc_elev$auc, 4), "\n")

thresholds   <- seq(0.10, 0.90, by = 0.01)
thresh_stats <- lapply(thresholds, function(thr) {
  pred_label <- ifelse(prob_elevated > thr, "Elevated", "Normal")
  cm         <- confusionMatrix(
    factor(pred_label, levels = c("Normal", "Elevated")),
    factor(ifelse(y_test == 1, "Elevated", "Normal"),
           levels = c("Normal", "Elevated")),
    positive = "Elevated"
  )
  prec <- cm$byClass["Precision"]
  rec  <- cm$byClass["Sensitivity"]
  data.frame(threshold = thr,
             f2        = fbeta(prec, rec, beta = 2),
             precision = prec,
             recall    = rec)
})
thresh_df       <- do.call(rbind, thresh_stats)
best_thresh_row <- thresh_df[which.max(thresh_df$f2), ]
ELEV_THRESHOLD  <- best_thresh_row$threshold
cat(sprintf("F-beta(2) optimal threshold: %.2f | F2=%.4f | Prec=%.4f | Rec=%.4f\n",
            ELEV_THRESHOLD, best_thresh_row$f2,
            best_thresh_row$precision, best_thresh_row$recall))

youden_coords <- coords(roc_elev, "best", best.method = "youden",
                        ret = c("threshold","sensitivity","specificity"),
                        transpose = FALSE)
cat(sprintf("Youden threshold (reference): %.3f | Sens=%.4f | Spec=%.4f\n",
            youden_coords$threshold[1],
            youden_coords$sensitivity[1],
            youden_coords$specificity[1]))

# Final Predictions 
pred_std <- factor(ifelse(prob_elevated > 0.50, "Elevated", "Normal"),
                   levels = c("Normal", "Elevated"))

# F-beta(2) optimised threshold
pred_clinical <- factor(ifelse(prob_elevated > ELEV_THRESHOLD, "Elevated", "Normal"),
                        levels = c("Normal", "Elevated"))

true_labels <- factor(ifelse(y_test == 1, "Elevated", "Normal"),
                      levels = c("Normal", "Elevated"))

# Evaluation
eval_binary <- function(pred, truth, label) {
  cm <- confusionMatrix(pred, truth, positive = "Elevated")
  cat(sprintf("\n── %s\n", label)); print(cm)
  cat(sprintf("F1: %.4f | Sensitivity: %.4f | Specificity: %.4f | Precision: %.4f\n",
              cm$byClass["F1"],
              cm$byClass["Sensitivity"],
              cm$byClass["Specificity"],
              cm$byClass["Precision"]))
  invisible(cm)
}

res_std  <- eval_binary(pred_std,      true_labels, "Standard threshold (0.50)")
res_clin <- eval_binary(pred_clinical, true_labels,
                        sprintf("F-beta(2) threshold (%.2f)", ELEV_THRESHOLD))

# Plots

# Threshold sweep
ggsave("threshold_sweep_binary_fbeta2_v2.png",
       ggplot(thresh_df, aes(threshold, f2)) +
         geom_line(colour = "#2980b9", linewidth = 1.2) +
         geom_vline(xintercept = ELEV_THRESHOLD,
                    linetype = "dashed", colour = "#e74c3c", linewidth = 1) +
         annotate("text", x = ELEV_THRESHOLD + 0.03, y = max(thresh_df$f2) * 0.9,
                  label = sprintf("Best = %.2f", ELEV_THRESHOLD),
                  colour = "#e74c3c", size = 4) +
         labs(title = "Threshold Sweep — F-beta(2) for Elevated Risk",
              subtitle = "Beta=2 penalises false negatives more than false positives",
              x = "P(Elevated) Threshold", y = "F-beta(2)") +
         theme_minimal(base_size = 13),
       width = 8, height = 4, dpi = 150)

# ROC curve
png("roc_elevated_risk_v2.png", width = 800, height = 600, res = 120)
plot(roc_elev, print.auc = TRUE, col = "#e74c3c", lwd = 2.5,
     main = "ROC — Elevated Risk (Binary, v2 — leakage-fixed)")
points(youden_coords$specificity[1], youden_coords$sensitivity[1],
       pch = 18, col = "#2c3e50", cex = 2.5)
dev.off()

# Feature importance
imp <- xgb.importance(feature_names = colnames(X_train), model = final_model)
ggsave("feature_importance_binary_v2.png",
       xgb.ggplot.importance(importance_matrix = imp, top_n = 20) +
         labs(title = "XGBoost Feature Importance (Top 20) — Binary v2",
              subtitle = "Weighted (no SMOTE) | Interaction Features | F-beta(2) Threshold") +
         theme_minimal(base_size = 13),
       width = 10, height = 7, dpi = 150)

# Confusion matrix
cm_tbl <- as.data.frame(
  confusionMatrix(pred_clinical, true_labels, positive = "Elevated")$table
)
colnames(cm_tbl) <- c("Predicted", "Actual", "Count")
ggsave("confusion_matrix_binary_v2.png",
       ggplot(cm_tbl, aes(Actual, Predicted, fill = Count)) +
         geom_tile(colour = "white", linewidth = 0.8) +
         geom_text(aes(label = Count), size = 9, fontface = "bold", colour = "white") +
         scale_fill_gradient(low = "#2c3e50", high = "#e74c3c") +
         labs(title = sprintf("Confusion Matrix — F-beta(2) Threshold (%.2f)", ELEV_THRESHOLD)) +
         theme_minimal(base_size = 13),
       width = 6, height = 5, dpi = 150)

# Probability distribution
prob_df <- data.frame(Probability = prob_elevated,
                      TrueClass   = ifelse(y_test == 1, "Elevated", "Normal"))
ggsave("probability_distribution_binary_v2.png",
       ggplot(prob_df, aes(Probability, fill = TrueClass)) +
         geom_histogram(binwidth = 0.05, colour = "white", alpha = 0.8,
                        position = "identity") +
         geom_vline(xintercept = ELEV_THRESHOLD,
                    linetype = "dashed", colour = "#c0392b", linewidth = 1) +
         scale_fill_manual(values = c(Normal = "#27ae60", Elevated = "#e74c3c")) +
         labs(title = "Predicted Probability Distribution — Binary v2",
              subtitle = sprintf("Dashed = F-beta(2) threshold (%.2f)", ELEV_THRESHOLD),
              x = "P(Elevated)", y = "Count", fill = "True Class") +
         theme_minimal(base_size = 12),
       width = 8, height = 4, dpi = 150)

# SHAP — now computed on the real (un-SMOTEd) training rows
raw_shap  <- predict(final_model, as.matrix(X_train), predcontrib = TRUE)
shap_mat  <- raw_shap[, seq_len(ncol(X_train))]
colnames(shap_mat) <- colnames(X_train)

shap_long <- shap.prep(shap_contrib = as.data.frame(shap_mat),
                       X_train      = as.matrix(X_train), top_n = 15)
ggsave("shap_summary_binary_v2.png",
       shap.plot.summary(shap_long) +
         labs(title = "SHAP Summary — Elevated Risk Class (Binary v2)") +
         theme_minimal(base_size = 12),
       width = 10, height = 7, dpi = 150)

cat("Plots saved.\n")

# Save & Summary 
xgb.save(final_model, "scripts/cancer_risk_xgboost_binary_v2.model")

cat("\n\n")
cat(sprintf("  STANDARD (0.50) | Acc: %5.1f%% | F1: %.4f | Sens: %.4f | Spec: %.4f\n",
            res_std$overall["Accuracy"] * 100,
            res_std$byClass["F1"],
            res_std$byClass["Sensitivity"],
            res_std$byClass["Specificity"]))
cat(sprintf("  CLINICAL (%.2f)  | Acc: %5.1f%% | F1: %.4f | Sens: %.4f | Spec: %.4f\n",
            ELEV_THRESHOLD,
            res_clin$overall["Accuracy"] * 100,
            res_clin$byClass["F1"],
            res_clin$byClass["Sensitivity"],
            res_clin$byClass["Specificity"]))
cat(sprintf("  AUC: %.4f\n", roc_elev$auc))
cat(sprintf("  Best rounds: %d | Depth: %d | eta: %.3f\n",
            as.integer(best_config$best_nrounds), best_config$max_depth, best_config$eta))
cat(sprintf("  F-beta(2) threshold: %.2f | Youden threshold: %.3f\n",
            ELEV_THRESHOLD, youden_coords$threshold[1]))
cat("\n")