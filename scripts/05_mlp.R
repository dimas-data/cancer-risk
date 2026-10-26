# ============================================================
# M5: MULTI-LAYER PERCEPTRON (MLP) FOR CANCER RISK PREDICTION
# ============================================================

# ------------------------------------------------------------
# 0. LOAD REQUIRED PACKAGES
# ------------------------------------------------------------
# Uncomment the install lines below if you haven't installed them yet
# install.packages("caret")
# install.packages("dplyr")
# install.packages("ggplot2")
# install.packages("keras3")
# install.packages("tensorflow")

library(caret)
library(dplyr)
library(ggplot2)
library(keras3)
library(tensorflow)

# Optional: Verify TensorFlow is configured correctly
# tensorflow::tf_config()


# ------------------------------------------------------------
# 1. LOAD DATASET DIRECTLY (No pop-up window)
# ------------------------------------------------------------
# Ensure "cancer_risk_M1_preprocessed.csv" is in your R working directory.
# If you get an error here, run getwd() to check your folder, or use the full path.
csv_file_path <- "data/cancer_risk_M1_preprocessed.csv"

if (!file.exists(csv_file_path)) {
  stop("ERROR: File not found! Please ensure 'cancer_risk_M1_preprocessed.csv' is in your working directory: ", getwd())
}

data <- read.csv(csv_file_path, stringsAsFactors = FALSE)

# Basic checking
cat(" Dataset loaded successfully.\n")
cat("Dimensions:", dim(data)[1], "rows,", dim(data)[2], "columns\n")

# Clean column names (removes hidden leading/trailing spaces that cause errors)
names(data) <- trimws(names(data))

# Verify target column exists
if (!"risk_level" %in% names(data)) {
  stop("ERROR: Column 'risk_level' not found! Check the CSV headers.")
}


# ------------------------------------------------------------
# 2. DEFINE TARGET AND REMOVE POSSIBLE LEAKAGE COLUMN
# ------------------------------------------------------------

# Convert target into an ORDERED factor (Best practice for Low < Medium < High)
data$risk_level <- factor(
  data$risk_level,
  levels = c("Low", "Medium", "High"),
  ordered = TRUE 
)

# Convert cancer_type into a standard factor for one-hot encoding
data$cancer_type <- as.factor(data$cancer_type)

# Create M5 dataset
m5_data <- data

# Remove possible leakage variable
# Reason: risk_level may be directly derived from overall_risk_score_norm
if ("overall_risk_score_norm" %in% names(m5_data)) {
  m5_data <- m5_data %>% select(-overall_risk_score_norm)
  cat(" Successfully removed 'overall_risk_score_norm' to prevent data leakage.\n")
}

cat("\nFinal M5 Data Class Distribution:\n")
print(table(m5_data$risk_level))


# ------------------------------------------------------------
# 3. ONE-HOT ENCODE CANCER TYPE
# ------------------------------------------------------------
dummy_model <- dummyVars(~ cancer_type, data = m5_data)
cancer_dummy <- predict(dummy_model, newdata = m5_data) %>% as.data.frame()

m5_model_data <- m5_data %>%
  select(-cancer_type) %>%
  bind_cols(cancer_dummy)

cat("\n Modelling Data Prepared. Dimensions:", dim(m5_model_data)[1], "x", dim(m5_model_data)[2], "\n")


# ------------------------------------------------------------
# 4. TRAIN / TEST SPLIT (80% Train, 20% Test)
# ------------------------------------------------------------
set.seed(123)

train_index <- createDataPartition(
  m5_model_data$risk_level,
  p = 0.8,
  list = FALSE
)

train_data <- m5_model_data[train_index, ]
test_data  <- m5_model_data[-train_index, ]

cat("Train/Test Split Complete (80/20).\n")


# ------------------------------------------------------------
# 5. SEPARATE FEATURES AND TARGET
# ------------------------------------------------------------
x_train <- train_data %>% select(-risk_level) %>% as.matrix()
x_test  <- test_data  %>% select(-risk_level) %>% as.matrix()

y_train_class <- train_data$risk_level
y_test_class  <- test_data$risk_level


# ------------------------------------------------------------
# 6. CONVERT TARGET INTO INTEGER LABELS (For Keras)
# ------------------------------------------------------------
class_names <- levels(y_train_class) # c("Low", "Medium", "High")

# Mapping: Low = 0, Medium = 1, High = 2
y_train_int <- as.numeric(y_train_class) - 1
y_test_int  <- as.numeric(y_test_class) - 1


# ------------------------------------------------------------
# 7. CREATE INTERNAL VALIDATION SET (For Early Stopping)
# ------------------------------------------------------------
set.seed(123)

fit_index <- createDataPartition(y_train_class, p = 0.8, list = FALSE)

x_fit <- x_train[fit_index, ]
y_fit <- y_train_int[fit_index]

x_val <- x_train[-fit_index, ]
y_val <- y_train_int[-fit_index]

cat("Internal Validation Set Created for Early Stopping.\n")


# ------------------------------------------------------------
# 8. CREATE CLASS WEIGHTS (Handle Imbalance)
# ------------------------------------------------------------
class_count <- table(y_fit)
total_count <- sum(class_count)

class_weight <- list(
  "0" = as.numeric(total_count / (3 * class_count["0"])),
  "1" = as.numeric(total_count / (3 * class_count["1"])),
  "2" = as.numeric(total_count / (3 * class_count["2"]))
)

cat("\n Class Weights Applied:\n")
print(class_weight)


# ------------------------------------------------------------
# 9. BUILD FIXED MLP MODEL
# ------------------------------------------------------------
set.seed(123)
tensorflow::set_random_seed(123)

model_fixed <- keras_model_sequential(input_shape = ncol(x_train)) |>
  layer_dense(units = 64, activation = "relu") |>
  layer_dropout(rate = 0.30) |>
  layer_dense(units = 32, activation = "relu") |>
  layer_dropout(rate = 0.20) |>
  layer_dense(units = 3, activation = "softmax")

model_fixed |> compile(
  optimizer = optimizer_adam(learning_rate = 0.001),
  loss = "sparse_categorical_crossentropy",
  metrics = list(metric_sparse_categorical_accuracy(name = "accuracy"))
)

cat("\n MLP Model Compiled.\n")
summary(model_fixed)


# ------------------------------------------------------------
# 10. TRAIN FIXED MLP MODEL
# ------------------------------------------------------------
cat("\n Starting Model Training...\n")
history_fixed <- model_fixed |> fit(
  x = x_fit,
  y = y_fit,
  epochs = 100,
  batch_size = 32,
  validation_data = list(x_val, y_val),
  class_weight = class_weight,
  callbacks = list(
    callback_early_stopping(
      monitor = "val_loss",
      patience = 15,
      restore_best_weights = TRUE
    )
  ),
  verbose = 1
)

# Plot training history
plot(history_fixed)


# ------------------------------------------------------------
# 11. EVALUATE MODEL ON TEST SET
# ------------------------------------------------------------
test_eval_fixed <- model_fixed |> evaluate(x_test, y_test_int)
cat("\nTest Set Evaluation:\n")
print(test_eval_fixed)


# ------------------------------------------------------------
# 12. MAKE PREDICTIONS
# ------------------------------------------------------------
pred_prob_fixed <- model_fixed |> predict(x_test)
pred_prob_fixed <- as.array(pred_prob_fixed)

# Get the index of the max probability (1, 2, or 3)
pred_class_index_fixed <- apply(pred_prob_fixed, 1, which.max)

# Map back to class names
pred_class_fixed <- factor(
  class_names[pred_class_index_fixed],
  levels = class_names
)

cat("\n Predictions Generated.\n")
cat("Prediction Distribution:\n")
print(table(pred_class_fixed))


# ------------------------------------------------------------
# 13. CONFUSION MATRIX
# ------------------------------------------------------------
cm_m5_fixed <- confusionMatrix(
  data = pred_class_fixed,
  reference = y_test_class
)

print(cm_m5_fixed)

cm_table <- table(
  Prediction = pred_class_fixed,
  Reference = y_test_class
)


# ------------------------------------------------------------
# 14. MANUAL PRECISION, RECALL, F1-SCORE
# ------------------------------------------------------------
accuracy_fixed <- sum(diag(cm_table)) / sum(cm_table)

precision_per_class <- diag(cm_table) / rowSums(cm_table)
recall_per_class <- diag(cm_table) / colSums(cm_table)

precision_per_class[is.nan(precision_per_class)] <- 0
recall_per_class[is.nan(recall_per_class)] <- 0

f1_per_class <- 2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class)
f1_per_class[is.nan(f1_per_class)] <- 0

metrics_m5_fixed <- data.frame(
  Class = names(precision_per_class),
  Precision = as.numeric(precision_per_class),
  Recall = as.numeric(recall_per_class),
  F1 = as.numeric(f1_per_class)
)

macro_precision_fixed <- mean(precision_per_class)
macro_recall_fixed <- mean(recall_per_class)
macro_f1_fixed <- mean(f1_per_class)

m5_summary_metrics <- data.frame(
  Accuracy = accuracy_fixed,
  Macro_Precision = macro_precision_fixed,
  Macro_Recall = macro_recall_fixed,
  Macro_F1 = macro_f1_fixed
)

cat("\n M5 Summary Metrics Calculated:\n")
print(m5_summary_metrics)


# ------------------------------------------------------------
# 15. PLOT CONFUSION MATRIX
# ------------------------------------------------------------
cm_df <- as.data.frame(cm_table)
cm_df$Reference <- factor(cm_df$Reference, levels = c("Low", "Medium", "High"))
cm_df$Prediction <- factor(cm_df$Prediction, levels = c("High", "Medium", "Low"))

confusion_plot <- ggplot(cm_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 5) +
  labs(
    title = "M5 MLP Confusion Matrix",
    x = "Actual Risk Level",
    y = "Predicted Risk Level"
  ) +
  scale_x_discrete(limits = c("Low", "Medium", "High")) +
  scale_y_discrete(limits = c("High", "Medium", "Low")) +
  theme_minimal(base_size = 13)

print(confusion_plot)


# ------------------------------------------------------------
# 16. SAVE MODEL, RESULTS, AND PLOTS (For Shiny & Report)
# ------------------------------------------------------------
cat("\n Saving all artifacts to working directory...\n")

# 1. Save the trained Keras model
save_model(model_fixed, "m5_mlp_model_fixed.keras")

# 2. Save metadata for Shiny prediction preprocessing
saveRDS(
  list(
    dummy_model = dummy_model,
    class_names = class_names,
    feature_names = colnames(x_train),
    class_weight = class_weight
  ),
  "m5_metadata_fixed.rds"
)

# 3. Save result tables for your Technical Article
write.csv(as.data.frame(cm_table), "m5_confusion_matrix.csv", row.names = FALSE)
write.csv(metrics_m5_fixed, "m5_class_metrics.csv", row.names = FALSE)
write.csv(m5_summary_metrics, "m5_summary_metrics.csv", row.names = FALSE)

# 4. Save training history plot
png("m5_training_history.png", width = 900, height = 600)
plot(history_fixed)
dev.off()

# 5. Save confusion matrix plot
ggsave("m5_confusion_matrix_plot_final.png", plot = confusion_plot, width = 7, height = 5, dpi = 300)

# 6. Save complete R environment for M5 (Backup)
save(
  data, m5_data, m5_model_data, train_data, test_data,
  x_train, x_test, y_train_class, y_test_class, y_train_int, y_test_int,
  x_fit, y_fit, x_val, y_val, class_names, class_weight,
  model_fixed, history_fixed, test_eval_fixed, pred_prob_fixed, pred_class_fixed,
  cm_table, cm_m5_fixed, metrics_m5_fixed, m5_summary_metrics,
  file = "M5_final_complete.RData"
)

cat("\n M5 Execution Complete! All models, metrics, and plots have been saved.\n")
cat("Files generated in your working directory:\n")
print(list.files(pattern = "m5|M5"))