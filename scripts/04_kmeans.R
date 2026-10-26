# ============================================================
# PROJECT: Cancer Risk Prediction
# MODULE 4: UNSUPERVISED LEARNING
# METHOD: K-MEANS CLUSTERING
# OBJECTIVE:
# Segmenting patients into clinical clusters based on
# biometric similarity.
# ============================================================


# ============================================================
# STEP 1: INSTALL PACKAGES (RUN ONCE ONLY)
# ============================================================

# install.packages("factoextra")
# install.packages("ggplot2")

library(factoextra)
library(ggplot2)


# ============================================================
# STEP 2: IMPORT DATASET
# ============================================================

data <- read.csv(file.choose())


# ============================================================
# STEP 3: DATA INSPECTION
# ============================================================

head(data)
str(data)
dim(data)
names(data)


# ============================================================
# STEP 4: DATA PREPARATION
# Remove target variables
# ============================================================

m4_data <- subset(
  data,
  select = -c(risk_level, cancer_type)
)


# ============================================================
# STEP 5: CONVERT CATEGORICAL VARIABLES TO NUMERIC
# ============================================================

m4_data[] <- lapply(m4_data, function(x) {
  if (is.character(x) || is.factor(x)) {
    as.numeric(as.factor(x))
  } else {
    x
  }
})


# ============================================================
# STEP 6: CHECK MISSING VALUES
# ============================================================

colSums(is.na(m4_data))


# ============================================================
# STEP 7: STANDARDIZE DATA
# ============================================================

m4_scaled <- scale(m4_data)


# ============================================================
# STEP 8: ELBOW METHOD
# OUTPUT: Figure 4.1
# ============================================================

wss <- numeric(10)

for(k in 1:10){
  set.seed(123)
  wss[k] <- kmeans(
    m4_scaled,
    centers = k,
    nstart = 25
  )$tot.withinss
}

plot(
  1:10,
  wss,
  type = "b",
  pch = 19,
  xlab = "Number of Clusters (k)",
  ylab = "Within-Cluster Sum of Squares",
  main = "Elbow Method for K-Means"
)


# ============================================================
# STEP 9: APPLY K-MEANS CLUSTERING
# ============================================================

set.seed(123)

km <- kmeans(
  m4_scaled,
  centers = 3,
  nstart = 25
)


# ============================================================
# OUTPUT 1: CLUSTER DISTRIBUTION TABLE
# Table 4.1
# ============================================================

cluster_count <- as.data.frame(table(km$cluster))

colnames(cluster_count) <- c(
  "Cluster",
  "Number_of_Patients"
)

cluster_count

View(cluster_count)

write.csv(
  cluster_count,
  "Cluster_Distribution.csv",
  row.names = FALSE
)


# ============================================================
# STEP 10: ADD CLUSTER LABELS
# ============================================================

data$KMeans_Cluster <- km$cluster


# ============================================================
# OUTPUT 2: COMPLETE CLUSTER SUMMARY
# ============================================================

cluster_summary <- aggregate(
  m4_data,
  by = list(Cluster = km$cluster),
  mean
)

cluster_summary

View(cluster_summary)

write.csv(
  cluster_summary,
  "Cluster_Summary.csv",
  row.names = FALSE
)


# ============================================================
# OUTPUT 3: IMPORTANT RISK FACTOR SUMMARY
# Table 4.2
# ============================================================

focused_summary <- aggregate(
  cbind(
    age_norm,
    bmi_norm,
    smoking_norm,
    obesity_norm,
    alcohol_use_norm,
    overall_risk_score_norm
  ) ~ KMeans_Cluster,
  data = data,
  mean
)

focused_summary

View(focused_summary)
write.csv(
  focused_summary,
  "Focused_Summary.csv",
  row.names = FALSE
)


# ============================================================
# OUTPUT 4: K-MEANS CLUSTER VISUALIZATION
# Figure 4.2
# ============================================================

cluster_plot <- fviz_cluster(
  km,
  data = m4_scaled,
  geom = "point",
  palette = c("#A4133C", "#7FFF00", "#003566"),
  ellipse.type = "convex",
  ellipse.alpha = 0.15,
  pointsize = 1.5
) +
theme_classic() +
labs(
  title = "K-Means Clustering Results",
  subtitle = "Patient segmentation based on biometric similarity"
)

cluster_plot


# ============================================================
# SAVE CLUSTER PLOT
# ============================================================

png(
  "M4_KMeans_Cluster_Plot.png",
  width = 900,
  height = 700
)

print(cluster_plot)

dev.off()


# ============================================================
# OUTPUT 5: FINAL DATASET
# ============================================================

write.csv(
  data,
  "M4_KMeans_Result.csv",
  row.names = FALSE
)


# ============================================================
# CHECK SAVED FILE LOCATION
# ============================================================

getwd()


# ============================================================
# END OF MODULE 4
# ============================================================