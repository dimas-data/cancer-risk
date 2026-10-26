# OncoShield-AI

Cancer risk prediction and patient segmentation using an integrated machine learning and deep learning framework in R.

## Overview
Early identification of cancer risk is crucial for timely medical intervention. This project delivers a comprehensive 5-module analytical framework and an interactive R-Shiny dashboard ("OncoShield AI"). The system utilizes a dual-engine architecture: a high-sensitivity XGBoost binary classifier for preliminary screening (Elevated vs. Normal) to minimize false negatives, coupled with a Multi-Layer Perceptron (MLP) for granular 3-tier risk stratification (Low, Medium, High). This work aligns with Sustainable Development Goal 3 (Good Health and Well-Being).

## Dataset
- **Source:** [Cancer Risk Factors Dataset](https://www.kaggle.com/datasets/tarekmasryo/cancer-risk-factors-dataset) by Tarek Masryo (Kaggle).
- **Observations:** 2,000 patients, 21 raw columns (clinical, lifestyle, genetic, and demographic factors).
- **Target:** `risk_level` (Low: 16.2% / Medium: 78.7% / High: 5.1%).
- **Note:** The dataset contains mixed data types (numerical and categorical) and satisfies the multivariate requirements for the 5-module framework.

## Project Structure
```text
OncoShield-AI/
├── data/       Preprocessed and raw dataset files
├── models/     Trained models (XGBoost & Keras MLP)
├── reports/    Analytical output folder
│   ├── plots/  Visual charts, ROC curves, and cluster figures (.png)
│   └── tables/ CSV metrics, summary tables, and confusion matrices (.csv)
├── scripts/    Analysis pipeline (run in order)
├── app/        R-Shiny dashboard code
└── README.md
Pipeline (M1 - M5)
Script	Module	What it does
01_dataprep.R	M1	Cleaning (median/mode imputation), factor encoding, Min-Max normalization.
02_xgboost.R	M2	Binary classifier (Elevated vs Normal), 5-fold CV, F-beta(2) threshold tuning, SHAP.
03_apriori.R	M3	Association rules (support 0.005, conf 0.5, lift > 1) for high-risk profiles.
04_kmeans.R	M4	K-Means clustering (k=3) to segment patients based on biometric similarity.
05_mlp.R	M5	3-class neural network (Low/Medium/High) using Keras/TensorFlow.
Key Results
Model	Metric	Value
XGBoost (M2)	AUC	0.9448
XGBoost (M2)	Sensitivity (Recall)	99.41% (Threshold = 0.10)
XGBoost (M2)	Precision / Specificity	90.13% / 38.33%
MLP (M5)	Test Accuracy	80.9% (Loss: 0.429)
MLP (M5)	Macro Recall / F1-Score	79.9% / 68.9%
K-Means (M4)	Cluster Sizes (k=3)	Cluster 1: 954, Cluster 2: 65, Cluster 3: 981
Visualizations
ROC Curve (XGBoost - Binary v2)
Shows the Youden optimal point (threshold=0.648) vs. the chosen clinical threshold (0.10).
https://reports/plots/roc_elevated_risk_v2.png

XGBoost Feature Importance (Top 20)
Interaction features (e.g., alcohol + obesity, smoking + air pollution) dominate the decision-making process.
https://reports/plots/xgboost_feature_importance.png

SHAP Summary (Elevated Risk Class)
https://reports/plots/shap_summary.png

K-Means Clustering (PCA Projection)
Visualization of the three distinct biometric patient profiles.
https://reports/plots/kmeans_clusters.png

M5 MLP Confusion Matrix (3-Class)
Highlights misclassifications primarily occurring at the boundary between Low and Medium risk.
https://reports/plots/mlp_confusion_matrix.png

OncoShield AI (R-Shiny App)
The framework includes a reactive, web-based dashboard built using Shiny and Shinydashboard.

Cohort Registry: Allows real-time Create, Read, and Update (CRUD) operations for patient data.

Dual-Engine Inference: Parses user inputs through two pre-trained models simultaneously: XGBoost (threshold 0.35 for dashboard inference) and the Keras MLP tensor.

Exploratory Intelligence: Displays live Apriori association rules and interactive K-Means cluster projections.

How to Run
Open the project as an RStudio Project (so relative paths work).

Install R packages: tidyverse, janitor, xgboost, caret, pROC, SHAPforxgboost, arules, arulesViz, factoextra, keras3, shiny, shinydashboard.

For the MLP: Python 3.11 plus keras3::install_keras() (TensorFlow backend).

Run scripts 01 to 05 in order to generate models and outputs.

Launch the dashboard via shiny::runApp("app/").

Limitations
Class Imbalance: Medium risk accounts for 78.7% of samples, which naturally expands the decision boundary for this class and pulls down the macro precision (64.1%) in the MLP model.

Sensitivity vs. Specificity Trade-off: The XGBoost model uses a highly conservative threshold (0.10) to achieve 99.41% sensitivity. This results in a high false positive rate (specificity 38.33%), intentionally redirecting the risk of error away from dangerous missed diagnoses at the cost of secondary diagnostic testing.

Boundary Overlap: Patient risk exists on a continuous spectrum; border cases between Low and Medium risk exhibit overlapping features that challenge discrete classification.

Small Sample Rules: Apriori association rules for "High" risk rest on relatively small counts.

Not a clinical tool: For educational and research purposes only.

Disclaimer
This project is for educational and research purposes only. The models and insights generated here do not constitute medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for medical concerns.

Authors
Department of Mathematics and Statistics, Faculty of Applied Sciences and Technology, Universiti Tun Hussein Onn Malaysia (UTHM), Pagoh, Johor, Malaysia.
Corresponding author: logen@uthm.edu.my