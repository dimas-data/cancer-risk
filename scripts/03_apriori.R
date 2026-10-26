# =====================================================
# 1. LOAD PACKAGES
# =====================================================
library(arules)
library(arulesViz)
library(stringr)
library(igraph)
library(tidygraph)
library(ggraph)
library(ggplot2)
library(dplyr)
library(openxlsx) 

# =====================================================
# 2. LOAD DATA
# =====================================================
data_m3 <- read.csv("data/cancer_risk_M1_preprocessed.csv")

# =====================================================
# 3. CONVERT NUMERIC VARIABLES TO CATEGORIES (DISCRETIZATION)
# =====================================================
data_m3$Age_Group <- cut(
  data_m3$age_norm,
  breaks = c(0, 0.33, 0.66, 1),
  labels = c("Young", "Middle_Aged", "Senior"),
  include.lowest = TRUE
)

data_m3$Smoking_Level <- cut(
  data_m3$smoking_norm,
  breaks = c(0, 0.33, 0.66, 1),
  labels = c("Low_Smoking", "Medium_Smoking", "High_Smoking"),
  include.lowest = TRUE
)

data_m3$Alcohol_Level <- cut(
  data_m3$alcohol_use_norm,
  breaks = c(0, 0.33, 0.66, 1),
  labels = c("Low_Alcohol", "Medium_Alcohol", "High_Alcohol"),
  include.lowest = TRUE
)

data_m3$Obesity_Level <- cut(
  data_m3$obesity_norm,
  breaks = c(0, 0.33, 0.66, 1),
  labels = c("Low_Obesity", "Medium_Obesity", "High_Obesity"),
  include.lowest = TRUE
)

data_m3$BMI_Level <- cut(
  data_m3$bmi_norm,
  breaks = c(0, 0.33, 0.66, 1),
  labels = c("Low_BMI", "Medium_BMI", "High_BMI"),
  include.lowest = TRUE
)

data_m3$Physical_Activity <- cut(
  data_m3$physical_activity_norm,
  breaks = c(0, 0.33, 0.66, 1),
  labels = c("Low_Activity", "Medium_Activity", "High_Activity"),
  include.lowest = TRUE
)

# =====================================================
# 4. SELECT VARIABLES & FORMAT AS FACTORS
# =====================================================
data_arm <- data_m3[, c(
  "cancer_type",
  "gender",
  "family_history",
  "brca_mutation",
  "h_pylori_infection",
  "risk_level",
  "Age_Group",
  "Smoking_Level",
  "Alcohol_Level",
  "Obesity_Level",
  "BMI_Level",
  "Physical_Activity"
)]

# Prepare raw metrics dataset for the final Excel export sheet
final_excel_data <- data_m3[, c(
  "cancer_type", "gender", "family_history", "brca_mutation", 
  "h_pylori_infection", "risk_level", "Age_Group", "Smoking_Level", 
  "Alcohol_Level", "Obesity_Level", "BMI_Level", "Physical_Activity", 
  "overall_risk_score_norm"
)]

data_arm[] <- lapply(data_arm, as.factor)

# =====================================================
# 5. CREATE TRANSACTIONS
# =====================================================
basket <- apply(
  data_arm,
  1,
  function(x) paste(paste(names(data_arm), x, sep="="), collapse=",")
)

trans <- read.transactions(
  textConnection(basket),
  format = "basket",
  sep = ","
)

# =====================================================
# 6. RUN APRIORI FOR GENERAL RULES (To use in Network Graph)
# =====================================================
rules <- apriori(
  trans,
  parameter = list(
    supp = 0.005,
    conf = 0.50,
    minlen = 2,
    maxlen = 5
  )
)

# Filter out the top 50 rules sorted by lift for the network graph
top_rules <- head(sort(rules, by = "lift"), 50)

# =====================================================
# 7. RUN APRIORI SPECIFIC FOR HIGH RISK
# =====================================================
rules_high <- apriori(
  trans,
  parameter = list(
    supp = 0.002,
    conf = 0.30,
    minlen = 2,
    maxlen = 5
  ),
  appearance = list(
    rhs = c("risk_level=High"),
    default = "lhs"
  )
)

# Inspect high risk rules in console
inspect(head(sort(rules_high, by = "lift"), 20))

rules_high_df <- DATAFRAME(rules_high)

# =====================================================
# 8. CONVERT RULES TO EDGE LIST (FOR GRAPH OBJECT)
# =====================================================
rule_labels <- labels(top_rules)
edges <- data.frame()

for(i in seq_along(rule_labels)){
  rule <- rule_labels[i]
  parts <- strsplit(rule, " => ")[[1]]
  
  lhs <- gsub("\\{|\\}", "", parts[1])
  rhs <- gsub("\\{|\\}", "", parts[2])
  
  lhs_items <- unlist(strsplit(lhs, ","))
  lhs_items <- trimws(lhs_items)
  
  for(item in lhs_items){
    edges <- rbind(
      edges,
      data.frame(
        from = item,
        to = rhs,
        stringsAsFactors = FALSE
      )
    )
  }
}

# =====================================================
# 9. CONVERT TO GRAPH OBJECTS & CENTRALITY ANALYSIS
# =====================================================
g <- graph_from_data_frame(edges, directed = TRUE)

degree_df <- data.frame(Node = names(degree(g)), Degree = degree(g))
degree_df <- degree_df[order(-degree_df$Degree), ]

between_df <- data.frame(Node = names(betweenness(g)), Betweenness = betweenness(g))
between_df <- between_df[order(-between_df$Betweenness), ]

print("--- Top 5 Central Nodes (Degree) ---")
print(head(degree_df, 5))

# =====================================================
# 10. ENHANCED NETWORK VISUALIZATION (GGRAPH)
# =====================================================
tg <- as_tbl_graph(g) %>%
  mutate(node_type = ifelse(
    grepl("risk_level|cancer_type", name), 
    "Outcome / Disease Target", 
    "Lifestyle / Risk Factor"
  ))

print(
  ggraph(tg, layout = "fr") +
    geom_edge_link(
      alpha = 0.4, 
      color = "gray60",
      arrow = arrow(length = unit(2.5, 'mm')), 
      end_cap = circle(4, 'mm') 
    ) +
    geom_node_point(aes(color = node_type), size = 5) +
    geom_node_text(aes(label = name), repel = TRUE, size = 3, fontface = "bold") +
    scale_color_manual(values = c(
      "Outcome / Disease Target" = "#D95F02", 
      "Lifestyle / Risk Factor" = "#1F78B4"
    )) +
    theme_graph() +
    theme(
      legend.position = "bottom",
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5)
    ) +
    labs(
      title = "Association Rule Network Map: Cancer Outcomes vs Risk Profiles",
      color = "Node Classification"
    )
)

# =====================================================
# 11. EXPORT DATA TO EXCEL (ULTRA-STABLE ALTERNATIVE METHOD)
# =====================================================
wb <- createWorkbook()
sheet_name <- "Cancer Risk Data"
addWorksheet(wb, sheet_name)

# Generate professional corporate styling guidelines
header_style <- createStyle(
  fontName = "Segoe UI", fontSize = 11, fontColour = "#FFFFFF", 
  halign = "center", valign = "center", textDecoration = "bold",
  fgFill = "#1F4E78", border = "TopBottomLeftRight", borderColour = "#5B9BD5"
)

data_style <- createStyle(fontName = "Segoe UI", fontSize = 11)

decimal_style <- createStyle(fontName = "Segoe UI", fontSize = 11, numFmt = "0.00", halign = "right")

# Write core dataset matrix into active worksheet
writeData(wb, sheet = sheet_name, x = final_excel_data)

# Add drop-down filters across the top row safely via core component calls
addFilter(wb, sheet = sheet_name, rows = 1, cols = 1:ncol(final_excel_data))

# Render styling configurations onto Header Row 1
addStyle(wb, sheet = sheet_name, style = header_style, rows = 1, cols = 1:ncol(final_excel_data), gridExpand = TRUE)

# Render standard typography spacing across all patient observations
total_rows <- nrow(final_excel_data) + 1
addStyle(wb, sheet = sheet_name, style = data_style, rows = 2:total_rows, cols = 1:(ncol(final_excel_data) - 1), gridExpand = TRUE)

# Format the final normalized float score column into a uniform decimal width structure
addStyle(wb, sheet = sheet_name, style = decimal_style, rows = 2:total_rows, cols = ncol(final_excel_data), gridExpand = TRUE)

# Auto-optimize widths to dynamically fit categorical strings without truncation
setColWidths(wb, sheet = sheet_name, cols = 1:ncol(final_excel_data), widths = "auto")
showGridLines(wb, sheet = sheet_name, showGridLines = TRUE)

# Save the workbook array directly into your working directory
saveWorkbook(wb, "Cancer_Risk_Assosiation.xlsx", overwrite = TRUE)
cat("\n>>> Process Complete. Excel file saved as 'Cancer_Risk_Assosiation.xlsx' <<<\n")