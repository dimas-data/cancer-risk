library(tidyverse)
library(janitor)

cancer_raw <- read_csv(file.choose())

dim(cancer_raw)
glimpse(cancer_raw)
summary(cancer_raw)

cancer_raw <- cancer_raw %>%
  clean_names()

glimpse(cancer_raw)

colSums(is.na(cancer_raw))
sum(duplicated(cancer_raw))

cancer_clean <- cancer_raw %>%
  distinct()

sum(duplicated(cancer_clean))
dim(cancer_clean)

table(cancer_clean$cancer_type)
table(cancer_clean$gender)
table(cancer_clean$family_history)
table(cancer_clean$brca_mutation)
table(cancer_clean$h_pylori_infection)
table(cancer_clean$risk_level)

cancer_clean <- cancer_clean %>%
  select(-patient_id)

cancer_clean <- cancer_clean %>%
  mutate(
    cancer_type = as.factor(cancer_type),
    gender = as.factor(gender),
    family_history = as.factor(family_history),
    brca_mutation = as.factor(brca_mutation),
    h_pylori_infection = as.factor(h_pylori_infection),
    risk_level = as.factor(risk_level)
  )

glimpse(cancer_clean)

normalize <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

cancer_final <- cancer_clean %>%
  mutate(
    age_norm = normalize(age),
    smoking_norm = normalize(smoking),
    alcohol_use_norm = normalize(alcohol_use),
    obesity_norm = normalize(obesity),
    diet_red_meat_norm = normalize(diet_red_meat),
    diet_salted_processed_norm = normalize(diet_salted_processed),
    fruit_veg_intake_norm = normalize(fruit_veg_intake),
    physical_activity_norm = normalize(physical_activity),
    air_pollution_norm = normalize(air_pollution),
    occupational_hazards_norm = normalize(occupational_hazards),
    calcium_intake_norm = normalize(calcium_intake),
    overall_risk_score_norm = normalize(overall_risk_score),
    bmi_norm = normalize(bmi),
    physical_activity_level_norm = normalize(physical_activity_level)
  )

cancer_final <- cancer_final %>%
  select(
    cancer_type,
    gender,
    family_history,
    brca_mutation,
    h_pylori_infection,
    age_norm,
    smoking_norm,
    alcohol_use_norm,
    obesity_norm,
    diet_red_meat_norm,
    diet_salted_processed_norm,
    fruit_veg_intake_norm,
    physical_activity_norm,
    air_pollution_norm,
    occupational_hazards_norm,
    calcium_intake_norm,
    overall_risk_score_norm,
    bmi_norm,
    physical_activity_level_norm,
    risk_level
  )

dim(cancer_final)
glimpse(cancer_final)
summary(cancer_final)

colSums(is.na(cancer_final))
sum(duplicated(cancer_final))

table(cancer_final$risk_level)

summary(cancer_final$age_norm)
summary(cancer_final$bmi_norm)
summary(cancer_final$overall_risk_score_norm)
summary(cancer_final$smoking_norm)

write_csv(cancer_final, "data/cancer_risk_M1_preprocessed.csv")