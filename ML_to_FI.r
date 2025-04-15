if (!require(tidyverse)) {
  install.packages("tidyverse")
}
if (!require(lubridate)) {
  install.packages("lubridate")
}
if (!require(xgboost)) {
  install.packages("xgboost")
}
if (!require(pROC)) {
  install.packages("pROC")
}

library(tidyverse)
library(lubridate)
library(xgboost)
library(pROC)

# ---- Data Preprocessing ----
# replace the path to your data_ml.RData file
load("/Users/melmeng/Library/Mobile Documents/com~apple~CloudDocs/Mel/4B/AFM 423/H5/data_ml.RData")

data_ml <- data_ml %>%
  filter(date > "2017-06-30", date < "2019-06-30") %>%
  arrange(stock_id, date)

# Filter only most complete stocks
stock_days <- data_ml %>% group_by(stock_id) %>% summarize(nb = n())
stock_ids_short <- stock_days$stock_id[stock_days$nb == max(stock_days$nb)]
data_ml <- filter(data_ml, stock_id %in% stock_ids_short)

# Select Features
features <- c("Mom_11M_Usd", "Debtequity", "Vol1Y_Usd", "Mkt_Cap_12M_Usd",
              "Div_Yld", "Pb", "Return_On_Capital", "Ebit_Ta", "Eps", "Advt_12M_Usd")

# Scale Features
data_ml <- data_ml %>%
  group_by(date) %>%
  mutate(across(all_of(features), ~ scale(.)[, 1])) %>%
  ungroup()

# Define Binary Label
q75 <- quantile(data_ml$R3M_Usd, 0.75, na.rm = TRUE)
q25 <- quantile(data_ml$R3M_Usd, 0.25, na.rm = TRUE)

data_ml <- data_ml %>%
  mutate(R3M_Usd_C = case_when(
    R3M_Usd >= q75 ~ 1,
    R3M_Usd <= q25 ~ 0,
    TRUE ~ NA_real_
  )) %>%
  drop_na(R3M_Usd_C)

# ---- Train-Test Split ----
separation_date <- as.Date("2018-11-30")
train_data <- filter(data_ml, date < separation_date)
test_data  <- filter(data_ml, date >= separation_date)

x_train <- as.matrix(train_data[, features])
x_test  <- as.matrix(test_data[, features])
y_train <- train_data$R3M_Usd_C
y_test  <- test_data$R3M_Usd_C

dtrain <- xgb.DMatrix(data = x_train, label = y_train)
dtest  <- xgb.DMatrix(data = x_test, label = y_test)

# ---- Class Imbalance Handling ----
scale_pos_weight <- sum(y_train == 0) / sum(y_train == 1)

# ---- Model Setup ----
params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 6,
  eta = 0.01,
  subsample = 0.8,
  colsample_bytree = 0.8,
  scale_pos_weight = scale_pos_weight,
  lambda = 0.5,
  alpha = 0.5,
  min_child_weight = 3,
  gamma = 0.1
)

# ---- Model Tuning (Cross-Validation) ----
set.seed(42)
cv <- xgb.cv(
  params = params,
  data = dtrain,
  nrounds = 500,
  nfold = 5,
  stratified = TRUE,
  early_stopping_rounds = 25,
  verbose = 1,
  maximize = TRUE
)

best_nrounds <- cv$best_iteration

# ---- Train Model ----
final_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = best_nrounds,
  watchlist = list(train = dtrain, test = dtest),
  early_stopping_rounds = 25,
  verbose = 1,
  print_every_n = 10
)

# ---- Model Evaluation ----
# Generate predictions, confusion matrix, and AUC
preds <- predict(final_model, dtest)
pred_class <- ifelse(preds > 0.5, 1, 0)

confusionMatrix(factor(pred_class, levels = c(0, 1)), factor(y_test, levels = c(0, 1)))

roc_obj <- roc(y_test, preds)
roc_df <- data.frame(
  TPR = roc_obj$sensitivities,
  FPR = 1 - roc_obj$specificities
)

ggplot(roc_df, aes(x = FPR, y = TPR)) +
  geom_line(color = "steelblue", size = 1) +
  geom_abline(linetype = "dashed", color = "gray") +
  labs(title = paste0("ROC Curve (AUC = ", round(auc(roc_obj), 3), ")"),
       x = "False Positive Rate", y = "True Positive Rate") +
  theme_minimal()

# ---- Training Log Visualization ----
eval_log <- data.frame(final_model$evaluation_log)

ggplot(eval_log, aes(iter)) +
  geom_line(aes(y = train_auc, color = "Train AUC")) +
  geom_line(aes(y = test_auc, color = "Test AUC")) +
  labs(title = "XGBoost AUC over Iterations", y = "AUC", x = "Boosting Rounds") +
  scale_color_manual(name = "Legend", values = c("Train AUC" = "blue", "Test AUC" = "red")) +
  theme_minimal()

# ---- Feature Importance ----
# Visualize which features were most important to the model
importance_matrix <- xgb.importance(model = final_model)
xgb.plot.importance(importance_matrix, top_n = 10, rel_to_first = TRUE, xlab = "XGBoost Feature Importance")

