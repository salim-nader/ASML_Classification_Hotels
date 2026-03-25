
required_packages <- c(
  "dplyr","ggplot2","caret","glmnet","pROC",
  "ranger","xgboost","tidymodels","rpart","rpart.plot"
)

invisible(lapply(required_packages, function(pkg) {
  if (!require(pkg, character.only = TRUE)) {
    stop(paste("Package not installed:", pkg))
  }
}))



library(dplyr)
library(ggplot2)
library(caret)
library(glmnet)
library(pROC)
library(ranger)
library(xgboost)
library(tidymodels)
library(xgboost)
library(rpart)
library(rpart.plot)


hotels <- read.csv("hotels.csv")

  
  
# Dimensions, Column Names, and Dataset view
  

cat("DIMENSIONS: \n\n")
dim(hotels)
cat("\n\n\n")
cat("---------------\n\n")



cat("COLUMNS NAMES: \n\n")
names(hotels)
cat("\n\n\n")
cat("---------------\n\n")


head(hotels)
cat("\n\n\n")




cat("SUMMARY: \n\n")
summary(hotels)
cat("\n\n\n")
cat("---------------\n\n")


cat("RESPONSE VARIABLE: \n\n")
table(hotels$is_canceled)
cat("\n\n\n")
cat("---------------\n\n")


cat("RESPONSE VARIABLE PROPORTIONS: \n\n")
prop.table(table(hotels$is_canceled))
cat("\n\n\n")
cat("---------------\n\n")



cat("MISSING VALUES: \n\n")
colSums(is.na(hotels))
cat("---------------\n\n")


  
# Remove missing rows
  
hotels$children[is.na(hotels$children)] <- 0


# Data Leakage - remove variables


hotels$reservation_status <- NULL
hotels$reservation_status_date <- NULL
hotels$assigned_room_type <- NULL
hotels$booking_changes <- NULL



# Data Types


for(i in seq_along(hotels)) {
  cat(names(hotels)[i], ":", class(hotels[[i]]), "\n")
}


# Convert data types from character to categorical where approrpriate

factor_vars <- c(
"hotel","arrival_date_month","meal","country",
"market_segment","distribution_channel",
"reserved_room_type",
"deposit_type","customer_type"
)

hotels[factor_vars] <- lapply(hotels[factor_vars], factor)

hotels$is_canceled <- factor(hotels$is_canceled)




# Check constant / near constant variables


nearZeroVar(hotels, saveMetrics = TRUE)



# Check perfect relationships/redundant variables

numeric_vars <- hotels %>% select(where(is.numeric))

cor_matrix <- cor(numeric_vars, use = "complete.obs")

cor_matrix



findCorrelation(cor_matrix, cutoff = 0.54)




# Cardinality of Categorical Variables

categorical_vars <- hotels %>% select(where(is.factor) | where(is.character))

sapply(categorical_vars, function(x) length(unique(x)))



# Univariate Predictive Power (Categorical)

for (var in names(categorical_vars)) {
  print(var)
  print(prop.table(table(hotels[[var]], hotels$is_canceled),1))
}



# Univariate Predictive Power (Numeric)

numeric_vars %>%
  mutate(is_canceled = hotels$is_canceled) %>%
  group_by(is_canceled) %>%
  summarise(across(everything(), mean, na.rm = TRUE))



# Visual quick check for numeric predictors


numeric_names <- names(numeric_vars)

for (v in numeric_names) {
  print(
    ggplot(hotels, aes_string(x = "is_canceled", y = v)) +
      geom_boxplot() +
      ggtitle(v)
  )
}





# EDA Plots

ggplot(hotels, aes(hotel, fill=is_canceled)) +
  geom_bar(position="fill") +
  labs(y="Proportion", title="Cancellation Rate by Hotel Type")


ggplot(hotels, aes(lead_time, fill=is_canceled)) +
  geom_histogram(bins=50) +
  labs(title="Lead Time Distribution by Cancellation")


ggplot(hotels, aes(is_canceled, lead_time)) +
  geom_boxplot()


ggplot(hotels, aes(market_segment, fill=is_canceled)) +
  geom_bar(position="fill") +
  theme(axis.text.x=element_text(angle=45,hjust=1))


ggplot(hotels, aes(deposit_type, fill=is_canceled)) +
  geom_bar(position="fill")


ggplot(hotels, aes(is_canceled, adr)) +
  geom_boxplot()


ggplot(hotels, aes(total_of_special_requests, fill=is_canceled)) +
  geom_bar(position="fill")


ggplot(hotels, aes(lead_time, fill=is_canceled)) +
  geom_density(alpha=0.4)



p1 <- ggplot(hotels, aes(hotel, fill=is_canceled)) +
  geom_bar(position="fill") +
  labs(y="Proportion", title="Cancellation Rate by Hotel Type")

p2 <- ggplot(hotels, aes(lead_time, fill=is_canceled)) +
  geom_histogram(bins=50) +
  labs(title="Lead Time Distribution by Cancellation")

p3 <- ggplot(hotels, aes(is_canceled, lead_time)) +
  geom_boxplot()

p4 <- ggplot(hotels, aes(market_segment, fill=is_canceled)) +
  geom_bar(position="fill") +
  theme(axis.text.x=element_text(angle=45,hjust=1))

p5 <- ggplot(hotels, aes(deposit_type, fill=is_canceled)) +
  geom_bar(position="fill")

p6 <- ggplot(hotels, aes(is_canceled, adr)) +
  geom_boxplot()

p7 <- ggplot(hotels, aes(total_of_special_requests, fill=is_canceled)) +
  geom_bar(position="fill")

p8 <- ggplot(hotels, aes(lead_time, fill=is_canceled)) +
  geom_density(alpha=0.4)




ggsave("plot1.png", p1, width=6, height=4, dpi=300)
ggsave("plot2.png", p2, width=6, height=4, dpi=300)
ggsave("plot3.png", p3, width=6, height=4, dpi=300)
ggsave("plot4.png", p4, width=6, height=4, dpi=300)
ggsave("plot5.png", p5, width=6, height=4, dpi=300)
ggsave("plot6.png", p6, width=6, height=4, dpi=300)
ggsave("plot7.png", p7, width=6, height=4, dpi=300)
ggsave("plot8.png", p8, width=6, height=4, dpi=300)


  
  
# LOGISTIC = BASELINE MODEL.
  
exclude_vars <- c(
  "reservation_status",
  "reservation_status_date",
  "assigned_room_type",
  "booking_changes",
  "country",
  "agent",
  "company"
)


new_vars <- setdiff(names(hotels), exclude_vars)

new_data <- hotels[, new_vars]

new_data$is_canceled <- as.factor(new_data$is_canceled)

new_data <- new_data %>%
  mutate(across(where(is.character), as.factor))

names(new_data)
str(new_data)


# Train Test Split

set.seed(1)

train_index <- createDataPartition(new_data$is_canceled, p = 0.7, list = FALSE)

train <- new_data[train_index, ]
test  <- new_data[-train_index, ]

train$is_canceled <- factor(train$is_canceled, levels = c(0,1), labels = c("No","Yes"))
test$is_canceled  <- factor(test$is_canceled,  levels = c(0,1), labels = c("No","Yes"))


# Model

set.seed(1)

cat("-------------------------------")
cat("Training Baseline Logistic Regression.")
       
baseline_model <- glm(
  is_canceled ~ .,
  data = train,
  family = binomial
)

summary(baseline_model)


# Predictions


prob_pred <- predict(baseline_model, test, type = "response")

class_pred <- ifelse(prob_pred > 0.35, "Yes", "No")
class_pred <- factor(class_pred)

table(Predicted = class_pred, Actual = test$is_canceled)
cat("\n\n\n")


# Confusion Matrix


cm <- table(Predicted = class_pred, Actual = test$is_canceled)

TP <- cm["Yes","Yes"]
TN <- cm["No","No"]
FP <- cm["Yes","No"]
FN <- cm["No","Yes"]

precision <- TP / (TP + FP)
recall <- TP / (TP + FN)
f1 <- 2 * (precision * recall) / (precision + recall)


metrics <- data.frame(
  Precision = precision,
  Recall = recall,
  F1 = f1
)

metrics


# ROC Curve


roc_obj <- roc(test$is_canceled, prob_pred)

plot(roc_obj)

auc(roc_obj)


# SINGLE TREE





cat("-------------------------------")
cat("Training Baseline Single Decision Tree.")
       
set.seed(1)

tree_model <- rpart(
  is_canceled ~ .,
  data = train,
  method = "class"
)


rpart.plot(tree_model)


tree_prob <- predict(tree_model, test, type = "prob")[, "Yes"]

tree_class <- ifelse(tree_prob > 0.35, "Yes", "No")
tree_class <- factor(tree_class, levels = c("No", "Yes"))


table(Predicted = tree_class, Actual = test$is_canceled)



roc_tree <- roc(test$is_canceled, tree_prob)
auc_tree <- auc(roc_tree)
auc_tree



# Confusion matrix
cm_tree <- table(Predicted = tree_class, Actual = test$is_canceled)

TP <- cm_tree["Yes", "Yes"]
FP <- cm_tree["Yes", "No"]
FN <- cm_tree["No", "Yes"]
TN <- cm_tree["No", "No"]


# Precision
precision <- TP / (TP + FP)

# Recall
recall <- TP / (TP + FN)

# F1 Score
f1 <- 2 * (precision * recall) / (precision + recall)

# Print results
precision
recall
f1


# RANDOM FOREST



cat("-------------------------------")
cat("Training Random Forest.")
       

set.seed(1)

control <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

tunegrid <- expand.grid(
  mtry = c(3, 5, 7, 10),
  splitrule = "gini",
  min.node.size = c(1, 5, 10)
)


trees_vals <- c(100, 200, 300)
min.node.size = c(1, 5, 10)


rf_cv <- train(
  is_canceled ~ .,
  data = train,
  method = "ranger",
  trControl = control,
  tuneGrid = tunegrid,
  metric = "ROC",
  num.trees = 300
)



# PREDICTIONS + ROC-AUC etc.

rf_prob <- predict(rf_cv, newdata = test, type = "prob")[, "Yes"]

rf_class <- ifelse(rf_prob > 0.35, "Yes", "No")
rf_class <- factor(rf_class, levels = c("No", "Yes"))



# Confusion Matrix


cm_rf <- table(Predicted = rf_class, Actual = test$is_canceled)
cm_rf

TP <- cm_rf["Yes","Yes"]
TN <- cm_rf["No","No"]
FP <- cm_rf["Yes","No"]
FN <- cm_rf["No","Yes"]


accuracy_rf <- mean(rf_class == test$is_canceled)

precision_rf <- TP / (TP + FP)

recall_rf <- TP / (TP + FN)

f1_rf <- 2 * (precision_rf * recall_rf) / (precision_rf + recall_rf)


roc_rf <- roc(test$is_canceled, rf_prob)

plot(roc_rf)

auc_rf <- auc(roc_rf)


rf_results <- data.frame(
  Model = "Random Forest",
  Accuracy = accuracy_rf,
  Precision = precision_rf,
  Recall = recall_rf,
  F1 = f1_rf,
  AUC = auc_rf
)

rf_results



# XGB - tidymodels




cat("-------------------------------")
cat("Training XGBoost Model.")
       

set.seed(1)


xgb_spec <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  min_n = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")


xgb_grid <- grid_random(
  trees(range = c(200, 1000)),
  tree_depth(range = c(3, 10)),
  learn_rate(range = c(-3, -1), trans = scales::log10_trans()),
  min_n(range = c(2, 20)),
  size = 20
)


ctrl <- control_grid(
  verbose = TRUE,
  save_pred = TRUE
)


folds <- vfold_cv(train, v = 5, strata = is_canceled)

xgb_results <- tune_grid(
  xgb_spec,
  is_canceled ~ .,
  resamples = folds,
  grid      = xgb_grid,
  metrics   = metric_set(roc_auc),
  control = ctrl
)


best_params <- select_best(xgb_results, metric = "roc_auc")

final_spec <- finalize_model(xgb_spec, best_params)

final_fit <- final_spec %>%
  fit(is_canceled ~ ., data = train)


pred_prob <- predict(final_fit, test, type = "prob")$.pred_Yes
pred_class <- ifelse(pred_prob > 0.35, "Yes", "No")
pred_class <- factor(pred_class, levels = c("No","Yes"))


print(conf_mat(
  data = data.frame(truth = test$is_canceled, estimate = pred_class),
  truth = truth,
  estimate = estimate
))

cat("AUC:", roc(test$is_canceled, pred_prob, levels = c("No", "Yes"), direction = "<")$auc, "\n")

roc_auc(
  data = data.frame(truth = test$is_canceled, .pred_Yes = pred_prob),
  truth = truth,
  .pred_Yes,
  event_level = "second"
)



cat("-------------------------------")       




# PLOTS

# ROC curves (all models overlaid)
roc_lr   <- roc(test$is_canceled, prob_pred, levels = c("No","Yes"), direction = "<")
roc_tree <- roc(test$is_canceled, tree_prob, levels = c("No","Yes"), direction = "<")
roc_rf   <- roc(test$is_canceled, rf_prob,   levels = c("No","Yes"), direction = "<")
roc_xgb  <- roc(test$is_canceled, pred_prob, levels = c("No","Yes"), direction = "<")

png("roc_curves.png", width = 800, height = 600, res = 150)
plot(roc_lr,   col = "blue",   lwd = 2, main = "ROC Curves - Model Comparison")
plot(roc_tree, col = "green",  lwd = 2, add = TRUE)
plot(roc_rf,   col = "orange", lwd = 2, add = TRUE)
plot(roc_xgb,  col = "red",    lwd = 2, add = TRUE)
legend("bottomright",
       legend = c(
         paste("Logistic Regression (AUC =", round(auc(roc_lr),   3), ")"),
         paste("Decision Tree       (AUC =", round(auc(roc_tree), 3), ")"),
         paste("Random Forest       (AUC =", round(auc(roc_rf),   3), ")"),
         paste("XGBoost             (AUC =", round(auc(roc_xgb),  3), ")")
       ),
       col = c("blue","green","orange","red"),
       lwd = 2)
dev.off()


# Calibration plot (XGBoost final model)
cal_data <- data.frame(
  prob  = pred_prob,
  truth = ifelse(test$is_canceled == "Yes", 1, 0)
)

cal_data$bin <- cut(cal_data$prob, breaks = seq(0, 1, by = 0.1),
                    include.lowest = TRUE)

cal_summary <- cal_data %>%
  group_by(bin) %>%
  summarise(
    mean_pred   = mean(prob),
    mean_actual = mean(truth),
    n           = n()
  )

p_cal <- ggplot(cal_summary, aes(x = mean_pred, y = mean_actual)) +
  geom_point(aes(size = n), colour = "steelblue") +
  geom_line(colour = "steelblue") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", colour = "grey50") +
  labs(
    title = "Calibration Plot - XGBoost",
    x     = "Mean Predicted Probability",
    y     = "Observed Cancellation Rate",
    size  = "n"
  ) +
  theme_minimal()

ggsave("calibration_plot.png", p_cal, width = 6, height = 5, dpi = 300)


