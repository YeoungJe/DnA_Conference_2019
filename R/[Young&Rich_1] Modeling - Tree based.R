rm(list=ls())
### setwd
setwd("~/4-2/stock price/data/price")

### library
if(!require(dplyr)) install.packages("dplyr"); library(dplyr)
if(!require(caret)) install.packages("caret"); library(caret)
if(!require(data.table)) install.packages("data.table"); library(data.table)
if(!require(reshape)) install.packages("reshape"); library(reshape)
if(!require(stringr)) install.packages("stringr"); library(stringr)
if(!require(LaplacesDemon)) install.packages("LaplacesDemon"); library(LaplacesDemon)
if(!require(tidyverse)) install.packages("tidyverse"); library(tidyverse)
if(!require(ggplot2)) install.packages("ggplot2"); library(ggplot2)
if(!require(e1071)) install.packages("e1071"); library(e1071)
if(!require(viridis)) install.packages("viridis"); library(viridis)
if(!require(stringr)) install.packages("stringr"); library(stringr)
if(!require(lightgbm)) install.packages("lightgbm"); library(lightgbm)

### function
first_preproc <- function(data) {
  colnames(data)[which(colnames(data) == "날짜")] <- "date"
  data$date <- as.Date(as.character(data$date), format = "%Y-%m-%d")
  data$Y <- as.factor(data$Y)
  return(data)
}

make_lgbm_data <- function(data, passing_col, val_set = FALSE) {
  if (val_set) {
    train_label <- 2 - (data$Y[train_idx] %>% as.numeric())
    val_label <- 2 - (data$Y[val_idx] %>% as.numeric())
    test_label <- data$Y[test_idx]
    
    whole_train <- data[,!colnames(data) %in% passing_col] %>% as.matrix()
    train_data <- whole_train[train_idx,]
    val_data <- whole_train[val_idx,]
    test_data <- whole_train[test_idx,]
    
    dtrain <- lgb.Dataset(rbind(train_data),label=c(train_label))
    dvalid <- lgb.Dataset(cbind(val_data),label=val_label)
    valids <- list(test = dvalid)
    output <- list("dtrain" = dtrain, "valids" = valids, "test_label" = test_label, "test_data" = test_data)
    return(output)
  } else {
    train_label <- 2 - (data$Y[train_idx] %>% as.numeric())
    test_label <- data$Y[-train_idx]
    whole_train <- data[,!colnames(data) %in% passing_col] %>% as.matrix()
    train_data <- whole_train[train_idx,]
    test_data <- whole_train[-train_idx,]
    
    dtrain <- lgb.Dataset(rbind(train_data),label=c(train_label))
    dvalid <- lgb.Dataset(cbind(train_data),label=train_label)
    valids <- list(test = dvalid)
    output <- list("dtrain" = dtrain, "valids" = valids, "test_label" = test_label, "test_data" = test_data)
    return(output)
  }
}

### load data
whole_data <- read.csv("whole_data.csv", stringsAsFactors = F) %>% first_preproc
whole_data_af <- read.csv("whole_data_minmax.csv", stringsAsFactors = F) %>% first_preproc

### data split
train_idx <- which(whole_data$date <= as.Date("2017-6-30", format = "%Y-%m-%d"))
val_idx <- train_idx[length(train_idx)] + which(whole_data$date[-train_idx] <= as.Date("2018-01-01", format = "%Y-%m-%d"))
test_idx <- which(whole_data$date > as.Date("2018-01-01", format = "%Y-%m-%d"))

### modeling
## LightGBM
# data preproc for lgbm
lgbm_data_list <- make_lgbm_data(whole_data, c("date","Y"))
dtrain <- lgbm_data_list$dtrain
valids <- lgbm_data_list$valids
test_label <- lgbm_data_list$test_label
test_data <- lgbm_data_list$test_data

# modeling
params <-  list(objective = "binary", metric = "AUC", device = "gpu")
lgbm_model <- lgb.train(params, dtrain, 2000, valids,  eval_freq = 100, learning_rate = 0.05, early_stopping_rounds = 100)
tmp_predicted_value <- predict(lgbm_model, test_data)
predicted_value_lgbm <- ifelse(tmp_predicted_value > 0.5, "상승", "하락") %>% as.factor
confusionMatrix(predicted_value_lgbm, test_label)

## XGBoost
# set params
none <- trainControl(method = "none",verboseIter = TRUE)
cv <- trainControl(method = "cv",number = 5, verboseIter = TRUE)

train_xgb <- whole_data[-test_idx,]
test_xgb <- whole_data[test_idx,]

train_xgb[is.na(train_xgb)] <- 0
test_xgb[is.na(test_xgb)] <- 0

bool_vector <- !colnames(whole_data) %in% c("date")

xgb_grid <- expand.grid(
  nrounds = 500, # 
  eta = 0.055, # learning rate
  gamma = 2.62, # less error for pruning
  max_depth = 12, # 3
  min_child_weight = 4, # high value -> over fitting
  colsample_bytree = 0.6, # feature sampling rate per tree
  subsample = 0.7836 # row sampling rate per tree
)

xgb_model <- train(Y~., train_xgb[,bool_vector], method  = "xgbTree", trControl = cv, tuneGrid = xgb_grid)
predicted_value_xgb <- predict(xgb_model, test_xgb)
confusionMatrix(predicted_value_xgb, test_xgb$Y)




### end
