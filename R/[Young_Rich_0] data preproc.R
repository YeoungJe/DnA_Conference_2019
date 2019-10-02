rm(list=ls())
### setwd ###
setwd("~/4-2/stock price/data/price")

### library ###
if(!require(dplyr)) install.packages("dplyr"); library(dplyr)
if(!require(caret)) install.packages("caret"); library(caret)
if(!require(data.table)) install.packages("data.table"); library(data.table)
if(!require(reshape)) install.packages("reshape"); library(reshape)
if(!require(stringr)) install.packages("stringr"); library(stringr)
if(!require(LaplacesDemon)) install.packages("LaplacesDemon"); library(LaplacesDemon)
if(!require(tidyverse)) install.packages("tidyverse"); library(tidyverse)
if(!require(ggplot2)) install.packages("ggplot2"); library(ggplot2)
if(!require(viridis)) install.packages("viridis"); library(viridis)
if(!require(stringr)) install.packages("stringr"); library(stringr)
if(!require(lightgbm)) install.packages("lightgbm"); library(lightgbm)
if(!require(readxl)) install.packages("readxl"); library(readxl)

### load data ###
list.files()
trading_data <- read.csv("KOSPI_트레이딩알고리즘.csv", fileEncoding = "cp949", stringsAsFactors = F)
kospi_data <- read.csv("KOSPI_국면분석.csv", fileEncoding = "cp949", stringsAsFactors = F)
eco_data <- read.csv("ECONOMIC.csv", fileEncoding = "cp949", stringsAsFactors = F)
ex_data <- read.csv("EXCHANGE.csv", fileEncoding = "cp949", stringsAsFactors = F)
market_data <- read.csv("MARKET.csv", fileEncoding = "cp949", stringsAsFactors = F)
per_data <- read.csv("PER.csv", fileEncoding = "cp949", stringsAsFactors = F)

# alg_kospi <- read_xlsx("데이터구조_국면분석.xlsx")
# alg_trading <- read_xlsx("데이터구조_트레이딩알고리즘.xlsx")
data_name <- c("trading_data","kospi_data","eco_data","ex_data","market_data","per_data")

### function ###
make_date <- function(data) {
  colnames(data)[which(colnames(data) == "날짜")] <- "date"
  data$date <- as.Date(as.character(data$date), format = "%Y-%m-%d")
  return(data)
}

check_minus <- function(x) {
  if ("-" %in% x) {
    return(TRUE)
  } else {
    return(FALSE)
  }
}

change_num <- function(data) {
  for (col_idx in 1:ncol(data)) {
    if (class(data[,col_idx]) %in% c("character","integer")) {
      data[,col_idx] <- data[,col_idx] %>% as.numeric()
    } 
  }
  return(data)
}

box_plot <- function(df,variable,kld_vector) {
  for_graph <- df[,c("Y",variable)]
  colnames(for_graph)[2] <- "value"
  kl_divergence <- kld_vector[variable]
  for_graph %>%
    ggplot(aes(x=Y, y=value, fill=Y)) +
    geom_boxplot() +
    scale_fill_viridis(discrete = TRUE, alpha=0.6) +
    theme(
      legend.position="none",
      plot.title = element_text(size=16)
    ) +
    ggtitle(paste0(variable," box plot"), subtitle = paste0("KL.D : ", kl_divergence)) +
    xlab("")
}

### data preproc ###
# colname to date for inner_join
for (data in data_name) {
  assign(data,make_date(get(data)))
}

# inner_join
whole_data <- kospi_data  %>% inner_join(ex_data, by = "date") %>%
  inner_join(market_data, by = "date") %>% inner_join(per_data, by = "date") #%>% inner_join(eco_data, by = "date")
whole_data %>% head()

# NA check
minus_vector <- which(apply(whole_data,2,check_minus) == TRUE)
for (col_idx in minus_vector) {
  whole_data[,col_idx] <- ifelse("-" == whole_data[,col_idx],NA,whole_data[,col_idx])
}
(whole_data %>% is.na %>% sum) / (whole_data %>% dim %>% prod)

# data type change
whole_data$Y <- as.factor(whole_data$Y)
whole_data <- change_num(whole_data)
whole_data %>% str

### EDA ###
# scaling
#preProcValues <- preProcess(whole_data, method = c("center", "scale"))
#whole_data_af <- predict(preProcValues, whole_data)

# min_max
preProcValues <- preProcess(whole_data, method = "range")
whole_data_af <- predict(preProcValues, whole_data)

# check kl.D
kld_vector <- c()
for (idx in 3:ncol(whole_data_af)) {
  tmp_kld_value <- c()
  name <- colnames(whole_data_af)[idx]
  colnames(whole_data_af)[idx] <- "tmp"
  tmp_a <- whole_data_af %>% group_by(Y) %>% summarise(name = mean(tmp, na.rm = T))
  tmp_b <- whole_data_af %>% group_by(Y) %>% summarise(name = sd(tmp, na.rm = T))
  for (waste in 1:20) {
    dist_a <- rnorm(n = 50000,mean = tmp_a$name[1],sd = tmp_b$name[1])
    dist_b <- rnorm(n = 50000,mean = tmp_a$name[2],sd = tmp_b$name[2])
    tmp_kld_value[waste] <- KLD(dist_a,dist_b)$mean.sum.KLD
  }
  kld_vector[idx-2] <- mean(tmp_kld_value)
  names(kld_vector)[idx-2] <- name
  colnames(whole_data_af)[idx] <- name
}
kld_vector %>% sort(decreasing = T) %>% head

# box plot
box_plot(whole_data_af, colnames(whole_data_af)[3], kld_vector)

### write.csv
write.csv(whole_data, "whole_data.csv", row.names = F)
write.csv(whole_data_af, "whole_data_minmax.csv", row.names = F)

### end