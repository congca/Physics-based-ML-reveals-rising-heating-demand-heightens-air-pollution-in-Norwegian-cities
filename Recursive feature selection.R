 
library(caret)
library(randomForest)
library("readxl")
library(ggplot2)
library(descriptr)
library(corrgram) 
library(mice)
library(reshape2) 
library(openxlsx)
library(e1071)
library(rpart)
library(rpart.plot)
library(xlsx2dfs)
library("corrplot")
# pak::pak("tidyverse/readxl")
library(readxl)
library(party)
library("randomForest")
library(vivid) # for visualisations 
library(randomForest) # for model fit
library(ranger)       # for model fit
library(vegan)
library(rfUtilities)
library(patchwork)
library(rfPermute)
library(export)
library(tidyverse)
library(reshape2)
library(corrplot)
library(ggpubr)
library("readxl")
library(ggplot2)
library(descriptr)
library(corrgram) 
library(mice)
library(reshape2) 
library(openxlsx)
library(e1071)
library(rpart)
library(rpart.plot)
library(xlsx2dfs)
library("corrplot")
library(readxl)
library(party)
library("randomForest")
library(vivid) # for visualisations 
library(randomForest) # for model fit
library(ranger)       # for model fit
library(vegan)
library(rfUtilities)
library(patchwork)
library(rfPermute)
library(export)
library(tidyverse)
library(reshape2)
library(corrplot)
library(broom.mixed)
library(ggpubr)
library(jtools)
setwd("C:/Users/congc/OneDrive - California Institute of Technology/5th")

NOx <- read_excel("NOx.xlsx")
PM25 <- read_excel("PM2.5.xlsx")
NOxOSLO <- read_excel("NOxOSLO.xlsx")
NOxTrondheim  <- read_excel("NOxTrondheim.xlsx")
NOxBergen <- read_excel("NOxBergen.xlsx")
PM25OSLO <- read_excel("PM25OSLO.xlsx")
PM25Bergen <- read_excel("PM25Bergen.xlsx")
PM25Trondheim <- read_excel("PM25Trondheim.xlsx")
# 假设目标变量的名称为 'target_column'
target <- 'NOx' # 替换为实际目标变量的列名
data <- NOx
# 检查数据中是否包含该列
if (!(target %in% colnames(data))) {
  stop("目标变量列名不存在于数据中")
}

# 切分数据
set.seed(123) # 确保结果可重复
trainIndex <- createDataPartition(data[[target]], p = 0.8, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# 定义控制参数，减少折数和打印进度
ctrl <- rfeControl(functions = rfFuncs, method = "cv", number = 5, verbose = TRUE)

# 执行RFE
sizes <- c(1:10)  # 选择特征数量的范围
results <- rfe(
  x = trainData[, !(names(trainData) %in% target)],
  y = trainData[[target]],
  sizes = sizes,
  rfeControl = ctrl
)

# 打印结果
print(results)

# 查看最优特征
optimal_features <- results$optVariables
print(optimal_features)

# 使用最优特征进行模型训练
finalModel <- randomForest(
  x = trainData[, optimal_features],
  y = trainData[[target]]
)
library(caret)
 


# 评估模型性能
predictions <- predict(finalModel, newdata = testData[, optimal_features])
#confusionMatrix(predictions, testData[[target]])


# 假设目标变量的名称为 'target_column'
target <- 'PM25' # 替换为实际目标变量的列名
data <- PM25
# 检查数据中是否包含该列
if (!(target %in% colnames(data))) {
  stop("目标变量列名不存在于数据中")
}

# 切分数据
set.seed(123) # 确保结果可重复
trainIndex <- createDataPartition(data[[target]], p = 0.8, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# 定义控制参数，减少折数和打印进度
ctrl <- rfeControl(functions = rfFuncs, method = "cv", number = 5, verbose = TRUE)

# 执行RFE
sizes <- c(1:10)  # 选择特征数量的范围
results <- rfe(
  x = trainData[, !(names(trainData) %in% target)],
  y = trainData[[target]],
  sizes = sizes,
  rfeControl = ctrl
)

# 打印结果
print(results)

# 查看最优特征
optimal_features <- results$optVariables
print(optimal_features)

# 使用最优特征进行模型训练
finalModel <- randomForest(
  x = trainData[, optimal_features],
  y = trainData[[target]]
)
library(caret)


