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
NOx <- read_excel("NOx.xlsx")
PM25 <- read_excel("PM2.5.xlsx")
NOxOSLO <- read_excel("NOxOSLO.xlsx")
NOxTrondheim  <- read_excel("NOxTrondheim.xlsx")
NOxBergen <- read_excel("NOxBergen.xlsx")
PM25OSLO <- read_excel("PM25OSLO.xlsx")
PM25Bergen <- read_excel("PM25Bergen.xlsx")
PM25Trondheim <- read_excel("PM25Trondheim.xlsx")
#### draw Radom forest graph
library(randomForest)

set.seed(42)  # 设置随机种子以确保结果可重复
PM25.rf <- randomForest(PM25~.,data=PM25,mtyr=3,importance=T,proximity=F)
# round(importance(PM25.rf),2)
# varImpPlot(PM25.rf)

NOx.rf <- randomForest(NOx~.,data=NOx,mtyr=3,importance=T,proximity=F)
round(importance(NOx.rf),2)
varImpPlot(NOx.rf)


# 获取特征重要性
importance_scoresPM25 <- importance(PM25.rf)
importance_scoresNOX <- importance(NOx.rf)
colnames(importance_scoresPM25)
colnames(importance_scoresNOX)
print(importance_scoresPM25)
print(importance_scoresNOX)
# 对特征重要性进行排序
scoresPM25 <- importance_scoresPM25[order(-importance_scoresPM25[, "MeanDecreaseGini"]), ]
scoresNOX <- importance_scoresNOX[order(-importance_scoresNOX[, "MeanDecreaseGini"]), ]

# 打印排序后的特征重要性
print(scoresPM25)
print(scoresNOX)



model <- PM25.rf
data <- PM25
set.seed(1701)
viviRf  <- vivi(fit = model, 
                data = data, 
                response = "PM25",
                gridSize = 50,
                importanceType = "agnostic",
                nmax = 500,
                reorder = TRUE,
                predictFun = NULL,
                numPerm = 4,
                showVimpError = FALSE)
viviHeatmap(mat = viviRf)
library("network")
library("sna")
library("intergraph")
viviNetwork(mat = viviRf)
viviNetwork(mat = viviRf, intThreshold = 0.12, removeNode = FALSE)
viviNetwork(mat = viviRf, intThreshold = 0.12, removeNode = TRUE)
viviNetwork(mat = viviRf, 
            layout = cbind(c(1,1,1,1,2,2,2,2,2), c(1,2,4,5,1,2,3,4,5)))



#### draw corrlation graph

cor_matr = cor(NOx)
 

tdc <- cor (NOx, method="pearson")
corrplot(tdc)
corrplot(tdc, method = "ellipse", 
         type = "upper",
         tl.col = "black", tl.cex = 1.2, tl.srt = 45
)

addcol <- colorRampPalette(c("red", "white", "blue"))

testRes = cor.mtest(NOx, method="pearson",conf.level = 0.95)
corrplot(tdc, method = "color", col = addcol(100), 
         tl.col = "black", tl.cex = 0.8, tl.srt = 45,tl.pos = "lt",
         p.mat = testRes$p, diag = T, type = 'upper',
         sig.level = c(0.001, 0.01, 0.05), pch.cex = 1.2,
         insig = 'label_sig', pch.col = 'grey20', order = 'AOE')
corrplot(tdc, method = "number", type = "lower",col = addcol(100), 
         tl.col = "n", tl.cex = 0.8, tl.pos = "n",order = 'AOE',
         add = T)
