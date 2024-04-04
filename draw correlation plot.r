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

NOx <- read_excel("NOx.xlsx")
PM25 <- read_excel("PM2.5.xlsx")
################# plot for NOX  LINEAR ###################
#### draw corrlation graph
df <- PM25[,c(11:20)]
cor_matr = cor(df)
tdc <- cor (df, method="pearson")
corrplot(tdc)
# corrplot(tdc, method = "ellipse", 
#          type = "upper",
#          tl.col = "black", tl.cex = 1.2, tl.srt = 45
# )

corrplot(tdc)
addcol <- colorRampPalette(c("red", "white", "blue"))
testRes = cor.mtest(df, method="pearson",conf.level = 0.95)

corrplot(tdc, method = "number", type = "lower",col = addcol(100), 
         tl.col = "n", tl.cex = 0.6, tl.pos = "n",order = 'AOE',
         add = T)


png('Picutre1.png',
    height = 15,
    width = 25,
    units = 'cm',
    res = 300)
corrplot(tdc)
addcol <- colorRampPalette(c("red", "white", "blue"))
testRes = cor.mtest(df, method="pearson",conf.level = 0.95)
corrplot(tdc, method = "number", type = "lower",col = addcol(100), 
         tl.col = "n", tl.cex = 0.6, tl.pos = "n",order = 'AOE',
         add = T)

dev.off()
######################
################# plot for PM2.5 LINEAR ###################
#### draw corrlation graph
df <- PM25[,c(10,12:20)]
cor_matr = cor(df)
tdc <- cor (df, method="pearson")
corrplot(tdc)
# corrplot(tdc, method = "ellipse", 
#          type = "upper",
#          tl.col = "black", tl.cex = 1.2, tl.srt = 45
# )

corrplot(tdc)
addcol <- colorRampPalette(c("red", "white", "blue"))
testRes = cor.mtest(df, method="pearson",conf.level = 0.95)

corrplot(tdc, method = "number", type = "lower",col = addcol(100), 
         tl.col = "n", tl.cex = 0.6, tl.pos = "n",order = 'AOE',
         add = T)


png('PM25cu.png',
    height = 15,
    width = 25,
    units = 'cm',
    res = 300)
corrplot(tdc)
addcol <- colorRampPalette(c("red", "white", "blue"))
testRes = cor.mtest(df, method="pearson",conf.level = 0.95)
corrplot(tdc, method = "number", type = "lower",col = addcol(100), 
         tl.col = "n", tl.cex = 0.6, tl.pos = "n",order = 'AOE',
         add = T)

dev.off()
####################
NOxOSLO <- read_excel("NOxOSLO.xlsx")
NOxTrondheim  <- read_excel("NOxTrondheim.xlsx")
NOxBergen <- read_excel("NOxBergen.xlsx")
PM25OSLO <- read_excel("PM25OSLO.xlsx")
PM25Bergen <- read_excel("PM25Bergen.xlsx")
PM25Trondheim <- read_excel("PM25Trondheim.xlsx")
################# plot for LINEAR for each city ###################
#### draw corrlation graph
df <- NOxTrondheim ### Only need to change this line
cor_matr = cor(df)
tdc <- cor (df, method="pearson")
corrplot(tdc)
addcol <- colorRampPalette(c("red", "white", "blue"))
testRes = cor.mtest(df, method="pearson",conf.level = 0.95)
corrplot(tdc, method = "number", type = "lower",col = addcol(100), 
         tl.col = "n", tl.cex = 0.6, tl.pos = "n",order = 'AOE',
         add = T)


png('NOxTrondheim.png',##########only need to change this line
    height = 800,
    width = 60,
    units = 'cm',
    res = 300)
corrplot(tdc)
addcol <- colorRampPalette(c("red", "white", "blue"))
testRes = cor.mtest(df, method="pearson",conf.level = 0.95)

par(pin=c(4, 3))  # 设置图形尺寸为宽4英寸，高3英寸


corrplot(tdc, method = "number", type = "lower", col = addcol(100), 
         tl.col = "n", tl.cex = 0.6, tl.pos = "n", order = 'AOE',
         add = TRUE, main = "Correlation Plot for NOxTrondheim", xlab = "Correlation coefficient")


dev.off()
######################

