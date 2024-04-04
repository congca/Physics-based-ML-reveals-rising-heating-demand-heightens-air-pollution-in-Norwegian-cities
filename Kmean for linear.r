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
dat <- read_excel("data for feature.xlsx")
######################## K mean clusering ###########################################
set.seed(12345)
library(ggplot2)
library(factoextra)
library(factoextra)
library(cluster)
CluData <- dat[,10:20]
dat <- cor(CluData,use="complete.obs")

fviz_nbclust(dat,kmeans,method = "silhouette")

CluR<-kmeans(x=dat,centers=2,iter.max=10,nstart=25)
cluster_labels <- CluR$cluster
cluster_labels
fviz_cluster(CluR, dat, ellipse.type = "norm")

bc.scaled <- scale(CluData)
data<-bc.scaled
#利用k-mean是进行聚类
png('KM2Sim.png',
    height = 15,
    width = 25,
    units = 'cm',
    res = 300)
plot.new()


fviz_nbclust(dat, kmeans, method = "wss")
df1<-scale(dat)
km <- kmeans(df1, 2, nstart = 25)
fviz_cluster(object = km, df1,
             ellipse.type = "euclid",
             star.plot = TRUE,
             repel = TRUE,
             geom = c("point", "text"),
             palette = 'jco',
             main = "",
             ggtheme = theme_minimal() +
               theme(text = element_text(size = 18))) +
  theme(axis.title = element_blank())
dev.off()


png('KM3Sim.png',
    height = 15,
    width = 25,
    units = 'cm',
    res = 300)
plot.new()

km <-kmeans(df1,centers=3,nstart = 25)
fviz_cluster(object=km,df1,
             ellipse.type = "euclid",star.plot=T,repel=T,
             geom = c("point","text"),palette='jco',main="",
             ggtheme=theme_minimal()+
               theme(text = element_text(size = 18))) +
  theme(axis.title = element_blank())
dev.off()

png('KM4Sim.png',
    height = 15,
    width = 25,
    units = 'cm',
    res = 300)
plot.new()
km <-kmeans(df1,centers=4,nstart = 25)
fviz_cluster(object=km,df1,
             ellipse.type = "euclid",star.plot=T,repel=T,
             geom = c("point","text"),palette='jco',main="",
             ggtheme=theme_minimal()+
               theme(text = element_text(size = 18))) +
  theme(axis.title = element_blank())
dev.off()

