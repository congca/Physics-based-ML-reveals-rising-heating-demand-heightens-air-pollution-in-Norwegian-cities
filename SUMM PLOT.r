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
############################### PM25###########################
dfPM25 <- dat[,c(10,12:36)]
fit <- lm(PM25 ~.,dfPM25)
# summ(fit, robust = "HC1")
# step <- step(fit)
# stepresult <- lm( PM25 ~ TV +   HDD + WS + meanRH + PP + TV2 + meanRH2 + HDD2 + WS2 + SD2 + TV3 + meanRH3  + WS3 + SD3 + PP3
#                   ,dfPM25)
# summ(stepresult, robust = "HC1")
# summ(stepresult, scale = TRUE)
# summ(stepresult, confint = TRUE, digits = 3)
# summ(stepresult, confint = TRUE, ci.width = .5)
#  
# fitg <- glm(PM25  ~ TV +   HDD + WS + meanRH + PP + TV2 + meanRH2 + HDD2 + WS2 + SD2 + TV3 + meanRH3  + WS3 + SD3 + PP3
#             ,dfPM25, family = gaussian)
# 
# summ(fitg)
# summ(fitg, exp = TRUE)
# 
# plot_summs(stepresult, robust = TRUE)
# plot_summs(stepresult, robust = TRUE, inner_ci_level = .9)
# dfnox <- dat[,c(11:36)]
fit2 <- lm(NOx~ .,dfnox)

# 
# summ(fit2, robust = "HC1")
# step2 <- step(fit2)
# stepresult2 <- lm( NOx ~ TV + Tmean + HDD + WS +  SD + PP + TV2 
#                    + HDD2 + WS2 + SD2 + TV3  + HDD3 + 
#                     WS3 + SD3 + PP3,dfnox)
# summary(stepresult2)
# summ(stepresult2, robust = "HC1")
# summ(stepresult2, confint = TRUE, digits = 3)
# summ(stepresult2, confint = TRUE, ci.width = .5)
# PM2.5model  <- stepresult
# NOxmodel  <- stepresult2
png('SUMM PLOT.png',
    height = 15,
    width = 25,
    units = 'cm',
    res = 300)
plot.new()
title("Whole dataset of three cities")

plot_summs(fit, fit2, robust = list(FALSE, "HC0", "HC5"),
           model.names = c("NOx model", "PM2.5 model"))

dev.off()
# library(huxtable )
# export_summs(NOxmodel, PM2.5model, scale = TRUE)
# export_summs(NOxmodel, PM2.5model, scale = TRUE,             error_format = "[{conf.low}, {conf.high}]")
# export_summs(NOxmodel, PM2.5model, scale = TRUE, to.file = "docx", file.name = "test.docx")
##############three cities#############
NOxOSLO <- dat[dat$id==1,c(11:36)]
NOxBergen <- dat[dat$id==2,c(11:36)]
NOxTrondheim  <- dat[dat$id==3,c(11:36)]
PM25OSLO <- dat[dat$id==1,c(10,12:36)]
PM25Bergen <- dat[dat$id==2,c(10,12:36)]
PM25Trondheim <- dat[dat$id==3,c(10,12:36)]

PM25BergenM <- lm(PM25 ~.,PM25Bergen)
PM25TrondheimM <- lm(PM25 ~.,PM25Trondheim)
PM25OSLOM <- lm(PM25 ~.,PM25OSLO)
NOxOSLOM <- lm(NOx~ .,NOxOSLO)
NOxBergenM <- lm(NOx~ .,NOxBergen)
NOxTrondheimM <- lm(NOx~ .,NOxTrondheim)

png('SUMM PLOT NOX.png',
    height = 15,
    width = 25,
    units = 'cm',
    res = 300)
plot.new()
 
custom_colors <- c("#93cc82","#e8c559","#ea9c9d","#005496")

plot_summs(list( fit2, NOxOSLOM, NOxBergenM, NOxTrondheimM), 
           robust = list(FALSE, "HC0", "HC5"),
           model.names = c("NOx of three cities" ,"OSLO NOx", "Bergen NOx", "Trondheim NOx"),
           colors = custom_colors)


dev.off()
png('SUMM PLOT PM25.png',
    height = 15,
    width = 25,
    units = 'cm',
    res = 300)
plot.new()
custom_colors <- c("#93cc82","#e8c559",
                   "#c74546","#88c4e8")

plot_summs(list( fit, PM25OSLOM, PM25BergenM, PM25TrondheimM), 
           robust = list(FALSE, "HC0", "HC5"),
           model.names = c( "PM2.5 of three cities", "OSLO PM2.5", "Bergen PM2.5", "Trondheim PM2.5"),
           colors = custom_colors)

 
dev.off()
