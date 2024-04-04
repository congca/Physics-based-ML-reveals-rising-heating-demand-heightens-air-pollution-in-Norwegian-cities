
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
#########
mydata <-read_excel('data.xlsx')
Oslo <- mydata[which(mydata$id==1),]  
Bergen <- mydata[which(mydata$id==2),]  
Trondheim<- mydata[which(mydata$id==3),]  
## For NOX
OsloNOX <- Oslo[,c(7,11)]
BergenNOX <- Bergen[,c(7,11)]
TrondheimNOX <- Trondheim[,c(7,11)]
sp<-ggplot(OsloNOX , aes(x=factor(monthoftheyear),y= NOx)) +  geom_bar(aes(color = factor(monthoftheyear), fill = factor(monthoftheyear)),
                                                                     stat = "identity", position = position_dodge()  )  
sp + ggtitle("Monthly variation of Oslo NOx concentration")+xlab("Month") + ylab("Traffic volume (PCU)")

as.numeric(BergenNOX$NOx)
sp<-ggplot(BergenNOX , aes(x=factor(monthoftheyear),y= NOx)) +  geom_bar(aes(color = factor(monthoftheyear), fill = factor(monthoftheyear)),
                                                                       stat = "identity", position = position_dodge()  )  
sp + ggtitle("Monthly variation of Bergen NOx concentration ")+xlab("Month") + ylab("Traffic volume (PCU)")


as.numeric(TrondheimNOX$NOx)
sp<-ggplot(TrondheimNOX , aes(x=factor(monthoftheyear),y= NOx)) +  geom_bar(aes(color = factor(monthoftheyear), fill = factor(monthoftheyear)),
                                                                          stat = "identity", position = position_dodge()  )  
sp + ggtitle("Monthly variation of Trondheim NOx concentration")+xlab("Month") + ylab("Traffic volume (PCU)")

## For PM2.5
OsloPM <- Oslo[,c(7,10)]
BergenPM <- Bergen[,c(7,10)]
TrondheimPM <- Trondheim[,c(7,10)]
sp<-ggplot(OsloPM , aes(x=factor(monthoftheyear),y= PM25)) +  geom_bar(aes(color = factor(monthoftheyear), fill = factor(monthoftheyear)),
                                                                     stat = "identity", position = position_dodge()  )  
sp + ggtitle("Monthly variation of Oslo PM2.5 concentration")+xlab("Month") + ylab("Traffic volume (PCU)")

as.numeric(BergenPM$PM)
sp<-ggplot(BergenPM , aes(x=factor(monthoftheyear),y= PM25)) +  geom_bar(aes(color = factor(monthoftheyear), fill = factor(monthoftheyear)),
                                                                       stat = "identity", position = position_dodge()  )  
sp + ggtitle("Monthly variation of Bergen PM2.5 concentration")+xlab("Month") + ylab("Traffic volume (PCU)")


as.numeric(TrondheimPM$PM)
sp<-ggplot(TrondheimPM , aes(x=factor(monthoftheyear),y= PM25)) +  geom_bar(aes(color = factor(monthoftheyear), fill = factor(monthoftheyear)),
                                                                          stat = "identity", position = position_dodge()  )  
sp + ggtitle("Monthly variation of Trondheim PM2.5 concentration")+xlab("Month") + ylab("Traffic volume (PCU)")

####Done


