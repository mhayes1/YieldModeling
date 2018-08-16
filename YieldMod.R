library(agridat)
library(caret)
library(xgboost)

dat<-lasrosas.corn


par(mfrow=c(3,3))
for(x in 1:ncol(dat)){
  if(is.numeric(dat[,x])){
    hist(dat[,x],main=colnames(dat)[x])
  }
}

dat$NitBin<-ifelse(dat$nitro<20,1,
                   ifelse((dat$nitro>=20 & dat$nitro<=50),2,
                          ifelse((dat$nitro>=50 & dat$nitro<=80),3,
                                 ifelse((dat$nitro>=80 & dat$nitro<=120),4,5))))
dvars <- as.data.frame(predict(dummyVars(~ topo + rep+nf+as.factor(year), data = dat),newdata=dat))

dat<-cbind(dat,dvars)

dat<-dat[,c(4,2,3,7,10:25)]

#dat$YieldCat<-ifelse(dat$yield<80,0,1)

ind<-createDataPartition(dat$yield,p=0.8,list=F)

train<-dat[ind,]
test<-dat[-ind,]

rm(list=setdiff(ls(), c('train','test')))







rf<-randomForest::randomForest(x=train[,2:20],
                               y=train$yield,ntree=200)


bestVars<-as.data.frame(rf$importance)
bestVars$Names<-row.names(bestVars)
bestVars<-bestVars[order(bestVars[,1],decreasing=T),]
bestVars<-bestVars[1:10,]

tcon<-trainControl(method='cv',
                   number=5,
                   allowParallel = T)

tuneGrid<-expand.grid(nrounds=c(50,100,150,200),
                      max_depth=c(8,10,12,14),
                      eta=c(0.01,0.1,0.2),
                      gamma=c(0,0.1,0.2),
                      subsample=c(0.6,0.8,1),
                      colsample_bytree=c(0.75,0.8,1),
                      min_child_weight=c(0.75,0.8,1)
)


tuneGrid<-tuneGrid[sample(nrow(tuneGrid),200),]

library(parallel)
library(doParallel)
cl <- makeCluster(8) # convention to leave 1 core for OS
registerDoParallel(cl)

system.time( finMod<-train(x=as.matrix(train[,bestVars$Names]),
              y=train$yield,
              method='xgbTree',
              trControl=tcon,
              tuneGrid=tuneGrid
              
)
)
stopCluster(cl)
registerDoSEQ()

test$Pred<-predict(finMod$finalModel,as.matrix(test[,bestVars$Names]))

lmod<-lm(test$yield~test$Pred)
summary(lm(test$yield~test$Pred))


plot(test$yield,test$Pred,pch=20)
abline(lm(test$yield~test$Pred),col='red',lwd=3)
abline(a=0,b=1,col='blue',lwd=3,lty=2)
