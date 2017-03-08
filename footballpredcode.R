##Data Import
train = read.csv("/Users/Bhat/Downloads/train.csv",header = TRUE,sep=",")
test = read.csv("/Users/Bhat/Downloads/test.csv",header = TRUE,sep=",")

numeric.sum = function(x)
{
  return (sum(as.numeric(x)))     
}

hist = tapply(train$HomeTeam,train$FTR,table)


##considering only odds and ignoring all the teams

library(tree)

tree.model=tree(FTR~.-ID-Date-HomeTeam-AwayTeam,data = train) 

tree.model
plot(tree.model)
text(tree.model, pretty=0)

#Basic tree model

summary(tree.model)
final.result = c()
for (i in (1:dim(test)[1]))
{
 predict.result = predict(tree.model,test[i,])
  position = which(predict.result==max(predict.result))
  if (position==1)
  {
    final.result=append(final.result,"A")
  }
  else if (position==2)
  {
    final.result=append(final.result,"D")
  }
  else
  {
    final.result=append(final.result,"H")
  }
}

summary(predict.result)
plot(predict.result)
test.tree=cbind(test,final.result)
summary(test.tree)
#plot(test.tree)
submission.1 = test.tree[,c(1,23)]
colnames(submission.1) = c("ID","FTR")
write.table(submission.1, file = "submission1.csv", col.names = TRUE, row.names = FALSE, sep = ",")

##Using cross-validation to improve the model

set.seed(10)
cv.tree.model = cv.tree(tree.model,FUN=prune.misclass)
cv.tree.model
summary(cv.tree.model)
par(mfrow=c(1,2))
plot(cv.tree.model$size,cv.tree.model$dev,type="b")
plot(cv.tree.model$k,cv.tree.model$dev,type="b")

prune.train = prune.misclass(tree.model,best=2)
plot(prune.train)
text(prune.train,pretty=0)
summary(prune.train)
tree.pred = predict(prune.train,test,type="class")
plot(tree.pred)
summary(tree.pred)
submission.2=data.frame(test$ID,tree.pred)
colnames(submission.2)=c("ID","FTR")
write.table(submission.2,file="submission2.csv",col.names = TRUE, row.names = FALSE, sep = ",")


##Bagging, Random Forest
library(ggplot2)
library(caret)
library(randomForest)

#removing missing values
bag.train = randomForest(FTR~.-ID-Date-HomeTeam-AwayTeam,data = train, ntree=500,importance = TRUE, na.action = na.exclude) 
bag.train
plot(bag.train)

importance(bag.train)
varImpPlot(bag.train)

#qplot(HomeTeam, AwayTeam, color=FTR, data=train)

predict.rf = predict(bag.train,test)
predict.rf
plot(predict.rf)

#test$FTR <- as.factor(test$FTR)
#trainlabels <- as.factor(test$FTR)
#plot(trainLabels)

submission.3=data.frame(test$ID,predict.rf)
colnames(submission.3)=c("ID","FTR")
write.table(submission.3,file="submission3.csv",col.names = TRUE, row.names = FALSE, sep = ",")


##Neural Network
library(nnet)
library(neuralnet)

set.seed(1234567890)
train = read.csv("/Users/Bhat/Downloads/train.csv",header = TRUE,sep=",")
dim(train)

#Return a logical vector indicating which cases are complete
str(train)
sum(complete.cases(train)) #count of complete cases
sum(!complete.cases(data)) #which cases are incomplete
scaled.train = train[complete.cases(train),]
scaled.train

#set or retrieve the column names of matrix
colname = colnames(scaled.train) 
colname

#output to produce an atomic vector and find numeric variables in a data frame
df = sapply(scaled.train,is.numeric) 
df

#Now we have to do Normalization
scaled.train[df] = apply(scaled.train[df],2,scale) 
scaled.train[df]

#Generate a class indicator function from a given factor.
scaled.train$FTR #change this 
scaled.train <- cbind(scaled.train[,6:23], class.ind(scaled.train$FTR))
scaled.train
new.output = as.formula(paste( "A + D + H ~",paste(colname[!colname %in% c("ID","Date","HomeTeam","AwayTeam","FTR","A","D","H")],collapse = " + ")) )
nnet = neuralnet(new.output,data=scaled.train,hidden=c(5),linear.output = FALSE,threshold=0.01,stepmax=1e6, rep = "1") 

nnet$net.result
nnet$result.matrix
print(nnet)
plot(nnet)
nnet$generalized.weights

test = read.csv("/Users/Bhat/Downloads/test.csv",header = TRUE,sep=",")
scaled.test = test[complete.cases(test),c(-2,-3,-4)]
ID = scaled.test$ID
ind2 = sapply(scaled.test,is.numeric)

#Normalization
scaled.test[ind2] = apply(scaled.test[ind2],2,scale) 
scaled.test=cbind(scaled.test,ID)
predict.nn = compute(nnet,scaled.test[,c(-1,-20)])
head(predict.nn$net.result,40)
result = as.data.frame(predict.nn$net.result)
result.nn=c()
for (i in (1:dim(result)[1]))
{
  position = which(result[i,]==max(result[i,]))
  if (position==1)
  {
    result.nn=append(result.nn,"A")
  }
  else if (position==2)
  {
    result.nn=append(result.nn,"D")
  }
  else
  {
    result.nn=append(result.nn,"H")
  }
}
table(result.nn)
scaled.test = cbind(scaled.test,result.nn)
scaled.test = scaled.test[,-1]

#Bind the result back to test
contains = c() 
for(i in (1:dim(test)[1])){
  if(i %in% scaled.test$ID){
    contains = append(contains,as.character(scaled.test[ID==i,20]))
  }
  else{
    contains = append(contains,NA)
  }
}
table(contains)
test = cbind(test,contains)


for(i in (which(is.na(test$contains)))){
  if( (which(test[i,]==max(test[i,c(5:22)],na.rm = TRUE))) %in% c(7,10,13,16,19,22)){
    test$contains[i] = "A"
  }
  else if ( (which(test[i,]==max(test[i,c(5:22)],na.rm = TRUE))) %in% c(7,10,13,16,19,22)-1){
    test$contains[i] = "D"
  }
  else{
    test$contains[i] = "H"
  }
}

predict.nn$net.result

submission.4 = test[,c(1,23)]
colnames(submission.4)[2] = "FTR"
write.table(submission.4,file="submission.4 - c(10).csv",col.names = TRUE, row.names = FALSE, sep = ",")

###XGBoost using softprob

require(xgboost) #model
require(caret) #for dummyVars
require(Metrics) #calculate errors
require(corrplot)
require(Rtsne)
require(stats)
require(knitr)
require(ggplot2)

train = read.csv("/Users/Bhat/Downloads/train.csv",header = TRUE,sep=",")
test = read.csv("/Users/Bhat/Downloads/test.csv",header = TRUE,sep=",")
str(train)
train.xg = train[,6:23]
FTR = as.matrix(as.numeric(train$FTR)-1)

##Build machine learning model
train.xg = as.matrix(train.xg)
mode(train.xg) = 'numeric'
test.xg = test[,5:22]
test.xg = as.matrix(test.xg)

##Parameters
param <- list("objective" = "multi:softprob",    # multiclass classification 
              "num_class" = 3,    # number of classes 
              "eval_metric" = "merror",    #evaluation/loss metric
              "max_depth" = 12,    # max tree depth
              "eta" = 0.01,    # # learning rate
              "gamma" = 0,    # minimum loss reduction 
              "subsample" = 0.7,    # part of data instances to grow tree 
              "colsample_bytree" = 0.7,  # subsample ratio of columns when constructing each tree 
              "min_child_weight" = 30  # minimum sum of instance weight needed in a child 
)

###k-fold validaton, with timing
set.seed(12345)
cv.nround= 20   # # max number of trees to build
cv.nfold = 5     #number of folds in K-fold
bst.cv = xgb.cv(param=param, data=train.xg, label=FTR, nfold=cv.nfold, nrounds=cv.nround, missing = NaN)
head(bst.cv)
summary(bst.cv)
str(bst.cv)
#nround=which(bst.cv$test.merror.mean==min(bst.cv$test.merror.mean))
min.merror = which.min(bst.cv$test.merror.mean) 
min.merror
plot(log(bst.cv$test.merror.mean),type = "l")


##Model Training
bst.mt = xgboost(param=param, data=train.xg, label=FTR,nrounds=min.merror,verbose=0,missing = NaN)
summary(bst.mt)
head(bst.mt)
str(bst.mt)



pr.softprob = predict(bst.mt,test.xg,missing = NaN)
summary(pr.softprob)
pr.softprob


pr.softprob.mat = matrix(pr.softprob,ncol = 3,byrow = T)
pr.softprob.mat
pr.softprob.mat.xg = max.col(pr.softprob.mat,"last")
pr.softprob.mat.xg = as.data.frame(pr.softprob.mat.xg)
pr.softprob.mat.xg$pr.softprob.mat.xg[pr.softprob.mat.xg$pr.softprob.mat.xg == 1] = "A"
pr.softprob.mat.xg$pr.softprob.mat.xg[pr.softprob.mat.xg$pr.softprob.mat.xg == 2] = "D"
pr.softprob.mat.xg$pr.softprob.mat.xg[pr.softprob.mat.xg$pr.softprob.mat.xg == 3] = "H"
pr.softprob.mat.xg = cbind(pr.softprob.mat.xg,1:610)
colnames(pr.softprob.mat.xg) = c("FTR","ID")
pr.softprob.mat.xg = pr.softprob.mat.xg[,c("ID","FTR")]
submission.5 = pr.softprob.mat.xg
colnames(submission.5)[2] = "FTR"
write.table(submission.5,file="submission.5 - c(10).csv",col.names = TRUE, row.names = FALSE, sep = ",")

###XGBoost using Softmax

library(xgboost)
train = read.csv("/Users/Bhat/Downloads/train.csv",header = TRUE,sep=",")
test = read.csv("/Users/Bhat/Downloads/test.csv",header = TRUE,sep=",")
library(lubridate)
convert = function(dt){
  dt$Date = as.POSIXct(dt$Date)
  dt$Month = as.numeric(as.factor(month(dt$Date)))-1
  dt$Wday = as.numeric(as.factor(wday(dt$Date)))-1
  dt$Date = NULL
  dt$HomeTeam = NULL
  dt$AwayTeam = NULL
  dt[is.na(dt)]=0 #replace all NA with 0
  return(dt)
}
train = convert(train)
test = convert(test)
levels = levels(train$FTR)
FTR = as.numeric(train$FTR)-1
ID = test$ID
train$ID = NULL
test$ID = NULL
train$FTR=NULL
train=as.matrix(train)
test=as.matrix(test)

##Model Training
param <- list("objective" = "multi:softmax",    # multiclass classification 
              "num_class" = 3,    # number of classes 
              "eval_metric" = "merror",    # evaluation metric 
              "nthread" = 5,   # number of threads to be used 
              "eta" = 0.1,    # step size shrinkage 
              "subsample" = 0.7,    # part of data instances to grow tree 
              "colsample_bytree" = 0.7  # subsample ratio of columns when constructing each tree
)

#k-fold cross-validation 
set.seed(123456789)
softmax.xg = xgb.cv(params = param,data = train,label = FTR,nfold = 5,nrounds = 20)
summary(softmax.xg)
head(softmax.xg)
str(softmax.xg)
min.merror.softmax <- which.min(softmax.xg$test.merror.mean)
min.merror.softmax
plot(log(softmax.xg$test.merror.mean),type = "l")


softmax.xgb <- xgboost(data = train, label = FTR, param = param, nround = min.merror.softmax,num_class = 3)
pr.softmax = predict(softmax.xgb,test)
summary(pr.softmax)
pr.softmax = ifelse(pr.softmax==0,"A",ifelse(pr.softmax==1,"D","H"))
submission.6 = data.frame(ID,pr.softmax)
colnames(submission.6) = c("ID","FTR")
write.table(submission.6,file="submission.6 - c(11).csv",col.names = TRUE, row.names = FALSE, sep = ",")




