rm(list=ls())
pop = 1e3
dummvars = 40 # binary dummy variables, e.g., yes/no for specific diagnostic codes
secovars = 30 # secondary dummy variables, whose existence is influenced by the above-noted dummy variables, e.g., more likely to be diabetic if also obese and hypertensive and hyperlipidemic
catevars = 20 # categorical variables, values 1 through 5, e.g., income category, education, race/ethnicity, etc.
contvars = 10 # continuous variables, e.g., copays, deductibles, charges, etc.
set.seed(100)
dummvarsmeans = runif(dummvars)/3
dummvarsmeans
set.seed(200)
x1=matrix(rbinom(pop*dummvars,1,dummvarsmeans),pop,dummvars, byrow=T)
summary(x1)
secovarselect = round(min(dummvars/2,10))
secovarselect
x2 = matrix(0,pop,secovars)
set.seed(300)
x2[,1] = rbinom(pop,1,(x1[,1])/secovarselect)
for (i in 2:secovars){
  set.seed(400)
  x2[,i]=rbinom(pop,1,rowMeans(x1[,1:i])/secovarselect)
}
set.seed(500)
catevarsshape = round(10*runif(catevars))
catevarsshape
set.seed(600)
x3=matrix(round(5*rbeta(pop*catevars, 2, catevarsshape)),pop,catevars, byrow=T)
summary(x3)
set.seed(700)
contvarsshape = round(10*runif(contvars))
contvarsshape
set.seed(800)
contvarsmag = round(1000*runif(contvars))
set.seed(900)
x4=matrix(round(contvarsmag*rbeta(pop*contvars, 2, contvarsshape)),pop,contvars, byrow=T)
summary(x4)
covars = cbind(x1,x2,x3,x4)
summary(covars)
dim = dummvars+catevars+contvars+secovars
influencers = seq(1,dim,round(dim/10))
influencers
set.seed(1000)
outprob = (rowMeans(covars[,influencers])/dim+covars[,influencers[1]]*covars[,influencers[2]]/2-covars[,influencers[3]]*covars[,influencers[4]]/2+rnorm(pop, mean =0, sd = 0.001))
outprob[outprob>1]=1
outprob[outprob<0]=0
set.seed(1100)
y = rbinom(pop,1,outprob)
alldata = data.frame(cbind(x1,x2,x3,x4,y))
colnames(alldata) = c(paste("X",c(1:dim),sep=""),"y")
alldata$y = factor(alldata$y)
library(caret)
set.seed(1200)
splitIndex <- createDataPartition(alldata$y, p = .5, list = FALSE, times = 1)
trainSplit <- alldata[ splitIndex,]
testSplit <- alldata[-splitIndex,]
prop.table(table(trainSplit$y))
library(DMwR)
set.seed(1300)
trainSplit <- SMOTE(y ~ ., trainSplit, perc.over = 100, perc.under=200)
prop.table(table(trainSplit$y))
prop.table(table(testSplit$y))
library(h2oEnsemble)  # recommend cloning Github repo and installing to get newest version
h2o.init(nthreads=-1, max_mem_size="50G")
h2o.removeAll() 
train <- as.h2o(trainSplit)
test <- as.h2o(testSplit)
y <- "y"
x <- setdiff(names(train), y)
train[,y] <- as.factor(train[,y])
test[,y] <- as.factor(test[,y])
h2o.glm.1 <- function(..., alpha = 0.0) h2o.glm.wrapper(..., alpha = alpha)
h2o.glm.2 <- function(..., alpha = 0.5) h2o.glm.wrapper(..., alpha = alpha)
h2o.glm.3 <- function(..., alpha = 1.0) h2o.glm.wrapper(..., alpha = alpha)
h2o.randomForest.1 <- function(..., ntrees = 200, nbins = 50, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, nbins = nbins, seed = seed)
h2o.randomForest.2 <- function(..., ntrees = 200, sample_rate = 0.75, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, sample_rate = sample_rate, seed = seed)
h2o.randomForest.3 <- function(..., ntrees = 200, sample_rate = 0.85, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, sample_rate = sample_rate, seed = seed)
h2o.randomForest.4 <- function(..., ntrees = 200, nbins = 50, balance_classes = TRUE, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, nbins = nbins, balance_classes = balance_classes, seed = seed)
h2o.gbm.1 <- function(..., ntrees = 100, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, seed = seed)
h2o.gbm.2 <- function(..., ntrees = 100, nbins = 50, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, nbins = nbins, seed = seed)
h2o.gbm.3 <- function(..., ntrees = 100, max_depth = 10, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth = max_depth, seed = seed)
h2o.gbm.4 <- function(..., ntrees = 100, col_sample_rate = 0.8, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.5 <- function(..., ntrees = 100, col_sample_rate = 0.7, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.6 <- function(..., ntrees = 100, col_sample_rate = 0.6, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.7 <- function(..., ntrees = 100, balance_classes = TRUE, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, balance_classes = balance_classes, seed = seed)
h2o.gbm.8 <- function(..., ntrees = 100, max_depth = 3, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth = max_depth, seed = seed)
h2o.deeplearning.1 <- function(..., hidden = c(500,500), activation = "Rectifier", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.2 <- function(..., hidden = c(200,200,200), activation = "Tanh", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.3 <- function(..., hidden = c(500,500), activation = "RectifierWithDropout", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.4 <- function(..., hidden = c(500,500), activation = "Rectifier", epochs = 50, balance_classes = TRUE, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, balance_classes = balance_classes, seed = seed)
h2o.deeplearning.5 <- function(..., hidden = c(100,100,100), activation = "Rectifier", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.6 <- function(..., hidden = c(50,50), activation = "Rectifier", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.7 <- function(..., hidden = c(100,100), activation = "Rectifier", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
family <- "binomial"
learner <- c("h2o.glm.1","h2o.glm.2","h2o.glm.3","h2o.randomForest.1", "h2o.randomForest.2","h2o.randomForest.3","h2o.randomForest.4","h2o.gbm.1","h2o.gbm.2","h2o.gbm.3","h2o.gbm.4","h2o.gbm.5","h2o.gbm.6","h2o.gbm.7","h2o.gbm.8","h2o.deeplearning.1","h2o.deeplearning.2","h2o.deeplearning.3","h2o.deeplearning.4","h2o.deeplearning.5","h2o.deeplearning.6", "h2o.deeplearning.7")
metalearner <- "h2o.glm.wrapper"
fit <- h2o.ensemble(x = x, y = y,training_frame = train,family = family,learner = learner,metalearner = metalearner,cvControl = list(V = 5))
perfml <- h2o.ensemble_performance(fit, newdata = test)
perfml
fulllogmodel = glm(y~., data =trainSplit, family=binomial())
aiclogmodel = step(fulllogmodel,trace=F)
library(pROC)
perffulllog = predict.glm(fulllogmodel,newdata=testSplit)
perfaiclog = predict.glm(aiclogmodel,newdata=testSplit)
perffull <- roc(testSplit$y, perffulllog)
perfaic <- roc(testSplit$y, perfaiclog)
perffull
perfaic
h2o.shutdown(prompt=FALSE)
