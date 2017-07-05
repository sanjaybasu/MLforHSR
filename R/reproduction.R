# install.packages('caret') # uncomment these first install.packages lines if you don't already have these R packages
# install.packages('DMwR')
# install.packages('devtools')
# install.packages('h2o')
# NOTE: TO CORRECTLY ADMINISTER ENSEMBLE AND CUSTOM DEEP LEARNING SCRIPTS AS BELOW, IT IS IMPORTANT TO download GitHub desktop from github.com, click 'clone' at the h2o-3 repo (https://github.com/h2oai/h2o-3), go to out the "h2oEnsemble_v0.1.9" branch from in the GitHub program, cd to your local h2o-r/ensemble directory using your terminal window (mac) or command prompt (windows), then type in terminal: R CMD install h2oEnsemble-package ; SEE: https://community.h2o.ai/questions/1588/issue-with-dl-h2o-ensembles-and-h2o-last-version.html
# install.packages('pROC')
rm(list=ls())
pop = 1e5
dummvars = 40 # binary dummy variables, e.g., yes/no for specific diagnostic codes
secovars = 30 # secondary dummy variables, whose existence is influenced by the above-noted dummy variables, e.g., more likely to be diabetic if also obese and hypertensive and hyperlipidemic
catevars = 20 # categorical variables, values 1 through 5, e.g., income category, education, race/ethnicity, etc.
contvars = 10 # continuous variables, e.g., copays, deductibles, charges, etc.
set.seed(100)
dummvarsmeans = runif(dummvars)/3 # producing dummy variables
set.seed(200)
x1=matrix(rbinom(pop*dummvars,1,dummvarsmeans),pop,dummvars, byrow=T) # putting dummy variables into matrix form
secovarselect = round(min(dummvars/2,10))
x2 = matrix(0,pop,secovars) # creating space for secondary dummy variables
set.seed(300)
x2[,1] = rbinom(pop,1,(x1[,1])/secovarselect)
for (i in 2:secovars){
  set.seed(400+i)
  x2[,i]=rbinom(pop,1,rowMeans(x1[,1:i])/secovarselect+runif(pop)/10) # producing secondary dummy variables with some noise to avoid perfect linear predictability
}
set.seed(500)
catevarsshape = round(10*runif(catevars)) # producing categorical variables
set.seed(600)
x3=matrix(round(5*rbeta(pop*catevars, 2, catevarsshape)),pop,catevars, byrow=T) # putting categorical variables into matrix form
set.seed(700)
contvarsshape = round(10*runif(contvars)) # producing con't variables by choosing shape parameters for beta distribution
set.seed(800)
contvarsmag = round(1000*runif(contvars)) # producing con't variables of various magnitudes to reflect claims' data variation in con't variable absolute numbers
set.seed(900)
x4=matrix(round(contvarsmag*rbeta(pop*contvars, 2, contvarsshape)),pop,contvars, byrow=T) # putting con't variables into matrix form
covars = cbind(x1,x2,x3,x4)
dim = dummvars+catevars+contvars+secovars
influencers = seq(1,dim,round(dim/10)) # choosing which variables are truly influential on the primary outcome
set.seed(1000)
coefs = runif(length(influencers))
set.seed(1100)
# simulating probability of the primary outcome based on influential variables, plus some complex interactions among other variables, producing positive and negative heterogeneity, and adding some noise
outprob = (rowMeans(coefs*covars[,influencers])/dim+covars[,round(3*dim/20)]*covars[,round(4*dim/20)]/2-covars[,round(5*dim/20)]*covars[,round(7*dim/20)]/2+rnorm(pop, mean =0, sd = .01)) 
outprob[outprob>1]=1
outprob[outprob<0]=0
set.seed(1200)
y = rbinom(pop,1,outprob) # making the primary outcome variable binary and probabilistic
alldata = data.frame(cbind(x1,x2,x3,x4,y))
colnames(alldata) = c(paste("X",c(1:dim),sep=""),"y")
alldata$y = factor(alldata$y)
library(caret)
options(warn=-1)
set.seed(1300)
splitIndex <- createDataPartition(alldata$y, p = .5, list = FALSE, times = 1) # randomly splitting the data into train and test sets
trainSplit <- alldata[ splitIndex,]
testSplit <- alldata[-splitIndex,]
prop.table(table(trainSplit$y))
library(DMwR)
set.seed(1400)
trainSplit <- SMOTE(y ~ ., trainSplit, perc.over = 100, perc.under=200) # making a balanced subset of the training data, which is known to improve ML methods' performance
prop.table(table(trainSplit$y))
prop.table(table(testSplit$y))
library(h2oEnsemble)  # NOTE: TO CORRECTLY ADMINISTER ENSEMBLE AND CUSTOM DEEP LEARNING SCRIPTS AS BELOW, IT IS IMPORTANT TO download GitHub desktop from github.com, click 'clone' at the h2o-3 repo (https://github.com/h2oai/h2o-3), go to out the "h2oEnsemble_v0.1.9" branch from in the GitHub program, cd to your local h2o-r/ensemble directory using your terminal window (mac) or command prompt (windows), then type in terminal: R CMD install h2oEnsemble-package ; SEE: https://community.h2o.ai/questions/1588/issue-with-dl-h2o-ensembles-and-h2o-last-version.html
h2o.init(nthreads=-1, max_mem_size="50G")
h2o.removeAll() 
train <- as.h2o(trainSplit)
test <- as.h2o(testSplit)
y <- "y"
x <- setdiff(names(train), y)
train[,y] <- as.factor(train[,y])
test[,y] <- as.factor(test[,y])
h2o.glm.1 <- function(..., alpha = 0.0) h2o.glm.wrapper(..., alpha = alpha) # generalized linear models with varying alpha to go from extremes of ridge and lasso
h2o.glm.2 <- function(..., alpha = 0.5) h2o.glm.wrapper(..., alpha = alpha)
h2o.glm.3 <- function(..., alpha = 1.0) h2o.glm.wrapper(..., alpha = alpha)
h2o.randomForest.1 <- function(..., ntrees = 200, max_depth = 10, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, max_depth = max_depth, seed = seed) # random forest
h2o.randomForest.2 <- function(..., ntrees = 200, col_sample_rate_per_tree = 0.75, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, col_sample_rate_per_tree = col_sample_rate_per_tree, seed = seed)
h2o.randomForest.3 <- function(..., ntrees = 200, col_sample_rate_per_tree = 0.85, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, col_sample_rate_per_tree = col_sample_rate_per_tree, seed = seed)
h2o.gbm.1 <- function(..., ntrees = 100, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, seed = seed) # gradient boosting machines with varied limitations on number of bins, learning rate, and max depth
h2o.gbm.2 <- function(..., ntrees = 100, nbins = 50, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, nbins = nbins, seed = seed)
h2o.gbm.3 <- function(..., ntrees = 100, max_depth = 10, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth = max_depth, seed = seed)
h2o.gbm.4 <- function(..., ntrees = 100, col_sample_rate = 0.8, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.5 <- function(..., ntrees = 100, col_sample_rate = 0.7, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.6 <- function(..., ntrees = 100, col_sample_rate = 0.6, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.7 <- function(..., ntrees = 100, learn_rate = 0.2, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, learn_rate = learn_rate, seed = seed)
h2o.gbm.8 <- function(..., ntrees = 100, max_depth = 3, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth = max_depth, seed = seed)
h2o.deeplearning.1 <- function(..., hidden = c(500,500), activation = "Rectifier", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed) # deep learners demonstrating variation in types of activation functions and hidden layer sizes
h2o.deeplearning.2 <- function(..., hidden = c(200,200,200), activation = "Maxout", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.3 <- function(..., hidden = c(500,500), activation = "RectifierWithDropout", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.4 <- function(..., hidden = c(500,500), activation = "Rectifier", epochs = 50, balance_classes = TRUE, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, balance_classes = balance_classes, seed = seed)
h2o.deeplearning.5 <- function(..., hidden = c(100,100,100), activation = "Rectifier", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.6 <- function(..., hidden = c(50,50), activation = "Rectifier", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.7 <- function(..., hidden = c(100,100), activation = "MaxoutWithDropout", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.glm_nn <- function(..., non_negative = T) h2o.glm.wrapper(..., non_negative = non_negative) # define meta-learner [GLM restricted to non-neg weights, which is shown in the literature to improve outcomes from ensembles]
# ensemble choosing the training data, list of learners defined above, and meta-learner 
fit <- h2o.ensemble(x = x, y = y,training_frame = train,family = "binomial",learner = c("h2o.glm.1","h2o.glm.2","h2o.glm.3","h2o.randomForest.1", "h2o.randomForest.2","h2o.randomForest.3","h2o.gbm.1","h2o.gbm.2","h2o.gbm.3","h2o.gbm.4","h2o.gbm.5","h2o.gbm.6","h2o.gbm.7","h2o.gbm.8","h2o.deeplearning.1","h2o.deeplearning.2","h2o.deeplearning.3","h2o.deeplearning.4","h2o.deeplearning.5","h2o.deeplearning.6", "h2o.deeplearning.7"),metalearner = "h2o.glm_nn",cvControl = list(V = 5))
# NOTE: if you get an error "Error in h2o.deeplearning(x = x, y = y, training_frame = training_frame,  : unused argument (max_confusion_matrix_size = max_confusion_matrix_size)", you need to download GitHub desktop from github.com, click 'clone' at the h2o-3 repo (https://github.com/h2oai/h2o-3), go to out the "h2oEnsemble_v0.1.9" branch from in the GitHub program, cd to your local h2o-r/ensemble directory using your terminal window (mac) or command prompt (windows), then type in terminal: R CMD install h2oEnsemble-package ; SEE: https://community.h2o.ai/questions/1588/issue-with-dl-h2o-ensembles-and-h2o-last-version.html
h2o.ensemble_performance(fit, newdata = test) # C-stat for each ML method and ensemble
fulllogmodel = glm(y~., data =trainSplit, family=binomial()) # conventional logistic model with all predictors
aiclogmodel = step(fulllogmodel,trace=F) # conventional logistic model with backwards variable selection by AIC
library(pROC)
perffulllog = predict.glm(fulllogmodel,newdata=testSplit) # apply conventional logistic models to test data
perfaiclog = predict.glm(aiclogmodel,newdata=testSplit)
roc(testSplit$y, perffulllog) # c-stat for conventional logistic model with all predictors
roc(testSplit$y, perfaiclog) # c-stat for conventional logistic model with backwards variable selection by AIC
