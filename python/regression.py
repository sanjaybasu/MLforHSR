"""

    Machine learning in health services research
=====================================================

An example of supervised learning for health services researchers. The dataset
is a synthetic claims database.

This script runs through regression models and prints C-statistics.

Authors: Sanjay Basu and Patrick Doupe

"""
print(__doc__)

import csv, numpy as np
from sklearn import linear_model
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

###################################################
# LOAD DATA
#
# data constructed in the R script 'reproduction.R'
# and stored in folder '../data'

training_data_location = '../data/trainData.csv'    

train_data = []
with open(training_data_location, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        train_data.append(row)
training_data = np.array(train_data[1:], dtype=float)
y = training_data[:,-1]
x = training_data[:, 1:-1]

# cross validation data
x_train, x_cv, y_train, y_cv = train_test_split(
        x, y, test_size=0.33, random_state=1996)

print('\nData loaded\n')

##########################
# LOGISTIC REGRESSION
# we only report cross validation errors in this script

# LOGISTIC WITH L1

# build model with parameters
# 'C' is the inverse of the regularization strenth
regression_l1 = linear_model.LogisticRegressionCV(
        penalty='l1', Cs=[0.3, 0.5, 1.0, 3.0, 5.0],
        solver='liblinear',
        scoring='roc_auc')

# fit the model
regression_l1.fit(x_train, y_train)

# get predictions (second dimension are predictions for class 1)
predictions_l1 = regression_l1.predict_proba(x_cv)[:,1]

# get C-statistics
c_statistic_l1 = roc_auc_score(y_cv, predictions_l1)

# LOGISTIC WITH L2

# build model with parameters
# 'C' is the inverse of the regularization strenth
regression_l2 = linear_model.LogisticRegressionCV(
        penalty='l2', Cs=[0.3, 0.5, 1.0, 3.0, 5.0],
        scoring='roc_auc')

# fit the model
regression_l2.fit(x_train, y_train)

# get predictions (second dimension are predictions for class 1)
predictions_l2 = regression_l2.predict_proba(x_cv)[:,1]

# get C-statistics
c_statistic_l2 = roc_auc_score(y_cv, predictions_l2)

##########################
# PRINT RESULTS
#

print('\nRegression model results' + ' ---' * 10)
print('\nL1 logistic regression C statistic: ', c_statistic_l1)
print('\nL1 best fitting parameter: ', regression_l1.C_[0])
print('\nL2 logistic regression C statistic: ', c_statistic_l2)
print('\nL2 best fitting parameter: ', regression_l2.C_[0])
print('\n' + '--- ' * 16)
#EOF
