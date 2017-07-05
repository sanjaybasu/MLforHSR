"""

    Machine learning in health services research
=====================================================

An example of supervised learning for health services researchers. The dataset
is a synthetic claims database.

This script runs through random forests and gradient boosting models and
prints C-statistics.

Authors: Sanjay Basu and Patrick Doupe

"""
print(__doc__)

import csv, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
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
# DECISION TREES

# RANDOM FOREST  
# we only report cross validation errors in this script

# build model with default parameters
random_forest = RandomForestClassifier()

# fit the model
random_forest.fit(x_train, y_train)

# get predictions (second dimension are predictions for class 1)
predictions_rf = random_forest.predict_proba(x_cv)[:,1]

# get C-statistics
c_statistic_rf = roc_auc_score(y_cv, predictions_rf)

# GRADIENT BOOSTING
# build model with default parameters
gradient_boost = GradientBoostingClassifier()

# fit the model
gradient_boost.fit(x_train, y_train)

# get predictions (second dimension are predictions for class 1)
predictions_gb = gradient_boost.predict_proba(x_cv)[:,1]

# get C-statistics
c_statistic_gb = roc_auc_score(y_cv, predictions_gb)

##########################
# PRINT RESULTS
#

print('\nDecision tree model results' + ' ---' * 9)
print('\nRandom forest C statistic: ', c_statistic_rf)
print('\nGradient boosting C statistic: ', c_statistic_gb)
print('\n' + '--- ' * 16)
#EOF
