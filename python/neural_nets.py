"""

    Machine learning in health services research
=====================================================

An example of supervised learning for health services researchers. The dataset
is a synthetic claims database.

This script runs through feed forward neural networks and prints C-statistics.

Authors: Sanjay Basu and Patrick Doupe

"""
print(__doc__)

import csv, numpy as np
from sklearn.neural_network import MLPClassifier
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
# NEURAL NETWORKS 
# we only report cross validation errors in this script

# MULTI LAYER PERCEPTRON WITH ONE HIDDEN LAYER

# build model with 128 hidden units and ReLU activation
mlp_128 = MLPClassifier(hidden_layer_sizes=(128,),
        activation='relu')

# fit the model
mlp_128.fit(x_train, y_train)

# get predictions (second dimension are predictions for class 1)
predictions_128 = mlp_128.predict_proba(x_cv)[:,1]

# get C-statistics
c_statistic_128 = roc_auc_score(y_cv, predictions_128)

# MULTI LAYER PERCEPTRON WITH TWO HIDDEN LAYERS

# build model with 2 x 128 hidden units and ReLU activation
mlp_128_128 = MLPClassifier(hidden_layer_sizes=(128,128),
        activation='relu')

# fit the model
mlp_128_128.fit(x_train, y_train)

# get predictions (second dimension are predictions for class 1)
predictions_128_128 = mlp_128_128.predict_proba(x_cv)[:,1]

# get C-statistics
c_statistic_128_128 = roc_auc_score(y_cv, predictions_128_128)

##########################
# PRINT RESULTS
#

print('\nNeural network results' + ' ---' * 10)
print('\nSingle hidden layer C statistic: ', c_statistic_128)
print('\nDouble hidden layer C statistic: ', c_statistic_128_128)
print('\n' + '--- ' * 16)
#EOF
