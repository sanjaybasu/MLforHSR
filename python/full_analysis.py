"""

    Machine learning in health services research
=====================================================

An example of supervised learning for health services researchers. The dataset
is a synthetic claims database.

This script runs through all models, and an ensemble of the models. We choose
the best performing model on a cross validation dataset using the C-statistic. 

A final C-statistic on the test set is presented using the best performing
model.

Authors: Sanjay Basu and Patrick Doupe

"""
print(__doc__)

import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

###################################################
# LOAD DATA
#
# data constructed in the R script 'reproduction.R'
# and stored in folder '../data'

# TRAINING DATA
training_data_location = '../data/training_data.csv'    

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

# TESTING DATA
testing_data_location = '../data/testing_data.csv'    

test_data = []
with open(testing_data_location, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        test_data.append(row)
testing_data = np.array(test_data[1:], dtype=float)
y_test = testing_data[:,-1]
x_test = testing_data[:, 1:-1]
print('Data loaded\n')

###################################################
# CREATE PIPELINE

# cross validation parameters
inverse_regularization = [0.3, 0.5, 1.0, 3.0]

classifiers = {'L1 A': LogisticRegression(C=inverse_regularization[0],
    penalty='l1', solver='liblinear'), 
    'L1 B': LogisticRegression(C=inverse_regularization[1], penalty='l1',
        solver='liblinear'), 
    'L1 C': LogisticRegression(C=inverse_regularization[2], penalty='l1',
        solver='liblinear'), 
    'L1 D': LogisticRegression(C=inverse_regularization[3], penalty='l1',
        solver='liblinear'),
    'L2 A': LogisticRegression(C=inverse_regularization[0], penalty='l2'),
    'L2 B': LogisticRegression(C=inverse_regularization[1], penalty='l2'),
    'L2 C': LogisticRegression(C=inverse_regularization[2], penalty='l2'),
    'L2 D': LogisticRegression(C=inverse_regularization[3], penalty='l2'),
    'RF 1': RandomForestClassifier(max_features='log2'),
    'RF 2': RandomForestClassifier(),
    'RF 3': RandomForestClassifier(n_estimators=100, max_features=1),
    'RF 4': RandomForestClassifier(max_depth=2, n_estimators=50),
    'GB 1': GradientBoostingClassifier(loss='exponential'),
    'GB 2': GradientBoostingClassifier(),
    'GB 3': GradientBoostingClassifier(n_estimators=500),
    'GB 4': GradientBoostingClassifier(n_estimators=50),
    'NN 1': MLPClassifier(hidden_layer_sizes=(128,), activation='relu'),
    'NN 2': MLPClassifier(hidden_layer_sizes=(128,128), activation='relu'),
    'NN 3': MLPClassifier(hidden_layer_sizes=(128,512), activation='relu'),
    'NN 4': MLPClassifier(hidden_layer_sizes=(128,128,128),
        activation='relu'),
    'NN 5': MLPClassifier(hidden_layer_sizes=(128,512), activation='tanh'),
    'NN 6': MLPClassifier(hidden_layer_sizes=(128,128), activation='tanh')
    }

print("\nC-statistics for each model")
print('--- ' * 16)
best_c_statistic = 0
ensemble_data = []
ensemble_data_cv = []
for index, (name, classifier) in enumerate(classifiers.items()):
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict_proba(x_cv)[:,1]
    c_stat = roc_auc_score(y_cv, y_pred)
    print("Model %s : %f " % (name, c_stat))
    # SAVE MODEL FROM BEST C_STATISTIC
    if c_stat > best_c_statistic:
        best_model = classifier
        best_name = name
        best_c_statistic = c_stat
    # store data for ensemble
    y_ensemble = classifier.predict_proba(x_train)[:,1]
    ensemble_data.append(y_ensemble)
    ensemble_data_cv.append(y_pred)

# rearrange data
ensemble_data = [list(d) for d in zip(*ensemble_data)]
ensemble_data_cv = [list(d) for d in zip(*ensemble_data_cv)]
# ensemble model
ensemble = LogisticRegression(penalty='l2')
ensemble.fit(ensemble_data, y_train)
y_pred = ensemble.predict_proba(ensemble_data_cv)[:,1]
c_stat = roc_auc_score(y_cv, y_pred)
print("Model %s : %f " % ('Ensemble', c_stat))
# SAVE MODEL FROM BEST C_STATISTIC
if c_stat > best_c_statistic:
    best_model = ensemble
    best_name = 'Ensemble'
    best_c_statistic = c_stat

best_name = 'Ensemble'
print('\nThe best model is: ', best_name)

# we have chosen our best performing model
# we can get estimate our model performance using the test statistic

# if the best model is the ensemble, the we have to estimate all models over
# full training set
if best_name == 'Ensemble':
    # train model
    ensemble_data = []
    ensemble_data_test = []
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(x, y)
        y_pred = classifier.predict_proba(x)[:,1]
        ensemble_data.append(y_pred)
        classifier.fit(x_test, y_test)
        y_pred = classifier.predict_proba(x_test)[:,1]
        ensemble_data_test.append(y_pred)
    
    x_e = [list(d) for d in zip(*ensemble_data)]
    x_e_test = [list(d) for d in zip(*ensemble_data_test)]
    
best_model.fit(x_e, y)
y_pred = best_model.predict_proba(x_e_test)[:,1]
c_stat = roc_auc_score(y_test, y_pred)

print("\n" + '--- ' * 16)
print("\nTest set C statistic for %s : %f " % (best_name, c_stat))

