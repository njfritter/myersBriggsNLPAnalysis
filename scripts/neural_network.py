#!/usr/bin/env python3

# Myers Briggs Personality Type Tweets Natural Language Processing
# By Nathan Fritter
# Project can be found at: 
# https://www.inertia7.com/projects/109 & 
# https://www.inertia7.com/projects/110

################

# Import Packages
import random
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
import re
import os
import sys
import helper_functions as hf
# Import libraries for model selection and feature extraction
from sklearn import (datasets, naive_bayes, feature_extraction, pipeline, linear_model,
metrics, neural_network, model_selection, feature_selection, svm)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = hf.train_test_split()

# NEURAL NETWORK
text_clf_nn = hf.build_pipeline(neural_network.MLPClassifier(
  hidden_layer_sizes=(50,), 
  activation = 'identity',
  max_iter=50, 
  alpha=1e-4,
  solver='sgd', 
  verbose=10, 
  tol=5e-4, 
  random_state=1,
  learning_rate_init=.1))

text_clf_nn.fit(X_train, y_train)

# Evaluate performance on test set
predicted_nn = text_clf_nn.predict(X_test)
print("Training set score: %f" % text_clf_nn.score(X_train, y_train))
print("Test set score: %f" % text_clf_nn.score(X_test, y_test))
print("Number of mislabeled points out of a total %d points for the Linear SVM algorithm: %d"
  % (X_test.shape[0],(y_test != predicted_nn).sum()))

# Test Crosstab results
test_crosstb_nn = pd.crosstab(index = y_test, columns = predicted_nn, rownames = ['class'], colnames = ['predicted'])
print(test_crosstb_nn)

# Cross Validation Score
mbtiposts, mbtitype = hf.read_split()
#cross_val(text_clf_nn, mbtiposts, mbtitype)

# Do a Grid Search to test multiple parameter values
#grid_search(text_clf_nn, parameters_nn, 1, X_train, y_train)
