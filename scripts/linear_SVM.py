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

# Linear Support Vector Machine w/ stochastic gradient descent (SGD) learning
# This model can be other linear models, but using "hinge" makes it a SVM
# Build Pipeline again
text_clf_svm = hf.build_pipeline(linear_model.SGDClassifier(
 loss='hinge',
 penalty='l2',
 alpha=1e-3,
 max_iter=5,
 learning_rate = 'optimal',
 verbose = 10,
 random_state=42))

text_clf_svm = text_clf_svm.fit(X_train, y_train)

# Evaluate performance on test set
predicted_svm = text_clf_svm.predict(X_test)
print("Training set score: %f" % text_clf_svm.score(X_train, y_train))
print("Test set score: %f" % text_clf_svm.score(X_test, y_test))
print("Test error rate: %f" % (1 - text_clf_svm.score(X_test, y_test)))
print("Number of mislabeled points out of a total %d points for the Linear SVM algorithm: %d"
  % (X_test.shape[0],(y_test != predicted_svm).sum()))

test_crosstb_svm = pd.crosstab(index = y_test, columns = predicted_svm, rownames = ['class'], colnames = ['predicted'])
print(test_crosstb_svm)

# Cross Validation
mbtiposts, mbtitype = tf.read_split()
#cross_val(text_clf_svm, mbtiposts, mbtitype)

# Do a Grid Search to test multiple parameter values
#grid_search(text_clf_svm, parameters_svm, 1, X_train, y_train)
