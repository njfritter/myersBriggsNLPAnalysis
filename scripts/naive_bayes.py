# Import Packages
import matplotlib as mpl
mpl.use('TkAgg') # Need to do this everytime for some reason
import matplotlib.pyplot as plt
import random
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
import re
import nltk
import wordcloud
import os
import sys
import helper_functions as hf
# Import libraries for model selection and feature extraction
from sklearn import (datasets, naive_bayes, feature_extraction, pipeline, linear_model,
metrics, neural_network, model_selection, feature_selection, svm)


# Split data into training and testing sets
X_train, X_test, y_train, y_test = hf.train_test_split()

# Naive Bayes model fitting and predictions
# Building a Pipeline; this does all of the work in intial_model()  
text_clf_nb = hf.build_pipeline(naive_bayes.MultinomialNB())
text_clf_nb = text_clf_nb.fit(X_train, y_train)

# Evaluate performance on test set
predicted_nb = text_clf_nb.predict(X_test)
print("Training set score: %f" % text_clf_nb.score(X_train, y_train))
print("Test set score: %f" % text_clf_nb.score(X_test, y_test))
print("Test error rate: %f" % (1 - text_clf_nb.score(X_test, y_test)))
print("Number of mislabeled points out of a total %d points for the Naive Bayes algorithm : %d"
  % (X_test.shape[0],(y_test != predicted_nb).sum()))

# Test set calculations
test_crosstb_nb = pd.crosstab(index = y_test, columns = predicted_nb, rownames = ['class'], colnames = ['predicted'])
print(test_crosstb_nb)

# Cross Validation
mbtiposts, mbtitype = hf.read_split()
#cross_val(text_clf_nb, mbtiposts, mbtitype)

# Do a Grid Search to test multiple parameter values
#grid_search(text_clf_nb, parameters_nb, 1, X_train, y_train)
