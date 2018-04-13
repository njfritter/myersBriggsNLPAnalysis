#!/usr/bin/env python3

# Myers Briggs Personality Type Tweets Natural Language Processing
# By Nathan Fritter
# Project can be found at: 
# https://www.inertia7.com/projects/109 & 
# https://www.inertia7.com/projects/110

################

# Import Packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import helper_functions as hf
# Import libraries for model selection and feature extraction
from sklearn import neural_network, feature_extraction, feature_selection


# Neural Network parameters for tuning
parameters_nn = {
  'vect__ngram_range': [(1, 1), (1, 2)],
  'tfidf__use_idf': (True, False),
  'clf__learning_rate_init': (1e-1, 5e-1),
  'clf__hidden_layer_sizes': (50, 100),
  'clf__activation': ['identity', 'tanh', 'relu']
}

# Split data into training and testing sets
X_train, X_test, y_train, y_test = hf.train_test_split(test_size = 0.33, random_state = 42)

# NEURAL NETWORK
text_clf_nn = hf.build_pipeline(feature_extraction.text.CountVectorizer(),
  feature_extraction.text.TfidfTransformer(),
  feature_selection.SelectKBest(feature_selection.chi2, k = 'all'),
  neural_network.MLPClassifier(
    hidden_layer_sizes=(50,), 
    activation = 'identity',
    max_iter=50, 
    alpha=1e-4,
    solver='sgd', 
    verbose=10, 
    tol=1e-4, 
    random_state=1,
    learning_rate_init=.1)
)

text_clf_nn.fit(X_train, y_train)

# Evaluate performance on test set
predicted_nn = text_clf_nn.predict(X_test)
print("Training set score: %f" % text_clf_nn.score(X_train, y_train))
print("Test set score: %f" % text_clf_nn.score(X_test, y_test))
print("Number of mislabeled points out of a total %d points for the Linear SVM algorithm: %d"
  % (X_test.shape[0],(y_test != predicted_nn).sum()))

# Display success rate of predictions for each type
hf.success_rates(y_test, predicted_nn)

# Test Crosstab results
test_crosstb_nn = pd.crosstab(index = y_test, columns = predicted_nn, rownames = ['class'], colnames = ['predicted'])
print(test_crosstb_nn)

# Cross Validation Score
#mbtiposts, mbtitype = hf.read_split()
#cross_val(text_clf_nn, mbtiposts, mbtitype)

# Do a Grid Search to test multiple parameter values
#grid_search(text_clf_nn, parameters_nn, n_jobs = 1, X_train, y_train)
