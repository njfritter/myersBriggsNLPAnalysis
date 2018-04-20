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
from helper_functions import mbtitype, mbtiposts
# Import libraries for model selection and feature extraction
from sklearn import naive_bayes, feature_extraction, feature_selection

# Split data into training and testing sets
X_train, X_test, y_train, y_test = hf.train_test_split(test_size = 0.33, random_state = 42)

# Naive Bayes model fitting and predictions
# Building a Pipeline; this does all of the work in intial_model()  
text_clf_nb = hf.build_pipeline(feature_extraction.text.CountVectorizer(ngram_range = (1, 1)),
  feature_extraction.text.TfidfTransformer(use_idf = False),
  feature_selection.SelectKBest(feature_selection.chi2, k = 'all'),
  naive_bayes.MultinomialNB(fit_prior = False, alpha = 1.0e-10)
)

text_clf_nb = text_clf_nb.fit(X_train, y_train)

# Evaluate performance on test set
predicted_nb = text_clf_nb.predict(X_test)
print("Training set score: %f" % text_clf_nb.score(X_train, y_train))
print("Test set score: %f" % text_clf_nb.score(X_test, y_test))
print("Test error rate: %f" % (1 - text_clf_nb.score(X_test, y_test)))
print("Number of mislabeled points out of a total %d points for the Naive Bayes algorithm : %d"
  % (X_test.shape[0],(y_test != predicted_nb).sum()))

# Display success rate of predictions for each type
rates = hf.success_rates(y_test, predicted_nb, return_results = True)
print(rates)

# Test set calculations
test_crosstb_nb = pd.crosstab(index = y_test, columns = predicted_nb, rownames = ['class'], colnames = ['predicted'])
print(test_crosstb_nb)

# Frequencies of personality types
labels, counts = hf.unique_labels(y_test, plot = False)
print(labels, counts)

# Plot success rate versus frequency
hf.scatter_plot(list(counts), list(rates.values())) 

# Cross Validation
#cross_val(text_clf_nb, mbtiposts, mbtitype)

# Do a Grid Search to test multiple parameter values
#grid_search(text_clf_nb, parameters_nb, n_jobs = 1, X_train, y_train)
