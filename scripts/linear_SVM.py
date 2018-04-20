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
from sklearn import linear_model, feature_extraction, feature_selection

# Linear Support Vector Machine parameters for tuning
parameters_svm = {
  'vect__ngram_range': [(1, 1), (1, 2)],
  'tfidf__use_idf': (True, False),
  'clf__alpha': (1e-2, 1e-3),
  'clf__penalty': ['l2', 'l1', 'elasticnet'],
  'clf__l1_ratio': (0, 0.5, 1),
  'clf__learning_rate': ['optimal'],
  'clf__eta0': (0.25, 0.5, 0.75)
}

# Split data into training and testing sets
X_train, X_test, y_train, y_test = hf.train_test_split(test_size = 0.33, random_state = 42)

# Linear Support Vector Machine w/ stochastic gradient descent (SGD) learning
# This model can be other linear models, but using "hinge" makes it a SVM
# Build Pipeline again
text_clf_svm = hf.build_pipeline(feature_extraction.text.CountVectorizer(ngram_range = (1, 1)),
  feature_extraction.text.TfidfTransformer(use_idf = True),
  feature_selection.SelectKBest(feature_selection.chi2, k = 'all'),
  linear_model.SGDClassifier(
   loss='hinge',
   penalty='l2',
   l1_ratio = 0,
   alpha=1e-3,
   eta0 = 0.25,
   max_iter=5,
   learning_rate = 'optimal',
   verbose = 10,
   random_state=42)
)

text_clf_svm = text_clf_svm.fit(X_train, y_train)

# Evaluate performance on test set
predicted_svm = text_clf_svm.predict(X_test)
print("Training set score: %f" % text_clf_svm.score(X_train, y_train))
print("Test set score: %f" % text_clf_svm.score(X_test, y_test))
print("Test error rate: %f" % (1 - text_clf_svm.score(X_test, y_test)))
print("Number of mislabeled points out of a total %d points for the Linear SVM algorithm: %d"
  % (X_test.shape[0],(y_test != predicted_svm).sum()))

# Display success rate of predictions for each type
rates = hf.success_rates(y_test, predicted_svm, return_results = True)
print(rates)

# Test set calculations
test_crosstb_nb = pd.crosstab(index = y_test, columns = predicted_svm, rownames = ['class'], colnames = ['predicted'])
print(test_crosstb_nb)

# Frequencies of personality types
labels, counts = hf.unique_labels(y_test, plot = False)
print(labels, counts)

# Plot success rate versus frequency
hf.scatter_plot(list(counts), list(rates.values())) 

# Cross Validation
#cross_val(text_clf_svm, mbtiposts, mbtitype)

# Do a Grid Search to test multiple parameter values
#grid_search(text_clf_svm, parameters_svm, n_jobs = 1, X_train, y_train)
