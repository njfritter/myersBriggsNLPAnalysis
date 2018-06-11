#!/usr/bin/env python3

# Myers Briggs Personality Type Tweets Natural Language Processing
# By Nathan Fritter
# Project can be found at: 
# https://www.inertia7.com/projects/109 & 
# https://www.inertia7.com/projects/110

################

# This is a script that will be our "helper functions"
# Each model script uses them, so I will be setting up the framework here
# And then importing them into each model script

# But first import necessary packages
import matplotlib as mpl
mpl.use('TkAgg') # Need to do this everytime for some reason
import matplotlib.pyplot as plt
import random
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import nltk
import wordcloud
# Import libraries for model selection and feature extraction
from sklearn import (datasets, naive_bayes, feature_extraction, pipeline, linear_model,
metrics, neural_network, model_selection, feature_selection)


# Confirm we are in the correct directory, otherwise break script 
# and prompt user to move to correct directory
filepath = os.getcwd()
if not filepath.endswith('myersBriggsNLPAnalysis'):
    print('\nYou do not appear to be in the correct directory,\
    you must be in the \'myersBriggsNLPAnalysis\' directory\
    in order to run these scripts. Type \'pwd\' in the command line\
    if you are unsure of your location in the terminal.')
    sys.exit(1)


def plot_frequency(labels, freq, data):
    # Horizontal Boxplots
    labels = np.array(labels)
    freq = np.array(freq)
    fig, ax = plt.subplots()
    width = 0.5
    ind = np.arange(len(labels))
    ax.barh(ind, freq, width, color = 'red')
    ax.set_yticks(ind + width / 2)
    ax.set_yticklabels(labels, minor = False)
    for i, v in enumerate(freq):
    ax.text(v + 2, i - 0.125, str(v), color = 'blue', fontweight = 'bold')
    if data == 'Types': 
        plt.title('Personality Type Frequencies')
        plt.xlabel('Frequency')
        plt.ylabel('Type')
        plt.savefig('images/typefrequencylabeled.png')
    if data == 'Words':
        plt.title('Top 25 Word Frequencies')
        plt.xlabel('Frequency')
        plt.ylabel('Word')
        plt.savefig('images/wordfrequencylabeled.png')
    
    plt.show()


def build_pipeline(vectorizer, tfidf, kbest, model):
  # Build pipelines for models
  text_clf = pipeline.Pipeline([
   ('vect', vectorizer),
   ('tfidf', tfidf),
   ('chi2', kbest),
   ('clf', model),
  ])

  return text_clf

def grid_search(clf, parameters, jobs, X, y):  
  # Perform grid search
  gs_clf = model_selection.GridSearchCV(clf, 
    param_grid = parameters, 
    n_jobs = jobs,
    verbose = 7
    )
  gs_clf = gs_clf.fit(X, y)

  best_parameters, score, _ = max(gs_clf.grid_scores_, key = lambda x: x[1])
  for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))
  print(score)

  print(gs_clf.cv_results_)

def gather_words(posts):
  
  words = []
  for tweet in posts:
    # Split tweet into words by comma
    # Or else iterator splits by letter, not word
    tweet_words = tweet.split(',')
    for word in tweet_words:
      # Remove brackets at end of tweet and quotes
      word = re.sub(r"]", "", word)
      word = re.sub(r"\'", "", word)
      words.append(word)

  return words

def scatter_plot(x, y):
  # Scatterplot
  plt.scatter(x, y)
  # Make trendline 
  trend = np.polyfit(x, y, 1)
  p = np.poly1d(trend)
  # Add to graph
  plt.plot(x, p(x), 'r--')

  plt.show()


def cross_val(clf, X_train, y_train):
  # Cross Validation Score
  scores = model_selection.cross_val_score(clf, X_train, y_train, cv = 5)
  print(scores)
  print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def success_rates(labels, predictions, return_results):
# Display success rate of predictions for each type
  labels_pred = pd.DataFrame(labels, columns = ['label'])
  labels_pred['predicted'] = predictions
  labels_pred['success'] = (labels_pred['predicted'] == labels)

  fracs = {}
  for name, group in labels_pred.groupby('label'):
    frac = sum(group['success'])/len(group)
    fracs[name] = frac
    if not return_results:
      print('Success rate for labeling personality type %s: %f' % (name, frac))
  
  if return_results:
    return fracs
      
