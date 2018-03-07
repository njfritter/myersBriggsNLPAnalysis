# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import random
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
import re
import nltk
# Only need this
import wordcloud
import os
import sys
# Import libraries for model selection and feature extraction
from sklearn import (datasets, naive_bayes, feature_extraction, pipeline, linear_model,
metrics, neural_network, model_selection, feature_selection)

# Any results you write to the current directory are saved as output.
unprocessed_data = '/Users/nathanfritter/myProjects/dataScience/myRepos/myersBriggsNLPAnalysis/mbti_1.csv'
# random_data = '~/myProjects/dataScience/myRepos/myersBriggsNLPAnalysis/mbti_random.csv'
processed_data = '/Users/nathanfritter/myProjects/dataScience/myRepos/myersBriggsNLPAnalysis/mbti_2.csv'
local_stopwords = []
columns = np.array(['type', 'posts'])
file = pd.read_csv(unprocessed_data, names = columns)
csv_file = csv.reader(open(unprocessed_data, 'rt'))


def show_stuff():
  # Basic stuff
  print(file.columns)
  print(file.shape)
  print(file.head(5))
  print(file.tail(5))

def tokenize_data():
  # Tokenize words line by line
  # Download stopwords here?
  nltk.download('stopwords')
  # And write to new file so we don't have to keep doing this
  tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
  processed = csv.writer(open(processed_data, 'w+'))

  i = 0
  for line in csv_file:
    (ptype, posts) = line
    # Regular expressions
    posts = re.sub(r"(?:\@|https?\://)\S+", "", posts)
    # Tokenize
    words = [word.lower() for word in tokenizer.tokenize(posts)]
    words = [word for word in words if word not in nltk.corpus.stopwords.words('english') and word not in local_stopwords]
    #posts = words

    if i % 100 == 0:
        print(i)
    i += 1

    processed.writerow([ptype] + [words])

def read_split():
  # Split up into types and posts
  processed_file = pd.read_csv(processed_data, names = columns)
  mbtitype = np.array(processed_file['type'])
  mbtiposts = np.array(processed_file['posts'])

  return mbtitype, mbtiposts

def train_test_split():
  # Split data into training and testing sets
  mbtitype, mbtiposts = read_split()

  X_train, X_test, y_train, y_test = model_selection.train_test_split(
  mbtiposts, mbtitype, test_size=0.33, random_state=42)

  print("X train length: %d" % len(X_train))
  print("X test length: %d" % len(X_test))
  print("Y train length: %d" % len(y_train))
  print("Y test length: %d" % len(y_test))

  return X_train, X_test, y_train, y_test


def unique_labels_word_freq():
  mbtitype, mbtiposts = read_split()

  print(type(mbtitype))
  print(type(mbtiposts))

  X_train, X_test, y_train, y_test = model_selection.train_test_split(
  mbtiposts, mbtitype, test_size=0.33, random_state=42)

  # Show unique labels
  unique, counts = np.unique(y_train, return_counts=True)
  print(np.asarray((unique, counts)).T)

  words = []
  for word in mbtiposts:
    words += word

  word_features = nltk.FreqDist(words)
  print("\nMost frequent words with counts:")
  for word, frequency in word_features.most_common(25):
    print('%s;%d' % (word, frequency))
  print("\n")

#print(word_features_train.most_common(25).keys())
#print(word_features_test.most_common(25).keys())


# Now to make bar graphs
# plt.plot(file['type'], type = 'bar')

#direc = path.dirname(__file__)
#text = open(file['posts']).read()

def word_cloud():

  mbtitype, mbtiposts = read_split()

  words = []
  for word in mbtiposts:
    words += word

  wordcloud_words = " ".join(words)
  # Lower max font size
  cloud = wordcloud.WordCloud(max_font_size = 40).generate(wordcloud_words)
  plt.figure()
  plt.imshow(cloud, interpolation = 'bilinear')
  plt.axis("off")
  plt.show()


# LINK IMAGE HERE 

def initial_model():
  # Split data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split()

  # Extract features from text files
  count_vect = feature_extraction.text.CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)
  X_train_counts = count_vect.fit_transform(X_train)
  print(X_train_counts.shape)

  tfidf_transformer = feature_extraction.text.TfidfTransformer()
  X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
  print(X_train_tfidf.shape)

  # Training a classifer
  clf = naive_bayes.MultinomialNB()
  clf = clf.fit(X_train_tfidf, y_train)
  INTJ_sentence = ['Writing college essays is stressful because I have to give a stranger a piece of myself and that piece has to incorporate all of who I am']
  INTJ_X_new_counts = count_vect.transform(INTJ_sentence)
  INTJ_X_new_tfidf = tfidf_transformer.transform(INTJ_X_new_counts)

  ENFP_sentence = ['Our favorite friendships are the ones where you can go from talking about the latest episode of the Bachelorette to the meaning of life']
  ENFP_X_new_counts = count_vect.transform(ENFP_sentence)
  ENFP_X_new_tfidf = tfidf_transformer.transform(ENFP_X_new_counts)
  # Make a prediction of test sentence
  predictedINTJ = clf.predict(INTJ_X_new_tfidf)
  predictedENFP = clf.predict(ENFP_X_new_tfidf)
  for words, category in zip(INTJ_sentence, predictedINTJ):
    print('%r => %s' % (INTJ_sentence, category))
  for words, category in zip(ENFP_sentence, predictedENFP):
    print('%r => %s' % (ENFP_sentence, category))


def naive_bayes_model():
  # Split data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split()

  # Naive Bayes model fitting and predictions
  # Building a Pipeline; this does all of the work in extract_and_train() at once  
  text_clf = pipeline.Pipeline([
   ('vect', feature_extraction.text.CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)),
   ('tfidf', feature_extraction.text.TfidfTransformer()),
   ('chi2', feature_selection.SelectKBest(feature_selection.chi2, k = 'all')),
   ('clf', naive_bayes.MultinomialNB()),
  ])

  text_clf = text_clf.fit(X_train, y_train)

  # Evaluate performance on test set
  predicted = text_clf.predict(X_test)
  print("The accuracy of a Naive Bayes algorithm is: ") 
  print(np.mean(predicted == y_test))
  print("Number of mislabeled points out of a total %d points for the Naive Bayes algorithm : %d"
        % (X_test.shape[0],(y_test != predicted).sum()))

  # Tune parameters
  parameters = {
    'vect__ngram_range': [(1, 1), (1, 2)],
    'tfidf__use_idf': (True, False),
    'clf__alpha': (1e-2, 1e-3),
    }

  print("Parameters?")
  gs_clf = model_selection.GridSearchCV(text_clf, parameters, n_jobs=-1)
  gs_clf = gs_clf.fit(X_train, y_train)
  print("Model fit")

  best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
  for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))
  print(score)

"""
None yet
"""

def linear_SVM_model():
  # Split data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split()

  # Linear Support Vector Machine
  # Build Pipeline again
  text_clf_two = pipeline.Pipeline([
   ('vect', feature_extraction.text.CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)),
   ('tfidf', feature_extraction.text.TfidfTransformer()),
   ('chi2', feature_selection.SelectKBest(feature_selection.chi2, k = 'all')),
   ('clf', linear_model.SGDClassifier(
     loss='hinge',
     penalty='l2',
     alpha=1e-3,
     max_iter=5,
     random_state=42)),
    ])
  text_clf_two = text_clf_two.fit(X_train, y_train)
  predicted_two = text_clf_two.predict(X_test)
  print("The accuracy of a Linear SVM is: ")
  print(np.mean(predicted_two == y_test))
  print("Number of mislabeled points out of a total %d points for the Linear SVM algorithm: %d"
        % (X_test.shape[0],(y_test != predicted_two).sum()))


  # Tune parameters Linear Support Vector Machine
  parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
  'tfidf__use_idf': (True, False),
  'clf__alpha': (1e-2, 1e-3),
  }

  gs_clf_two = model_selection.GridSearchCV(text_clf_two, parameters, n_jobs=-1)
  gs_clf_two = gs_clf_two.fit(X_train, y_train)

  best_parameters, score, _ = max(gs_clf_two.grid_scores_, key=lambda x: x[1])
  for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))

  print(score)

def neural_network_model():
  # Split data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split()

  # NEURAL NETWORK
  text_clf_three = pipeline.Pipeline([
    ('vect', feature_extraction.text.CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)),
    ('tfidf', feature_extraction.text.TfidfTransformer()),
    ('chi2', feature_selection.SelectKBest(feature_selection.chi2, k = 'all')),
    ('clf', neural_network.MLPClassifier(
      hidden_layer_sizes=(100,), 
      max_iter=50, 
      alpha=1e-4,
      solver='sgd', 
      verbose=10, 
      tol=1e-4, 
      random_state=1,
      learning_rate_init=.1)),
    ])

  text_clf_three.fit(X_train, y_train)
  print("Training set score: %f" % text_clf_three.score(X_train, y_train))
  print("Test set score: %f" % text_clf_three.score(X_test, y_test))

  # Parameter Tuning
  parameters = {
  'vect__ngram_range': [(1, 1), (1, 2)],
  'tfidf__use_idf': (True, False),
  'clf__alpha': (1e-2, 1e-3),
  }

  gs_clf_three = model_selection.GridSearchCV(text_clf_three, parameters, n_jobs=-1)
  gs_clf_three = gs_clf_three.fit(X_train, y_train)

  best_parameters, score, _ = max(gs_clf_three.grid_scores_, key=lambda x: x[1])
  for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))
  print(score)

  # Cross Validation Score
  scores = model_selection.cross_val_score(text_clf_three, X_train, y_train, cv = 5)
  print(scores)
  print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


if __name__ == '__main__':
  if len(sys.argv) == 2:
    if sys.argv[1] == 'basic':
      show_stuff()
    elif sys.argv[1] == 'tokenize':
      tokenize_data()
    elif sys.argv[1] == 'split':
      train_test_split()
    elif sys.argv[1] == 'unique':
      unique_labels_word_freq()
    elif sys.argv[1] == 'cloud':
      word_cloud()
    elif sys.argv[1] == 'initial':
      initial_model()
    elif sys.argv[1] == 'NB':
      naive_bayes_model()
    elif sys.argv[1] == 'SVM':
      linear_SVM_model()
    elif sys.argv[1] == 'NN':
      neural_network_model()
    else:
      print("Incorrect keyword; please try again.")
  else:
    print("Incorrect number of keywords; please enter only one keyword.")