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
try:
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
  # Import libraries for model selection and feature extraction
  from sklearn import (datasets, naive_bayes, feature_extraction, pipeline, linear_model,
  metrics, neural_network, model_selection, feature_selection)
except ImportError:
  print("The necessary packages do not seem to be installed",
    "Please make sure to pip install the necessary packages in \'requirements.txt\'")

# Set variables for files and file objects
try:
  unprocessed_data = './data/mbti_unprocessed.csv'
  processed_data = './data/mbti_processed.csv'
  local_stopwords = np.empty(shape = (10, 1))
  columns = np.array(['type', 'posts'])
  file = pd.read_csv(unprocessed_data, names = columns)
  csv_file = csv.reader(open(unprocessed_data, 'rt'))
except FileNotFoundError:
  print("The necessary files do not seem to exist",
    "Please make sure \'mbti_unprocessed.csv\' and \'mbti_processed.csv\'\
    exist in the proper file paths states above")

# Processed data
# Split up into types and posts
processed_file = pd.read_csv(processed_data, names = columns, skiprows = [0])
mbtitype = np.array(processed_file['type'])
mbtiposts = np.array(processed_file['posts'])

def basic_output():
  # Basic stuff
  try:
    print(file.columns)
    print(file.shape)
    print(file.head(5))
    print(file.tail(5))
  except AttributeError:
    print("The files do not seem to be in the proper format.",
    "Did you read in the files as a pandas object?")

def tokenize_data():
  # Tokenize words line by line
  # Download stopwords here
  nltk.download('stopwords')
  # And write to new file so we don't have to keep doing this
  tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
  processed = csv.writer(open(processed_data, 'w+'))

  try:
    i = 0
    for line in csv_file:
      (ptype, posts) = line
      # Regular expressions filter out hyperlinks & emojis
      words = re.sub(r"(?:\@|https?\://)\S+", "", posts)
      # Tokenize
      words = [word.lower() for word in tokenizer.tokenize(words)]
      words = [word for word in words if word not in nltk.corpus.stopwords.words('english') and word not in local_stopwords]

      if i % 100 == 0:
          print(i)
      i += 1

      processed.writerow([ptype] + [words])
  except AttributeError:
    print("The files do not seem to be in the proper format.",
    "Did you read in the files as a pandas object?")

def train_test_split(test_size, random_state):
  # Split data into training and testing sets
  try:
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
    mbtiposts, mbtitype, test_size = 0.33, random_state = 42)

    return X_train, X_test, y_train, y_test
  except ImportError:
    print("The necessary packages do not seem to be installed")

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
  """
  gs_clf = model_selection.GridSearchCV(clf, 
    param_grid = parameters, 
    n_jobs = jobs,
    verbose = 7
    )
  gs_clf = gs_clf.fit(X, y)
  """
  try:
    # Create Spark session
    sc = spark_sklearn.util.createLocalSparkSession().sparkContext
    # Perform grid search
    gs_clf = spark_sklearn.GridSearchCV(sc, estimator = clf, 
      param_grid = parameters, 
      n_jobs = jobs,
      verbose = 7
      )
    gs_clf = gs_clf.fit(X, y)
  except java.lang.IllegalArgumentException:
    print("The necessary java environment is not installed",
      "Make sure that you have a JDK set up for Spark distributed computing.")

  """
  best_parameters, score, _ = max(gs_clf.grid_scores_, key = lambda x: x[1])
  for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))
  print(score)
  """
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

def plot_frequency(labels, freq, data):
  # Horizontal Boxplots
  fig, ax = plt.subplots()
  width = 0.5
  ind = np.arange(len(labels))
  ax.barh(ind, freq, width, color = 'red')
  ax.set_yticks(ind + width / 2)
  ax.set_yticklabels(labels, minor = False)
  for i, v in enumerate(freq):
    ax.text(v + 2, i - 0.125, str(v), color = 'blue', fontweight = 'bold')
  if data == 'Type': 
    plt.title('Personality Type Frequencies')
    plt.xlabel('Frequency')
    plt.ylabel('Type')
  if data == 'Words':
    plt.title('Top 25 Word Frequencies')
    plt.xlabel('Frequency')
    plt.ylabel('Word')
  plt.show()

def unique_labels(labels, plot):
  # Show counts of personality types of tweets
  unique, counts = np.unique(labels, return_counts=True)

  if plot:
    # Now to make bar graphs
    # First for the type frequencies
    plot_frequency(unique, counts, 'Type')
    print(np.asarray((unique, counts)).T)
  else:
    return unique, counts

def word_freq(word_data, plot):
  # Gather list of words
  words = gather_words(word_data)

  words_top_25 = []
  freq_top_25 = []
  word_features = nltk.FreqDist(words)
  print("\nMost frequent words with counts:")
  for word, frequency in word_features.most_common(25):
    print('%s: %d' % (word, frequency))
    words_top_25.append(word.title())
    freq_top_25.append(frequency)

  if plot:
    # Now top 25 word frequencies
    unique, counts = np.array(words_top_25), np.array(freq_top_25)
    plot_frequency(unique, counts, 'Words')
  else:
    return unique, counts

def word_cloud():
  # Gather list of words
  words = gather_words(mbtiposts)

  wordcloud_words = " ".join(words)
  # Lower max font size
  cloud = wordcloud.WordCloud(max_font_size = 40).generate(wordcloud_words)
  plt.figure()
  plt.imshow(cloud, interpolation = 'bilinear')
  plt.axis("off")
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
      
