# This is a script that will be our "helper functions"
# Each model uses them, so I will be setting up the framework here
# And then importing them into each model script

unprocessed_data = './data/mbti_1.csv'
processed_data = './data/mbti_2.csv'
local_stopwords = np.empty(shape = (10, 1))
columns = np.array(['type', 'posts'])
file = pd.read_csv(unprocessed_data, names = columns)
csv_file = csv.reader(open(unprocessed_data, 'rt'))

def read_split():
  # Split up into types and posts
  processed_file = pd.read_csv(processed_data, names = columns, skiprows = [0])
  mbtitype = np.array(processed_file['type'])
  mbtiposts = np.array(processed_file['posts'])

  return mbtiposts, mbtitype

def train_test_split():
  # Split data into training and testing sets
  mbtiposts, mbtitype = read_split()

  X_train, X_test, y_train, y_test = model_selection.train_test_split(
  mbtiposts, mbtitype, test_size = 0.33, random_state = 42)

  return X_train, X_test, y_train, y_test

def build_pipeline(model):
  # Build pipelines for models
  text_clf = pipeline.Pipeline([
   ('vect', feature_extraction.text.CountVectorizer()),
   ('tfidf', feature_extraction.text.TfidfTransformer()),
   ('chi2', feature_selection.SelectKBest(feature_selection.chi2, k = 'all')),
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

def plot_frequency(labels, freq):
  fig, ax = plt.subplots()
  width = 0.5
  ind = np.arange(len(labels))
  ax.barh(ind, freq, width, color = 'red')
  ax.set_yticks(ind + width / 2)
  ax.set_yticklabels(labels, minor = False)
  for i, v in enumerate(freq):
    ax.text(v + 2, i - 0.125, str(v), color = 'blue', fontweight = 'bold')
  plt.title('Personality Type Frequencies')
  plt.xlabel('Frequency')
  plt.ylabel('Type')
  plt.show()

def unique_labels_word_freq():

  mbtiposts, mbtitype = read_split()

  # Show counts of personality types of tweets
  unique, counts = np.unique(mbtitype, return_counts=True)
  print(np.asarray((unique, counts)).T)

  # Now to make bar graphs
  # First for the type frequencies
  plot_frequency(unique, counts)

  # Gather list of words
  words = gather_words(mbtiposts)

  words_top_25 = []
  freq_top_25 = []
  word_features = nltk.FreqDist(words)
  print("\nMost frequent words with counts:")
  for word, frequency in word_features.most_common(25):
    print('%s: %d' % (word, frequency))
    words_top_25.append(word.title())
    freq_top_25.append(frequency)

  # Now top 25 word frequencies
  unique, counts = np.array(words_top_25), np.array(freq_top_25)
  plot_frequency(unique, counts)

def word_cloud():

  mbtiposts, mbtitype = read_split()

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
