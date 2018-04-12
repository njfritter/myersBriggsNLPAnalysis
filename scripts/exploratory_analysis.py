# This is a script that is dedicated to the exploratory analysis of this project
# This includes words and type frequencies, word clouds, and the tokenization process

unprocessed_data = './data/mbti_1.csv'
processed_data = './data/mbti_2.csv'
local_stopwords = np.empty(shape = (10, 1))
columns = np.array(['type', 'posts'])
file = pd.read_csv(unprocessed_data, names = columns)
csv_file = csv.reader(open(unprocessed_data, 'rt'))

def basic_output():
  # Basic stuff
  print(file.columns)
  print(file.shape)
  print(file.head(5))
  print(file.tail(5))

def tokenize_data():
  # Tokenize words line by line
  # Download stopwords here
  nltk.download('stopwords')
  # And write to new file so we don't have to keep doing this
  tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
  processed = csv.writer(open(processed_data, 'w+'))

  i = 0
  for line in csv_file:
    (ptype, posts) = line
    # Regular expressions
    words = re.sub(r"(?:\@|https?\://)\S+", "", posts)
    # Tokenize
    words = [word.lower() for word in tokenizer.tokenize(words)]
    words = [word for word in words if word not in nltk.corpus.stopwords.words('english') and word not in local_stopwords]

    if i % 100 == 0:
        print(i)
    i += 1

    processed.writerow([ptype] + [words])


def plot_frequency(labels, freq, data):
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

def unique_labels_word_freq():

  mbtiposts, mbtitype = read_split()

  # Show counts of personality types of tweets
  unique, counts = np.unique(mbtitype, return_counts=True)
  print(np.asarray((unique, counts)).T)

  # Now to make bar graphs
  # First for the type frequencies
  plot_frequency(unique, counts, 'Type')

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
  plot_frequency(unique, counts, 'Words')
