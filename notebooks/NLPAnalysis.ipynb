# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

%matplotlib inline
import matplotlib.pyplot as plt
import random
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

"""
RESULT:
"mbti_1.csv"
"""

# Any results you write to the current directory are saved as output.
unprocessed_data = '~/myProjects/ucsbDataScience/myersBriggsNLPAnalysis/mbti_1.csv'
random_data = '~/myProjects/ucsbDataScience/myersBriggsNLPAnalysis/mbti_random.csv'



# Basic stuff
columns = ['type', 'posts']
file = pd.read_csv(unprocessed_data)
print(file.columns)
print(file.shape)
print(file.head(5))
print(file.tail(5))


"""
Index(['type', 'posts'], dtype='object')
(8675, 2)
   type                                              posts
0  INFJ  'http://www.youtube.com/watch?v=qsXHcwe3krw|||...
1  ENTP  'I'm finding the lack of me in these posts ver...
2  INTP  'Good one  _____   https://www.youtube.com/wat...
3  INTJ  'Dear INTP,   I enjoyed our conversation the o...
4  ENTJ  'You're fired.|||That's another silly misconce...
      type                                              posts
8670  ISFP  'https://www.youtube.com/watch?v=t8edHB_h908||...
8671  ENFP  'So...if this thread already exists someplace ...
8672  INTP  'So many questions when i do these things.  I ...
8673  INFP  'I am very conflicted right now when it comes ...
8674  INFP  'It has been too long since I have been on per...
"""

# How about filter the words first
import csv
import re
from nltk import corpus
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
local_stopwords = []

# Tokenize words line by line
tokenizer = RegexpTokenizer(r'\w+')
i = 0
for index, line in file.iterrows():
    # Regular expressions
    line['posts'] = re.sub(r"(?:\@|https?\://)\S+", "", line['posts'])
    # Tokenize
    words = [word.lower() for word in tokenizer.tokenize(line['posts'])]
    words = [word for word in words if word not in stopwords.words('english') and word not in local_stopwords]
    line['posts'] = words
    if i % 100 == 0:
        print(i)
    i += 1

print(file.head(10))


"""
0
100
200
300
400
500
600
700
800
900
1000
1100
1200
1300
1400
1500
1600
1700
1800
1900
2000
2100
2200
2300
2400
2500
2600
2700
2800
2900
3000
3100
3200
3300
3400
.
.
.
7800
7900
8000
8100
8200
8300
8400
8500
8600
   type                                              posts
0  INFJ  [intj, moments, sportscenter, top, ten, plays,...
1  ENTP  [finding, lack, posts, alarming, sex, boring, ...
2  INTP  [good, one, _____, course, say, know, blessing...
3  INTJ  [dear, intp, enjoyed, conversation, day, esote...
4  ENTJ  [fired, another, silly, misconception, approac...
5  INTJ  [18, 37, perfect, scientist, claims, scientifi...
6  INFJ  [draw, nails, haha, done, professionals, nails...
7  INTJ  [tend, build, collection, things, desktop, use...
8  INFJ  [sure, good, question, distinction, two, depen...
9  INTP  [position, actually, let, go, person, due, var...
"""



# Import libraries for model selection and feature extraction
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.feature_selection import SelectKBest, chi2

mbtitype = np.array(file['type'])
mbtiposts = np.array(file['posts'])

X_train, X_test, y_train, y_test = train_test_split(
mbtiposts, mbtitype, test_size=0.33, random_state=42)

print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))

"""
5812
2863
5812
2863
"""

# Show unique labels
import nltk
unique, counts = np.unique(y_train, return_counts=True)
print(np.asarray((unique, counts)).T)
"""
# Get word frequencies
wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return wordlist
"""
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





"""
[['ENFJ' 128]
 ['ENFP' 463]
 ['ENTJ' 158]
 ['ENTP' 465]
 ['ESFJ' 32]
 ['ESFP' 36]
 ['ESTJ' 19]
 ['ESTP' 62]
 ['INFJ' 995]
 ['INFP' 1213]
 ['INTJ' 752]
 ['INTP' 853]
 ['ISFJ' 101]
 ['ISFP' 178]
 ['ISTJ' 128]
 ['ISTP' 229]]

Most frequent words with counts:
like;69675
think;49836
people;47854
one;37166
know;36934
really;35291
would;35015
get;30804
time;27610
feel;23336
much;23120
well;22928
love;21030
good;20719
things;20487
say;20267
way;19653
something;19538
want;19378
see;19134
also;18330
type;17149
even;16914
always;16809
lot;16440
"""



# Now to make bar graphs
# plt.plot(file['type'], type = 'bar')
from wordcloud import WordCloud
from os import path

#direc = path.dirname(__file__)
#text = open(file['posts']).read()
print(type(words))
print(type(mbtiposts))

wordcloud_words = " ".join(words)
# Lower max font size
wordcloud = WordCloud(max_font_size = 40).generate(wordcloud_words)
plt.figure()
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis("off")
plt.show()


# LINK IMAGE HERE 



# Extract features from text files
count_vect = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)
X_train_counts = count_vect.fit_transform(X_train)
print(X_train_counts.shape)


tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)



"""
(5812, 87237)
(5812, 87237)
"""



# Training a classifer
clf = MultinomialNB().fit(X_train_tfidf, y_train)
INTJ_sentence = ['Writing college essays is stressful because I have to give a stranger a piece of myself and that piece has to incorporate all of who I am']
INTJ_X_new_counts = count_vect.transform(INTJ_sentence)
INTJ_X_new_tfidf = tfidf_transformer.transform(INTJ_X_new_counts)
​
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


"""
['Writing college essays is stressful because I have to give a stranger a piece of myself and that piece has to incorporate all of who I am'] => INFP
['Our favorite friendships are the ones where you can go from talking about the latest episode of the Bachelorette to the meaning of life'] => INFP
"""



# Naive Bayes model fitting and predictions
# Building a Pipeline; this does all of the work in extract_and_train() at once  
text_clf = Pipeline([('vect', CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)),
                     ('tfidf', TfidfTransformer()),
                     ('chi2', SelectKBest(chi2, k = 'all')),
                     ('clf', MultinomialNB()),
                     ])
​
text_clf = text_clf.fit(X_train, y_train)
​
# Evaluate performance on test set
predicted = text_clf.predict(X_test)
print("The accuracy of a Naive Bayes algorithm is: ") 
print(np.mean(predicted == y_test))
print("Number of mislabeled points out of a total %d points for the Naive Bayes algorithm : %d"
      % (X_test.shape[0],(y_test != predicted).sum()))


"""
The accuracy of a Naive Bayes algorithm is: 
0.216556060077
Number of mislabeled points out of a total 2863 points for the Naive Bayes algorithm : 2243

"""


# Tune parameters
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                  'tfidf__use_idf': (True, False),
                  'clf__alpha': (1e-2, 1e-3),
                  }
​
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(X_train, y_train)
​
​
best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))
​
​
print(score)


"""
None yet
"""





# Linear Support Vector Machine
# Build Pipeline again
text_clf_two = Pipeline([('vect', CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)),
                         ('tfidf', TfidfTransformer()),
                         ('chi2', SelectKBest(chi2, k = 'all')),
                         ('clf', SGDClassifier(
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
​



"""
The accuracy of a Linear SVM is: 
0.668180230527
Number of mislabeled points out of a total 2863 points for the Linear SVM algorithm: 950
"""

# Tune parameters Linear Support Vector Machine
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                  'tfidf__use_idf': (True, False),
                  'clf__alpha': (1e-2, 1e-3),
                  }
​
gs_clf_two = GridSearchCV(text_clf_two, parameters, n_jobs=-1)
gs_clf_two = gs_clf_two.fit(X_train, y_train)
​
​
best_parameters, score, _ = max(gs_clf_two.grid_scores_, key=lambda x: x[1])
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))
​
​
print(score)


# NEURAL NETWORK
text_clf_three = Pipeline([('vect', CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)),
                            ('tfidf', TfidfTransformer()),
                            ('chi2', SelectKBest(chi2, k = 'all')),
                            ('clf', MLPClassifier(
                                    hidden_layer_sizes=(100,), 
                                    max_iter=50, 
                                    alpha=1e-4,
                                    solver='sgd', 
                                    verbose=10, 
                                    tol=1e-4, 
                                    random_state=1,
                                    learning_rate_init=.1)),
                            ])
​
text_clf_three.fit(X_train, y_train)
print("Training set score: %f" % text_clf_three.score(X_train, y_train))
print("Test set score: %f" % text_clf_three.score(X_test, y_test))

"""
Iteration 1, loss = 2.41922957
Iteration 2, loss = 2.26409348
Iteration 3, loss = 2.23200054
Iteration 4, loss = 2.18698508
Iteration 5, loss = 2.10344828
Iteration 6, loss = 1.98161336
Iteration 7, loss = 1.83373022
Iteration 8, loss = 1.68431646
Iteration 9, loss = 1.54573919
Iteration 10, loss = 1.41954076
Iteration 11, loss = 1.31342309
Iteration 12, loss = 1.21830070
Iteration 13, loss = 1.12815700
Iteration 14, loss = 1.05345863
Iteration 15, loss = 0.97787602
Iteration 16, loss = 0.90871745
Iteration 17, loss = 0.84880009
Iteration 18, loss = 0.78214876
Iteration 19, loss = 0.72302585
Iteration 20, loss = 0.66995744
Iteration 21, loss = 0.61389314
Iteration 22, loss = 0.56691956
Iteration 23, loss = 0.51917463
Iteration 24, loss = 0.48391800
Iteration 25, loss = 0.44175561
/opt/conda/lib/python3.6/site-packages/sklearn/neural_network/multilayer_perceptron.py:564: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (25) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
Training set score: 0.911562
Test set score: 0.621725
"""


# Parameter Tuning
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3),
              }
​
gs_clf_three = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf_three = gs_clf_three.fit(x_train, y_train)
​
​
best_parameters, score, _ = max(gs_clf_three.grid_scores_, key=lambda x: x[1])
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))
​
​
    print(score)


# Cross Validation Score
scores = cross_val_score(text_clf_three, X_train, y_train, cv = 5)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


"""
Iteration 1, loss = 2.43849082
Iteration 2, loss = 2.26501837
Iteration 3, loss = 2.23500018
Iteration 4, loss = 2.19875539
Iteration 5, loss = 2.14235469
Iteration 6, loss = 2.05900125
Iteration 7, loss = 1.95088979
Iteration 8, loss = 1.82999925
Iteration 9, loss = 1.70392465
Iteration 10, loss = 1.57981588
Iteration 11, loss = 1.46337494
Iteration 12, loss = 1.35717152
Iteration 13, loss = 1.25902679
Iteration 14, loss = 1.17291272
Iteration 15, loss = 1.09415957
Iteration 16, loss = 1.01959976
Iteration 17, loss = 0.94847289
Iteration 18, loss = 0.88546836
Iteration 19, loss = 0.82294339
Iteration 20, loss = 0.76329302
Iteration 21, loss = 0.70728825
Iteration 22, loss = 0.65237540
Iteration 23, loss = 0.60274807
Iteration 24, loss = 0.55683390
Iteration 25, loss = 0.51348644
/opt/conda/lib/python3.6/site-packages/sklearn/neural_network/multilayer_perceptron.py:564: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (25) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
Iteration 1, loss = 2.46354189
Iteration 2, loss = 2.27008724
Iteration 3, loss = 2.25222430
Iteration 4, loss = 2.22665928
Iteration 5, loss = 2.18904700
Iteration 6, loss = 2.13007811
Iteration 7, loss = 2.04455390
Iteration 8, loss = 1.93436386
Iteration 9, loss = 1.81056109
Iteration 10, loss = 1.68312665
Iteration 11, loss = 1.56308890
Iteration 12, loss = 1.44871713
Iteration 13, loss = 1.34591602
Iteration 14, loss = 1.25526571
Iteration 15, loss = 1.16521554
Iteration 16, loss = 1.08522540
Iteration 17, loss = 1.01449962
Iteration 18, loss = 0.94884817
Iteration 19, loss = 0.88260790
Iteration 20, loss = 0.82140326
Iteration 21, loss = 0.76258094
Iteration 22, loss = 0.70788178
Iteration 23, loss = 0.65661715
Iteration 24, loss = 0.60803656
Iteration 25, loss = 0.56124116
/opt/conda/lib/python3.6/site-packages/sklearn/neural_network/multilayer_perceptron.py:564: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (25) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
Iteration 1, loss = 2.46155059
Iteration 2, loss = 2.26621912
Iteration 3, loss = 2.23940457
Iteration 4, loss = 2.20196965
Iteration 5, loss = 2.14744658
Iteration 6, loss = 2.06725995
Iteration 7, loss = 1.95970085
Iteration 8, loss = 1.83557185
Iteration 9, loss = 1.70658471
Iteration 10, loss = 1.58212920
Iteration 11, loss = 1.46457504
Iteration 12, loss = 1.36294465
Iteration 13, loss = 1.26708350
Iteration 14, loss = 1.18385212
Iteration 15, loss = 1.10311480
Iteration 16, loss = 1.03241457
Iteration 17, loss = 0.96198425
Iteration 18, loss = 0.90023140
Iteration 19, loss = 0.83844484
Iteration 20, loss = 0.77731215
Iteration 21, loss = 0.71900004
Iteration 22, loss = 0.66593109
Iteration 23, loss = 0.61603300
Iteration 24, loss = 0.57079787
Iteration 25, loss = 0.52663781
/opt/conda/lib/python3.6/site-packages/sklearn/neural_network/multilayer_perceptron.py:564: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (25) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
Iteration 1, loss = 2.45601849
Iteration 2, loss = 2.27234404
Iteration 3, loss = 2.23977016
Iteration 4, loss = 2.20569408
Iteration 5, loss = 2.15234271
Iteration 6, loss = 2.07354839
Iteration 7, loss = 1.96703571
Iteration 8, loss = 1.84535607
Iteration 9, loss = 1.71508957
Iteration 10, loss = 1.58866515
Iteration 11, loss = 1.46719376
Iteration 12, loss = 1.35701565
Iteration 13, loss = 1.25944480
Iteration 14, loss = 1.17366358
Iteration 15, loss = 1.09095521
Iteration 16, loss = 1.01672356
Iteration 17, loss = 0.94845862
Iteration 18, loss = 0.88112686
Iteration 19, loss = 0.82041990
Iteration 20, loss = 0.76230751
Iteration 21, loss = 0.70678327
Iteration 22, loss = 0.65321705
Iteration 23, loss = 0.60541488
Iteration 24, loss = 0.55889194
Iteration 25, loss = 0.51545046
/opt/conda/lib/python3.6/site-packages/sklearn/neural_network/multilayer_perceptron.py:564: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (25) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
Iteration 1, loss = 2.45101860
Iteration 2, loss = 2.27142830
Iteration 3, loss = 2.24719260
Iteration 4, loss = 2.21567556
Iteration 5, loss = 2.16788106
Iteration 6, loss = 2.09294304
Iteration 7, loss = 1.99083503
Iteration 8, loss = 1.86731509
Iteration 9, loss = 1.73481456
Iteration 10, loss = 1.60832036
Iteration 11, loss = 1.48441926
Iteration 12, loss = 1.37818254
Iteration 13, loss = 1.27744912
Iteration 14, loss = 1.19001471
Iteration 15, loss = 1.10586666
Iteration 16, loss = 1.03612322
Iteration 17, loss = 0.96529008
Iteration 18, loss = 0.90056503
Iteration 19, loss = 0.83814943
Iteration 20, loss = 0.78186819
Iteration 21, loss = 0.72599661
Iteration 22, loss = 0.67264493
Iteration 23, loss = 0.62725204
Iteration 24, loss = 0.57967404
Iteration 25, loss = 0.53460886
/opt/conda/lib/python3.6/site-packages/sklearn/neural_network/multilayer_perceptron.py:564: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (25) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
[ 0.62703165  0.63496144  0.64347079  0.66637857  0.63809524]
Accuracy: 0.64 (+/- 0.03)
