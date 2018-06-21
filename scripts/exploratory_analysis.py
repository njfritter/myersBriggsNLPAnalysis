#!/usr/bin/env python3

################################################################
# Myers Briggs Personality Type Tweets Natural Language Processing
# By Nathan Fritter
# Project can be found at: 
# https://www.inertia7.com/projects/109 & 
# https://www.inertia7.com/projects/110
################################################################

##################
# Import packages
##################
import helper_functions as hf
import numpy as np
import pandas as pd
import os, sys
from nltk import FreqDist, bigrams
from data_subset import raw_df, raw_type, raw_posts
from data_subset import token_df, token_type, token_posts
from data_subset import clean_df, clean_type, clean_posts
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

# Confirm we are in the correct directory, otherwise break script 
# and prompt user to move to correct directory
filepath = os.getcwd()
if not filepath.endswith('myersBriggsNLPAnalysis'):
    print('\nYou do not appear to be in the correct directory,\
    you must be in the \'myersBriggsNLPAnalysis\' directory\
    in order to run these scripts. Type \'pwd\' in the command line\
    if you are unsure of your location in the terminal.')
    sys.exit(1)

# First print out basic information about the different data we have
data_dfs = {
	'Raw Data': raw_df,
	'Tokenized Data': token_df,
	'Tokenized Data w/o Stopwords': clean_df 
}

print('''
	-----------------------------------------------
	- BASIC STRUCTURE AND DATATYPES OF DATAFRAMES -
	-----------------------------------------------
    ''')

for desc, df in data_dfs.items():
	print('Columns of %s:' % desc, df.columns)
	print('Shape of %s' % desc, df.shape)
	print('Data types of %s' % desc, df.dtypes)
	print('Head of %s' % desc, df.head(5))
	print('Tail of %s' % desc, df.tail(5))
"""
print('''
	------------------------------
	- WORD AND LABEL FREQUENCIES -
	------------------------------
	''')

# Frequency of Personality Types
type_count = raw_type.value_counts()
idx = type_count.index.tolist()
freq = type_count.tolist()

print('Counts of Personality Types:\n')
for idx, freq in zip(idx, freq):
    print(idx + ': ' + str(freq))

type_count.plot(kind = 'barh')
plt.title('Personality Type Frequencies')
plt.xlabel('Frequency')
plt.ylabel('Type')
plt.show()
plt.savefig('images/typefrequencylabeled.png')

# Frequencies of words in tokenized data
token_words = hf.gather_words(token_posts)
token_freq = FreqDist(token_words)
print('Top 25 Tokenized Words without Stopwords:\n')
for word, frequency in token_freq.most_common(25):
	print('%s: %d' % (word, frequency))

token_freq.plot(25, title = 'Top 25 Word Frequencies', cumulative = False)

# Frequencies of words in cleaned data
clean_words = hf.gather_words(clean_posts)
clean_freq = FreqDist(clean_words)
print('Top 25 Tokenized Words without Stopwords:\n')
for word, frequency in clean_freq.most_common(25):
	print('%s: %d' % (word, frequency))

clean_freq.plot(25, title = 'Top 25 Word Frequencies No Stopwords', cumulative = False)

print('''
	---------------------------------
	- DISPLAYING CLEANED WORD CLOUD -
	---------------------------------
	''')
hf.plot_wordcloud(clean_posts, save_image = True)

"""
print('''
	-------------------
	- BIGRAM ANALYSIS -
	-------------------
	''')
token_bigram = bigrams(hf.gather_words(token_posts))
token_bigram_counts = FreqDist(token_bigram)
print('\nTokenized Bigram Frequencies:')
for word, freq in token_bigram_counts.most_common(25):
	print('%s: %d' % (word, freq))

print('''
	----------------------------------------------
	- HASHTAG, RETWEET, MENTION, URL FREQUENCIES -
	----------------------------------------------
	''')

# Find every instance of hashtags, retweets, mentions
# Using string matching patterns
mention_tokens = hf.find_pattern(token_df, 'posts', '@')
mention_counts = FreqDist(mention_tokens)
print('\nMention Frequencies:')
for word, freq in mention_counts.most_common(25):
	print('%s: %d' %(word, freq))

#hashtag_tokens = hf.find_pattern(token_df, 'posts', r'#.*?(?=\s|$)')
hashtag_tokens = hf.find_pattern(token_df, 'posts', '#')
hashtag_counts = FreqDist(hashtag_tokens)
print('\nHashtag Frquencies:')
for word, freq in hashtag_counts.most_common(25):
	print('%s: %d' % (word, freq))

retweet_tokens = hf.find_pattern(token_df, 'posts', 'rt')
retweet_counts = FreqDist(retweet_tokens)
print('\nRetweet Frequencies:')
for word, freq in retweet_counts.most_common(50):
	print('%s: %s' % (word, freq))

url_tokens = hf.find_pattern(token_df, 'posts', 'https')
url_counts = FreqDist(url_tokens)
print('\nUrl Frequencies:')
for word, freq in url_counts.most_common(25):
	print('%s: %d' % (word, freq))