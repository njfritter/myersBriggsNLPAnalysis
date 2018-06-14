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
from data_subset import raw_df, raw_type, raw_posts
from data_subset import token_df, token_type, token_posts
from data_subset import clean_df, clean_type, clean_posts
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import os, sys
import nltk
from nltk import bigrams
import wordcloud
from collections import Counter

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
	'Long Data': token_df,
	'Wide Data': clean_df 
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
token_freq = nltk.FreqDist(token_words)
print('Top 25 Tokenized Words without Stopwords:\n')
for word, frequency in token_freq.most_common(25):
	print('%s: %d' % (word, frequency))

token_freq.plot(25, title = 'Top 25 Word Frequencies', cumulative = False)

# Frequencies of words in cleaned data
clean_words = hf.gather_words(clean_posts)
clean_freq = nltk.FreqDist(clean_words)
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

print('''
	----------------------------------------------
	- HASHTAG, RETWEET, MENTION, URL FREQUENCIES -
	----------------------------------------------
	''')

# Find every instance of hashtags, retweets, mentions
# Using string matching patterns
hashtag_rows = hf.find_pattern(token_df, 'posts', r'#.*?(?=\s|$)')
print('Shape of remaining data with hashtags: ', hashtag_rows.shape)
print(hashtag_rows['posts'])

retweet_rows = hf.find_pattern(token_df, 'posts', 'rt')
print('Shape of remaining data with retweets: ', retweet_rows.shape)
print(retweet_rows['posts'])

mention_rows = hf.find_pattern(token_df, 'posts', '@')
print('Shape of remaining data with mentions: ', mention_rows.shape)
print(mention_rows['posts'])

url_rows = hf.find_pattern(token_df, 'posts', 'https')
print('Shape of remaining data with urls: ', url_rows.shape)
print(url_rows['posts'])