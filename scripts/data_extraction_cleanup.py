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
import time
import sys, os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from multiprocessing import cpu_count, Pool
import helper_functions as hf

# Confirm we are in the correct directory, otherwise break script 
# and prompt user to move to correct directory
filepath = os.getcwd()
if not filepath.endswith('myersBriggsNLPAnalysis'):
	print('\nYou do not appear to be in the correct directory,\
	you must be in the \'myersBriggsNLPAnalysis\' directory\
	in order to run these scripts. Type \'pwd\' in the command line\
	if you are unsure of your location in the terminal.')
	sys.exit(1)

raw_data = 'data/mbti_raw.csv'
wide_data = 'data/mbti_wide.csv'
token_data = 'data/mbti_tokenized.csv'
clean_data = 'data/mbti_cleaned.csv'
columns = np.array(['type', 'posts'])
raw_df = pd.read_csv(raw_data, header = 0)
wide_df = pd.read_csv(wide_data, header = 0)

################
# Tokenize data 
################

'''
Explanation
-----------
Every user gets a line of data with all tweets put together, separated by three pipes (|||)
I will split each line by that string pattern and create a new line for each tweet
This is data in 'long' format; my justification is here: 
https://sejdemyr.github.io/r-tutorials/basics/wide-and-long/

And tokenize the line as well (keeping hashtags, urls, etc.)
I used the regex package, but the nltk.tokenize package has many different tokenizers available
https://www.nltk.org/_modules/nltk/tokenize/regexp.html
Check helper_function for more detail on tokenization method used
'''

start = time.time()
token_df = hf.parallelize(func = hf.tokenize_data, df = raw_df)	
end = time.time()
elapsed = end - start
print('Time elapsed: %.2f' % elapsed)

print(token_df.shape)
print(token_df.head(10))

# Write to csv
token_df.to_csv(token_data, columns = list(token_df.columns.values), index = False)
"""
##################
# Remove stopwords
##################

'''
Explanation
-----------
In every language there are words that are used commonly but don't have meaning by themselves or in other contexts
Articles, conjections and some adverbs are all candidates for this (i.e. 'to', 'and', 'on')
I will be analyzing the data with stopwords included, but would also like a copy of data without stopwords
For analysis as well as use for the machine learning models I will be implementing
'''
start = time.time()
clean_df = hf.remove_stopwords(raw_df)
end = time.time()
elapsed = end - start
print('Time elapsed: %.2f' % elapsed)

print(clean_df.shape)
print(clean_df.head(10))

# Write to csv
clean_df.to_csv(clean_data, columns = list(clean_df.columns.values), index = False)

##################################################
# Make different versions of our data for analysis
##################################################

'''
Explanation
-----------
Now we will have various versions of our data:
- Raw, unfiltered data
- Tokenized data with hashtags, mentions, retweets, etc.
- Cleaned tokenized data with stopwords removed

We will now subset the data into various parts to be used in the other scripts
'''

# Declare different processed and unprocessed objects for further analysis
raw_type = raw_df['type']
raw_posts = raw_df['posts']

# Split wide data (every user has a line, can have multiple tweets)
wide_type = wide_df['type']
wide_posts = wide_df['posts']

# Split token data (each tweet has an individual line)
token_type = token_df['type']
token_posts = token_df['posts']

# Split cleaned (no stopwords) data
cleaned_type = cleaned_df['type']
cleaned_posts = cleaned_df['posts']

# Split up data into training and testing datasets
# To evaluate effectiveness of model training
X_train_token, X_test_token, y_train_token, y_test_token = train_test_split(
	token_posts, token_type, test_size = 0.30, random_state = 42)

X_train_wide, X_test_wide, y_train_wide, y_test_wide = train_test_split(
	wide_posts, wide_type, test_size = 0.30, random_state = 42)
"""