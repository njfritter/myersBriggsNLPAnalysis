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
import helper_functions as hf

# Confirm the correct directory; break script and prompt user to move to correct directory otherwise
filepath = os.getcwd()
if not filepath.endswith('myersBriggsNLPAnalysis'):
	print('\nYou do not appear to be in the correct directory,\
	you must be in the \'myersBriggsNLPAnalysis\' directory\
	in order to run these scripts. Type \'pwd\' in the command line\
	if you are unsure of your location in the terminal.')
	sys.exit(1)

raw_data = 'data/mbti_1.csv'
wide_data = 'data/mbti_wide.csv'
token_data = 'data/mbti_tokenized.csv'
clean_data = 'data/mbti_cleaned.csv'
columns = np.array(['type', 'posts'])
raw_df = pd.read_csv(raw_data, header = 0)

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
# I used the regex package, but the nltk.tokenize package has many different tokenizers
https://www.nltk.org/_modules/nltk/tokenize/regexp.html
Check helper_function for more detail on tokenization method used
'''

# Check if tokenized file exists; if not, create it
# If it does, ask user if they want to make it again
file_exists = os.path.isfile(token_data)
if file_exists:
    prompt_user = input('\nThe tokenized data appears to already exist.\n' +
        'Would you like to generate it again? (Y/n) ')
if prompt_user == 'n':
    print('\nContinuing to clean data without stopwords.')
elif (prompt_user == 'Y') or (not file_exists):
    start = time.time()
    token_df = hf.parallelize(func = hf.tokenize_data, df = raw_df)
    end = time.time()
    elapsed = end - start
    print('Time elapsed: %.2f' % elapsed)
    print('Tokenized Data Shape:', token_df.shape)
    print('Head of Tokenized Data:', token_df.head(10))

    # Write to csv
    token_df.to_csv(token_data, columns = list(token_df.columns.values), index = False)

else:
    print('\nInvalid input, please run script again with valid input to create tokenized data.')
    sys.exit(1)

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
file_exists = os.path.isfile(clean_data)
if file_exists:
    prompt_user = input('\nThe cleaned data appears to already exist.\n' +
        'Would you like to generate it again? (Y/n) ')
if (prompt_user == 'Y') or (not file_exists):
    start = time.time()
    #clean_df = hf.parallelize(func = hf.tokenize_data(raw_df, filter_stopwords = True), df = raw_df)
    clean_df = hf.tokenize_data(raw_df, filter_stopwords = True)
    end = time.time()
    elapsed = end - start
    print('Time elapsed: %.2f' % elapsed)
    print('Cleaned Data Shape:', clean_df.shape)
    print('Head of Cleaned Data:', clean_df.head(10))

    # Write to csv
    clean_df.to_csv(clean_data, columns = list(clean_df.columns.values), index = False)
elif prompt_user == 'n':
    print('\nContinuing through rest of script.')
else:
    print('\nInvalid input, please run script again with a valid input\
    to create the cleaned data.')
    sys.exit(1)