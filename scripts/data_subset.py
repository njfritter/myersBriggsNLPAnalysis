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
import sys, os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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

# First check if the data has been generated
# If not prompt user to make it
token_file_exists = os.path.isfile(token_data)
clean_file_exists = os.path.isfile(clean_data)

if not token_file_exists or not clean_file_exists:
    print('It looks like no processed data has been generated.\n',
        'Please run the \'data_generation.py\' file and follow the prompts.')
    sys.exit(1)

# Declare different processed and unprocessed objects for further analysis
raw_df = pd.read_csv(raw_data, header = 0)
raw_type = raw_df['type']
raw_posts = raw_df['posts']

wide_df = pd.read_csv(wide_data, header = 0)
wide_type = wide_df['type']
wide_posts = wide_df['posts']

token_df = pd.read_csv(token_data, header = 0)
token_type = token_df['type']
token_posts = token_df['posts']

clean_df = pd.read_csv(clean_data, header = 0)
clean_type = clean_df['type']
clean_posts = clean_df['posts']

# Split up data into training and testing datasets
# To evaluate effectiveness of model training
X_train_token, X_test_token, y_train_token, y_test_token = train_test_split(
    token_posts, token_type, test_size = 0.30, random_state = 42)

X_train_wide, X_test_wide, y_train_wide, y_test_wide = train_test_split(
    wide_posts, wide_type, test_size = 0.30, random_state = 42)

X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(
    clean_posts, clean_type, test_size = 0.30, random_state = 42)
