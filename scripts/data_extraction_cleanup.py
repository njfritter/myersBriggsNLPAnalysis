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
import nltk, re
from nltk.tokenize import RegexpTokenizer, TweetTokenizer
from nltk.corpus import stopwords
from multiprocessing import cpu_count, Pool

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
columns = np.array(['type', 'posts'])
raw_df = pd.read_csv(raw_data, header = 0)
wide_df = pd.read_csv(wide_data, header = 0)

# Every user gets a line of data with all tweets put together, separated by three pipes (|||)
# I will split each line by that string pattern and create a new line for each tweet
# with each line being cleaned and tokenized as well

# For this I will need to do this in parallel because theres tons of tweet
# Declare variables and function for parallel processing

def split_data(df):

	# Create empty dataframe for results
    split_df = pd.DataFrame(columns = columns)

    # Declare tokenizer and its parsing instructions
    # Check here for some other different string parsing patterns
    # https://www.nltk.org/_modules/nltk/tokenize/regexp.html
    # Specific string pattern to filter out hyperlinks & emojis
    tokenizer = TweetTokenizer()
    #words = re.sub(r"(?:\@|https?\://)\S+", "", posts)

    # Download stopwords here
    nltk.download('stopwords')

    # Iterate through rows of data
    for idx, row in df.iterrows():

    	# Split tweets into individual tweets
        tweets_split = row['posts'].split('|||')

        # Grab personality type since we're analyzing one row of data at a time
        ptype = pd.Series(row['type'], name = 'type')

        # Iterate through list of tweets for each user
        # Tokenize and remove stopwords as well
        for tweet in tweets_split:

            # Tokenize data (feel free to change tokenization method)
            # This will take a while, so we will measure the time taken
            tokenized_tweets = [word.lower() for word in tokenizer.tokenize(tweet)]

            # Now we will filter out stopwords, words that have little meaning alone
            tokenized_tweets = [word for word in tokenized_tweets if word not in stopwords.words('english')]
            
            # Turn tweets into a series object, then append
            tweet_object = pd.Series(tokenized_tweets, name = 'posts')
            line = pd.concat([ptype, tweet_object], axis = 1)
            split_df = pd.concat([split_df, line])

        print('Row %s of %s done' % (idx, raw_df.shape[0]))

    return split_df


start = time.time()
long_df = split_data(raw_df)
"""
partitions = cpu_count()

df_subsets = np.array_split(raw_df, partitions)
pool = Pool(partitions)
long_df = pd.concat(pool.map(split_data, df_subsets))
pool.close()
pool.join()
"""
end = time.time()
elapsed = end - start
print('Time elapsed: %.2f' % elapsed)

print(long_df.head(10))
print(long_df.shape)

long_file = 'data/mbti_long.csv'
long_df.to_csv(long_file, columns = list(long_df.columns.values), index = False)


# Declare different processed and unprocessed objects for further analysis
raw_type = raw_df['type']
raw_posts = raw_df['posts']

# Split wide data (every user has a line, can have multiple tweets)
wide_type = wide_df['type']
wide_posts = wide_df['posts']

# Split long data (each tweet has an individual line)
long_type = long_df['type']
long_posts = long_df['posts']

# Split up data into training and testing datasets
# To evaluate effectiveness of model training
X_train_long, X_test_long, y_train_long, y_test_long = train_test_split(
	long_posts, long_type, test_size = 0.30, random_state = 42)

X_train_wide, X_test_wide, y_train_wide, y_test_wide = train_test_split(
	wide_posts, wide_type, test_size = 0.30, random_state = 42)
