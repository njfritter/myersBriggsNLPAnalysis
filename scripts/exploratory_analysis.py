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
from data_extraction_cleanup import raw_df, raw_type, raw_posts
from data_extraction_cleanup import long_df, long_type, long_posts
from data_extraction_cleanup import clean_df, clean_type, clean_posts
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
	'Long Data': long_df, # All tweets on their own line
	'Wide Data': wide_df # All tweets by same user on one line
}

print('''
	-----------------------------------------------
	- BASIC STRUCTURE AND DATATYPES OF DATAFRAMES -
	-----------------------------------------------
    ''')

for desc, df in data_dfs:
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
type_freq = raw_type.value_counts()
print('Counts of Personality Types:', type_freq)

# Frequencies of Words (Using long data) 
# We will look at the top 25
word_features = nltk.FreqDist(long_posts)
words_top_25 = []
freq_top_25 = []
for word, frequency in word_features.most_common(25):
	print('%s: %d' % (word, frequency))
	words_top_25.append(word.title())
	freq_top_25.append(frequency)


# Now make use of helper functions to plot the frequencies
hf.plot_frequency(raw_type, type_freq, 'Types')
hf.plot_frequency(words_top_25, freq_top_25, 'Words')


print('''
	--------------
	- WORD CLOUD -
	--------------
	''')

# Gather list of words using helper function
individual_words = hf.gather_words(mbtiposts)
wordcloud_words = ' '.join(words)

# Lower max font size
cloud = wordcloud.WordCloud(max_font_size = 40).generate(wordcloud_words)
plt.figure()
plt.imshow(cloud, interpolation = 'bilinear')
plt.axis("off")
plt.savefig('images/wordcloud.png')
plt.show()

print('''
	----------------------------------------------
	- HASHTAG, RETWEET, MENTION, URL FREQUENCIES -
	----------------------------------------------
	''')

# Find every instance of hashtags, retweets, mentions
# Using string matching patterns

