#!/usr/bin/env python3

# Myers Briggs Personality Type Tweets Natural Language Processing
# By Nathan Fritter
# Project can be found at: 
# https://www.inertia7.com/projects/109 & 
# https://www.inertia7.com/projects/110

################

# This is a script that is dedicated to the exploratory analysis of this project
# This includes words and type frequencies, word clouds, and the tokenization process
# But first import necessary packages (all we need are the helper functions)
import helper_functions as hf
from helper_functions import mbtitype, mbtiposts

hf.basic_output()

#hf.tokenize_data()

hf.unique_labels(mbtitype)
hf.word_freq(mbtiposts)