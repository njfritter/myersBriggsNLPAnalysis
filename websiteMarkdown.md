# Load Modules

First, we want to load the appropriate modules into our **Python** environment. 
For this we use the `import` method, followed by the module argument. 

Make sure to have these modules installed in your local environment first.  
For this, you can use `sudo pip3 install MODULE_NAME` in your console, replacing `MODULE_NAME` with the package you would like to install.

```python
# Make a .py file for analysis
# Or clone the project off Github and use for reference
# INSTALL any modules not already on local environment
sklearn
csv
re
nltk 
matplotlib
random
numpy
pandas

```
*
Any modules included in the main `myersBriggsNLPAnalysis.py` file that are not above should already be included with your local environment.
*

More than just surface level knowledge of the modules is encouraged; you can find the official documentation for each **Python** module by typing `python WRITE_MODULE_HERE`

# Get Data
## Collecting Data
The dataset can be downloaded off of the Github link at the bottom, or can be found here:

["Link to dataset on Kaggle"](https://www.kaggle.com/datasnaek/mbti-type)

## Scraping Data (Optional)

**
IF YOU ARE NOT PLANNING TO SCRAPE YOUR OWN DATA, GO TO NEXT SECTION
**

Using the **SQL** software described in [“Scraping Twitter”](https://github.com/inertia7/sentiment_ClintonTrump_2016#scraping-twitter) Section of my other NLP project in the link, one can look through the raw output of the **Twitter Scraper** (which has a lot of extraneous information) and select what specifically you would like to look at.

For the project above, we selected the following sections: 


```
rowid, 
tweet_id, 
created_at, 
query, 
content, 
possibly_sensitive
```

*
The last section was chosen arbitrarily to create a column for the labels, and was manually changed.
*

This way, if you'd like to scrape data from some popular personality type specific twitter accounts, this is the way!
 
## Loading Data

Here are what some tweets look like:

```
1,"Hello ENFJ7. Sorry to hear of your distress. It's only natural for a relationship to not be perfection all the time in every moment of existence. Try to figure the hard times as times of growth, as...",INFJ
2,"Yes you are geniuses, athletes, wildly attractive with genitals at least five times the size of the average of those of other types.  You can do anything you want with little to no effort involved.",INFP
3,"Sex can be boring if it's in the same position often. For example me and my girlfriend are currently in an environment where we have to creatively use cowgirl and missionary. There isn't enough..",ENTP
4,"Have you noticed how peculiar vegetation can be? All you have to do is look down at the grass: dozens of different plant species there.    And now imagine that hundreds of years later (when/if soil...",INTP
5,"Dear INTP,   I enjoyed our conversation the other day.  Esoteric gabbing about the nature of the universe and the idea that every rule and social code being arbitrary constructs created...",INTJ
6,"That's another silly misconception. That approaching is logically is going to be the key to unlocking whatever it is you think you are entitled to.   Nobody wants to be approached with BS...",ENTJ
7,"My ISFJ friend almost always instantly shuts down and becomes upset whenever someone disagrees with her POV or the way she does things. She can't seem to understand why a certain behaviour, though...",INFJ
8,"Well I personally don't go that much for attractiveness in general but I can see you have the will to change that and that's good already. May I ask if you want to be with them in a merely sexual...",ENFJ
9, "I think we do agree. I personally don't consider myself Alpha, Beta, or Foxtrot (lol at my own joke). People are people. We both agree that having emotions isn't the same as being weak, whiny,...",INFP
10,"Poker face for sure, accompanied by some sarcasm probably! But inside I'm running a pretty vivid list of pros and cons starting with asking myself 'do I like them back'? I probably know the answer...",INTJ
```

Here we load the data using `pandas.read_csv()` function:

```python
# How to read in files
# Change if necessary

unprocessed_data = 'mbti_1.csv'
processed_data = 'mbti_2.csv'
        
columns = ['type', 'posts']
file = pd.read_csv(unprocessed_data)
print(file.columns)
print(file.shape)
print(file.head(5))
print(file.tail(5))
```

And now we will go through the process of removing symbols and punctuation (regular expressions), as well as tokenizing the words:

```python
import csv
import re
from nltk import corpus
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
local_stopwords = []

# Tokenize words line by line
tokenizer = RegexpTokenizer(r'\w+')
i = 0
for index, line in file_unsep.iterrows():
    # Regular expressions
    line['posts'] = re.sub(r"(?:\@|https?\://)\S+", "", line['posts'])
    # Tokenize
    words = [word.lower() for word in tokenizer.tokenize(line['posts'])]
    words = [word for word in words if word not in stopwords.words('english') and word not in local_stopwords]
    line['posts'] = words
    if i % 100 == 0:
        print(i)
    i += 1

print(file_unsep.head(50))
```

This process takes the longest; there are ALOT of tweets that have to be processed. So this is probably the best time to catch up on your favorite dank memes/Shrek fantasies. Be an hero for us all...


# Do Exploratory Analysis

## Bar (Frequency) Charts

Performing unsupervised learning would normally be a critical step in the exploratory analysis phase. This phase will highlight the relationships between explanatory features and determine which features are the most significant. 

However, most forms of unsupervised learning come via continuous numerical features and since this is not the case (they are tokenized words and are therefore categorical) we cannot perform many of the typical analyses. I have graphed the tweets by personality type:

** Frequency of Words: All Personality Types**
<iframe width="100%" height=415  frameborder="0" scrolling="no" src="https://plot.ly/~raviolli77/69.embed?autosize=True&width=90%&height=100%"></iframe>




## Word Clouds

A word cloud is a collection of words that highlights the frequency of said words through size (bigger words show up more, smaller ones show up less).

Here is the word cloud generated by the whole dataset:



Here we use the `random` module in Python to generate random indices for every data point and then put it back together in order to effectively create a random train/test split.

Random sampling is desired (not just here, but in all experiments and statistical problems) because it minimizes bias by given all inputs an equal chance to be chosen and used in an experiment.

But the most important reason? 

*"The mathematical theorems which justify most frequentist statistical procedures apply only to random samples."* ([source](https://www.ma.utexas.edu/users/mks/statmistakes/RandomSampleImportance.html))
