# Myers Briggs Personality Type Natural Language Processing Project

## Inspired by the Kaggle competition [here](https://www.kaggle.com/datasnaek/mbti-type)

### Introduction
My name is Nathan Fritter and I am obsessed with data. I have a B.S. in Applied Statistics from UCSB and was an active member of the Data Science club @ UCSB. I have completed numerous projects utilizing various methods, have presented my findings to others and sat on the executive board of the club. Feel free to take a look at any and all other projects and make suggestions. 

### The Project
This project is based on the Kaggle competition linked above. Tweets from various accounts on Twitter (all with an accompanying personality type) were gathered together; each account has anywhere from a couple to 20+ tweets each. 

My goal here is to remove the noise (stop words), categorize the data (tokenize), and use things such as word frequencies and sentence orientation to try and predict the personality type of the user. 

I have attempted to make comments outlining my thought process. Feel free to clone the repo, make a branch, leave comments or contribute in any way you'd like.

### What is the Myers Briggs Personality Type?

The Myers Briggs Personality Type is based on psychological theory around how people perceive their world and then make accompanying judgements about these perceptions. 

It is a rather simplistic view, but one that does have surprising effectiveness at predicting people's behaviors in general when taking into account the limitations of the theory. There are four categories:

+ Extroverted/Introverted (E/I):
	+ Rather than the mainstream view that this distinction means talkative versus antisocial (and other similar variations), this difference actually stems from where one derives their energy from.
		+ Extroverts gain energy from being around other people: talking, conversing, being noticed, etc. They are capable of being alone, but will get tired out without contact with others.
		+ Introverts gain energy from being alone. Being alone and allowed to let their thoughts flow is very energizing, and allows one to clear their head (coming from personal experience). Opposite to extroverts, introverts have the capability to socialize and be the center of attention quite effectively; but after a while, even a five minute break alone may be necessary.

+ Intuitive/Sensory (N/S):
	+ Here, the differences lie in how the individual perceives their world. The two domains here are either through the five senses (immediate environment) or within their mind.
		+ Intuitives (N) are better at perceiving their world through their mind and imagining possibilities in the world. 
		+ Sensories (S) are better at perceiving their world through their five senses

+ Thinking/Feeling (T/F):
	+ This domain deals with how the individual judges the information they have perceived: Either the individual makes judgments in a Thinking (T) way or a Feeling (F) way
		+ Thinkers make conclusions about their world through logic and reasoning
		+ Feelers make conclusions about their world through emotion and gut feeling

+ Judgmental/Perceiving (J/P):
	+ Lastly (and a little more complicated), this domain basically states whether the perceiving trait or the judging trait is the dominant trait of the individual
		+ Judgers (J) will have their judging trait be their dominant overall trait
		+ Perceivers (P) will have their perceiving trait be their dominant overall trait

[Here](http://www.myersbriggs.org/my-mbti-personality-type/mbti-basics/) is more in depth detail on the theory and the complexities behind it. 

### General Analysis & Findings
In a sentiment analysis project, there are some limitations on the types of analysis one can do. What we CAN do, is things like word frequencies/clouds, label frequencies, remove stop words and use these remaining words to build a model predicting the target class (personality type).

#### Word Frequencies (Top 25; no change after stop words are removed) 

<img src="https://raw.githubusercontent.com/Njfritter/myersBriggsNLPAnalysis/master/images/wordfrequencylabeled.png" style = "width: 100px;"/>

#### WordCloud (based on frequencies above; this can also be found in the *images* folder)

<img src="https://raw.githubusercontent.com/Njfritter/myersBriggsNLPAnalysis/master/images/wordcloud.png" style = "width: 100px;"/>

#### Label (Personality Type) Frequencies

<img src="https://raw.githubusercontent.com/Njfritter/myersBriggsNLPAnalysis/master/images/typefrequencylabeled.png" style = "width: 100px;"/>


Clearly this may be an issue down the line; "INFP", "INFJ", "INTP", and "INTJ" shows up the most, and disproportionally so. Because of this, there will likely be something called "class imbalance": this is where some classes are represented much more highly than others.  

Also, complexity does not always make models better. The fact that there are sixteen different classes would impact any model's performance.

As a next step, I will alter the types to look at specific type combination differences, which may include:
+ E vs I, N vs S, T vs F, J vs P
+ NT vs NF vs SF vs ST
+ NJ vs NP vs SJ vs SP
+ Etc.

Whichever method I choose should reduce error and increase accuracy due to increased simplicity of the model.

#### Model results

For this project, I utilized three different methods known for success in Natural Language Processing (NLP): 
+ Multinomial Naive Bayes Model (specifically multinomial due to the number of classes)
+ Linear Support Vector Machine
+ Multi Layer Perceptron (simple Neural Network)

Using the original, four letter types (16 classes) here are the model results:

|  Model  |  Accuracy  |  Test Error Rate |  Cross Validation Score   |  Hyperparameter Optimization | Optimized Accuracy |
| ------  |  --------- |  --------------  |  ------------------------ |   ---------------    |  ----------|
| Multinomial Naive Bayes |  0.2169   |  0.7831   | Accuracy: 0.21 (+/- 0.00)  |  {'vect__ngram_range': (1, 1), 'tfidf__use_idf': False, 'clf__alpha': 0, 'clf__fit_prior': False} 	|  Not sure  |
| Linear Support Vector Machine  | 0.6717  |   0.3283  |  Accuracy: 0.67 (+/- 0.03)  |  {'clf__alpha': 0.001, 'clf__eta0': 0.25, 'clf__l1_ratio': 0, 'clf__learning_rate': 'optimal', 'clf__penalty': 'l2', 'tfidf__use_idf': True, 'vect__ngram_range': (1, 1)}  |   0.6569   |
| Multi Layer Perceptron  |   0.6577  |   0.3423  |  Accuracy: 0.66 (+/- 0.02)   |  Blank     | Blank  |

*The accuracy and test error rate are based on one train test split model fitting. 

As we can see, the accuracy of these methods will be fairly limited due to the large number of classes (and the shortcomings of using tweets as data, where slang, hyperlinks and more all interfere with data quality).

** Will be doing hyperparameter tuning shortly **

I will be showcasing a few of the type combinations mentioned earlier; may do more later.

### Steps to Reproduction

+ Clone the repo onto your machine (instructions [here](https://help.github.com/articles/cloning-a-repository/) if you are not familiar)
+ Download the package `virtualenv`
	+ Assuming you have made it this far you should have experience with one of `pip/pip3`, `brew` or another package manager
	+ Here we will use `pip3` (python3 compatible)
+ Run the following commands in your terminal (commands compatible with Linux and Mac):
	+ `virtualenv venv` 
		+ This creates a virtual environment called "venv"
		+ Feel free to change the second word to whatever you'd like to call it
	+ `source venv/bin/activate` 
		+ This turns on the virtual environment and puts you in it
	+ `pip3 install -r requirements.txt`
		+ Installs every python package listed in the requirements.txt file
+ I've made various scripts and places them in the `scripts` folder so one can look at each part individually (`naive_bayes.py` for the Naive Bayes model, `exploratory_analysis.py` which is self-explanatory, etc.)
	+ Run `python3 scripts/*anyscript*.py and watch the magic happen!
	+ Also feel free to change my train test split size, grid search parameters, etc.
+ However, for those that want everything in one I've made `NLPAnalysis.py`
	+ Run `python3 scripts/NLPAnalysis.py *text*`
		+ Examine the code at the bottom of each part
		+ Replace *text* with whichever part you'd like to run
		+ E.g. "tokenize" to tokenize data, "cloud" to create a word cloud, etc.
+ **IMPORTANT:** To make sure these scripts run properly, run the code above from the main directory after cloning (not in the `scripts` folder). I have defined the file path to the data 

### Contributing

If you would like to contribute:
+ Create a new branch to work on (instructions [here](https://github.com/Kunena/Kunena-Forum/wiki/Create-a-new-branch-with-git-and-manage-branches))
	+ By default the branch you will receive after cloning is the "master" branch
	+ This is where the main changes are deployed, so it should not contain any of your code unless the changes have been agreed upon by all parties (which is the purpose of creating the new branch
	+ This way you must submit a request to push changes to the main branch)
+ Submit changes as a pull request with any relevant information commented
	+ Reaching out in some way would also be great to help speed up the process

If you do contribute, I will add your name as a contributor!

### Iterations

This project is a work in progress; first it started as an ipython notebook on Kaggle, then I translated the work to a script.

It initially started with no functions just to make sure that it worked. however, I have now separated out the parts into different functions, allowing for the user to input what part they'd like to run. 

I will implement "try/except" functions since this seems to be best practice, and I will also eventually split up the models into different scripts, with one script being a "helper functions" script.


### Sources Cited

+ Will update