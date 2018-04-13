# Myers Briggs Personality Type Natural Language Processing Project

## Inspired by the Kaggle competition [here](https://www.kaggle.com/datasnaek/mbti-type)

### Introduction
My name is Nathan Fritter and I am obsessed with data. I have a B.S. in Applied Statistics from UCSB and was an active member of the Data Science club @ UCSB. I have completed numerous projects utilizing various methods, have presented my findings to others and sat on the executive board of the club. Feel free to take a look at any and all other projects and make suggestions. 

### The Project
This project is based on the Kaggle competition linked above. Tweets from various accounts on Twitter (all with an accompanying personality type) were gathered together; each account has anywhere from a couple to 20+ tweets each. 

My goal here is to remove the noise (stop words), categorize the data (tokenize), and use things such as word frequencies and sentence orientation to try and predict the personality type of the user. 

For this project, I utilized three different methods known for success in Natural Language Processing (NLP): 
+ Multinomial Naive Bayes Model (target variable has more than 2 classes)
+ Linear Support Vector Machine
+ Multi Layer Perceptron (simple Neural Network)

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

<img src="https://raw.githubusercontent.com/Njfritter/myersBriggsNLPAnalysis/master/images/wordfrequencylabeled.png" style = "width: 150px;"/>

#### WordCloud (based on frequencies above; this can also be found in the *images* folder)

<img src="https://raw.githubusercontent.com/Njfritter/myersBriggsNLPAnalysis/master/images/wordcloud.png" style = "width: 150px;"/>

#### Label (Personality Type) Frequencies

<img src="https://raw.githubusercontent.com/Njfritter/myersBriggsNLPAnalysis/master/images/typefrequencylabeled.png" style = "width: 150px;"/>


Clearly this may be an issue down the line; "INFP", "INFJ", "INTP", and "INTJ" shows up the most, and disproportionally so. Because of this, there will likely be something called "class imbalance": this is where some classes are represented much more highly than others.  

Also, complexity does not always make models better. The fact that there are sixteen different classes would impact any model's performance.

As a next step, I will alter the types to look at specific type combination differences, which may include:
+ E vs I, N vs S, T vs F, J vs P
+ NT vs NF vs SF vs ST
+ NJ vs NP vs SJ vs SP
+ Etc.

Whichever method I choose should reduce error and increase accuracy due to increased simplicity of the model.

#### Model Metrics

Using the original, four letter types (16 classes) here are the model results:

|  Model  |  Accuracy  |  Test Error Rate |  Cross Validation Score   |  Hyperparameter Optimization | Optimized Accuracy |
| ------  |  --------- |  --------------  |  ------------------------ |   ---------------    |  ----------|
| Multinomial Naive Bayes |  0.2169   |  0.7831   | Accuracy: 0.21 (+/- 0.00)  |  {'vect__ngram_range': (1, 1), 'tfidf__use_idf': False, 'clf__alpha': 0, 'clf__fit_prior': False} 	|  0.3210  |
| Linear Support Vector Machine  | 0.6615  |   0.3385  |  Accuracy: 0.67 (+/- 0.03)  |  {'clf__alpha': 0.001, 'clf__eta0': 0.25, 'clf__l1_ratio': 0, 'clf__learning_rate': 'optimal', 'clf__penalty': 'l2', 'tfidf__use_idf': True, 'vect__ngram_range': (1, 1)}  |   0.6716   |
| Multi Layer Perceptron  |   0.5777  |   0.3423  |  Accuracy: 0.66 (+/- 0.02)   |  Blank     | Blank  |

*The accuracy and test error rate are based on one train test split with model fitting and default parameters(simplest method).*
* The Optimized Accuracy is the accuracy of the model chosen with the best parameters after optimization, as well as the cross validation results.* 

As we can see, the accuracy of these methods will be fairly limited due to the large number of classes (and the shortcomings of using tweets as data, where slang, hyperlinks and more all interfere with data quality).

I will be showcasing a few of the type combinations mentioned earlier; may do more later.

#### Personality Type Prediction Results

Here are the success rates of each model predicting each personality type:

*This is used tuned model results*

|   Personality Type      |  Multinomial Naive Bayes  |  Linear Support Vector Machine  |   Multi Layer Perceptron  |
|  ---------  |   ---------   |  ---------  |  ----------   |
|  ENFJ  |  0.000000  |  0.129032  |  0.370968 |
|  ENFP  |  0.051887  |  0.561321  |  0.584906 |
|  ENTJ  |  0.000000  |  0.342466  |  0.561644 |
|  ENTP  |  0.027273  |  0.613636  |  0.586364 |
|  ESFJ  |  0.000000  |  0.100000  |  0.100000 |
|  ESFP  |  0.000000  |  0.000000  |  0.000000 |
|  ESTJ  |  0.000000  |  0.000000  |  0.000000 |
|  ESTP  |  0.000000  |  0.222222  |  0.296296 |
|  INFJ  |  0.475789  |  0.751579  |  0.698947 |
|  INFP  |  0.676898  |  0.886914  |  0.754443 |
|  INTJ  |  0.230088  |  0.657817  |  0.678466 |
|  INTP  |  0.396896  |  0.815965  |  0.778271 |
|  ISFJ  |  0.000000  |  0.415385  |  0.538462 |
|  ISFP  |  0.000000  |  0.225806  |  0.494624 |
|  ISTJ  |  0.000000  |  0.272727  |  0.480519 |
|  ISTP  |  0.000000  |  0.583333  |  0.703704 |

We can deduce a couple of things here:
+  The Naive Bayes model is NOT a good choice for this data, as it consistently has trouble predicting the underrepresented classes 
	+ The model only was able to give six different classes in its predictions (all of the "INxx" types plus "ENFP" and "ENTP")
	+ These six types happened to be the top 6 types in terms of number of tweets
	+ This is a case where the model's simplicity ends up hurting it; it is highly reliant on seeing many examples of a class to predict it accuractely
+ No matter what model, the four "INxx" personality types seem to be predicted pretty well. Within that, the "INFP" type is predicted very well
	+ This is consistent with the fact that those four have the most tweets, as well as "INFP" having the most overall
	+ The "ENxx" types have the second most, and they have reasonable performance (including actual predictions from Naive Bayes)
	+ Next are "ISxx" types, then "ESxx" types did the worst overall (also having the lowest amount of tweets)
+ In fact, there seems to be a noticable relationship between the number of tweets (data points) per type and the model's predictive success with them
	+ There are some exceptions ("ENTJ", "ISFJ", and "ISTP"), but for the most part this is to be expected: The more data you have of a certain class, the better a model will be able to correctly predict it on new data


### Steps to Reproduction

1. Clone the repo onto your machine (instructions [here](https://help.github.com/articles/cloning-a-repository/) if you are not familiar)

2. Download the package `virtualenv`
	+ Assuming you have made it this far you should have experience with one of `pip/pip3`, `brew` or another package manager
	+ Here we will use `pip3` (python3 compatible)

3. Run the following commands in your terminal (commands compatible with Linux and Mac):
	+ `virtualenv venv` 
		+ This creates a virtual environment called "venv"
		+ Feel free to change the second word to whatever you'd like to call it
	+ `source venv/bin/activate` 
		+ This turns on the virtual environment and puts you in it
	+ `pip3 install -r requirements.txt`
		+ Installs every python package listed in the requirements.txt file

4. I've made various scripts and places them in the `scripts` folder so one can look at each part individually 
	+ I.e. `naive_bayes.py` for the Naive Bayes model, `exploratory_analysis.py` for exploratory analysis, etc.
	+ Run `python3 scripts/*anyscript*.py and watch the magic happen!
	+ Also feel free to change my train test split size, model parameters and see how things change compared to the default parameters in `NLPAnalysis.py`
		+ These scripts all make use of the general `helper_functions.py` script, which gives the user more flexibility in choosing inputs
		+ Here, I have specifically chosen inputs that seem to improve model performance. Feel free to change these as well

5. For those that want everything in one I've made `NLPAnalysis.py`
	+ Run `python3 scripts/NLPAnalysis.py *text*`
		+ Examine the code at the bottom of each part
		+ Replace *text* with whichever part you'd like to run
		+ E.g. "tokenize" to tokenize data, "cloud" to create a word cloud, etc.
	+ Here there is less stuff to modify, as the point of this script is to get a feel for the models and how they perform with default/chosen parameters 
		+ The ONLY variables set initially are: 
			+ `random_state`: ensures the models train on the same subset of data 
				+ This does not apply to the Naive Bayes model (no such parameter)
			+ `verbose`: Declares the level of model training progress output
		+ If you'd like to tune model parameters among other changes, modify the individual scripts mentioned above
			+ The default parameters can be found in the docs [here](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.base)

i. **IMPORTANT:** To make sure these scripts run properly, run the code above from the main directory exactly how I have written them out (should read `.../..../.../myersBriggsNLPAnalysis` when called the `pwd` command) after cloning (NOT while in the `scripts` folder). I have defined the file path to the data based on being in the main directory.
	+ If you are a pro and would like to change the file path, go for it!

### Contributing

If you would like to contribute:
+ Create a new branch to work on (instructions [here](https://github.com/Kunena/Kunena-Forum/wiki/Create-a-new-branch-with-git-and-manage-branches))
	+ By default the branch you will receive after cloning is the "master" branch
	+ This is where the main changes are deployed, so it should not contain any of your code unless the changes have been agreed upon by all parties (which is the purpose of creating the new branch)
	+ This way you must submit a request to push changes to the main branch
+ Submit changes as a pull request with any relevant information commented
	+ Reaching out in some way would also be great to help speed up the process

If you do contribute, I will add your name as a contributor!

### Iterations

This project is a work in progress; first it started as an ipython notebook on Kaggle, then I translated the work to a script.

It initially started with no functions just to make sure that it worked. however, I have now separated out the parts into different functions, allowing for the user to input what part they'd like to run. 

I will implement "try/except" functions since this seems to be best practice, and I will also eventually split up the models into different scripts, with one script being a "helper functions" script.


### Sources Cited

+ http://sdsawtelle.github.io/blog/output/week4-andrew-ng-machine-learning-with-python.html
+ Will update