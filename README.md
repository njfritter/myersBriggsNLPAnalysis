# Myers Briggs Personality Type Natural Language Processing Project

## Inspired by the Kaggle competition [here](https://www.kaggle.com/datasnaek/mbti-type)

### Introduction
My name is Nathan Fritter and I am obsessed with data. I have a B.S. in Applied Statistics from UCSB and was an active member of the Data Science club @ UCSB. I have completed numerous projects utilizing various methods, have presented my findings to others and sat on the executive board of the club. Feel free to take a look at any and all other projects and make suggestions. 

### The Project
This project is based on the Kaggle competition linked above. Tweets from various accounts on Twitter (all with an accompanying personality type) were gathered together; each account has anywhere from a couple to 20+ tweets each. 

My goal here is to remove the noise (stop words), categorize the data (tokenize), and use things such as word frequencies and sentence orientation to try and predict the personality type of the user. 

I have attempted to make comments outlining my thought process. Feel free to clone the repo, make a branch, leave comments or contribute in any way you'd like.

### What is the Myers Briggs Personality Type?

The Myers Briggs Personality Type is 

### General Analysis
In a sentiment analysis project, there are some limitations on the types of analysis one can do. What we CAN do, is things like word frequencies/clouds, label frequencies, remove stop words and analyze

#### Word Frequencies (Top 25; no change after stop words are removed) 

|  Word  | Frequency |
| ------ | --------- |
|  Like  |   69675	 |
|  Think |   49836	 |
|  People |  47854  |
|  One |  37166  |
|  Know |  36934  |
|  Really |  35291  |
|  Would |  35015  |
|  Get |  30804  |
|  Time |  27610  |
|  Feel |  23336  |
|  Much |  23120  |
|  Well |  22928  |
|  Love |  21030  |
|  Good |  20719  |
|  Things |  20487  |
|  Say |  20267  |
|  Way |  19653  |
|  Something |  19538  |
|  Want |  19378  |
|  See |  19134  |
|  Also |  18330  |
|  Type |  17149  |
|  Even |  16914  |
|  Always |  16809  |
|  Lot |  16440  |


#### Label (Personality Type) Frequencies

|  Type  |  Frequency  |
|  ENFJ  | 128   |
|  ENFP  | 463   |
|  ENTJ  | 158   |
|  ENTP  | 465   |
|  ESFJ  | 32   |
|  ESFP  | 36   |
|  ESTJ  | 19   |
|  ESTP  | 62   |
|  INFJ  | 995   |
|  INFP  | 1213   |
|  INTJ  | 752   |
|  INTP  | 853   |
|  ISFJ  | 101   |
|  ISFP  | 178   |
|  ISTJ  | 128   |
|  ISTP  | 229   |


Clearly this may be an issue down the line; "INFP" 

### Steps to Reproduction

+ Clone the repo onto your machine (instructions [here](https://help.github.com/articles/cloning-a-repository/) if you are not familiar)
+ Download the package `virtualenv`
	+ Assuming you have made it this far you should have experience with one of `pip/pip3`, `brew` or another package manager
	+ Here we will use `pip3` (python3 compatible)
+ Run the following commands:
	+ `virtualenv venv` 
		+ This creates a virtual environment called "venv"
		+ Feel free to change the second word to whatever you'd like to call it
	+ `source venv/bin/activate` 
		+ This turns on the virtual environment and puts you in it
	+ `pip3 install -r requirements.txt`
		+ Installs every python package listed in the requirements.txt file
	+ `python3 NLPAnalysis.py *text*`
		+ Examine the code at the bottom of each part
		+ Replace *text* with whichever part you'd like to run
		+ E.g. "tokenize" to tokenize data, "cloud" to create a word cloud, etc.

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

I will attempt to implement "try/except" functions as well; will update.

Also, there are some parts that do not work (word cloud, word frequencies) because for whatever reason, the tokenization is happening at the letter level rather than the word level. Will investigate and update as well.

### Sources Cited

+ Will update