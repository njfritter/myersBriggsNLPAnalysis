# Python Scripts 

## Brief Overview
Here are the different scripts I have included in this project. They each represent a part of the project and have varying levels of depth.

### Scripts w/ Abstract Functions

The scripts `data_extraction_cleanup.py` and  `helper_functions.py` have been made in order to simplify the other scripts and remove some of the more involved tasks. 

All other scripts import either one or both of the above, and make use of the variables and/or functions saved to these scripts. 

For example, `data_extraction_cleanup.py` will using `mbti_1.csv` (the raw data) and create a tokenized file (`mbti_tokenized.csv`) and a tokenized file with stopwords removed (`mbti_cleaned.csv`). It will then turn the CSVs into dataframes and subset the data into variables that I import into the other scripts.

The `helper_functions.py` script does many different things, from plotting frequencies to tokenizing data to creating a pipeline connecting many different parts together to create a fully functional machine learning model. 

It is getting big however, so I may divide it up based on what category the function falls under (data cleaning/manipulating, exploratory analysis or predictive modeling)

### The Other Scripts

<img src="https://raw.githubusercontent.com/Njfritter/myersBriggsNLPAnalysis/master/images/otherguys.jpeg">

#### Exploratory Analysis

The `exploratory_analysis.py` script goes in depth in the different nuggets of insight that are located within the tweets, along with basic data shape and viewing. 

Frequencies and various other insights are visualized, and a neat specialized visual called a **word cloud** is used to visualized term frequency. 

I will even go in depth on the different numbers within various personality types, so we may analyze how often retweets, hashtags and more are used within different personality types.

#### Predictive Modeling/Machine Learning

These three scripts are used to generate models and make predictions:
+ `naive_bayes.py`
+ `linear_SVM.py`
+ `neural_network.py`

The first one creates a **Multinomial Naive Bayes** model, which is based on Bayes theorem of conditional probablity. More in depth explanations can be found [here](https://www.analyticsvidhya.com/blog/2017/09/naive-bayes-explained/) and [here](https://monkeylearn.com/blog/practical-explanation-naive-bayes-classifier/).

The second one creates a **Linear Support Vector machine** model, which is able to utlize complex decision boundaries (non-linear) in order to make more effective decision planes with which the model groups observations of different classes. More on the topic can be found [here](http://www.statsoft.com/Textbook/Support-Vector-Machines) and [here](https://en.wikipedia.org/wiki/Support_vector_machine).

The third one created a **Neural Network** model, specifically a **Multi-Layer Perceptron**. This is the simplest type of Neural network, a classifier that utilizes a linear combination of inputs times various weights (which are adjusted during training) and produces a single output using a (usually non-linear) activation function. More on Multi-Layer Perceptrons [here](https://machinelearningmastery.com/neural-networks-crash-course/) and [here](https://en.wikipedia.org/wiki/Multilayer_perceptron).

#### Opportunities for Modification

The models themselves have been initialized with default parameters for the sake of simplicity. I have also included various parameters and values for model tuning, but it is not an exhaustive list. 

Each of the model, their parameters and more can be found in the docs [here](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.base).

I also intend to implement more deep learning on this data, as specific deep learning algorithms (specifically LSTMs) are more effective with Natural Language Processing tasks. Stay tuned.