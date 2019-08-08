# AML-project
## Description of python files:

### city_rev.py: 
Performs the web scrapping of a city using parallel processing.

### content_class.py: 
Code to analyze the content of a topic model, generates tables and plots.
### frequencies.py: 
Code to perfom basictext mining and NLP tasks, among others.
### ngram_analysis.py: 
This code allows to get n-grams of documents and produce metrics of them, inlcuding sentiments analysis. 
In this case, I use the code  to get sentiment of segments using Core NLP.
### topic_modeling.py: 
This code contain a class to create different topics models.
### util.py:
In this file I stored most of the utility functions used to develop the project, from the web scrapping proces, 
segmentation, and sentiment analysis. It also includes the code to use Allen NLP models. 

## Description of Jupyter Notebooks:
### master_corpus:
In this notebook, I do the segmentation and classifications of comments by city.

### log_sent:
Computes the sentiment of segments using logistic regression.

### results_sentiment:
In thi notebook, I compute the sentimen analysis using Core NLP

### topic_modeling_states:
In this notebook, I create several topic models and analyze the results of the model selected.

## Data
The data from the trip advisor were stored as json files with the name of the city. 
Additionally, the re are some backup files that contain intermiate versions of the data set in diffrent stages of
the project in order to avoid to re run computationaly complex process many times.
The file segments.json in home directory contains the final version of segmentation.
