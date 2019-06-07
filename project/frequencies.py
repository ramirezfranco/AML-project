'''
Utility functions to do get frequencies of n-grams

Author: Jesus I. Ramirez Franco
December 2018
'''
import nltk
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from pycorenlp import StanfordCoreNLP
import math
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import string

nlp = StanfordCoreNLP('http://localhost:9000/')
pos_not_included = ['CC', 'CD', 'DT', 'FW', 'IN', 'LS', 'PP', 'PP$', 'WP', 'WP$', 'WRB', 'WDT', '#', '$', '“', '``', '(', ')', ',', ':']
pos_not_included_1 = ['NN', 'NNS','NP', 'NPS','CC', 'CD', 'DT', 'FW', 'IN', 'LS', 'PP', 'PP$', 'WP', 'WP$', 'WRB', 'WDT', '#', '$', '“', '``', '(', ')', ',', ':']
stemmer = SnowballStemmer("english")
#regex_tokenizer = RegexpTokenizer(r'\w+') # Tokenizer that removes punctuation

def clean_doc(text, language='english'):
	'''
	Removes unknown characters and punctuation, change capital to lower letters and remove
	stop words. If stem=False
	Inputs:
	sentence (string): a sting to be cleaned
	Returns: a string
	'''
	#tokens = regex_tokenizer.tokenize(text)
	tokens = nltk.word_tokenize(text)
	tokens = [t.lower() for t in tokens]
	tokens = [t for t in tokens if t not in stopwords.words(language)+[p for p in string.punctuation]]
	return ' '.join(tokens)


def csv_as_text(file_name):
	'''
	Opens a csv file with sentences and creates a string
	Inputs:
		- file_name (str): name of the file to open
	Returns a string
	'''
	try:
		df = pd.read_csv(file_name)
		texts_list = set(list(df['0']))
		return ' '.join(texts_list)
	except:
		pass


def gettin_all_text(list_of_files):
	'''
	Opens all csv files with sentences and returns a corpus
	Inputs:
		-list_of_files (list): a list with the names of the files to open
	Returns a string
	'''
	all_text = [csv_as_text(file) for file in list_of_files]
	all_text = [text for text in all_text if type(text) == str]
	all_str = ' '.join(all_text)
	return all_str


def all_text_list(list_of_files):
	'''
	Opens all csv files with sentences and returns a list of texts
	Inputs:
		-list_of_files (list): a list with the names of the files to open
	Returns a list
	'''
	all_text = [csv_as_text(file) for file in list_of_files]
	all_text = [text for text in all_text if type(text) == str]
	return all_text

def pos_filter(list_of_texts, filter_list=pos_not_included):
	'''
	Removes the words identified with the Part of Speech included 
	in the filter list, from every text in the list of texts.
	Inputs:
		- list_of_texts (list of strings): list with the texts to be analyzed
		- filter_list (list of strings): list with part of speech to eliminate
	Returns a list of cleaned texts
	'''
	filtered_texts = []
	for text in list_of_texts:
		pos = nlp.annotate(text, properties={'annotators': 'pos', 'outputFormat': 'json'})['sentences'][0]['tokens']
		filtered_words = [stemmer.stem(token['word']) for token in pos if token['pos'] not in filter_list]
		filtered_str = ' '.join(filtered_words)
		filtered_texts.append(filtered_str)
	return filtered_texts


def pos_filter_text(text, filter_list=pos_not_included):
	'''
	Removes the words identified with the Part of Speech included 
	in the filter list, from a given text.
	Inputs:
		- text (str): text to be analyzed
		- filter_list (list of strings): list with part of speech to eliminate
	Returns a cleaned text
	'''
	text_list = make_chunks(text)
	temp = []
	for t in text_list:
		pos = nlp.annotate(t, properties={'annotators': 'pos', 'outputFormat': 'json'})['sentences'][0]['tokens']
		filtered_words = [stemmer.stem(token['word']) for token in pos if token['pos'] not in filter_list]
		filtered_str = ' '.join(filtered_words)
		temp.append(filtered_str)
	final_text = ' '.join(temp)
	return final_text


def pos_filter_corpus(corpus):
	'''
	Removes the words identified with the Part of Speech included 
	in the filter list, from every text in the corpus.
	Inputs:
		- corpus (dict): Dictionary where every key is an starting link and 
		  and every valu is a text associated with the starting link.
	Returns a dictionary with the cleaned texts.
	'''
	results = {}
	for k, v in corpus.items():
		results[k] = pos_filter_text(v)
	return results

def make_chunks(text, max_size=95000):
	'''
	Creates chunks of text with lenght less than or equal to the 
	defined maximum size, from an original text.
	Inputs:
		- text (str):
		- max_size (int):
	Returns a list of chunks
	'''
	tokens = nltk.word_tokenize(text)
	chunks = []
	chunk = []
	count = 0
	for word in tokens:
		if count < max_size-len(word):
			chunk.append(word)
			count += len(word)+1
		else:
			chunks.append(' '.join(chunk))
			count = len(word)
			chunk = []
			chunk.append(word)
	chunks.append(' '.join(chunk))
	return chunks


def tokens_freq(corpus, size):
	'''
	Computes the frequency of n-grams according to size and
	retuns an ordered data frame.
	Inputs:
		corpus (string): text to be analized
		size (int): size of n-grams
	Returns: a data frame
	'''
	
	tokens = nltk.word_tokenize(corpus)
	frequencies = {}
	complete = tokens + tokens[:size - 1]

	n_grams = []
	for i in range(len(tokens)):

		l = i
		h = i + size-1
		n_grams.append(', '.join(complete[l:h+1]))

	for ng in n_grams:
		if ng not in frequencies.keys():
			frequencies[ng] = 1
		else:
			frequencies[ng] += 1

	freq_list = [(k, v) for k, v in frequencies.items()]
	df = pd.DataFrame(freq_list, columns=[str(size)+'-gram', 'Frequency'])
	return df.sort_values(by='Frequency', ascending=False)[:20]
