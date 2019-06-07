#import box
import pandas as pd
import topic_modeling as tm
import frequencies as fr
import nltk
from pycorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('http://localhost:9000/')

def read_survey_xlsx(file):
	'''
	Opens a xlsx file with text from surveys. 
	It assumes thet the first column is the ID.
	imputs:
		- file(str): name of the file that contains the information.
	returns a pandas data frame.
	'''
	df = pd.read_excel(file)
	print('This are the columns available in the file:')
	print(list(df.columns))
	return df

# def get_corpus(df):
# 	corpus = {column:{} for column in list(df.columns)[1:]}
# 	for column in corpus.keys():
# 		content = list(df[column].dropna(axis=0))
# 		for i in range(len(content)):
# 			t = content[i]
# 			corpus[column][i]= {'text':t, 'clean':fr.clean_doc(t)}
# 	return corpus 

# def corpus(file):
# 	df = read_survey_xlsx(file)
# 	corpus = get_corpus(df)
# 	return corpus


def concat_corpus(corpus):
	texts = [t['clean'] for t in corpus.values()]
	concat_str = ' '.join(texts)
	return concat_str

def integrated_corpus(file):
	'''
	Creates a dictionary with a corpus considering all the columns in the file
	out of the id column, that is considered to be in the first column.
	Imputs:
		- file(str): name of the file that contains the information.
	returns a dictionary.
	'''
	df = read_survey_xlsx(file)
	all_text = []
	for column in df.columns[1:]:
		all_text+=list(df[column].dropna(axis=0))
	size = range(len(all_text))
	corpus= {i:{'text':all_text[i], 'clean':fr.clean_doc(all_text[i])}for i in size}
	return corpus


def partial_corpus(file, columns):
	'''
	Creates a dictionary with a corpus considering the columns that were
	indicated in the "columns" input variable.
	Imputs:
		- file(str): name of the file that contains the information.
	returns a dictionary.
	'''
	df = read_survey_xlsx(file)
	all_text = []
	for column in columns:
		all_text+=list(df[column].dropna(axis=0))
	size = range(len(all_text))
	corpus= {i:{'text':all_text[i], 'clean':fr.clean_doc(all_text[i])}for i in size}
	return corpus


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


def freq_ngrams(corpus, n, threshold):
	'''
	Identifies the most important n-grams in a corpus dictionary.
	Imputs:
		- corpus (dict): A dictionary where every key is an id and every value
		  is a dictionary containing two keys: 'Text' and 'Clean'. The first
		  one contains the original text and the second one contains the cleaned
		  text.
		- n (int): number of words to condider for n-grams.
		-threshold (int): minimum frequency to be consider.
	Returns a list with the most important n-grams.
	'''
	corpus_text = concat_corpus(corpus)
	fr_df = tokens_freq(corpus_text, n)
	fr_df = fr_df[fr_df['Frequency']>= threshold]
	important_ngrams = [fr_df[str(n)+'-gram'].iloc[i] for i in range(len(fr_df ))]
	return important_ngrams


def ngram_texts(corpus, freq_ngrams_list):
	'''
	Identifies the sentences that contains the most frequent n-grams 
	in a corpus.
	Inputs:
		- corpus (dict): Corpus to be analyzed.
		- freq_ngrams_list (list): List of strings with the most 
		  important n-grams.
	Returns a dictionary.
	'''
	results = {bi:[] for bi in freq_ngrams_list}
	for bi in freq_ngrams_list:
		for v in corpus.values():
			if ''.join(bi.split(',')) in v['clean']:
				results[bi].append(v['text'])
	return results


def review_sent(text):
	'''
	Computes the average value of the sentiment of an opinion.
		- text (str): text to be analized.
	Returns an integer
	'''
	try:
		if type (nlp.annotate(text, properties={'annotators': 'sentiment', 'outputFormat': 'json'}))!=str:
			sentiment = nlp.annotate(text, properties={'annotators': 'sentiment', 'outputFormat': 'json'})['sentences']
			sentiment_values = [int(s['sentimentValue']) for s in sentiment] 
			value_avg = sum(sentiment_values)/len(sentiment_values)
		else:
			return int(-1)
	except UnicodeEncodeError:
		return int(-1)
	return int(value_avg)


def reviews_sentiment(ngram_revs):
	'''
	Computes the sentiment value for every review in a corpus with 
	n-gram calssifications and its texts list.
	Inputs:
		- ngram_revs (dict): Dictionary where every k is a n-gram 
		  class and every value is a list with the sentences from
		  the original corpus that contains that n-gram.
	'''
	results = {k:[review_sent(s) for s in ngram_revs[k] if review_sent(s) > 0] for k in ngram_revs.keys()}
	return results


def sentiment_ratios(rev_sent):
	'''
	Computes the sentiment ratios for a given corpus.
	Inputs:
		- rev_sent (dict): dictionary with sentiment values for every sentence in 
		  an n-gram class.
	Returns a dictionary. 
	'''
	ratios = {k:{
	'negative':round(len([j for j in rev_sent[k] if j < 2])/len(rev_sent[k]),2),
	'neutral': round(len([j for j in rev_sent[k] if j == 2])/len(rev_sent[k]),2),
	'positive':round(len([j for j in rev_sent[k] if j > 2])/len(rev_sent[k]),2)} 
	for k in rev_sent.keys()}
	return ratios 


def get_interest_revs(reviews, rev_sent):
	revs_classified = {k:{
	'negative':[reviews[k][j] for j in range(len(rev_sent[k])) if rev_sent[k][j] < 2],
	'neutral': [reviews[k][j] for j in range(len(rev_sent[k])) if rev_sent[k][j] == 2],
	'positive':[reviews[k][j] for j in range(len(rev_sent[k])) if rev_sent[k][j] > 2]
	} 
	for k in rev_sent.keys()}
	return revs_classified


class Ngram_Analysis:
	def __init__(self, file, columns_list, n=2, threshold = 5):
		'''
		Inputs:
			- file (str): name of the xlsx file to analyze.
			- columns (list of strings): List of columns to use in the analysis
			- n (int): number of words to consider in the n-grams.
			- threshold (int): number of minimum frequence to condider an 
			  n-gram as "important".
		'''
		self.n = n
		self.corpus = partial_corpus(file, columns_list)
		self.important_ngrams = freq_ngrams(self.corpus, self.n, threshold)
		self.important_reviews = ngram_texts(self.corpus, self.important_ngrams)
		self.reviews_sentiment = reviews_sentiment(self.important_reviews)
		self.reviews_classified = get_interest_revs(self.important_reviews, self.reviews_sentiment)
		self.sentiment_ratios = pd.DataFrame.from_dict(sentiment_ratios(self.reviews_sentiment), orient='index').sort_values(by='negative', ascending=False)


	def specific_reviews(self, n_gram_class, sentiment):
		revs =  self.reviews_classified[n_gram_class][sentiment]
		if len(revs) == 0:
			print('There are zero {} reviews in the n-gram class you selected'.format(sentiment))
		else:
			print('{} reviews in {} class:'.format(sentiment, n_gram_class))
			for r in revs:
				print()
				print(r)
				print('***************************************************************************')
			return revs 

	def n_grams_in_class(self, n_gram_class, sentiment='all'):
		if sentiment =='all':
			corpus = self.important_reviews[n_gram_class]
		else:
			corpus = self.reviews_classified[n_gram_class][sentiment]
		corpus = [fr.clean_doc(t) for t in corpus]
		corpus = ' '.join(corpus)
		return tokens_freq(corpus, self.n)
