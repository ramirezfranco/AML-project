'''
Class to compute topics from a corpus using different methods

Jesus I. Ramirez Franco
January, 2019
'''

import frequencies as fr
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd 


class Topics:
	'''
	Creates an object that run different topic models and stire and show the 
	relevant information of that models.
	Inputs: 
		- corpus(dict): A dictionary where every key is an starting link and
		  every value is a text related with the starting link.
		- num_topics (int): the number of topics to create.
	Instances:
		- self.num_topics: the numbers of tocpics built in the model.
		- self.corpus: the original corpus.
		- self.links_considered: the keys of the original corpus.
		- self.vectorizer: A TF-IDF vectorizer.
		- self.data_vectorized: The data transofmed using the vectorizer.
		- self.lda: The LDA model using the data vectorized.
		- self.nmf: The NMF model using the data vectorized.
		- self.lsi: The LSI model using the data vectorized.
	'''
	def __init__(self, corpus, num_topics):
		self.num_topics = num_topics
		self.corpus = corpus
		self.links_considered = list(corpus.keys())
		self.vectorizer = TfidfVectorizer(min_df=5, max_df=0.9)
		self.data_vectorized = self.vectorizer.fit_transform(list(corpus.values()))
		self.lda = LatentDirichletAllocation(n_topics=self.num_topics, max_iter=300, learning_method='online')
		self.nmf = NMF(n_components=self.num_topics)
		self.lsi = TruncatedSVD(n_components=self.num_topics, n_iter=300)


	def z_matrix(self, model):
		'''
		Creates the document-topic matrix.
		Inputs:
			- model(str): the model to be considere. It could be 'lda', 
			  'nmf' or 'lsi'
		Returns a data frame
		'''
		if model == 'lda':
			matrix = self.lda.fit_transform(self.data_vectorized)
		elif model == 'nmf':
			matrix = self.nmf.fit_transform(self.data_vectorized)
		elif model == 'lsi':
			matrix = self.lsi.fit_transform(self.data_vectorized)
		else:
			print('Please, chose a model among "lda", "nmf" and "lsi"')
		topics = list(range(len(self.links_considered)))
		col = ['starting_link'] + list(range(self.num_topics))
		results = []
		for i in range(len(self.links_considered)):
			results.append([self.links_considered[i]]+list(matrix[i]))
		df = pd.DataFrame(results, columns = col)#.round(2)
		df = df.set_index('starting_link')
		return df


	def topic_words(self, model):
		'''
		Creates the words-topics matrix.
		Inputs:
			- model(str): the model to be considered. It could be 'lda', 
			  'nmf' or 'lsi'
		Returns a data frame
		'''
		results = []
		if model == 'lda':
			compo = enumerate(self.lda.components_)
		elif model == 'nmf':
			compo = enumerate(self.nmf.components_)
		elif model == 'lsi':
			compo = enumerate(self.lsi.components_)
		else:
			print('Please, chose a model among "lda", "nmf" and "lsi"')
		for i, topic in compo:
			results.append([i, ', '.join([(self.vectorizer.get_feature_names()[i])for i in topic.argsort()[:-10 - 1:-1]])])
		df = pd.DataFrame(results, columns=['Topic', 'Content'])
		df = df.set_index('Topic')
		return df

	def vectorizer_transform(self, text):
		vec = self.vectorizer.transform(text)
		return vec

