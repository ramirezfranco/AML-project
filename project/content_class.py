'''
Class to analyze tagged text, filtered by keywords and POS, and show the results.

Jesus I. Ramirez Franco
February, 2019
'''
import frequencies as fr
import topic_modeling as tm
#import search
#import box
import pandas as pd
from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, LabelSet
output_notebook()


class Analysis:
	'''
	Creates an object that read text, reduce it considering tags, key words 
	and parts of speech. Then count the frequencies of n-grams and creates a topic
	model. Fianlly, show important results.
	Inputs:
		- tag (str): One of the following calssifications: Location, 
		  Person, Organization, Money, Percent, Date, Time.
		- keywords (list of strings or string): a list with keywords or a phrase to find in text.
		- n_topics (int): Number of topics to be built in the model.
		- method (str): The model to be considered. It could be 'lda', 
		  'nmf' or 'lsi'.
	Instances: 
		- self.keywords (list): The list of key words given as input.
		- self.corpus (dict): the corpus resulted after considering 
		  the tag and keywords.
		- self.links_considered (list): the list of the keys in self.corpus.
		- self.filtered_corpus (dict): the new corpus after POS filter.
		- self.topics: Topics class object, defined in 'topic_modeling.py'.
		- self.topics_document (pandas DataFrame): topics-coument matrix.
		- self.topics_content (pandas DataFrame): words-topics matrix.
		- self.important_topics: None when the object is created, a list
		  of strings after calling the 'set_importan_topics' method.
	'''

	def __init__(self, Topics, method):
		self.topics = Topics
		if method == 'lda':
			self.model = self.topics.lda
		elif method == 'nmf':
			self.model = self.topics.nmf
		elif method == 'lsi':
			self.model == self.topics.lsi
		else:
			print('the model does not exist')
		self.topics_document = self.topics.z_matrix(method)
		self.topics_content = self.topics.topic_words(method)
		self.important_topics = None


	def set_importan_topics(self):
		'''
		Method to define the important topics.
		'''
		important_topics = input('Enter the indeces of the topics you consider important, separated by commas:')
		important_topics = [int(t) for t in important_topics.split(',')]
		self.important_topics = important_topics

	def imp_topics_content(self):
		'''
		Shows the words-topics matrix considering only the self.important_topics.
		Returns a pandas DataFrame
		'''
		if not self.important_topics:
			print('You have to define the important topics befor using this method, by calling the "self.set_importan_topics()"')
		return self.topics_content.iloc[self.important_topics]


	def imp_topics_doc(self):
		'''
		Shows the documents-topics matrix considering only the self.important_topics.
		Returns a pandas DataFrame
		'''
		if not self.important_topics:
			print('You have to define the important topics befor using this method, by calling the "self.set_importan_topics()"')
		return self.topics_document[self.important_topics]


	def top_doc_in_topic(self, topic, top):
		'''
		Shows the top documents that contain a defined topic.
		Inputs:
			- topic (int): the topic to be condidered.
			- top (int): the number of top documents to be shown.
		Returns a pandas DataFrame.
		'''
		df = self.topics_document[[topic]]
		return df.sort_values(by=topic, ascending=False)[:top]


	def topics_in_doc(self, starting_link):
		'''
		Shows the topics in a defined document, sorted in not ascending mode.
		Inputs:
			- starting_link (str): starting link to be considered.
		Returns a pandas DataFrame
		'''
		# if starting_link not in self.links_considered:
		# 	print('The starting link you chose is not included in "self.links_considered" list')
		df = self.topics_document.loc[starting_link]
		df = df.sort_values(ascending=False).reset_index()
		return df

	def twotopics_vis(self, topic_1, topic_2):
		'''
		Creates visualization of the distribution of documents with 
		respect to two defined topics.
		Inputs: 
			- topic_1, topic_2 (int): indices of the topics to be considered.
		Returns a 'bokeh' plot.
		'''
		df = self.topics_document[[topic_1, topic_2]].reset_index()
		df.columns = ['starting_link', 'x', 'y']
		source = ColumnDataSource(ColumnDataSource.from_df(df))
		labels = LabelSet(x="x", y="y", text="starting_link", y_offset=8,
                  text_font_size="8pt", text_color="#555555",
                  source=source, text_align='center')
		plot = figure(plot_width=600, plot_height=600)
		plot.circle("x", "y", size=12, source=source, line_color="black", fill_alpha=0.8)
		plot.add_layout(labels)
		show(plot, notebook_handle=True)

	def topic_content(self, text):
		vec = self.topics.vectorizer_transform(text)
		topics_text = self.model.transform(vec)
		return topics_text
