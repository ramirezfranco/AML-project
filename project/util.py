import nltk
import time
import re
from nltk.corpus import stopwords
import pandas as pd
import os
import numpy as np
from pycorenlp import StanfordCoreNLP
import pandas as pd 
import urllib3
import urllib
import json
import bs4
import re
import nltk
from selenium import webdriver
from nltk.tokenize import RegexpTokenizer
import json
#from allennlp import pretrained

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

path = 'https://www.tripadvisor.com'

#model = pretrained.neural_coreference_resolution_lee_2017()

def make_soup(myurl):
    '''
    Creates a BeautifulSoup object
    inputs:
        myurl - string with the url
    output:
        BeautifulSoup object
    '''
    pm = urllib3.PoolManager()
    html = pm.urlopen(url = myurl, method = 'GET', redirect= False).data
    return bs4.BeautifulSoup(html, "lxml")


def put_city(number, tup, dictionary):
    dictionary[number] = {'name':tup[0], 'state':tup[1], 'url':tup[2]}


def attraction_url(attraction_part):
	return path+attraction_part

def number_of_reviews(attraction_soup):
	try:
		s = attraction_soup.find('span', class_="reviews_header_count").text
		s = re.search(r'\((.*?)\)', s).group(1)
		s = ''.join(s.split(','))
		return int(s)
	except AttributeError as e:
		return 0

def last_page(attraction_soup):
	try:
		last = attraction_soup.find('a', class_="pageNum last taLnk ").text
		return int(last)
	except AttributeError as e:
		return 0

def get_reviews(attraction_soup, ids_dict):
	page_reviews = attraction_soup.find_all('div', class_="reviewSelector")
	for review in page_reviews:
		#ids_dict[review.get('data-reviewid')] = review.find('a', class_="title ").get('href')
		try:
			href = path + review.find('a', class_="title ").get('href')
			show_review = make_soup(href)
			ids_dict[review.get('data-reviewid')] = json.loads(show_review.find('script', type="application/ld+json").text)
		except AttributeError:
			pass

class Attraction:
	def __init__(self, attraction_part):
		self.url = path + attraction_part
		self.g = re.search(r'-g(.*?)-', self.url).group(1)
		self.d = re.search(r'-d(.*?)-', self.url).group(1)
		self.name = re.search(r'Reviews-(.*?).html', self.url).group(1)
		self.soup = make_soup(self.url)
		self.num_reviews = number_of_reviews(self.soup)
		self.last_page = last_page(self.soup)
		self.base_morereviews_url = path+'/Attraction_Review-g'+self.g+'-d'+self.d+'-Reviews-or{}0-'+self.name+'.html'
		#self.base_review_url = path'/ShowUserReviews-g'+self.g'-d'+self.d'-r{}-'+self.name+'.html'
		self.url_list = [self.url]+[self.base_morereviews_url.format(i) for i in range(1,self.last_page)]
		self.reviews_json = {}

		# def morereviews_urls(self):
		# 	url_list =
		# 	return url_list

	def attraction_reviews(self):
		total = len(self.url_list)
		count = 1
		name = re.search(r'(.*)(?:-)', self.name).group(1)
		for page in self.url_list:
			if count <= 100:
				s = make_soup(page)
				get_reviews(s, self.reviews_json)
				print("{}:              ***REVIEW {}/{}***                  ".format(name, count, total))
				count+=1

def open_json(file):
	'''
	Opens a json file
	Inputs: 
		- file (str): path an dname to the file to open
	Returns a dictionary
	'''
	with open(file, encoding='utf-8', mode = 'r') as file:
		r = json.load(file)
	return r

def save_json(file, data):
	'''
	Opens a json file
	Inputs: 
		- file (str): path an dname to the file to save
	Returns a dictionary
	'''
	with open(file, mode = 'w') as file:
		json.dump(data, file)

def open_city_json(name):
	'''
	Opens a json file of an specifuc city.
	Inputs:
		- name (str): name of the city.
	Returns a dictionary
	'''
	file = 'data/{}.json'.format(name)
	city = open_json(file)
	return city

def get_city_reviews(name):
	'''
	Obtains the id and bosies of the reviews in a city
	Inputs:
		- name (str): name of the city.
	Returns a dictionary
	'''
	city_json = open_city_json(name)
	city_revs = {k:v['reviewBody'].encode().decode('unicode-escape') for val in city_json.values() for k,v in val.items()}
	return city_revs


def sentences_ranges(rev_sent_tokenize):
	ends = [len(nltk.word_tokenize(sent)) for sent in rev_sent_tokenize]
	ends_index = [0]
	e = 0
	for i in ends:
		e+=i
		ends_index.append(e)
	ranges = [[ends_index[j],ends_index[j+1]] for j in range(len(ends_index)-1)]
	return ranges

#un comment this an the import packaes to use
# def get_clusters(rev):
# 	try:
# 		results = model.predict(rev)
# 	except UnicodeEncodeError:
# 		return []
# 	clusters = results['clusters']
# 	return clusters


def segment_review(rev):
	clusters = get_clusters(rev)
	sentences = nltk.sent_tokenize(rev)
	ranges = sentences_ranges(sentences)
	results = []
	for c in clusters:
		p = ''
		agg = []
		for cc in c:
			for r in range(len(ranges)):
				if cc[0]>= ranges[r][0] and cc[1]<= ranges[r][1]:
					if r not in agg:
						p+=' '+sentences[r]
						agg.append(r)
		results.append(p)

	results = list(set(results))
	if len(results) == 0:
		return [sentences[0]]
	return results


def get_city_segments(segments_dict, city_group):
	res = {}
	for city in city_group:
		clean = []
		seg = []
		for k,v in segments_dict.items():
			if v['city']==city:
				clean+=(v['clean_seg'])
				seg+=(v['segments'])
		res[city]={'clean_seg':clean, 'segments':seg}
	return res


def topics_in_segments(content_object, segments_list):
	res= content_object.topic_content(segments_list)
	return res

def segments_to_process(content_object, segments_dict, city_group, topic, threshold):
	order_segments = get_city_segments(segments_dict, city_group)
	res={}
	for k, v in order_segments.items():
		topics = topics_in_segments(content_object, v['clean_seg'])
		to_process = [v['segments'][i] for i in range(len(topics)) if topics[i][topic]>threshold]
		res[k] = to_process
	return res 