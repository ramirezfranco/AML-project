import multiprocessing as mp 
import util
import json
import sys
import os

with open('data/cities.json', 'r') as f:
	cities = json.load(f)

if len(sys.argv) > 1:
	city_index = sys.argv[1]
	city = cities[city_index]

else:
	print('Provide an index between 1 and 100')

mono ='''
	______________________
	|                    |
          INITIALIZING
           {}
	|____________________| 
        (\__/) ||
        (• @•) ||
        / 　  v^
	'''.format(city['name'])

def put_urls(url_list, q):
	'''
	Insert the starting links in a queue to be used as input by processes
	p2, p3 and p4
	Inputs:
		- url_list (list): list that store the url strings to start the 
		  crawling.
		- q (multiprocessing queue): an object to store the starting links 
		  and pass them to the next processes.
	Returns: nothing, just creates the queue. 
	'''
	for url in url_list:
		q.put(url)


def city_reviews(q, d):
	while True:
		attraction_link = q.get()
		attraction = util.Attraction(attraction_link)
		print('Core number '+ str(os.getpid()) +  ' now processing: ' +attraction.name)
		print()
		attraction.attraction_reviews()
		d[attraction.name] = attraction.reviews_json
		print()
		print(' * '+attraction.name + ' Done ;)')
		if q.empty():
			q.close()
			print("Queue closed")
			break


if __name__ == '__main__':
	print(mono)
	print()
	manager = mp.Manager()
	d = manager.dict()
	q = mp.Queue()

	p1 = mp.Process(name='extracting_urls', target=put_urls, args=(city['links'], q))
	p2 = mp.Process(name='getting_revs_a', target=city_reviews, args=(q,d))
	p3 = mp.Process(name='getting_revs_b', target=city_reviews, args=(q,d))
	p4 = mp.Process(name='getting_revs_c', target=city_reviews, args=(q,d))

	p1.start()
	p2.start()
	p3.start()

	p1.join()
	p4.start()
	p2.join()
	p3.join()
	p4.join()


	print()
	print('Ya casi, ya nada mas guardo esta cosa y ya eh')
	print()
	with open('data/'+city['name']+'.json', 'w') as outfile:
		json.dump(d.copy(), outfile)
	print("         *\o/*         ")
	print('listo patron, algo mas?')