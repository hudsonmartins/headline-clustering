import numpy as np
import csv, glob, os, extract_features, nltk

def create_sentences_csv():
	fn = 'dataset/sentences.csv'		
	remove = ['TO', 'IN', 'PRP', 'PRP$'] #Tags to remove undesired words
	data = []
	dataset = np.genfromtxt('dataset/news_headlines.csv', delimiter=',', dtype = None)
	row_count = 0
	for row in dataset:
		if row_count == 0:
			with open(fn, 'a') as csvfile:
				writer = csv.writer(csvfile, delimiter = ',', quoting=csv.QUOTE_NONE)
				writer.writerow(row)
			row_count +=1
			continue
		tokens = nltk.word_tokenize(row[1])
		tagged = nltk.pos_tag(tokens) #Getting tags
		sentence = ''
		
		#Removing tags from the sentences
		count_tag = 0
		for tags in tagged:
			if tags[1] not in remove:	
				if count_tag != len(tagged) - 1:
					sentence += tags[0]+' '
				else:
					sentence += tags[0]
			count_tag += 1 			
		row_count +=1
				
		with open(fn, 'a') as csvfile:
			writer = csv.writer(csvfile, delimiter = ',', quoting=csv.QUOTE_NONE)
			writer.writerow([row[0], sentence])

def split_years_csv():
	dataset = np.genfromtxt('dataset/sentences.csv', delimiter=',', dtype = None)

	data = []
	row_count = 0
	year = 2003
	for row in dataset:
		fn = 'dataset/sentences_'+str(year)+'.csv'
		if row_count == 0:
			with open(fn, 'a') as csvfile:
				writer = csv.writer(csvfile, delimiter = ',', quoting=csv.QUOTE_NONE)
				writer.writerow(row)
			row_count +=1
			continue
		
		if str(year) == row[0][0:4]:			
			with open(fn, 'a') as csvfile:
				writer = csv.writer(csvfile, delimiter = ',', quoting=csv.QUOTE_NONE)
				writer.writerow(row)
		else:
			year += 1

def get_headlines():

	"""
		Collect the headlines from the csv file, returns as a list of strings
	"""
	data = []
	dataset = np.genfromtxt('dataset/sentences.csv', delimiter=',', dtype = None)
	row_count = 0
	
	for row in dataset:
		if row_count == 0:
			row_count +=1
			continue 			
		data.append(row[1])
		row_count +=1	
	return data
	

data = get_headlines()
#create_sentences_csv()
#split_years_csv()
extract_features.get_features_2gram(data)
