import numpy as np
import csv, glob, os, extract_features

def get_headlines():
	"""
		Collect the headlines from the csv file, returns as a list of strings
	"""
	data = []
	dataset = np.genfromtxt('dataset/news_headlines.csv', delimiter=',', dtype = None)
	row_count = 0

	for row in dataset:
		if row_count == 0:
			row_count +=1
			continue
		data.append(row[1])
		row_count +=1	
	return data
	

data = get_headlines()
extract_features.get_features_2gram(data)
