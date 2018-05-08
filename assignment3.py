import numpy as np
import csv, glob, os, extract_features
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE




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

def get_headlines(year=None):

	"""
		Collect the headlines from the csv file, returns as a list of strings
	"""
	data = []
	dataset = np.genfromtxt('dataset/news_headlines.csv', delimiter=',', dtype = None)
	row_count = 0
	if year == None:
		for row in dataset:
			if row_count == 0:
				row_count +=1
				continue 			
			data.append(row[1])
			row_count +=1	
	else:
		for row in dataset:
			if row_count == 0:
				row_count +=1
				continue
			if str(year) == row[0][0:4]:			 			
				data.append(row[1])
				row_count +=1	
	return data


def clusterization(data, features, k):
	#Clusterization
	kmeans = KMeans(n_clusters = k, verbose = 1)
	kmeans.fit(features)

	print "Cost: ", kmeans.inertia_
	print "Centers: ", kmeans.cluster_centers_

	pred = kmeans.predict(features)
	result_dict = dict()

	i = 0
	for result in pred:
		if(result in result_dict):  #save in the dictionary
			result_dict[result].append(data[i])
		else:
			result_dict[result] = [data[i]]
		i += 1

	
	
	for i in range(len(result_dict)):
		print "Group ", i, " lenght ", len(result_dict[i])
		count = 0
		top_words = extract_features.get_top_n_words(20, result_dict[i])
		print('Cluster '+str(i) + ' : ' + ', '.join(word for word in top_words))

		"""
		for sentence in result_dict[i]:
			print sentence	
			count += 1
			if(count > 20):
				print "..."
				break
		"""
	#tsne_visulatization(kmeans.transform(features), k)
	return kmeans

def tsne_visulatization(matrix, n_topics):
	tsne = TSNE(n_components=2).fit_transform(matrix)
	print tsne.shape

def elbow_rule(data, features):
	k = 2
	cost = []
	while k < 20:
		print "Clustering with k = ", k
		kmeans = clusterization(data, features, k)
		cost.append(kmeans.inertia_)
		print "Final cost = ", kmeans.inertia_
		k+=1
		
	x = np.arange(2, 20, 1)
	plt.plot(x, cost)
	plt.show()
	


data = get_headlines(year=2003)
print "Number of headlines: ", len(data)
features = extract_features.get_features_4char_gram(data)
clusterization(data, features,k=5)
#elbow_rule(data, features)

