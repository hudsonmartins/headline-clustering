import numpy as np
import csv, glob, os, extract_features, random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize

def get_headlines(year=None, subset=None):

	"""
		Collect the headlines from the csv file, returns as a list of strings
	"""
	data = []
	headlines = []
	dataset = np.genfromtxt('dataset/news_headlines.csv', delimiter=',', dtype = None)
	row_count = 0
	if subset != None:
		num_dados = int(len(dataset)*subset)
		print "Getting a random subset with ", num_dados, " data"
		for i in range(num_dados):
			index = random.randrange(1, len(dataset))
			sentence = dataset[index]
			headlines.append(sentence)
	else:
		headlines = dataset
		
	if year == None:
		print "Getting headlines from all years"
			
		for row in headlines:
			if row_count == 0:
				row_count +=1
				continue
			sentence = get_root_sentence(row[1]) 			
			data.append(sentence)
			row_count +=1	
	else:
		print "Getting headlines from ", year
			
		for row in headlines:
			if row_count == 0:
				row_count +=1
				continue

			if str(year) == row[0][0:4]:
				sentence = get_root_sentence(row[1])
				data.append(sentence)
				row_count +=1	
	return data

def get_root_sentence(sentence):
	ps = SnowballStemmer("english")
	words = word_tokenize(sentence)				 			
	root_sentence = ''	

	for word in words:
		if len(word) < 3:
			continue
		root_sentence += ps.stem(word) + ' '
		
	return root_sentence

def clusterization(data, features, k):
	#Clusterization
	kmeans = KMeans(n_clusters = k, verbose = 1)
	kmeans.fit(features)

	print "Cost: ", kmeans.inertia_/features.shape[0]
	#print "Centers: ", kmeans.cluster_centers_

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
	print "Training t-sne..."
	tsne = TSNE(n_components=2).fit_transform(matrix)
	print tsne.shape
	vis_x = tsne[:,0]
	vis_y = tsne[:,1]
	plt.scatter(vis_x, vis_y)
	plt.show()
	
def elbow_rule(data, features, max_k=10):
	k = 2
	cost = []

	while k < max_k:
		print "Clustering with k = ", k
		kmeans = clusterization(data, features, k)
		cost.append(kmeans.inertia_/features.shape[0])
		print "Final cost = ", kmeans.inertia_/features.shape[0]
		k+=1
	#x = np.arange(2, max_k, 1)
	#plt.plot(x, cost)
	
	#plt.show()	
	return cost		
	
	


data = get_headlines()
print "Number of headlines: ", len(data)
features = extract_features.get_features_4char_gram(data)
clusterization(data, features,k=8)
"""
for i in range(5):
	cost = elbow_rule(data, features, max_k=20)
	x = np.arange(2, 20, 1.0)
	plt.plot(x, cost)
plt.ylabel("Cost")
plt.xlabel("Number of Clusters")
plt.show()
"""
