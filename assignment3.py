import numpy as np
import csv, glob, os, extract_features
import matplotlib.pyplot as plt
#<<<<<<< HEAD
from sklearn.manifold import TSNE
from sklearn.decomposition import LatentDirichletAllocation
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
output_notebook()

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
#>>>>>>> e79f1e776d82fcb716fbd6a9473fce5ad5ffead5

	"""
		Collect the headlines from the csv file, returns as a list of strings
	"""
	data = []
	dataset = np.genfromtxt('dataset/news_headlines.csv', delimiter=',', dtype = None)
	row_count = 0

#<<<<<<< HEAD
	for row in dataset:
		if row_count == 0:
			row_count +=1
			continue
		data.append(row[1])
		row_count +=1
#=======
	if year == None:
		print "Getting all headlines"
		for row in dataset:
			if row_count == 0:
				row_count +=1
				continue
			sentence = get_root_sentence(row[1]) 			
			data.append(sentence)
			row_count +=1	
	else:
		print "Getting headlines from ", year
		for row in dataset:
			if row_count == 0:
				row_count +=1
				continue

			if str(year) == row[0][0:4]:
				sentence = get_root_sentence(row[1])
				data.append(sentence)
				row_count +=1	
#>>>>>>> e79f1e776d82fcb716fbd6a9473fce5ad5ffead5
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

#<<<<<<< HEAD




def tsne_visulatization(matrix, n_topics):
	tsne = TSNE(n_components=2).fit_transform(matrix)
	print tsne.shape

def elbow_rule(data, features, max_k=10):
#>>>>>>> e79f1e776d82fcb716fbd6a9473fce5ad5ffead5
	k = 2
	cost = []
	while k < max_k:
		print "Clustering with k = ", k
		kmeans = clusterization(data, features, k)
		cost.append(kmeans.inertia_/features.shape[0])
		print "Final cost = ", kmeans.inertia_/features.shape[0]
		k+=1
#<<<<<<< HEAD

	x = np.arange(2, 20, 1)
	plt.plot(x, cost, 'b*-')
	plt.grid(True)
	plt.xlabel('Number of clusters')
	plt.ylabel('Average within-cluster sum of squares')
	plt.title('Elbow for KMeans clustering')
#=======
		
	x = np.arange(2, max_k, 1)
	plt.plot(x, cost)
#>>>>>>> e79f1e776d82fcb716fbd6a9473fce5ad5ffead5
	plt.show()




def create_TSNE(features,k = 7):

	features = float(features)
	colormap = np.array([
		"#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
		"#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
		"#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
		"#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"])
	colormap = colormap[:k]

	tsne_model = TSNE(n_components=2, perplexity=50, learning_rate=100,
					  n_iter=2000, verbose=1, random_state=0, angle=0.75)
	tsne_vectors = tsne_model.fit_transform(features)

	plot = figure(title="t-SNE Clustering of {} LSA Topics".format(k), plot_width=700, plot_height=700)
	plot.scatter(x=features[:, 0], y=features[:, 1], color=colormap[k])









#<<<<<<< HEAD
# data = get_headlines()
# features = extract_features.get_features_2gram(data)
# elbow_rule(data, features)
#=======
#split_years_csv()
data = get_headlines(year=2017)
print data
#print "Number of headlines: ", len(data)
features = extract_features.get_features_2gram(data)

#features = extract_features.get_features_4char_gram(data)
#clusterization(data, features,k=8)
#elbow_rule(data, features, max_k=20)
#>>>>>>> e79f1e776d82fcb716fbd6a9473fce5ad5ffead5

# kmeans = clusterization(data,features,7)
# create_TSNE(kmeans)

