import numpy as np
import csv, glob, os, extract_features
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import LatentDirichletAllocation
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook
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
		for sentence in result_dict[i]:
			print sentence
			count += 1
			if(count > 20):
				print "..."
				break
	return kmeans





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
	plt.plot(x, cost, 'b*-')
	plt.grid(True)
	plt.xlabel('Number of clusters')
	plt.ylabel('Average within-cluster sum of squares')
	plt.title('Elbow for KMeans clustering')
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









data = get_headlines()
features = extract_features.get_features_2gram(data)
elbow_rule(data, features)

# kmeans = clusterization(data,features,7)
# create_TSNE(kmeans)

