from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import numpy as np
def get_features_2gram(data):
	"""
		Creates a vector of normalized (tf-idf) features from a 2-gram bag of words
	"""
	vectorizer = CountVectorizer(analyzer = 'word', stop_words = 'english', binary = True, ngram_range = (2,2), max_features = 10000) 
	features = vectorizer.fit_transform(data)
	print vectorizer.get_feature_names()
	return features
	"""
	steps = int(len(data)*0.001)
	for i in range(0, len(data), steps):
		print "From ", i, " to ", i+steps
		feat = features[i:i+steps]
		print feat.shape
		feat =	feat.toarray()
		pca = PCA()	
		components = pca.fit_transform(feat)	
		print "reduced: ", components.shape
	"""																																																																																																																																																																																																																				

	

def get_features_3gram(data):
	"""
		Creates a vector of normalized (tf-idf) features from a 3-gram bag of words
	"""
	feat = []
	vectorizer = CountVectorizer(analyzer = 'word', stop_words = 'english', binary = True, ngram_range = (3,3), max_features = 10000) 
	feat = vectorizer.fit_transform(data)
	return feat
	#print vectorizer.get_feature_names()
	#for v in feat:
	#	print v

def get_features_4char_gram(data):
	"""
		Creates a vector of normalized (tf-idf) features from a char-4-gram bag of words
	"""
	feat = []
	vectorizer = CountVectorizer(analyzer = 'char_wb', stop_words = 'english', binary = True, ngram_range = (4,4), max_features = 1000) 
	feat = vectorizer.fit_transform(data)
	return feat
	#print vectorizer.get_feature_names()
	#for v in feat:
	#	print v

def get_top_n_words(top, sentences):
	vectorizer = CountVectorizer(analyzer = 'word', stop_words = 'english', ngram_range = (1,1), max_features = top) 
	feat = vectorizer.fit_transform(sentences)
	
	vectorized_total = np.sum(feat, axis=0)
	word_indices = np.flip(np.argsort(vectorized_total)[0,:], 1)
	word_values = np.flip(np.sort(vectorized_total)[0,:],1)

	word_vectors = np.zeros((top, feat.shape[1]))
	for i in range(top):
		word_vectors[i,word_indices[0,i]] = 1

	words = [word[0].encode('ascii').decode('utf-8') for word in vectorizer.inverse_transform(word_vectors)]

	return words

