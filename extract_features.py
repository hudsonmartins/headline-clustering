from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

def get_features_2gram(data):
	"""
		Creates a vector of normalized (tf-idf) features from a 2-gram bag of words
	"""
	feat = []
	vectorizer = CountVectorizer(analyzer = 'word', binary = True, ngram_range = (2,2), max_features = 1000) 
	feat = vectorizer.fit_transform(data).toarray()
	print vectorizer.get_feature_names()
	print feat[0]
	#for v in feat:
	#	print v

def get_features_3gram(data):
	"""
		Creates a vector of normalized (tf-idf) features from a 3-gram bag of words
	"""
	feat = []
	vectorizer = CountVectorizer(analyzer = 'word', binary = True, ngram_range = (3,3), max_features = 1000) 
	feat = vectorizer.fit_transform(data).toarray()
	print vectorizer.get_feature_names()
	print feat[0]
	#for v in feat:
	#	print v

def get_features_4char_gram(data):
	"""
		Creates a vector of normalized (tf-idf) features from a 3-gram bag of words
	"""
	feat = []
	vectorizer = CountVectorizer(analyzer = 'char_wb', binary = True, ngram_range = (4,4), max_features = 1000) 
	feat = vectorizer.fit_transform(data).toarray()
	print vectorizer.get_feature_names()
	print feat[0]
	#for v in feat:
	#	print v
