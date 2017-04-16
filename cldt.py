from util import tokenize
from nltk import word_tokenize,FreqDist
import math
import sys
import dill

def give_loc_tf(text):
	ls_text = [x.lower() for x in tokenize(text)]
	freq = FreqDist(ls_text)
	ls = freq.most_common(len(freq))
	loc_tf = {}
	for x in ls:
		loc_tf[x[0]] = x[1]
	return loc_tf

def computeIDF(sentences):
	idf={}
	noOfDocs=len(sentences)
	j=0
	for sentence in sentences:
		words=set(tokenize(sentence))
		for word in words:
			if word not in idf.keys():
				idf[word]=0
				vocab.update(word)
			idf[word]+=1
		# if j%5000==0:
		# 	print "Done for sentences ",j
		j+=1

	for word in idf.keys():
		idf[word]=noOfDocs/idf[word]

	return idf

def normalize(vector):
	length = 0
	vocab = list(vector.keys())
	count = len(vocab)
	mod_vector = {}

	for word in vocab:
		length += vector[word]*vector[word]

	for word in vocab:
		mod_vector[word] = math.sqrt(((vector[word]*vector[word])/length))

	return mod_vector


def getTFIDF(sentence,lang):
	N=100#1806951
	
	if lang==0:
		with open('./db/idfF.pickle') as inp:
			idf=dill.load(inp)
	else:
		with open('./db/idfE.pickle') as inp:
			idf=dill.load(inp)
	
	loc_tf=give_loc_tf(sentence)
	
	TfIdf={}
	for word in loc_tf.keys():
		if word not in idf.keys():
			idf[word] = 1
		TfIdf[word]=math.log(1+loc_tf[word],10)*idf[word]

	return normalize(TfIdf)


def cosine_similarity(sentence1, sentence2,lang):
	vector1=getTFIDF(sentence1,lang)
	vector2=getTFIDF(sentence2,lang)
	intersect=set.intersection(set(vector1.keys()),set(vector2.keys()))
	dot_product = sum([vector1[word]*vector2[word] for word in intersect])
	return dot_product


def jaccard_similarity(doc1, doc2):
	d1 = set(tokenize(doc1))
	d2 = set(tokenize(doc2))
	d1ud2 = d1 | d2
	d1id2 = d1 & d2
	return float(len(d1id2))/len(d1ud2)
