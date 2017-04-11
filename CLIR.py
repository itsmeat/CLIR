import re
from nltk import word_tokenize,FreqDist
import math
import sys

reload(sys)
sys.setdefaultencoding('cp1252')

file1='../parallel_corpus_IR2/english.txt'
file2='../parallel_corpus_IR2/french.txt'

with open(file2,'r') as inp:
	sentences=inp.readlines()

print len(sentences)

def tokenize(sentence):
	words=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\\\:"|<,./<>?,\n ]', sentence)
	return [w.lower() for w in words if w not in ['']]

# print tokenize(sentences[1]),"\n\n"

def give_loc_tf(text):

	ls_text = [x.lower() for x in tokenize(text)]
	freq = FreqDist(ls_text)

	ls = freq.most_common(len(freq))
	loc_tf = {}
	for x in ls:
		loc_tf[x[0]] = x[1]
	return loc_tf

# print give_loc_tf(sentences[1]),"\n\n"

vocab=set()

def computeIDF(sentences):
	idf={}
	noOfDocs=len(sentences)
	for sentence in sentences:
		words=set(tokenize(sentence))
		for word in words:
			if word not in idf.keys():
				idf[word]=0
				vocab.update(word)
			idf[word]+=1
	for word in idf.keys():
		idf[word]=noOfDocs/idf[word]

	return idf

idf = computeIDF(sentences[:100])
# print idf.keys()

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


def getTFIDF(sentence):
	N=100
	loc_tf=give_loc_tf(sentence)
	TfIdf={}
	for word in loc_tf.keys():
		TfIdf[word]=math.log(1+loc_tf[word],10)*idf[word]

	return normalize(TfIdf)


# for sentence in sentences[:100]:
# 	print getTFIDF(sentence)

def cosine_similarity(sentence1, sentence2):
	vector1=getTFIDF(sentence1)
	vector2=getTFIDF(sentence2)
	intersect=set.intersection(set(vector1.keys()),set(vector2.keys()))
	dot_product = sum([vector1[word]*vector2[word] for word in intersect])
	return dot_product


def jaccard_similarity(doc1, doc2):
	d1 = set(tokenize(doc1))
	d2 = set(tokenize(doc2))
	d1ud2 = d1 | d2
	d1id2 = d1 & d2
	return float(len(d1id2))/len(d1ud2)


print "Cosine Similarity :",cosine_similarity(sentences[0],sentences[1])

print "Jaccard Similarity :",jaccard_similarity(sentences[0],sentences[1])


