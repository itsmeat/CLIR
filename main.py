import dill 
import sys
import os
from util import tokenize
from cldt import jaccard_similarity, cosine_similarity

def identifyLanguage(doc, eng_vocab, fre_vocab):
	
	count_eng = 0
	count_fre = 0

	words=doc
	for word in words:
		if word in eng_vocab:
			count_eng+=1
		if word in fre_vocab:
			count_fre+=1

	return count_eng>=count_fre

def translate(src, Map):

	trg = []
	for word in src:

		if(word in Map.keys()):
			trg.append(Map[word])
		else:
			trg.append(word)
	return trg

if __name__ == '__main__':

	if len(sys.argv) > 3:
		print("Please specify two arguments: [source_document_path] [target_document_path]")
		sys.exit()
	elif len(sys.argv) == 3:
		src_path = sys.argv[0]
		trg_path = sys.argv[1]
	elif len(sys.argv) == 2:
		src_path = sys.argv[0]
		print("Enter target_document_path:")
		trg_path = input()
	elif len(sys.argv) == 1:
		print("Enter source_document_path:")
		src_path = input()
		print("Enter target_document_path:")
		trg_path = input()
	
	src = []
	trg = []

	if(os.path.isdir(src_path)):
		for file in sorted(os.listdir(src_path)):
			h = open(os.path.join(src_path, file), 'r')
			src.append(h.read())

		for file in sorted(os.listdir(trg_path)):
			h = open(os.path.join(trg_path, file), 'r')
			trg.append(h.read())
	else:
		h = open(src_path, 'r')
		src = h.readlines()
		h = open(trg_path, 'r')
		trg = h.readlines()

	no_of_docs = len(src)
	
	#load db
	db_path = './db/'
	with open(db_path + 'EnglishVocab.pickle','rb') as out:
		eng_vocab = dill.load(out)
	with open(db_path + 'FrenchVocab.pickle','rb') as out:
		fre_vocab = dill.load(out)
	
	with open(db_path + 'FrenchToEnglish.pickle','rb') as out:
		fre_eng = dill.load(out)
	with open(db_path + 'EnglishToFrench.pickle','rb') as out:
		eng_fre = dill.load(out)
	
	print("Loading")
	Map = {}
	j_ls = []
	c_ls = []
	avg_j = 0
	avg_c = 0
	lang = []
	trans_ls = []
	for x in range(no_of_docs):

		print("Processing document: ", x)
		a = tokenize(src[x])
		b = tokenize(trg[x])
	
		temp = identifyLanguage(a, eng_vocab, fre_vocab)	
		lang.append(temp)
		if(temp):
			Map = eng_fre
			print("    Language of source is: English")
			print("    Performing English->French conversion")
		else:
			Map = fre_eng
			print("    Language of source is: French")	
			print("    Performing French->English conversion")
		
		trans = translate(a, Map)
		trans_ls.append(trans)

		j = jaccard_similarity(' '.join(trans), trg[x])
		c = cosine_similarity(' '.join(trans) , trg[x], temp)
		j_ls.append(j)
		c_ls.append(c)

		avg_j += j
		avg_c += c
	
	avg_j /= no_of_docs
	avg_c /= no_of_docs	

	print("Done")

	a = 1
	while a:
		print("1. Cosine Similarity and Jaccard coefficient:")
		print("2. Translated Document:")
		print("3. Average cosine similarity and Jaccard Coefficient:")
		print("Enter option:")
		opt = input()

		if opt == 1:
			
			for x in range(no_of_docs):
				print("Document ", x, ": Similarity")
				print("    Jaccard:", j[x])
				print("	   Cosine:", c[x])

		elif opt == 2:

			for x in range(no_of_docs):
				print("Document ", x, ": Translation")
				print(trans_ls[x])

		elif opt == 3:

			print("Average Jaccard Similarity: ", avg_j*100)
			print("Average Cosine Similarity: ", avg_c*100)
			
		else:
			print("Enter valid option:")


		