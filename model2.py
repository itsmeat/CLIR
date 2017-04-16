import dill as pickle
import collections
from model1 import train1
import os
from util import tokenize


file1='../parallel_corpus_IR2/english.txt'
file2='../parallel_corpus_IR2/french.txt'

parallel_corpus=[]
size=0


class customDict(collections.defaultdict):
	
	def __missing__(self, key):
		if self.default_factory is None:
			raise KeyError(key)
		else:
			ret = self[key] = self.default_factory(key)
			return ret


def aux(key):
	
	eng_ptr, fre_ptr, eng_len, fre_len = key
	return (1.0/fre_len)

def train2(english, french, map, iters = 20):

	align = customDict(aux)

	for x in range(iters):

		print("Running Iteration..",x+1)
		
		count_map = collections.defaultdict(float)
		count_align = collections.defaultdict(float)
		total_map = collections.defaultdict(float)
		total_align = collections.defaultdict(float)
		total_map_s = collections.defaultdict(float)
		
		for (eng, fre) in zip(english, french):
			
			eng = tokenize(eng)
			fre = tokenize(fre)
			eng_len = len(english)
			fre_len = len(french)

			for eng_ptr, eng_word in enumerate(eng, 1):
				
				total_map_s[eng_word] = 0
				for fre_ptr, fre_word in enumerate(fre, 1):
					
					total_map_s[eng_word] += map[(eng_word, fre_word)] *\
										align[(eng_ptr, fre_ptr, eng_len, fre_len)]
			
			for eng_ptr, eng_word in enumerate(eng, 1):
				
				for fre_ptr, fre_word in enumerate(fre, 1):
					
					temp = map[(eng_word, fre_word)] *\
						 align[(eng_ptr, fre_ptr, eng_len, fre_len)] / total_map_s[eng_word]
					
					count_map[(eng_word, fre_word)] += temp
					total_map[fre_word] += temp
					count_align[(eng_ptr, fre_ptr, eng_len, fre_len)] += temp
					total_align[(eng_ptr, eng_len, fre_len)] += temp

		# update map
		for key in count_map.keys():
			
			try:
				map[key] = count_map[key] / total_map[key[1]]
			
			except decimal.DivisionByZero:
				print('Error at', key)
				raise
		
		#update align
		for key in count_align.keys():
			align[key] = count_align[key] / total_align[(key[0], key[2], key[3])]

		pickle.dump(map, \
					open('../OutputFiles/map2_'+str(size)+'_'+str(x+1)+'.pickle','wb'))
		pickle.dump(align, \
					open('../OutputFiles/align_'+str(size)+'_'+str(x+1)+'.pickle','wb'))


	return (map, align)

def _constant_factory(value):
    return lambda: value

if __name__ == '__main__':

	with open(file1,'r') as inp:
		english=inp.readlines()

	with open(file2,'r') as inp:
		french=inp.readlines()
	
	# english = english[:size]
	# french = french[:size]

	print("Training for a corpus size of ", size, "out of", len(english))
	
	# load ibmmodel1's weights
	
	# if(os.path.exists('../OutputFiles/map1'+str(size)+'.pickle')):
	# 	print("Found exisiting IBMModel .. using")
	# 	map = pickle.load(open('../OutputFiles/map1'+str(size)+'.pickle', 'rb'))
	# else:
	print("Pre-trained IBMModel1 not found..start training")
	map = collections.defaultdict(_constant_factory(1.0/163497))
	map = train1(english, french, map,20)
	# pickle.dump(map, \
	# 			open('../OutputFiles/map1'+str(size)+'.pickle','wb'))
	
	print("Done with IBMModel1's training...")


	final_map, final_align = train2(english, french, map, 20)
	
	# pickle.dump(final_map, \
	# 			open('../OutputFiles/map2'+str(size)+'.pickle','wb'))
	# pickle.dump(final_align, \
	# 			open('../OutputFiles/align'+str(size)+'.pickle','wb'))
	print("Done with IBMModel2's training...")
	