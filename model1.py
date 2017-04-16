import collections
from decimal import Decimal
import dill as pickle
from util import tokenize

parallel_corpus=[]
size=10000

file1='../parallel_corpus_IR2/english.txt'
file2='../parallel_corpus_IR2/french.txt'


def _constant_factory(value):
    return lambda: value

def train1(english, french, trans_prob, loop_count=20):
	
	for i in range(loop_count):
	
		print("Running Iteration..",i+1)

		count=collections.defaultdict(float)
		total=collections.defaultdict(float)
		count=collections.defaultdict(float)
		
		sum_total={}

		for(english,french) in zip(english, french):

			english = tokenize(english)
			french = tokenize(french)
			for e in english:
				sum_total[e]=0.0
				for f in french:
					sum_total[e]+=trans_prob[(e,f)]
					
			for e in english:
				for f in french:
					count[(e, f)] += trans_prob[(e, f)] / sum_total[e]
					total[f]+=trans_prob[(e,f)]/sum_total[e]

		for (e,f) in count.keys():
			trans_prob[(e,f)]=count[(e,f)]/total[f]

		pickle.dump(trans_prob, \
					open('../OutputFiles/map1_'+str(size)+'_'+str(i+1)+'.pickle','wb'))

	return trans_prob


def trainAll():
	trans_prob = collections.defaultdict(_constant_factory(1e-6))
	i=0

	while i<len(english):
		if(i%size==0):
			print( "Parallel corpus Added")
			global parallel_corpus
			trans_prob=train1(english, french, trans_prob)
			with open('OutputFiles/transProb'+str(i)+'.pickle','wb') as fp:
				pickle.dump(trans_prob,fp)
			print ("Done Training for lines ",i,"\n\n")
			parallel_corpus=[]
			
			print(trans_prob[('on', 'sur')])

		parallel_corpus.append((tokenize(english[i]),tokenize(french[i])))

		i+=1
	
	trans_prob=train1(english, french, trans_prob)
	
	return trans_prob

if __name__ == '__main__':
	
	with open(file1,'r') as inp:
		english=inp.readlines()

	with open(file2,'r') as inp:
		french=inp.readlines()


	final=trainAll()
	with open('OutputFiles/transProb'+str(i)+'.pickle','wb') as fp:
		pickle.dump(trans_prob,fp)