#specify file here
file1='../OutputFiles/map_xx.pickle'

outFile='../OutputFiles/Translate.pickle'

import dill

def storeFile(file1,outFile):
	print ("Loading File..")
	
	with open(file1,'rb') as inp:
		transProb=dill.load(inp)

	print("Done loading File..")

	translate={}

	for (e,f) in transProb.keys():
		if(transProb[(e,f)])>0.6:
			if e not in translate.keys():
				translate[e]=[]
			translate[e].append((f,transProb[(e,f)]))
	
	print("Done getting Translations...")
	englishToFrench={}
	frenchToEnglish={}
	
	for e in translate.keys():
	    translate[e]=sorted(translate[e], key=lambda x: x[1], reverse=True)
	    englishToFrench[e]=translate[e][0][0]
	    frenchToEnglish[translate[e][0][0]]=e

	print("Done getting word Translation...")

	with open(outFile,'wb') as out:
		dill.dump(translate,out)

	with open('./db/EnglishToFrench.pickle','wb') as out:
		dill.dump(englishToFrench,out)

	with open('./db/FrenchToEnglish.pickle','wb') as out:
		dill.dump(frenchToEnglish,out)

	print("Done Storing Translation Probabilites...")

storeFile(file1, outFile)
	
