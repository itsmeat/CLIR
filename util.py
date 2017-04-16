import re

def tokenize(sentence):
	words=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\\\:"|<,./<>?,\n\' ]', sentence)
	return [w.lower() for w in words if w not in [''] and not w.isnumeric()]