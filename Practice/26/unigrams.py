import nltk
from operator import itemgetter
import re
import math
from bs4 import BeautifulSoup
from pickle import dump, load
from nltk.corpus import cess_esp
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import stopwords
import glob

##############################################
#			REVIEWS
##############################################
def getReviews(path, code):
	names = [f for f in glob.glob(path + "**/*.txt", recursive=True)]
	text = ""
	for name in names:
		f = open(name, encoding = code)
		rev = f.read()
		f.close()
		text = text + rev + " "
		print(name)
	text = text.lower()
	return text

#Parameters: Text
#Return: List of original tokens
def getTokens(text):
	tokens = nltk.word_tokenize(text)
	return tokens

#Parameters: List of tuples of tokens
#Return: List of clean tokens and Tags
def getCleanTokens(tokens):
	clean = []
	for token in tokens:
		t = []
		for char in token:
			if re.match(r'[a-záéíóúñüA-ZÁÉÍÓÚÜÑ]', char):
				t.append(char)
		letterToken = ''.join(t)

		if letterToken != '':
			clean.append(letterToken)
	return clean
#Parameters: List of clean tokens, language of stopwords
#Return: List of tokens without stopwords
def removeStopwords(tokens, language):
	sw = stopwords.words(language)

	cleanTokens = []
	for tok in tokens:
		if tok not in sw:
			cleanTokens.append(tok)
	return cleanTokens

##############################################
#	     		LEMMAS
##############################################
def getWords(fpath, code):
	f = open(fpath, encoding = code) #Cod: utf-8, latin-1
	text = f.read()
	f.close()

	words = re.sub(" ", " ",  text).split()
	return words

# Return: Dictionary 
def createDicLemmas(tokensLemmas):
	lemmas = {}
	j = 0
	for i in range(0, len(tokensLemmas)- 2, 3):
		word = tokensLemmas[i]
		tag = tokensLemmas[i+1]
		val = tokensLemmas[i+2]
		l = word
		lemmas[l] = val
		j = j+1
	return lemmas

def printDictionary(dic, n):
	i = 0
	for j in dic:
		print(j, dic[j])
		i = i + 1
		if i > n:
			break
##############################################
#	     		UNIGRAMS
##############################################
def flatten_corpus(corpus):
	return ' '.join([document.strip() for document in corpus])

def compute_ngrams(sequence, n):
	return zip(*[sequence[index:] for index in range(n)])

def get_top_ngrams(tokens, ngram_val=1, limit=20):
	# tokens = nltk.word_tokenize(corpus)
	ngrams = compute_ngrams(tokens, ngram_val)
	ngrams_freq_dist = nltk.FreqDist(ngrams)
	sorted_ngrams_fd = sorted(ngrams_freq_dist.items(), key=itemgetter(1), reverse=True)
	sorted_ngrams = sorted_ngrams_fd[0:limit]
	sorted_ngrams = [(' '.join(text), freq) for text, freq in sorted_ngrams]
	return sorted_ngrams

fpathLemmas = '/Users/abiga/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/Normalize/generateClean.txt'
fpathName = 'generateClean.txt'
code = 'ISO-8859-1'
textLemmas = getWords(fpathLemmas, code)
print(textLemmas[:20])

# Get dictionary of tuples of 
lemmas = {}
lemmas = createDicLemmas(textLemmas)
print("dictionary of lemmas:")
lemmas["abaláncenosla"] # Check the first
lemmas["zutano"]		   # Check the last 
lemmas["acercarnos"]
printDictionary(lemmas, 10)


kind = 'moviles'
path = '/Users/abiga/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/Practice/26/corpus/'
reviews = getReviews(path + kind, code)
tokens = getTokens(reviews) #  24565
print(len(tokens))

cleanTokens = getCleanTokens(tokens) # 20951
print(len(cleanTokens))

tok = removeStopwords(cleanTokens, 'spanish') # 11463
print(len(tok))

n = 1
aux = get_top_ngrams(tok, n, 20)
#### UNIGRAMS
print('\n Most common ' + str(n) +'-grams:')
for i in range(len(aux)):
	print(i+1, ". ", aux[i])
	
n = 2
aux = get_top_ngrams(tok, n, 20)
#### UNIGRAMS
print('\n Most common ' + str(n) +'-grams:')
for i in range(len(aux)):
	print(i+1, ". ", aux[i])

n = 3
aux = get_top_ngrams(tok, n, 20)
#### UNIGRAMS
print('\n Most common ' + str(n) +'-grams:')
for i in range(len(aux)):
	print(i+1, ". ", aux[i])
	