import nltk
import re
import math
from bs4 import BeautifulSoup
from pickle import dump, load
from nltk.corpus import cess_esp
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

##############################################################
#					NORMALIZE TEXT
##############################################################

#Parameters: File path, encoding
#Return: String with only lower case letters
#Notes: path = '/Users/27AGO2019/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/corpus/e961024.htm'
def getText(corpusRoot, code):
	f = open(corpusRoot, encoding = code) #Cod: utf-8, latin-1
	text = f.read()
	f.close()
	soup = BeautifulSoup(text, 'lxml')
	text = soup.get_text()
	text = text.lower()
	return text

#Parameters: Text
#Return: List of original tokens
def getTokens(text):
	tokens = nltk.word_tokenize(text)
	return nltk.Text(tokens)	

#Parameters: List of tokens
#Return: List of clean tokens
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

#Parameters: List of normalize tokens
#Return: Set, vocabulary
def getVocabulary(tokens):
	vocabulary = sorted(set(tokens))
	return vocabulary

##############################################################
#						CONTEXT
##############################################################

#Parameters: clean Tokens, vocabulary
#Return: Map of positions of every word in vocabulary
def initializeContext(tokens, vocabulary):
	contexto = {}
	for word in vocabulary:
		contexto[word] = []

	for i in range(len(tokens)):
		token = tokens[i]
		if token in contexto:
			contexto[token].append(i)

	return contexto

#Parameters: Position of the word, size of window
#Return: Position of begin
def leftContextPosition(pos, window):
	pos = pos - window
	if(pos < 0):
		pos = 0
	return pos

#Parameters: Position of the word, size of window, size of window
#Return: Position of end
def rightContextPosition(pos, window, n):
	pos = pos + window
	if(pos > n):
		pos = n - 1 
	return pos

#Parameters: t
#Return: list of context
def getContext(token, positions, window, originalText):
	context = []
	if token in positions:
		for pos in positions[token]:
			lpos = leftContextPosition(pos, window)
			rpos = rightContextPosition(pos, window, len(originalText))
			#con = []
			for i in range(lpos, pos):
				context.append(originalText[i])
			#con.append(token)
			for i in range(pos + 1, rpos):
				context.append(originalText[i])			
			#context.append(con)
	return context

##############################################################
#				VECTOR OPERATIONS
##############################################################

#Parameters: vector a, vector b
#Return: int, point product
#Notes: Need vectors of the same size
def pointProduct(a, b):
	ans = 0
	for i in range(0, len(a)):
		ans += (a[i] * b[i])
	return ans

#Parameters: vector of int
#Return: mag of a vector
def mag(v):
	ans = 0
	for i in range(0, len(v)):
		ans += (v[i] * v[i])
	return math.sqrt(ans)

##############################################################
#						FRECUENCY
##############################################################

def sumElements(vector):
	sum = 0
	for v in vector:
		sum = sum + v
	return sum

def getDocumentFrecuency(vocabulary, contexts):
	documentFreq = {}
	for term in vocabulary:
		documentFreq[term] = 0

	for context in contexts:
		for word in contexts[context]:
			documentFreq[word] = documentFreq[word] + 1 

	return documentFreq

def getFrecuency(vocabulary, contexts, k, documentFreq):
	vectors = {}
	for term in vocabulary:
		context = contexts[term]
		vector = []
		for t in vocabulary:
			frec = context.count(t)
			vector.append(frec)
		vectors[term] = vector

	# Getting TF dictionary:
	TF = {}
	for term in vocabulary:
		vectorFrecuency = vectors[term]
		vTF = []
		for t in vectorFrecuency:
			frec = ((k + 1) * t) / (t + k)
			vTF.append(frec)
		TF[term] = vTF

	# Getting IDF:
	IDF = []
	for term in vocabulary:
		frec = 0
		if documentFreq[term] != 0:
			frec = math.log((len(vocabulary) + 1) / documentFreq[term])
		IDF.append(frec)

	# Gettin tf - IDF = tf * IDF
	finalFrecuency = {}
	for term in vocabulary:
		vector = []
		tf = TF[term]
		for i in range(0, len(vocabulary)):
			frec = tf[i] * IDF[i]
			vector.append(frec)
		finalFrecuency[term] = vector

	return finalFrecuency

##############################################################
#						SIMILITUD
##############################################################

def getSimilitud(vocabulary, vectors, word):
	similitud = {}
	v = vectors[word]
	for term in vocabulary:
		vec = vectors[term]
		similitud[term] = 0
		if mag(v) != 0 and mag(vec) != 0:
			cos = pointProduct(v, vec) / (mag(v) * mag(vec))
			similitud[term] = cos
	return similitud
	
##############################################################
#						CREATE FILE
##############################################################

#Parameters: Set, vocabulary
#Return: Nothing
def createFile(path, vocabulary):
	f = open(path, 'w')
	for word in vocabulary:
		f.write(word + '\n')
	f.close()

#Parameters: , vocabulary
#Return: Nothing
def createFileDic(path, l):
	f = open(path, 'w')
	for item in l:
		f.write(str(item))
		f.write('\n')
	f.close()

def printContext(context):
	for i in range(0, len(context)):
		aux = ''
		for j in range(0, len(context[i])):
			aux += context[i][j] + " "
		print(aux)

##############################################################
#					STEMMING
##############################################################
def convertStemms(tokens):
	ss = SnowballStemmer("spanish")
	text = []
	for t in tokens:
		text.append(ss.stem(t))
	return text


#######################################
# 				MAIN
#######################################

# Normalizing text
fpath = '/Users/27AGO2019/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/corpus/e961024.htm'
nameFile = '/Users/27AGO2019/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/Normalize/similitudTF_IDF.txt'
language = 'spanish'
code = 'utf-8'
textSource = getText(fpath, code)
tokensHtml = getTokens(textSource)
cleanTokens = getCleanTokens(tokensHtml)
tokensStop = removeStopwords(cleanTokens, language)

# Convert text to Stemms
tokens = convertStemms(tokensStop)
# print(tokens[:100])

vocabulary = getVocabulary(tokens)

positions = initializeContext(tokens, vocabulary) #Initialize Context

#Get contexts
contexts = {}
print(" Getting Context:")
for term in vocabulary:
	contexts[term] = getContext(term, positions, 8, tokens)
	
# Getting Document frecuency for each word
documentFreq = {}
documentFreq = getDocumentFrecuency(vocabulary, contexts)

# print("documentFreq:")
# i = 0
# for j in documentFreq:
# 	print(j, documentFreq[j])
# 	i = i + 1
# 	if i > 50:
# 		break

#Get frecuency, vectors = {}
k = 1.2
vectors = {}
vectors = getFrecuency(vocabulary, contexts, k, documentFreq)

# print("Vectors frecuency:")
# i = 0
# for j in vectors:
# 	print(j, vectors[j])
# 	i = i + 1
# 	if i > 50:
# 		break


word = "grande"
ss = SnowballStemmer("spanish")
stemWord = ss.stem(word)

similitud = {}
similitud = getSimilitud(vocabulary, vectors, stemWord)

l = list()
for key, val in similitud.items():
	l.append((val, key))
l.sort(reverse = True)
print(l[:10])

createFileDic(nameFile, l)
