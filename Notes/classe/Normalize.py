import nltk
import re
import math
from bs4 import BeautifulSoup
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import stopwords

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
#Return: List of tokens
#Notes:
def getTokens(text):
	tokens = nltk.word_tokenize(text)
	return nltk.Text(tokens)	

#Parameters: List of tokens
#Return: List of clean tokens
#Notes:
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

#Parameters: List of clean tokens, language
#Return: List of tokens without stopwords
#Notes:
def removeStopwords(tokens, language):
	sw = stopwords.words(language)

	cleanTokens = []
	for tok in tokens:
		if tok not in sw:
			cleanTokens.append(tok)
	return cleanTokens

#Parameters: List of normalize tokens
#Return: Set, vocabulary
#Notes:
def getVocabulary(tokens):
	vocabulary = sorted(set(tokens))
	return vocabulary

#Parameters: Set, vocabulary
#Return: Nothing
#Notes:	Create a new file
def createFile(path, vocabulary):
	f = open(path, 'w')
	for word in vocabulary:
		f.write(word + '\n')
	f.close()

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

def leftContextPosition(pos, window):
	pos = pos - window
	if(pos < 0):
		pos = 0
	return pos

def rightContextPosition(pos, window, n):
	pos = pos + window
	if(pos > n):
		pos = n - 1 
	return pos

#Parameters: string: 
#Return: 
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

def pointProduct(a, b):
	ans = 0
	for i in range(0, len(a)):
		ans += (a[i] * b[i])
	return ans

def printContext(context):
	for i in range(0, len(context)):
		aux = ''
		for j in range(0, len(context[i])):
			aux += context[i][j] + " "
		print(aux)

def mag(v):
	ans = 0
	for i in range(0, len(v)):
		ans += (v[i] * v[i])
	return math.sqrt(ans)

def createFileDic(path, dic):
	f = open(path, 'w')
	for key, value in dic.items():
		f.write(key + " ")
		f.write(str(value))
		f.write('\n')
	f.close()
 #    print (key, value)
	# for word in dic:
	# 	aux = dic[word]
	# 	f.write(word + " ")
	# 	f.write(aux)
	# 	f.write('\n')
	
	# f.close()

fpath = '/Users/27AGO2019/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/corpus/e961024.htm'
nameFile = '/Users/27AGO2019/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/Normalize/similitud.txt'
language = 'spanish'
code = 'utf-8'
textSource = getText(fpath, code)
tokensHtml = getTokens(textSource)
cleanTokens = getCleanTokens(tokensHtml)

tokens = removeStopwords(cleanTokens, language)
vocabulary = getVocabulary(tokens)

positions = initializeContext(tokens, vocabulary) #Initialize Context

#Get contexts
contexts = {}
print("Context:")
for term in vocabulary:
	contexts[term] = getContext(term, positions, 8, tokens)
	print(contexts[term])

#Get frecuency
vectors = {}
for term in vocabulary:
	context = contexts[term]
	vector = []
	for t in vocabulary:
		frec = context.count(t)
		vector.append(frec)
	vectors[term] = vector

#Get List
similitud = {}
word = "grande"
v = vectors[word]
for term in vocabulary:
	vec = vectors[term]
	if mag(v) != 0 and mag(vec) != 0:
		cos = pointProduct(v, vec) / (mag(v) * mag(vec))
		similitud[term] = cos
{k: v for k, v in sorted(similitud.items(), key=lambda item: item[1])}

# for i in sorted(similitud):
# 	print((i, similitud[i]), end = " ")

createFileDic(nameFile, similitud)

'''
For each term, get context

contexts[term] = list of context
vectors[term] = num vector

For each term, get context
	context = contexts[term]
	vector = []
	for each term 
		frec = contex.count(term)
	vector.append(frec)

vector = vectors[word]
for each ter
	vec = vectors[term]
	coseno = 
'''
# createFile(nameFile, vocabulary)