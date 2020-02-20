'''
	Implementar stemming:
	- Cada token original en el texto sustituir por su stem
		* Usar el archivo generate para hacer stem
		* Primero sin etiquetar, luego etiquetando

		La palabra extranjero tiene como stem: extra
		¿Cómo recupero de extra la palabra extranjero?
'''
import nltk
import re
import math
from bs4 import BeautifulSoup
from pickle import dump, load
from nltk.corpus import cess_esp
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import stopwords

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
def getVocabulary(tokens, stems):
	l = []
	for t in tokens:
		if t in stems:
			l.append(stems[t])

	vocabulary = sorted(set(l))
	return vocabulary

##############################################################
#				     		LEMMAS
##############################################################
def getWord(word):
	cleanWord = ''
	for char in word:
		if char != '#':
			cleanWord += char
	return cleanWord

def getStem(word):
	stem = ''
	for char in word:
		if char == '#':
			break
		stem += char 
	return stem

# Return: Dictionary 
def createDicStem(tokenStem):
	print("Tokens stem size:")
	print(len(tokenStem))
	stem = {}
	j = 0
	for i in range(0, len(tokenStem)- 2, 3):
		word = getWord(tokenStem[i])
		st = getStem(tokenStem[i])
		stem[word] = st
		j = j+1
	print("Iteraciones: " + str(j))
	return stem

def textStem(tokens, stems):
	l = []
	for t in tokens:
		if t in stems:
			l.append(stems[t])

	return l


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

def getFrecuency(vocabulary, contexts):
	vectors = {}
	for term in vocabulary:
		context = contexts[term]
		vector = []
		for t in vocabulary:
			frec = context.count(t)
			vector.append(frec)
		vectors[term] = vector
	return vectors

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

################################################
################################################
################################################

# nameFile = '/Users/27AGO2019/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/Normalize/similitudTags.txt'
# Get Tokens by Generate.txt to create dictionary of lemmas
fpathStem = '/Users/27AGO2019/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/Normalize/generate2.txt'
fpathName = 'generate2.txt'
code = 'ISO-8859-1'
textStem = getWords(fpathStem, code)
print(textStem[:20])

# Get dictionary of stem 
stem = {}
stem = createDicStem(textStem)
print("dictionary of stem:")
stem["abalancéis"]


i = 0
for j in stem:
	print(j, stem[j])
	i = i + 1
	if i > 200:
		break

fpath = '/Users/27AGO2019/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/corpus/e961024.htm'
language = 'spanish'
code = 'utf-8'
textSource = getText(fpath, code)
tokensHtml = getTokens(textSource)
cleanTokens = getCleanTokens(tokensHtml)
tokens = removeStopwords(cleanTokens, language)
print(tokens[:10])

vocabulary = getVocabulary(tokens, stem)
t = textStem(tokens, stem)
print(t[:50])

positions = initializeContext(t, vocabulary) #Initialize Context

#Get contexts
contexts = {}
print("Context:")
for term in vocabulary:
	contexts[term] = getContext(term, positions, 4, t)
	
#Get frecuency, vectors = {}
vectors = getFrecuency(vocabulary, contexts)

word = "gigante"
#Get List
#similitud = {}

similitud = getSimilitud(vocabulary, vectors, word)

l = list()
for key, val in similitud.items():
	l.append((val, key))
l.sort(reverse = True)
print(l[:10])

nameFile = '/Users/27AGO2019/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/Normalize/similitudStem.txt'
createFileDic(nameFile, l)