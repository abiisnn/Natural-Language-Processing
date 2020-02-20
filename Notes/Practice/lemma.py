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
	return tokens

#Parameters: List of normalize tokens
#Return: Set, vocabulary
def getVocabulary(tokens, lemmas):
	l = []
	for token in tokens:
		if token in lemmas:
			l.append(lemmas[token])

	vocabulary = sorted(set(l))
	return vocabulary

#Parameters: List of tuples of tokens
#Return: List of clean tokens and Tags
def getCleanTokensTags(tokens):
	clean = []
	for token in tokens:
		t = []
		for char in token[0]:
			if re.match(r'[a-záéíóúñüA-ZÁÉÍÓÚÜÑ]', char):
				t.append(char)
		letterToken = ''.join(t)

		if len(token[1]) > 0:
			tag = token[1]
			tag = tag[0].lower()

		if letterToken != '':
			l = (letterToken, tag)
			clean.append(l)
	return clean

#Parameters: List of clean tokens, language of stopwords
#Return: List of tokens without stopwords
def removeStopwords(tokens, language):
	sw = stopwords.words(language)

	cleanTokens = []
	for tok in tokens:
		l = ()
		if tok[0] not in sw:
			l = (tok[0], tok[1])
			cleanTokens.append(l)
	return cleanTokens

##############################################################
#				     GET TAGS SPEECH
##############################################################
def createFileCombinedTagger(fname):
	default_tagger = nltk.DefaultTagger('V')
	patterns = [ (r'.*o$', 'NMS'), # noun masculine singular
               (r'.*os$', 'NMP'), # noun masculine plural
               (r'.*a$', 'NFS'),  # noun feminine singular
               (r'.*as$', 'NFP')  # noun feminine singular
               ]
	regexpTagger = nltk.RegexpTagger(patterns, backoff = default_tagger)
	cessTagged = cess_esp.tagged_sents()
	combinedTagger = nltk.UnigramTagger(cessTagged, backoff = regexpTagger)
	output = open(fname, 'wb')
	dump(combinedTagger, output, -1)
	output.close()

def tag(fname, tokens):
	input = open(fname, 'rb')
	defaultTagger = load(input)
	input.close()

	text = []
	for token in tokens:
		tagged = defaultTagger.tag(tokens)
		text.append(tagged)
		# print(tagged)
	print("Type of tagged: I supposed is a tuple")
	print(type(tagged))

	return text

def tag_with_pos_tag_function_nltk(tokens):
	tagged = nltk.pos_tag(tokens)
	return tagged

def cleanTags(tagged):
	text = []
	for t in tagged:
		if t[1] is None:
			tag = "v"
		else:
			tag = t[1][0].lower()
		t = (t[0], tag)
		text.append(t)
	return text

##############################################################
#				     		LEMMAS
##############################################################
def getWord(word):
	cleanWord = ''
	for char in word:
		if char != '#':
			cleanWord += char
	return cleanWord

def getTag(word):
	c = 'v'
	if len(word) > 0:
		c = word[0]
	return c.lower()

# Return: Dictionary 
def createDicLemmas(tokensLemmas):
	print("Tokens lemmas size:")
	print(len(tokensLemmas))
	lemmas = {}
	j = 0
	for i in range(0, len(tokensLemmas)- 2, 3):
		word = tokensLemmas[i]
		tag = tokensLemmas[i+1]
		val = tokensLemmas[i+2]
		l = (word, tag[0].lower())
		lemmas[l] = val
		j = j+1
	print("Iteraciones: " + str(j))
	return lemmas

##############################################################
#						CONTEXT
##############################################################

#Parameters: clean Tokens, vocabulary
#Return: Map of positions of every word in vocabulary
def initializeContext(tokens, vocabulary, lemmas):
	contexto = {}
	for word in vocabulary:
		contexto[word] = []

	for i in range(len(tokens)):
		token = tokens[i]
		if token in lemmas:
			lemma = lemmas[token]
			if lemma in contexto:
				contexto[lemma].append(i)
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

#Parameters: List of list(size 2)
#Return: list of context
def getContext(token, positions, window, originalText, vocabulary):
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

def getFrecuency(vocabulary, contexts, lemmas):
	vectors = {}
	for term in vocabulary:
		context = contexts[term] #Its a list
		vector = []
		for t in vocabulary:
			frec = 0
			#frec = context.count(t)
			for c in context:
				if c in lemmas:
					if t == lemmas[c]:
						frec = frec + 1
			vector.append(frec)
		vectors[term] = vector
	return vectors

##############################################################
#						SIMILITUD
##############################################################
def getSimilitud(vocabulary, vectors, lemma):
	similitud = {}
	if lemma in vectors:
		v = vectors[lemma]
		for term in vocabulary:
			similitud[term] = 0
			if term in vectors:
				vec = vectors[term]
				if mag(v) != 0 and mag(vec) != 0:
					cos = pointProduct(v, vec) / (mag(v) * mag(vec))
					similitud[term] = cos
	return similitud

def getWords(fpath, code):
	f = open(fpath, encoding = code) #Cod: utf-8, latin-1
	text = f.read()
	f.close()

	words = re.sub(" ", " ",  text).split()
	# words = text.words(fname)
	# words = list(words) #Convertir a lista de palabras
	return words

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
fpathLemmas = '/Users/27AGO2019/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/Normalize/generate1.txt'
fpathName = 'generate1.txt'
code = 'ISO-8859-1'
textLemmas = getWords(fpathLemmas, code)
print(textLemmas[:20])

# Get dictionary of tuples of 
lemmas = {}
lemmas = createDicLemmas(textLemmas)
print("dictionary of lemmas:")
lemmas[("abaláncenosla", "v")]
# lemmas[("zutano", "n")]
# lemmas[("rasante", "a")]


# Read file of corpus
fpath = '/Users/27AGO2019/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/corpus/e961024.htm'
code = 'utf-8'
textSource = getText(fpath, code) 
tokensHtml = getTokens(textSource) #Get tokens with out html tags
print("Text with tags, stopwords and punctuation:")
print(tokensHtml[:10])

#Tagging
fnameCombinedTagger = '/Users/27AGO2019/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/Normalize/combinedTagger.pkl'
# createFileCombinedTagger(fnameCombinedTagger)
text = tag(fnameCombinedTagger, tokensHtml)
#text = tag_with_pos_tag_function_nltk(tokensHtml)
print("text with tags:")
print(text[:20])

textTags = cleanTags(text)
print(textTags[:10])

# Text in list with lists of size 2
cleanTokens = getCleanTokensTags(text)
print("text with tags corrected:")
print(cleanTokens[:10])

# Remove Stopwords
language = 'spanish'
tokens = removeStopwords(cleanTokens, language)
print("Text without stopwords:")
print(tokens[:50])

i = 0
for j in lemmas:
	print(j, lemmas[j])
	i = i + 1
	if i > 50:
		break


# Get vocabulary of lemmas
vocabulary = getVocabulary(tokens, lemmas)
print("vocabulary:")
print(vocabulary[:10])

#Get contexts
positions = initializeContext(tokens, vocabulary, lemmas) #Initialize Context

contexts = {}
for term in vocabulary:
	contexts[term] = getContext(term, positions, 4, tokens, vocabulary)
print("Context:")
i = 0
for j in contexts:
	print(j, contexts[j])
	i = i + 1
	if i > 20:
		break

#Get frecuency, vectors = {}
vectors = getFrecuency(vocabulary, contexts, lemmas)
print("frecuency:")
i = 0
for j in vectors:
	print(j, vectors[j])
	i = i + 1
	if i > 10:
		break

# Find lemma of my word
word = "grande"
l = (word, "a")
lemmas[("grande", "a")]
if l in lemmas:
	w = lemmas[l]

#Get List
similitud = getSimilitud(vocabulary, vectors, word)
print("Similitud:")
i = 0
for j in similitud:
	print(j, similitud[j])
	i = i + 1
	if i > 10:
		break

l = list()
for key, val in similitud.items():
	l.append((val, key))
l.sort(reverse = True)
print(l[:10])

nameFile = '/Users/27AGO2019/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/Normalize/similitudTags.txt'
createFileDic(nameFile, l)
#language = 'spanish'
#tokens = removeStopwords(cleanTokens, language)
#vocabulary = getVocabulary(tokens)