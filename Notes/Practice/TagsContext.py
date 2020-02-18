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
# def getCleanTokens(tokens):
# 	clean = []
# 	for token in tokens:
# 		t = []
# 		for char in token:
# 			if re.match(r'[a-záéíóúñüA-ZÁÉÍÓÚÜÑ]', char):
# 				t.append(char)
# 		letterToken = ''.join(t)
# 		if letterToken != '':
# 			clean.append(letterToken)
# 	return clean

# #Parameters: List of clean tokens, language of stopwords
# #Return: List of tokens without stopwords
# def removeStopwords(tokens, language):
# 	sw = stopwords.words(language)

# 	cleanTokens = []
# 	for tok in tokens:
# 		if tok not in sw:
# 			cleanTokens.append(tok)
# 	return cleanTokens

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
		l = []
		for char in token[0]:
			if re.match(r'[a-záéíóúñüA-ZÁÉÍÓÚÜÑ]', char):
				t.append(char)
		letterToken = ''.join(t)

		if len(token[1]) > 0:
			tag = token[1]
			tag = tag[0]

		if letterToken != '':
			l.append(letterToken)
			l.append(tag)
			clean.append(l)
	return clean

#Parameters: List of clean tokens, language of stopwords
#Return: List of tokens without stopwords
def removeStopwords(tokens, language):
	sw = stopwords.words(language)

	cleanTokens = []
	for tok in tokens:
		l = []
		if tok[0] not in sw:
			l.append(tok[0])
			l.append(tok[1])
			cleanTokens.append(l)
	return cleanTokens

##############################################################
#				     GET TAGS SPEECH
##############################################################
def createFileCombinedTagger(fname):
	default_tagger=nltk.DefaultTagger('V')
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

# Return: Dictionary of list of size 2: [word, tag]
def createDicLemmas(tokensLemmas):
	lemmas = {}
	t = 0 # t: type of operation
	for token in tokensLemmas:
		if token[0] != '[':
			if t == 0: # Is the first word
				word = getWord(token)
			if t == 1: # Is the tag
				tag = getTag(token) #Get tag in lowercase 
			if t == 2:
				l = []
				l.append(word)
				l.append(tag)
				lemmas[l] = token
			t = t + 1
			t = t % 3
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
					lemm = lemmas[c]
					if t == lemm:
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
			vec = vectors[term]
			similitud[term] = 0
			if mag(v) != 0 and mag(vec) != 0:
				cos = pointProduct(v, vec) / (mag(v) * mag(vec))
				similitud[term] = cos
	return similitud

################################################
################################################
################################################

# nameFile = '/Users/27AGO2019/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/Normalize/similitudTags.txt'
# Get Tokens by Generate.txt to create dictionary of lemmas
textLemmas = getText(fpathLemmas, code)
tokensLemmas = getTokens(textLemmas)
print("LEMMAS tokens")
print(tokensLemmas[:10])

# Get dictionary of tuples of Lemmas
lemmas = createDicLemmas(tokensLemmas)
print("dictionary of lemmas:")
print(lemmas[:10])

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
print("text with tags:")
print(text[:10])

# Text in list with lists of size 2
cleanTokens = getCleanTokensTags(text)
print("text with tags corrected:")
print(cleanTokens[:10])

# Remove Stopwords
language = 'spanish'
tokens = removeStopwords(cleanTokens, language)
print("Text without stopwords:")
print(tokens[:10])

# Get vocabulary of lemmas
vocabulary = getVocabulary(tokens, lemmas)

#Get contexts
positions = initializeContext(tokens, vocabulary, lemmas) #Initialize Context

contexts = {}
for term in vocabulary:
	contexts[term] = getContext(term, positions, 4, tokens, vocabulary)
print("Context:")
print(contexts[:10])

#Get frecuency, vectors = {}
vectors = getFrecuency(vocabulary, contexts, lemmas)

# Find lemma of my word
word = "grande"
l.append(word)
l.append("a")
if l in lemmas:
	w = lemmas[l]
#Get List
similitud = getSimilitud(vocabulary, vectors, word)

#language = 'spanish'
#tokens = removeStopwords(cleanTokens, language)
#vocabulary = getVocabulary(tokens)