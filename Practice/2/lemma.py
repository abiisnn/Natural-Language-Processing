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
def getVocabulary(tokens):
	vocabulary = sorted(set(tokens))
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
#						TAGGING
##############################################################
def make_and_save_combined_tagger(fname):
    default_tagger = nltk.DefaultTagger('v')
    patterns = [ (r'.*o$', 'n'),   # noun masculine singular
               	 (r'.*os$', 'n'),  # noun masculine plural
                 (r'.*a$', 'n'),   # noun feminine singular
                 (r'.*as$', 'n')   # noun feminine singular
               ]
    regexp_tagger = nltk.RegexpTagger(patterns, backoff=default_tagger)
    cess_tagged_sents = cess_esp.tagged_sents()
    combined_tagger = nltk.UnigramTagger(cess_tagged_sents, backoff=regexp_tagger)
    
    output = open(fname, 'wb')
    dump(combined_tagger, output, -1)
    output.close()

def tag(fname, text):
    input = open(fname, 'rb')
    default_tagger = load(input)
    input.close()

    s_tagged = default_tagger.tag(text)
    return s_tagged

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
	lemmas = {}
	j = 0
	for i in range(0, len(tokensLemmas)- 2, 3):
		word = tokensLemmas[i]
		tag = tokensLemmas[i+1]
		val = tokensLemmas[i+2]
		l = (word, tag[0].lower())
		lemmas[l] = val
		j = j+1
	return lemmas

def lemmatizeText(tokens, lemmas):
	text = []
	for token in tokens:
		lemma = token[0]
		if token in lemmas:
			lemma = lemmas[token]
		aux = (lemma, token[1])
		text.append(aux)
	return text

##############################################################
#						CONTEXT
##############################################################

#Parameters: clean Tokens, vocabulary
#Return: Map of positions of every word in vocabulary
def initializeContext(tokens, vocabulary, lemmas):
	contexto = {}
	for word in vocabulary:
		contexto[word] = []

	# for i in range(len(tokens)):
	# 	token = tokens[i]
	# 	lemma = token[0]
	# 	if token in lemmas:
	# 		lemma = lemmas[token]
	# 	if lemma in contexto:
	# 		contexto[lemma].append(i)
	for i in range(len(tokens)):
		contexto[tokens[i]].append(i)
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

def getIndexTuples(vocabulary):
	index = {}
	i = 0
	for term in vocabulary:
		index[term] = i 
		i = i + 1
	return index

def getFrecuency(vocabulary, contexts, lemmas, indexTuples):
	vectors = {}
	v = []
	for i in range(0, len(vocabulary) + 1):
		v.append(0)

	for term in vocabulary:
		vector = []
		for i in range(0, len(vocabulary) + 1):
			vector.append(0)
		for t in contexts[term]:
			vector[indexTuples[t]] = vector[indexTuples[t]] + 1
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

def filtredSimilitud(similitud, word):
	sim = {}
	for t in similitud:
		if t[1] == word[1]:
			sim[t] = similitud[t]
	return sim

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

def printDictionary(dic, n):
	i = 0
	for j in dic:
		print(j, dic[j])
		i = i + 1
		if i > n:
			break
################################################
################################################
################################################

# Get Tokens by Generate.txt to create dictionary of lemmas
fpathLemmas = '/Users/27AGO2019/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/Normalize/generateClean.txt'
fpathName = 'generateClean.txt'
nameFile = '/Users/27AGO2019/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/Normalize/similitudLemma.txt'
code = 'ISO-8859-1'
textLemmas = getWords(fpathLemmas, code)
print(textLemmas[:20])

# Get dictionary of tuples of 
lemmas = {}
lemmas = createDicLemmas(textLemmas)
print("dictionary of lemmas:")
lemmas[("abaláncenosla", "v")] # Check the first
lemmas[("zutano", "n")]		   # Check the last 
lemmas[("acercarnos", "v")]
printDictionary(lemmas, 10)

# Read file of corpus
fpath = '/Users/27AGO2019/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/corpus/e961024.htm'
code = 'utf-8'
textSource = getText(fpath, code) 
tokensHtml = getTokens(textSource) #Get tokens with out html tags
print("Text with tags, stopwords and punctuation:")
print(tokensHtml[:10])

# Tagging
fcombinedTagger = '/Users/27AGO2019/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/pkl/combined_tagger.pkl'
#make_and_save_combined_tagger(fcombinedTagger)
textTagged = tag(fcombinedTagger, tokensHtml)
print(textTagged[:10])

# Text in list with lists of size 2
cleanTokens = getCleanTokensTags(textTagged)
print("text with tags corrected:")
print(cleanTokens[:10])

# Remove Stopwords
language = 'spanish'
tokens = removeStopwords(cleanTokens, language)
print("Text without stopwords:")
print(tokens[:100])

# Lemmatize text
tokens = lemmatizeText(tokens, lemmas)
print(tokens[:100])

# Get vocabulary of lemmas
vocabulary = getVocabulary(tokens)
print("vocabulary:")
print(vocabulary[3100:3200])

indexTuples = {}
indexTuples = getIndexTuples(vocabulary)
printDictionary(indexTuples, 10)
len(indexTuples)

#Get contexts
positions = initializeContext(tokens, vocabulary, lemmas) #Initialize Context
printDictionary(positions, 100)
len(positions)

contexts = {}
for term in vocabulary:
	contexts[term] = getContext(term, positions, 4, tokens, vocabulary)
print("Context:")
printDictionary(contexts, 2)
len(contexts)


#Get frecuency, vectors = {}
vectors = {}
vectors = getFrecuency(vocabulary, contexts, lemmas, indexTuples)
print("frecuency:")
printDictionary(vectors, 2)
len(vectors)

# Find lemma of my word
word = ('grande', 'a')

lemmas[word]

#Get List
similitud = {}
similitud = getSimilitud(vocabulary, vectors, word)
print("Similitud:")
printDictionary(similitud, 10)


filtred = {}
filtred = filtredSimilitud(similitud, word)
print("Similitud filtrada:")
printDictionary(filtred, 10)

l = list()
for key, val in filtred.items():
	l.append((val, key))
l.sort(reverse = True)
print(l[:10])

nameFile = '/Users/27AGO2019/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/Normalize/similitudTagsLemma.txt'
createFileDic(nameFile, l)
