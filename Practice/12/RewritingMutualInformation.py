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
	s = set()
	for l in tokens:
		for element in l:
			s.add(element)
	vocabulary = sorted(s)
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

def getIndexTuples(vocabulary):
	index = {}
	i = 0
	for term in vocabulary:
		index[term] = i 
		i = i + 1
	return index

def getFrecuency(vocabulary, contexts, b, k, documentFreq, lemmas, indexTuples):
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

	sum = 0
	for element in vectors:
		for t in vectors[element]:
			sum = sum + t
	avdl = sum / len(vocabulary)
	
	# Getting IBM25 dictionary:
	IBM25 = {}
	for term in vocabulary:
		vectorFrecuency = vectors[term]
		magd1 = sumElements(vectorFrecuency)
		vIBM25 = []
		for t in vectorFrecuency:
			frec = 0
			num = (k + 1) * t 
			den = t + (k * (1 - b + ((b * magd1) / avdl)))
			if den != 0:
				frec = num / den
			vIBM25.append(frec)
		IBM25[term] = vIBM25

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
		ibm25 = IBM25[term]
		sumBM25 = sumElements(ibm25)
		for i in range(0, len(vocabulary)):
			frec = 0
			if sumBM25 != 0:
				frec = (ibm25[i] / sumBM25) * IDF[i]
			vector.append(frec)
		finalFrecuency[term] = vector

	return finalFrecuency

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

def makePKL(fname, aux):
	output = open(fname, 'wb')
	dump(aux, output, -1)
	output.close()

def getPKL(fname):
	input = open(fname, 'rb')
	aux = load(input)
	input.close()
	return aux

##############################################################
#						SYNTAGMATIC RELATION
##############################################################
def get_sentences(fname):
    f=open(fname, encoding = 'utf-8')
    t=f.read()
    soup = BeautifulSoup(t, 'lxml')
    text_string = soup.get_text()

    #get a list of sentences
    sent_tokenizer=nltk.data.load('nltk:tokenizers/punkt/english.pickle')
    sentences=sent_tokenizer.tokenize(text_string)
    return sentences

def getCount(sentences, vocabulary):
 	count = {}
 	for i in vocabulary:
 		count[i] = 0
 	for sentence in sentences:
 		for element in sentence:
 			if element in vocabulary:
	 			count[element] = count[element] + 1
 	return count

def getConjunta(sentences, w1, w2, p2):
	ans = 0
	for sentence in sentences:
		v = sentence
		w1count = v.count(w1)
		w2count = v.count(w2)
		if (w1count == 1) and (w2count == 1):
			ans = ans + 1
	return (ans + 0.25) / p2

def I(w1, w2, p11):
	w10 = 1 - w1
	w20 = 1 - w2
	p01 = w2 - p11
	p10 = w1 - p11
	p00 = w10 - p01
	a = 0
	b = 0
	c = 0
	d = 0
	if w10 != 0 and w20 != 0:
		l = p00 / (w10 * w20)
		if l > 0:
			a = p00 * math.log2(l)
	if w1 != 0 and w20 != 0:
		l = p10 / (w1 * w20)
		if l > 0:
			b = p10 * math.log2(l)
	if w10 != 0 and w2 != 0:
		l = p01 / (w10 * w2)
		if l > 0:
			c = p01 * math.log2(l)
	if w1 != 0 and w2 != 0:
		l = p11 / (w1 * w2)
		if l > 0:
			d = p11 * math.log2(l)

	return a + b + c + d

################################################
################################################
################################################

# Get Tokens by Generate.txt to create dictionary of lemmas
fpathLemmas = '/Users/abiga/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/Normalize/generateClean.txt'
fpathName = 'generateClean.txt'
nameFile = '/Users/abiga/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/Normalize/similitudLemma.txt'
code = 'ISO-8859-1'
textLemmas = getWords(fpathLemmas, code)
# print(textLemmas[:20])

# Get dictionary of tuples of 
lemmas = {}
lemmas = createDicLemmas(textLemmas)

# print("dictionary of lemmas:")
# lemmas[("abaláncenosla", "v")] # Check the first
# lemmas[("zutano", "n")]		   # Check the last 
# lemmas[("acercarnos", "v")]
# printDictionary(lemmas, 10)


# Read file of corpus and get Sentences
fpath = '/Users/abiga/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/corpus/e961024.htm'
sentences = get_sentences(fpath)

# Tagging
fcombinedTagger = '/Users/abiga/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/pkl/combined_tagger.pkl'
#make_and_save_combined_tagger(fcombinedTagger)
sentencesTag = []
for i in range(0, len(sentences)):
	aux = getTokens(sentences[i])
	auxTag = tag(fcombinedTagger, aux)
	sentencesTag.append(auxTag)

sentencesCleanTokens = []
for i in range(0, len(sentencesTag)):
	aux = getCleanTokensTags(sentencesTag[i])
	sentencesCleanTokens.append(aux)

# Remove Stopwords
language = 'spanish'
sentencesClean = []
for i in range(0, len(sentencesCleanTokens)):
	aux = removeStopwords(sentencesCleanTokens[i], language)
	sentencesClean.append(aux)

# Lemmatize text
tokens = []
for i in range(0, len(sentencesClean)):
	aux = lemmatizeText(sentencesClean[i], lemmas)
	tokens.append(aux)

tokensUnic = []
for i in range(0, len(tokens)):
	aux = sorted(set(tokens[i]))
	tokensUnic.append(aux)

# Get vocabulary of lemmas
vocabulary = getVocabulary(tokensUnic)

# Get count of each word in vocabulary
count = {}
count = getCount(tokensUnic, vocabulary)
printDictionary(count, 3)

w1 = ('economía', 'n')

p1 = (count[w1] + 0.5) / (len(sentences) + 1)

ans = {}
p1 = count[w1] / len(sentences)
for w2 in vocabulary:
	p2 = (count[w2] +  0.5) / (len(sentences) + 1)
	p1p2 = getConjunta(tokensUnic, w1, w2, len(sentences)+1)
	ans[w2] = I(p1, p2, p1p2)


l = list()
for key, val in ans.items():
	l.append((val, key))
l.sort(reverse = True)
print(l[:10])

nWord = "economia_n.txt"
nameFile = '/Users/abiga/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/Practice/12/'
createFileDic(nameFile + nWord, l)