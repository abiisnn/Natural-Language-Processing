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
def getWords(fpath, code):
	f = open(fpath, encoding = code) #Cod: utf-8, latin-1
	text = f.read()
	f.close()

	words = re.sub(" ", " ",  text).split()
	return words

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
#					MOST COMMON WORDS
##############################################################
# Parameter: vocabulary and tokens
# Return: dictionary with frecuency of each word in vocabulary
def getFrecuency(vocabulary, tokens):
	mostCommon = {}
	for v in vocabulary:
		mostCommon[v] = 0

	for token in tokens:
		mostCommon[token] = mostCommon[token] + 1
	
	return mostCommon

# Getting frecuency with TF - IDF
def getTFIDF(vocabulary, frecuency, k, sizeText):
	mostCommon = {}
	for v in vocabulary:
		mostCommon[v] = 0

	for token in vocabulary:
		mostCommon[token] = (((k + 1) * frecuency[token]) / (frecuency[token] + k)) * math.log((sizeText / frecuency[x]))
	return mostCommon

##############################################################
#							SORT
##############################################################
# Parameters: dictionary
# Return: list of elements sorted Highest to Lower
def sortHL(dic):
	l = list()
	for key, val in dic.items():
		l.append((val, key))
	l.sort(reverse = True)
	return l

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

##############################################################
#							PKL
##############################################################
def makePKL(fname, aux):
	output = open(fname, 'wb')
	dump(aux, output, -1)
	output.close()

def getPKL(fname):
	input = open(fname, 'rb')
	aux = load(input)
	input.close()
	return aux

################################################
################################################
################################################

# Get Tokens by Generate.txt to create dictionary of lemmas
fpathLemmas = '/Users/abiga/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/Normalize/generateClean.txt'
fpathName = 'generateClean.txt'
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
fpath = '/Users/abiga/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/corpus/e961024.htm'
code = 'utf-8'
textSource = getText(fpath, code) 
tokensHtml = getTokens(textSource) #Get tokens with out html tags
# print("Text with tags, stopwords and punctuation:")
print(tokensHtml[:10])

# Tagging
fcombinedTagger = '/Users/abiga/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/pkl/combined_tagger.pkl'
#make_and_save_combined_tagger(fcombinedTagger)
textTagged = tag(fcombinedTagger, tokensHtml)
print(textTagged[:10])

# Text in list with lists of size 2
cleanTokens = getCleanTokensTags(textTagged)
# print("text with tags corrected:")
print(cleanTokens[:10])

# Remove Stopwords
language = 'spanish'
tokens = removeStopwords(cleanTokens, language)
# print("Text without stopwords:")
print(tokens[:100])

# Lemmatize text
tokens = lemmatizeText(tokens, lemmas)
print(tokens[:100])

# Get vocabulary of lemmas
vocabulary = getVocabulary(tokens)
print("vocabulary:")
print(vocabulary[:10])

frecuency = {}
frecuency = getFrecuency(vocabulary, tokens)

k = 1.2
mostCommon = getTFIDF(vocabulary, frecuency, k, len(tokens))

printDictionary(mostCommon, 10)

l = []
l = sortHL(mostCommon)
print(l[:10])

nWord = "mostCommonWords.txt"
nameFile = '/Users/abiga/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/Practice/14/'
createFileDic(nameFile + nWord, l)