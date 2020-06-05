import nltk
import re
import math
from bs4 import BeautifulSoup
from pickle import dump, load
from nltk.corpus import cess_esp
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import stopwords

#############################################
#			GETTING DICTIONARY
#############################################
def getElement(s):
	word = ''
	j = s.find('">')
	j = j + 3
	while(s[j] != '<'):
		word += s[j]
		j = j + 1
	word = word[:len(word)-1]	
	return word

# Return a string
def getPOS(s):
	j = s.find('pos=')
	j = j + 5
	tag = ''
	while(s[j] != '"'):
		tag = tag + s[j]
		j = j + 1
	return tag

# Return a number that represent polarity
def getPolarity(s):
	j = s.find('pol=')
	j = j + 5
	pol = ''
	while(s[j] != '"'):
		pol = pol + s[j]
		j = j + 1
	pol = float(pol)
	return pol

# Return dictionary of pairs with polarity
def getDictionary(corpusRoot, code):
	dictionary = {}
	f = open(corpusRoot, encoding = code)
	text = f.readlines()
	f.close()

	for i in range(0, len(text)):
		word = getElement(text[i])
		POS = getPOS(text[i])
		polarity = getPolarity(text[i])
		pair = (word, POS)
		dictionary[pair] = polarity
	return dictionary

##############################################################
#					NORMALIZE TEXT
##############################################################
def getReviews(path):
	names = [f for f in glob.glob(path + "**/*.txt", recursive=True)]
	text = ""
	for name in names:
		f = open(name, encoding = 'ISO-8859-1')
		rev = f.read()
		f.close()
		text = text + rev + " "
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
			letterToken = letterToken.lower()
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

def getWords(fpath, code):
	f = open(fpath, encoding = code) #Cod: utf-8, latin-1
	text = f.read()
	f.close()

	words = re.sub(" ", " ",  text).split()
	return words
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
#						CREATE FILE
##############################################################
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

def get_sentences(fname):
    text_string = getReviews(fname)
    sent_tokenizer=nltk.data.load('nltk:tokenizers/punkt/english.pickle')
    sentences = sent_tokenizer.tokenize(text_string)
    return sentences

############################
#			POLARITY
############################
def getPolaritySen(dictionary, sentence):
	pol = 0
	cont = 0
	for word in sentence:
		if word in dictionary:
			pol = pol + dictionary[word]
			cont = cont + 1
	if cont:
		pol = pol / cont
	return pol

def getDictionaryPolarity(tokens, dictionary):
	dic = {}
	for i in range(0, len(tokens)):
		dic[i] = getPolaritySen(dictionary, tokens[i])
	return dic

def findWords(words, tokens):
	dic_words = {}
	for word in words:
		dic_words[word] = set()

	for i in range(0, len(tokens)):
		for j in range(0, len(tokens[i])):
			if tokens[i][j] in dic_words:
				dic_words[tokens[i][j]].add(i)
	return dic_words

def calculateTotalPolarity(list_words, words, polarity):
	ans = {}
	for word in list_words:
		total = 0
		cont = 0
		for i in words[word]:
			total = total + polarity[i]
			cont = cont + 1
		if cont:
			total = total / cont
		ans[word] = total
	return ans

################################################
################################################
################################################

# Get Tokens by Generate.txt to create dictionary of lemmas
fpathLemmas = '/Users/abiga/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/Normalize/generateClean.txt'
code = 'ISO-8859-1'
textLemmas = getWords(fpathLemmas, code)
lemmas = {}
lemmas = createDicLemmas(textLemmas)
# Get dictionary of polarity
fpath = '/Users/abiga/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/Practice/25/ML-SentiCon/es.xml'
dictionary = getDictionary(fpath, 'utf-8') # Dictionary of tuples with polarity

# Read file of corpus and get Sentences
kind = 'moviles'
words = [('nokia', 'n'), ('pantalla', 'n'), ('batería', 'n'), ('memoria', 'n'), ('gama', 'n'), ('cámara', 'n')]

path = '/Users/abiga/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/Practice/26/corpus/'
sentences = get_sentences(path+kind) # 755
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

sentences_polarity = getDictionaryPolarity(tokens, dictionary)
polarity_words = findWords(words, tokens)
result = calculateTotalPolarity(words, polarity_words, sentences_polarity)

print("\n\n Polarity of " + kind)
printDictionary(result, 10)