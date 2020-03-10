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
			l = (letterToken.lower(), tag)
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
#						ARTICLES
##############################################################
def getArticles(fpath, code):
	f = open(fpath, encoding = code)
	text_string = f.read()
	f.close()

	article_segments = re.split('<h3>', text_string)
	articles = []

	for art in article_segments:
		soup = BeautifulSoup(art, 'lxml')
		text = soup.get_text()
		articles.append(text)

	return articles

# Initialize dictionary of topics
def createDictionary(listTopics):
	topics = {}
	for topic in listTopics:
		topics[topic] = 0
	return topics

# Parameters: Dictionary of topics, article
def getFrecuency(listTopics, article):
	topics = createDictionary(listTopics)

	for token in article:
		if token in topics:
			topics[token] = topics[token] + 1

	return topics

# Parameters: Dictionary of frecuency for each topic
def sumFrecuency(table, j, lenTopics):
	sum = 0
	for i in range(1, lenTopics + 1):
		sum = sum + table[i][j]
	return sum

def initializeTable(topics, lenArticles):
	table = []
	aux = []
	for i in range(0, lenArticles + 1):
		aux.append(0)
	table.append(aux)

	for i in range(1, len(topics)+1):
		aux = []
		for j in range(0, lenArticles + 1):
			aux.append(0)
		aux[0] = topics[i-1][0]
		table.append(aux)
	
	for i in range(1, lenArticles + 1):
		table[0][i] = i

	return table

def printTable(table, show):
	for i in range(0, 7):
		for j in range(0, show):
			if i == 0:
				print(table[i][j], end = " ")
			elif j == 0:
				print(table[i][j], end = " ")
			elif table[i][j] == 0:
				print('{0:.0f}'.format(table[i][j]), end = " ")
			else:
				print('{0:.1f}'.format(table[i][j]), end = " ")
		print("")

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

def printDictionary(dic, n):
	i = 0
	for j in dic:
		print(j, dic[j])
		i = i + 1
		if i > n:
			break

#Parameters: List
#Return: Nothing
def createFileTable(path, table, show):
	f = open(path, 'w')
	for i in range(0, 7):
		for j in range(0, show):
			aux = " "
			if i == 0:
				aux = str(table[i][j]) + " "
				# print(table[i][j], end = " ")
			elif j == 0:
				aux = str(table[i][j]) + " "
				# print(table[i][j], end = " ")
			elif table[i][j] == 0:
				aux = '{0:.0f}'.format(table[i][j]) + " "
				# print('{0:.0f}'.format(table[i][j]), end = " ")
			else:
				aux = '{0:.1f}'.format(table[i][j]) + " "
				# print('{0:.1f}'.format(table[i][j]), end = " ")
			f.write(aux)
		f.write('\n')
	f.close()

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

# Get articles
fpath = '/Users/abiga/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/corpus/e961024.htm'
code = 'utf-8'
articles = getArticles(fpath, code)

# print(articles[2:3])

# Tagging
fcombinedTagger = '/Users/abiga/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/pkl/combined_tagger.pkl'
#make_and_save_combined_tagger(fcombinedTagger)
articlesTag = []
for i in range(0, len(articles)):
	aux = getTokens(articles[i])
	auxTag = tag(fcombinedTagger, aux)
	articlesTag.append(auxTag)

# print(articlesTag[:2])

articleCleanTokens = []
for i in range(0, len(articlesTag)):
	aux = getCleanTokensTags(articlesTag[i])
	articleCleanTokens.append(aux)

# print(articleCleanTokens[:2])

# Remove Stopwords
language = 'spanish'
articleClean = []
for i in range(0, len(articleCleanTokens)):
	aux = removeStopwords(articleCleanTokens[i], language)
	articleClean.append(aux)

# print(articleClean[:2])

# Lemmatize text
tokens = []
for i in range(0, len(articleClean)):
	aux = lemmatizeText(articleClean[i], lemmas)
	tokens.append(aux)

# print(tokens[:2])

taux = [('-------crisis', 'n'), ('privatización', 'n'), ('contaminación', 'n'), ('-----política', 'n'), ('-----economía', 'n'), ('---tecnología', 'n'), ('----televisa', 'n')]
t = [('crisis', 'n'), ('privatización', 'n'), ('contaminación', 'n'), ('política', 'n'), ('economía', 'n'), ('tecnología', 'n'), ('televisa', 'n')]

table = initializeTable(taux, len(tokens))
# print(table)

j = 1
for article in tokens:
	dicFrecuency = getFrecuency(t, article)
	i = 1
	# total = sumFrecuency(dicFrecuency)
	for frecuency in dicFrecuency:
		table[i][j] = dicFrecuency[frecuency]
		i = i + 1
	j = j + 1

# print(table)

j = 1
for article in tokens:
	total = sumFrecuency(table, j, len(t))
	for i in range(1, len(t) + 1):
		if total != 0:
			table[i][j] = (table[i][j] * 100) / total 
		else:
			table[i][j] = 0 
	j = j + 1

# print(table)
show = 50
printTable(table, show)

show = 10
nWord = "10.txt"
nameFile = '/Users/abiga/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/Practice/16/'
createFileTable(nameFile + nWord, table, show)

show = 25
nWord = "25.txt"
nameFile = '/Users/abiga/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/Practice/16/'
createFileTable(nameFile + nWord, table, show)

show = 50
nWord = "50.txt"
nameFile = '/Users/abiga/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/Practice/16/'
createFileTable(nameFile + nWord, table, show)

show = 75
nWord = "75.txt"
nameFile = '/Users/abiga/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/Practice/16/'
createFileTable(nameFile + nWord, table, show)

show = 77
nWord = "77.txt"
nameFile = '/Users/abiga/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/Practice/16/'
createFileTable(nameFile + nWord, table, show)
