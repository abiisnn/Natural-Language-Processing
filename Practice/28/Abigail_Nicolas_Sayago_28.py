from gensim.summarization.summarizer import summarize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import PlaintextCorpusReader
from scipy.sparse.linalg import svds
from nltk.corpus import stopwords
from nltk.corpus import cess_esp
from bs4 import BeautifulSoup
from pickle import dump, load
import pandas as pd
import numpy as np
import nltk
import glob
import re
import math
import glob

#############################################
#			GETTING DICTIONARY
#############################################
def getElement(s):
	word = ''
	j = s.find('">')
	j = j + 3
	while(ord(s[j]) != 9):
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
	positive = ""
	negative = ""
	for name in names:
		f = open(name, encoding = 'ISO-8859-1')
		rev = f.read()
		f.close()
		j = name.find("yes")
		if j != -1:
			positive = positive + rev + " "
		else:
			negative = negative + rev + " "
	return positive, negative

def get_sentences(positiveText, negativeText):
	sent_tokenizer = nltk.data.load('nltk:tokenizers/punkt/spanish.pickle')
	senPos = sent_tokenizer.tokenize(positiveText)
	senNeg = sent_tokenizer.tokenize(negativeText)
	return senPos, senNeg

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

def cleanAll(fcombinedTagger, sentences):
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
	return tokens
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
def makePKL(fname, aux):
	output = open(fname, 'wb')
	dump(aux, output, -1)
	output.close()

def getPKL(fname):
	input = open(fname, 'rb')
	aux = load(input)
	input.close()
	return aux

############################### TEXT OF EACH WORD
def findWords(words, tokens):
	dic_words = {}
	for word in words:
		dic_words[word] = set()

	for i in range(0, len(tokens)):
		for j in range(0, len(tokens[i])):
			if tokens[i][j] in dic_words:
				dic_words[tokens[i][j]].add(i)
	return dic_words

def getSentencesText(sentences, dic, word):
	text = ""
	for index in dic[word]:
		text = text + sentences[int(index)] + " "
	return text 


################################3

def normalize_document(doc):
    doc = doc.lower()
    doc = doc.strip()
    tokens = nltk.word_tokenize(doc)
    clean = []
    for token in tokens:
    	t = []
    	for char in token:
    		if re.match(r'[a-záéíóúñüA-ZÁÉÍÓÚÜÑ]', char):
    			t.append(char)
    	letterToken = ''.join(t)
    	if letterToken != '':
    		clean.append(letterToken.lower())

    sw = stopwords.words('spanish')
    filtered_tokens = [token for token in tokens if token not in sw]
    doc = ' '.join(filtered_tokens)
    return doc

# LSA
def low_rank_svd(matrix, singular_count=2):
    u, s, vt = svds(matrix, k=singular_count)
    return u, s, vt

def showResult(document, num_sentences):
	sentences = nltk.sent_tokenize(document)
	normalize_corpus = np.vectorize(normalize_document)
	norm_sentences = normalize_corpus(sentences)

	tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
	dt_matrix = tv.fit_transform(norm_sentences)
	dt_matrix = dt_matrix.toarray()
	vocab = tv.get_feature_names()
	td_matrix = dt_matrix.T

	##### Gensim
	print("-------------------------------- GENSIM")
	gensim = summarize(text, ratio=0.1, split=True)
	gensim = gensim[:num_sentences]
	print('\n'.join(gensim))

	#### LSA:
	print("------------------------------- LSA")
	num_topics = 3
	u, s, vt = low_rank_svd(td_matrix, singular_count=num_topics)
	term_topic_mat, singular_values, topic_document_mat = u, s, vt

	sv_threshold = 0.5
	min_sigma_value = max(singular_values) * sv_threshold
	singular_values[singular_values < min_sigma_value] = 0
	salience_scores = np.sqrt(np.dot(np.square(singular_values), 
                         np.square(topic_document_mat)))
	top_sentence_indices = (-salience_scores).argsort()[:num_sentences]
	top_sentence_indices.sort()
	print('\n'.join(np.array(sentences)[top_sentence_indices]))

	print("------------------------------- TEXT RANK")
	num_topics = 3
	similarity_matrix = np.matmul(dt_matrix, dt_matrix.T)
	similarity_graph = networkx.from_numpy_array(similarity_matrix)

	scores = networkx.pagerank(similarity_graph)
	ranked_sentences = sorted(((score, index) for index, score in scores.items()), reverse=True)
	
	top_sentence_indices = [ranked_sentences[index][1] 
	                        for index in range(num_sentences)]
	top_sentence_indices.sort()
	print('\n'.join(np.array(sentences)[top_sentence_indices]))
	


# Get Tokens by Generate.txt to create dictionary of lemmas
fpathLemmas = '/Users/abiga/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/Normalize/generateClean.txt'
code = 'ISO-8859-1'
textLemmas = getWords(fpathLemmas, code)
lemmas = {}
lemmas = createDicLemmas(textLemmas)

kind = 'moviles'
path = '/Users/abiga/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/Practice/26/corpus/'

positiveText, negativeText = getReviews(path+kind)
pos_sentences, neg_sentences = get_sentences(positiveText, negativeText)
# Tagging
fcombinedTagger = '/Users/abiga/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/pkl/combined_tagger.pkl'
pos_tok = cleanAll(fcombinedTagger, pos_sentences)
neg_tok = cleanAll(fcombinedTagger, neg_sentences)

words = [('nokia', 'n'), ('pantalla', 'n'), ('batería', 'n'), ('memoria', 'n'), ('gama', 'n'), ('cámara', 'n')]

pos_dic = findWords(words, pos_tok)
neg_dic = findWords(words, neg_tok)

for word in words:
	print("=============================> POSITIVAS" + word[0])
	text = getSentencesText(pos_sentences, pos_dic, word)
	showResult(text, 4)
	print("=============================> NEGATIVAS" + word[0])
	text = getSentencesText(neg_sentences, neg_dic, word)
	showResult(text, 4)