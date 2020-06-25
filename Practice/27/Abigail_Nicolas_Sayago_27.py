import nltk
import re
import math
import glob
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
	j = 0
	while(ord(s[j]) != 9):
		word = word + s[j]
		j = j + 1
	return word.lower()

def isPositive(s):
	pol = s[len(s) - 4:len(s)-1]
	if pol == 'pos':
		return True
	return False

# Return dictionary of pairs with polarity
def getDictionary(corpusRoot, code):
	dictionary = {}
	f = open(corpusRoot + 'a.txt', encoding = code)
	text = f.readlines()
	f.close()

	for i in range(0, len(text)):
		word = getElement(text[i])
		pol = isPositive(text[i])
		dictionary[word] = pol
	return dictionary

#############################################
#			GETTING TEXT
#############################################
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

def get_sentences(fpath):
	positiveText, negativeText = getReviews(fpath)
	sent_tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')
	senPos = sent_tokenizer.tokenize(positiveText)
	senNeg = sent_tokenizer.tokenize(negativeText)
	return senPos, senNeg

def getTokens(text):
	tokens = nltk.word_tokenize(text)
	return tokens

def getCleanTokens(tokens):
	clean = []
	for token in tokens:
		t = []
		for char in token:
			if re.match(r'[a-záéíóúñüA-ZÁÉÍÓÚÜÑ]', char):
				t.append(char)
		letterToken = ''.join(t)
		if letterToken != '':
			clean.append(letterToken.lower())
	return clean

def removeStopwords(tokens, language):
	sw = stopwords.words(language)
	cleanTokens = []
	for tok in tokens:
		if tok not in sw:
			cleanTokens.append(tok)
	return cleanTokens

def getpreTokens(sentences):
	cleanTokens = []
	language = 'spanish'
	for i in range(0, len(sentences)):
		aux = getTokens(sentences[i])
		clean = getCleanTokens(aux)
		stop = removeStopwords(clean, language)
		cleanTokens.append(stop)
	return cleanTokens

def clean(pos, neg):
	clean_pos = getpreTokens(pos)
	clean_neg = getpreTokens(neg)
	return clean_pos, clean_neg

#############################################
#			GETTING RESULTS
#############################################
def findWords(words, tokens):
	dic_words = {}
	for word in words:
		dic_words[word] = set()

	for i in range(0, len(tokens)):
		for j in range(0, len(tokens[i])):
			if tokens[i][j] in dic_words:
				dic_words[tokens[i][j]].add(i)
	return dic_words

def prePolarity(dictionary, sentences):
	ans = []
	for i in range(0, len(sentences)):
		pos = 0
		neg = 0
		for j in range(0, len(sentences[i])):
			if sentences[i][j] in dictionary:
				if dictionary[sentences[i][j]]:
					pos = pos + 1
				else: 
					neg = neg + 1
		ans.append((pos, neg))
	return ans

def getResults(positions, pre):
	ans = []
	print("pre size: ", len(pre))
	for key, value in positions.items():
		pos = 0
		neg = 0
		for index in value:
			ans_sen = pre[index]
			pos = pos + ans_sen[0]
			neg = neg + ans_sen[1]
		ans.append((pos, neg))
	return ans
def getVocabulary(toka, tokb):
	voc = set()
	for sentence in toka:
		for word in sentence:
			voc.add(word)
	for sentence in tokb:
		for word in sentence:
			voc.add(word)
	return voc

def getFrecuency(tokpos, tokneg, vocabulary):
	dic = {}
	for word in vocabulary:
		dic[word] = 0
	for sentence in tokpos:
		for word in sentence:
			dic[word] = dic[word] + 1
	for sentence in tokneg:
		for word in sentence:
			dic[word] = dic[word] + 1
	ans = {}
	size = len(vocabulary)
	for key, value in dic.items():
		ans[key] = (value / size)
	return ans

def getWordsInSentences(positions, sentences):
	ans = {}
	for key, value in positions.items():
		ans[key] = set()

	for key, value in positions.items():
		for index in value:
			vec = set(sentences[index])
			actual = ans[key]
			actual = actual.union(vec)
			ans[key] = actual
	return ans

def getWordsProbability(word_list, dictionary, frecuency):
	frec_pos = {}
	frec_neg = {}
	for key, value in word_list.items():
		frec_pos[key] = list()
		frec_neg[key] = list()

	for key, value in word_list.items():
		for word in value:
			if word in dictionary:
				if dictionary[word]:
					frec_pos[key].append((word, frecuency[word]))
				else:
					frec_neg[key].append((word, frecuency[word]))
	for key, value in frec_pos.items():
		l = sorted(value, key=lambda tup: tup[1])
		frec_pos[key] = l
	for key, value in frec_neg.items():
		l = sorted(value, key=lambda tup: tup[1])
		frec_neg[key] = l

	return frec_pos, frec_neg

def min(a, b):
	if a < b:
		return a
	return b
def printResults(dic, n):
	for key, value in dic.items():
		aux = min(n, len(value))
		value.reverse()
		print("--->", key)
		for i in range(0, aux):
			print(value[i])

################################################
################################################
################################################
# Get dictionary of polarity
fpath = '/Users/abiga/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/Practice/27/dictionary/'
dictionary = getDictionary(fpath, 'utf-8') # Dictionary of tuples with polarity

fpath = '/Users/abiga/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/Practice/26/corpus/moviles/'
sen_positive, sen_negative = get_sentences(fpath)
tok_positive, tok_negative = clean(sen_positive, sen_negative)

word_list = ['nokia', 'pantalla', 'fotos', 'memoria', 'batería']

polarity_rev_pos = prePolarity(dictionary, tok_positive)
polarity_rev_neg = prePolarity(dictionary, tok_negative)

positions_words_pos = findWords(word_list, tok_positive)
positions_words_neg = findWords(word_list, tok_negative)

result_pos = getResults(positions_words_pos, polarity_rev_pos)
result_neg = getResults(positions_words_neg, polarity_rev_neg)

print("\nPolarity in positive reviews:")
print(word_list)
print(result_pos)

print("\nPolarity in negative reviews:")
print(word_list)
print(result_neg)

vocabulary = getVocabulary(tok_positive, tok_negative)
frecuency_voc = getFrecuency(tok_positive, tok_negative, vocabulary)

words_pos_set = getWordsInSentences(positions_words_pos, tok_positive)
words_neg_set = getWordsInSentences(positions_words_neg, tok_negative)

result_prob_pos, result_prob_neg = getWordsProbability(words_pos_set, dictionary, frecuency_voc)

print("Positive words with best probability in positive reviews")
printResults(result_prob_pos, 5)

print("Negative words with best probability in positive reviews")
printResults(result_prob_neg, 5)

result_prob_pos, result_prob_neg = getWordsProbability(words_neg_set, dictionary, frecuency_voc)
print("Positive words with best probability in negative reviews")
printResults(result_prob_pos, 5)

print("Negative words with best probability in negative reviews")
printResults(result_prob_neg, 5)