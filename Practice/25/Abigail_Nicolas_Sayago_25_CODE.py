import nltk
import re
import math
import numpy as np
import mord
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn import metrics

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

#############################################
#			GETTING CORPUS TEXT
#############################################
# Return vector of vector of tuples that represent reviews
def getText(corpusRoot, code, n):
	reviews = list()
	for i in range(2, n):
		try:
			f = open(corpusRoot + str(i) + ".review.pos", encoding = code) #Cod: utf-8, latin-1
			text = f.readlines()
			tokens = [nltk.word_tokenize(line) for line in text]
			review = list()
			for line in tokens:
				if len(line) > 0:
					tag = 'n'
					if len(line) > 3:
						tag = line[2][0].lower()
					review.append((line[1], tag))
			reviews.append(review)
			f.close()
		except:
			continue
	return reviews

# Return vector of ranks
def getRank(corpusRoot, code, n):
	ranks = list()
	for i in range(2, n):
		try:
			f = open(corpusRoot + str(i) + ".xml")
			lines = f.readlines()
			j = lines[0].index( ' rank=' )
			ranks.append(int(lines[0][j+7]))
			f.close()
		except:
			continue
	return ranks

def removeStopwords(tokens, language):
	sw = stopwords.words(language)
	clean = list()
	for review in tokens:
		aux = list()
		for word in review:
			if word[0] not in sw:
				aux.append(word)
		clean.append(aux)
	return clean

#############################################
#			 	GET RESULT
#############################################
def getMatrix(dictionary, reviews, ranks):
	# Create matrix: [[], [], [], [], []]
	matrix = []
	for i in range(0, 5):
		matrix.append([])

	for i in range(len(reviews)):
		cont = 0
		polarity = 0
		for j in range(len(reviews[i])):
			if reviews[i][j] in dictionary:
				polarity = polarity + dictionary[reviews[i][j]]
				cont = cont + 1
		ans = 0
		if cont != 0:
			ans = polarity / cont
		rank = ranks[i] - 1
		matrix[rank].append(round(ans, 2))
	return matrix

# Getting polarity of every rank
def getTotalPolarity(m):
	result = [np.sum(row)/len(row) for row in m]
	return result

#############################################
#############################################
#############################################
# Get dictionary
fpath = '/Users/abiga/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/Practice/25/ML-SentiCon/es.xml'
dictionary = getDictionary(fpath, 'utf-8') # Dictionary of tuples with polarity

# Get tokens by corpus
fpath = '/Users/abiga/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/Practice/25/corpus/'
n = 4395
reviews = getText(fpath, 'ISO-8859-1', n) # List of tuples
cleanReviews = removeStopwords(reviews, 'english')
ranks = getRank(fpath, code, n)
matrix = getMatrix(dictionary, cleanReviews, ranks)
total = getTotalPolarity(matrix)

print("")
for i in range(len(matrix)):
	print("-------> RANK:", str(i+1), "  # Reviews:", len(matrix[i]), "  Total Polarity:", total[i])
	# print("\n")

print("")
for i in range(len(matrix)):
	print("-------> RANK:", str(i+1), "  # Reviews:", len(matrix[i]), "  Total Polarity:", total[i])
	print(matrix[i])
	print("\n")