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
def getWord(s):
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
		word = getWord(text[i])
		POS = getPOS(text[i])
		polarity = getPolarity(text[i])
		pair = (word, POS)
		dictionary[word] = polarity
	return dictionary

#############################################
#############################################
#############################################
# Get dictionary
fpath = '/Users/abiga/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/Practice/25/ML-SentiCon/senticon.en.xml'
dictionary = getDictionary(fpath, 'utf-8') # Dictionary of tuples with polarity

dictionary['especial']
i = 0
for key, value in dictionary.items():
	print(key, " ", value)
	i = i + 1
	if i == 10:
		break

aux = 'If I could give you one thing in life I would give you the ability to see yourself through my eyes only then would you realize how special you are to me '
# aux = 'You are the reason I wake up with a smile on my face every morning You are never off my mind You are in my dreams and even then you are still perfect in every way .'
aux = 'I would like to tell you that you are my love my happiness my heart my home my path to follow but the most important of all you are my future'
message = nltk.word_tokenize(aux)
polarity = 0.0
cont = 0
for word in message:
	word = word.lower()
	if word in dictionary:
		print(word)
		polarity = polarity + dictionary[word]
		cont = cont + 1

print("polarity:", polarity/cont)
