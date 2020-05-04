import nltk
import re
import math
from bs4 import BeautifulSoup
from pickle import dump, load
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

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

#############################################
#############################################
#############################################

#Read file spam/ham
fpathCorpus = '/Users/abiga/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/Practice/18/corpus.txt'
code = 'ISO-8859-1'
corpus = getText(fpathCorpus, code)
tokens = getTokens(corpus)

#Matrix and Y
xaux = list()
yaux = list()
xi = ""
for token in tokens:
	if token != 'spam' and token != 'ham':
		xi += token + " "
	else:
		typeMessage = 0
		if token == 'ham':
			typeMessage = 1
		yaux.append(typeMessage) 
		xaux.append(xi) #Create X
		xi = ""
x = np.array(xaux)
y = np.array(yaux)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(x)
matrix_X = X.toarray()
matrix_Y = y
# Split the data into training/testing sets
size = -500 
matrix_X_train = matrix_X[:size]
matrix_X_test = matrix_X[size:]
matrix_y_train = matrix_Y[:size]
matrix_y_test = matrix_Y[size:]

clf = LogisticRegression(random_state = 0) #Linear regression object
clf.fit(matrix_X_train, matrix_y_train) #Training the model
matrix_y_prediction = clf.predict(matrix_X_test) #Making predctions
print('########## Values ##########')
for i in range(0, 20):
    print("Real: %.2f" % matrix_y_test[i], "Prediction: %.2f" % matrix_y_prediction[i])