import nltk
import re
import math
from pickle import dump, load
from bs4   import BeautifulSoup
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
	return nltk.Text(tokens)	

#Parameters: List of tokens
#Return: List of clean tokens
def getCleanTokens(tokens):
	clean = []
	for token in tokens:
		t = []
		for char in token:
			if re.match(r'[a-záéíóúñüA-ZÁÉÍÓÚÜÑ]', char):
				t.append(char)
		letterToken = ''.join(t)
		if letterToken != '':
			clean.append(letterToken)
	return clean

#Parameters: List of clean tokens, language of stopwords
#Return: List of tokens without stopwords
def removeStopwords(tokens, language):
	sw = stopwords.words(language)

	cleanTokens = []
	for tok in tokens:
		if tok not in sw:
			cleanTokens.append(tok)
	return cleanTokens

#Parameters: List of normalize tokens
#Return: Set, vocabulary
def getVocabulary(tokens):
	vocabulary = sorted(set(tokens))
	return vocabulary

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

def tagToLower(text):
	taggedText = []
	for token in text:
		t = (token[0], token[1].lower())
		taggedText.append(t)
	return taggedText

################################################
################################################
################################################

fpath = '/Users/27AGO2019/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/corpus/e961024.htm'
fcombinedTagger = '/Users/27AGO2019/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/pkl/combined_tagger.pkl'
nameFile = '/Users/27AGO2019/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/Normalize/similitud4.txt'

# Normalize text
language = 'spanish'
code = 'utf-8'
textSource = getText(fpath, code)
tokensHtml = getTokens(textSource)
cleanTokens = getCleanTokens(tokensHtml)
tokens = removeStopwords(cleanTokens, language)
print("Tokens:")
print(tokens[:10])

# Tagging text
#make_and_save_combined_tagger(fcombinedTagger)
#textTagged = tag(fcombinedTagger, tokens)
textTagged = tag(fcombinedTagger, tokens)

textTagged = tagToLower(textTagged)
print(textTagged[:10])