# # # # # # # # # # # # # # # 
# 	Gutenberg Corpus
# # # # # # # # # # # # # # # 

import nltk
nltk.corpus.gutenberg.fileids()

emma = nltk.corpus.gutenberg.words('austen-emma.txt')
len(emma)

# # # # # #

from nltk.corpus import gutenberg
gutenberg.fileids()

for fileid in gutenberg.fileids():
	num_chars = len(gutenberg.raw(fileid))
	num_words = len(gutenberg.words(fileid))
	num_sents = len(gutenberg.sents(fileid))
	num_vocab = len(set([w.lower() for w in gutenberg.words(fileid)]))
	print(int(num_chars/ num_words), int(num_words/num_sents))

macbeth_sentences = gutenberg.sents('shakespeare-macbeth.txt')

# # # # # # # # # # # # # # # 
# 	Web and Chat Text
# # # # # # # # # # # # # # # 

from nltk.corpus import webtext
for fileid in webtext.fileids():
	print(fileid, webtext.raw(fileid)[:65], '...')

# # # # # #

from nltk.corpus import nps_chat
chatroom = nps_chat.posts('10-19-20s_706posts.xml')
chatroom[123]

# # # # # # # # # # # # # # # 
# 	Brown Corpus
# # # # # # # # # # # # # # # 
from nltk.corpus import brown
brown.categories()
brown.words(categories='news')
brown.words(fileids=['cg22'])
brown.sents(categories=['news', 'editorial', 'reviews'])

news_text = brown.words(categories='news')
fdist = nltk.FreqDist([w.lower() for w in news_text])
modals = ['can', 'could', 'may', 'might', 'must', 'will']

for m in modals:
	print(m + ':', fdist[m])

#Conditional frequency distributions
cfd = nltk.ConditionalFreqDist(
			(genre, word)
			for genre in brown.categories()
			for word in brown.words(categories=genre))

genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
modals = ['can', 'could', 'may', 'might', 'must', 'will']
cfd.tabulate(conditions=genres, samples=modals)

# # # # # # # # # # # # # # # 
# 	Reuters Corpus
# # # # # # # # # # # # # # # 
from nltk.corpus import reuters
reuters.fileids()
reuters.categories()

reuters.categories('training/9865')
reuters.categories(['training/9865', 'training/9880'])
reuters.fileids('barley')
reuters.fileids(['barley', 'corn'])

reuters.words('training/9865')[:14]
reuters.words(['training/9865', 'training/9880'])
reuters.words(categories='barley')
reuters.words(categories=['barley', 'corn'])

# # # # # # # # # # # # # # # # # # # # 
# 	Inaugural Address Corpus
# # # # # # # # # # # # # # # # # # # #
from nltk.corpus import inaugural
inaugural.fileids()
[fileid[:4] for fileid in inaugural.fileids()]

cfd = nltk.ConditionalFreqDist(
			(target, fileid[:4])
			for fileid in inaugural.fileids()
			for w in inaugural.words(fileid)
			for target in ['america', 'citizen']
			if w.lower().startswith(target))
cfd.plot()

# # # # # # # # # # # # # # # # # # # # 
# 	Corpora int Other Languages
# # # # # # # # # # # # # # # # # # # #
nltk.corpus.cess_esp.words()
nltk.corpus.floresta.words()
nltk.corpus.indian.words('hindi.pos')
nltk.corpus.udhr.fileids()
nltk.corpus.udhr.words('Javanese-Latin1')[11:]

from nltk.corpus import udhr
languages = ['Chickasaw', 'English', 'German_Deutsch', 'Greenlandic_Inuktikut', 'Hungarian_Magyar', 'Ibibio_Efik']
cfd = nltk.ConditionalFreqDist(
			(lang, len(word))
			for lang in languages
			for word in udhr.words(lang + '-Latin1'))
cfd.plot(cumulative=True)

#Exercise
raw_text = udhr.raw('Romani-Latin1')
nltk.FreqDist(raw_text).plot()

# # # # # # # # # # # # # # # # # # # # 
# 	Text Corpus Structure
# # # # # # # # # # # # # # # # # # # #

#Gutenberg
raw = gutenberg.raw('burgess-busterbrown.txt')
raw[1:20]
words = gutenberg.words('burgess-busterbrown.txt')
words[1:20]
sents = gutenberg.sents('burgess-busterbrown.txt')
sents[1:20]

#With own local copy
from nltk.corpus import BracketParseCorpusReader
corpus_root = r"C:\copora\penntreebank\parsed\mrg\wsj"
file_pattern = r".*/wsj_.*\.mrg"
ptb = BracketParseCorpusReader(corpus_root, file_pattern)
ptb.fileids()
len(ptb.sents())
ptb.sents(fileids='20/wsj_2013.mrg')[19]

# # # # # # # # # # # # # # # # # # # # 
# 	Counting Words by Genre
# # # # # # # # # # # # # # # # # # # #

from nltk.corpus import brown
cfd = nltk.ConditionalFreqDist(
			(genre, word)
			for genre in brown.categories()
			for word in brown.words(categories=genre))

#For categories: news and romance
genre_word = [(genre, word)
				for genre in ['news', 'romance']
				for word in brown.words(categories=genre)]
len(genre_word)
genre_word[:4]
genre_word[-4:]

cfd = nltk.ConditionalFreqDist(genre_word)
cfd
cfd.conditions()

cfd['news']
cfd['romance']
list(cfd['romance'])
cfd['romance']['could']

# # 
from nltk.corpus import udhr
languages = ['Chickasaw', 'English', 'German_Deutsch', 'Greenlandic_Inuktikut', 'Hungarian_Magyar', 'Ibibio_Efik']
cfd = nltk.ConditionalFreqDist(
			(lang, len(word))
			for lang in languages
			for word in udhr.words(lang + '-Latin1'))
cfd.plot(cumulative=True)
cfd.tabulate(conditions=['English', 'German_Deutsch'],
	samples=range(10), cumulative=True)

# # # # # # # # # # # # # # # # # # # # # # # # # 
# 	Generating Random Text with Bigrams
# # # # # # # # # # # # # # # # # # # # # # # # #
sent = ['In', 'the', 'beginning', 'God', 'created', 'the', 'heaven', 'and', 'the', 'earth', '.']
nltk.bigrams(sent)

#Generating random text
def generate_model(cfdist, word, num=15):
	for i in range(num):
		print (word)
		word = cfdist[word].max()

text = nltk.corpus.genesis.words('english-kjv.txt')
bigrams = nltk.bigrams(text)
cfd = nltk.ConditionalFreqDist(bigrams)

print(cfd['living'])

# # # # # # # # # # # # # # # # # # # # # # # # # 
# 					Functions
# # # # # # # # # # # # # # # # # # # # # # # # #
from __future__ import division

#Measure of the lexical richness
def lexical_diversity(text):
	return len(text) / len(set(text))

#How often a word occurs in a text
#percentage(text4.count('a'), len(text4))
def percentage(count, total):
	return 100 * count / total

#Tries to work out the plural form of any English noun
def plural(word):
	if word.endswith('y'):
		return word[:-1] + 'ies'
	elif word[-1] in 'sx' or word[-2:] in ['sh', 'ch']:
		return word + 'es'
	elif word.endswith('an'):
		return word[:-2] + 'en'
	else:
		return word + 's'

plural('fairy')
plural('woman')

# # # # # # # # # # # # # # # # # # # # # # # # # 
# 					Modules
# # # # # # # # # # # # # # # # # # # # # # # # #
#File called: text.py
from textpro import plural
plural('wish')
plural('fan')

# # # # # # # # # # # # # # # # # # # # # # # # # 
# 			Lexical Resources
# # # # # # # # # # # # # # # # # # # # # # # # #
#Filtering a text
def unusual_words(text):
	text_vocab = set(w.lower() for w in text if w.isalpha())
	english_vocab = set(w.lower() for w in nltk.corpus.words.words())
	unusual = text_vocab.difference(english_vocab)
	return sorted(unusual)

a = unusual_words(nltk.corpus.gutenberg.words('austen-sense.txt'))

from nltk.corpus import stopwords
stopwords.words('english')

#Fraction of words in a text are not in the stopwords
def content_fraction(text):
	stopwords = nltk.corpus.stopwords.words('english')
	content = [w for w in text if w.lower() not in stopwords]
	return len(content) / len(text)

content_fraction(nltk.corpus.reuters.words())

#Puzzle
puzzle_letters = nltk.FreqDist('egivrvonl')
obligatory = 'r'
wordlist = nltk.corpus.words.words()
[w for w in wordlist if len(w) >= 6
					 and obligatory in w 
					 and nltk.FreqDist(w) <= puzzle_letters]

#Ambiguous names
names = nltk.corpus.names
names.fileids()
male_names = names.words('male.txt')
female_names = names.words('female.txt')
ambiguous = [w for w in male_names if w in female_names]
ambiguous[:10]

#Names ending in the letter a
cfd = nltk.ConditionalFreqDist(
			(fileid, name[-1])
			for fileid in names.fileids()
			for name in names.words(fileid))
cfd.plot()

# # # # # # # # # # # # # # # # # # # # # # # # # 
# 			A Pronouncing Dictionary
# # # # # # # # # # # # # # # # # # # # # # # # #
#For each word, this lexicon provides a list of phonetic codes
entries = nltk.corpus.cmudict.entries()
len(entries)
for entry in entries[39943:39951]:
	print(entry)

#Looking for entries whose pronunciation consists of three phones
for word, pron in entries:
	if len(pron) == 3:
		ph1, ph2, ph3 = pron
		if ph1 == 'P' and ph3 == 'T':
			print(word, ph2)

#Find all words whose pronunciation ends with a syllable sounding like: nicks
syllable = ['N', 'IH0', 'K', 'S']
soundNick = [word for word, pron in entries if pron[-4:] == syllable]
soundNick[:10]

#Pronunciation with M
sound = [w for w, pron in entries if pron[-1] == 'M' and w[-1] == 'n']
sound

#Pronunctiarion with N
sound = sorted(set(w[:2] for w, pron in entries if pron[0] == 'N' and w[0] != 'n'))
sound

#Extract the stress digits
def stress(pron):
	return [char for phone in pron for char in phone if char.isdigit()]

a = [w for w, pron in entries if stress(pron) == ['0', '1', '0', '2', '0']]
a[:10]
a = [w for w, pron in entries if stress(pron) == ['0', '2', '0', '1', '0']]
a[:10]

#Find all the p words consisting of 3 sounds and group
#them according to their first and last sounds.

p3 = [(pron[0] + '-' + pron[2], word)
	   for (word, pron) in entries
	   if pron[0] == 'P' and len(pron) == 3]
cfd = nltk.ConditionalFreqDist(p3)
for template in cfd.conditions():
	words = cfd[template].keys()
	wordlist = ' '.join(words)
	print(template, wordlist[:70] + "...")   

#Using dictionary
prondict = nltk.corpus.cmudict.dict()
prondict['fire']

prondict['blog'] #ERROR
prondict['blog'] = [['B', 'L', 'AA1', 'G']]
prondict['blog']

#Text-to-speech function
text = ['natural', 'language', 'processing']
a = [ph for w in text for ph in prondict[w][0]]
a

# # # # # # # # # # # # # # # # # # # # # # # # # 
# 			Comparative Wordlists
# # # # # # # # # # # # # # # # # # # # # # # # #

#Common words in several languages
from nltk.corpus import swadesh
swadesh.fileids()

swadesh.words('en')
#Cognate words
es2en = swadesh.entries(['es', 'en']) #Spanish-English
fr2en[:10]
translate = dict(fr2en)
translate['volar']

#Update dictionary
de2en = swadesh.entries(['de', 'en']) #German-English
fr2en = swadesh.entries(['fr', 'en']) #Fra-English
translate.update(dict(de2en))
translate.update(dict(fr2en))

translate['Hund']
translate['perro']

#Compare words in various languages

languages = ['en', 'de', 'nl', 'es', 'fr', 'pt', 'la']
for i in [139, 140, 141, 142]:
	print(swadesh.entries(languages)[i])

# # # # # # # # # # # # # # # # # # # # # # # # # 
# 	Shoebox and Toolbox Lexicons
# # # # # # # # # # # # # # # # # # # # # # # # #

from nltk.corpus import toolbox
a = toolbox.entries('rotokas.dic')
a[:10]

# # # # # # # # # # # # # # # # # # # # # # # # # 
# 					WordNet
# # # # # # # # # # # # # # # # # # # # # # # # #

#Senses and Synonyms
from nltk.corpus import wordnet as wn 
wn.synsets('motocar')

wn.synset('car.n.01').lemma_names
wn.synset('car.n.01').definition
wn.synset('car.n.01').examples
wn.synset('car.n.01').lemmas