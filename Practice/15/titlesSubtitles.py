import nltk
import re
import math
from bs4 import BeautifulSoup
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import stopwords

##############################################################
#						CREATE FILE
##############################################################

#Parameters: List
#Return: Nothing
def createFile(path, titles):
	f = open(path, 'w')
	for title in titles:
		f.write(title + '\n')
	f.close()

################################################
################################################
################################################


# Read file of corpus
fpath = '/Users/abiga/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/corpus/e961024.htm'
code = 'utf-8'

f = open(fpath, encoding = code)
text = f.read()
f.close()

soup = BeautifulSoup(text, 'lxml')
titles = soup.find_all(['h3', 'b'])
# print(titles)

titles = [title.get_text() for title in titles]
titles = [title.strip() for title in titles]
# print(titles)

authors = [title for title in titles if title.isupper()] 
# print(authors)

titles = [title for title in titles if title not in authors]
# print(titles[:10])

clean = []
for title in titles:
	words = title.split()
	words = [word.strip() for word in words]
	t = ''
	for word in words:
		if word != '*':
			word.strip()
			t = t + word + " "
	clean.append(t)
print(clean)

# print(clean)
nWord = "list_titles.txt"
nameFile = '/Users/abiga/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/Practice/15/'
createFile(nameFile + nWord, clean)


