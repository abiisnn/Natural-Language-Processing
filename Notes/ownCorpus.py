#Epigrama
#C:\Users\27AGO2019\Desktop\AbiiSnn\GitHub\Natural-Language-Processing\corpus\e961024.htm
#C:\Users\27AGO2019\Desktop\AbiiSnn\GitHub\Natural-Language-Processing\corpus\e970428.htm
#C:\Users\27AGO2019\Desktop\AbiiSnn\GitHub\Natural-Language-Processing\corpus\e980315.htm

#Own corpus way 1
from nltk.corpus import PlaintextCorpusReader
corpus_root = '/Users/27AGO2019/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/corpus'
text = PlaintextCorpusReader(corpus_root, '.*')
text.fileids() #Muestra los archivos de tu ruta
words = text.words('e961024.htm')
words = list(words) #Convertir a lista de palabras


#Own corpus way 2
import nltk
path = '/Users/27AGO2019/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/corpus/e961024.htm'
f = open(path, encoding = 'utf-8') #Cod: utf-8, latin-1
text_string = f.read() 
f.close()

tokens = nltk.word_tokenize(text_string)
text = nltk.Text(tokens)
print(text[:100]) #Separo los simbolos

text.concordance('actividad')
text.similar('actividad')

print(type(text))
print(len(text))


# HTML
from bs4 import BeautifulSoup
soup = BeautifulSoup(text_string, 'lxml')
text = soup.get_text()
type(text)

tokens = nltk.word_tokenize(text)
tokens

print(tokens[:100])

#Write clear text in other file 
path = '/Users/27AGO2019/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/corpus/e961024_clear.txt'
f = open(path, encoding = 'utf-8')
f.write(text)
f.close()