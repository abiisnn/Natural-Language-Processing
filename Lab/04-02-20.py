import nltk
from nltk.book import *
from __future__ import division

#Generar las graficas de la distribucion de frecuencia 
#de 10000 palabras del vocabulario de text5
#Grafica acumulativa

fdist1 = FreqDist(text5) #Es un diccionario
fdist1.plot(1000)
fdist1.plot(1000, cumulative=True)
#Podemos observar que hay pocas palabras con alta frecuencia
#y muchas palabras con poca frecuencia, eso se llama frecuencia de Zipf

#Imprimir palabras de text5 cuya longitud es > 30
vocabulary = set(text5)
l = [w for w in vocabulary if len(w) > 30]
print(l)


# Ditribución para Brown Corpus
from nltk.corpus import brown
brown.categories()

#Conditional frequency distributions
cfd = nltk.ConditionalFreqDist(
			(genre, word)
			for genre in brown.categories()
			for word in brown.words(categories=genre))

genres = ['news', 'romance', 'humor']
modals = ['love', 'hate', 'speak', 'control', 'feel', 'great', 'president']
cfd.tabulate(conditions=genres, samples=modals)

#Imprimir los numeros de synsets de palabras: 
#Computer, machine, car, sandwich

from nltk.corpus import wordnet as wn 
computer = wn.synsets('computer')
machine = wn.synsets('machine')
car = wn.synsets('car')
sandwich = wn.synsets('sandwich')

#Similitud entre compute y machine
# car y machine, car y computer
# computer y sand

computer = wn.synset('computer.n.01')
machine = wn.synset('machine.n.01')
car = wn.synset('car.n.01')
sandwich = wn.synset('sandwich.n.01')
computer.path_similarity(machine)
car.path_similarity(machine)
car.path_similarity(computer)
computer.path_similarity(sandwich)