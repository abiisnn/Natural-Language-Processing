import nltk
from nltk.book import *
from nltk.collocations import *
from __future__ import division

#28-01-20
#Anaconda with python:
#Natural Language Toolkit: https://www.nltk.org/
	
	#Install NLTK
	#py -3 -m venv venv
	#venv\Scripts\activate
	#py
	import nltk
	nltk.download()
	#Escoger ALL

	#Capitulo 1 del libro
	#Descargar libros:
	from nltk.book import *
	#Aparecen en la pantalla los libros
	#Concordance: Nos interesa el contexto
	text1.concordance("monstrous")
	text5.concordance("great")
	#Solo muestra algunos
	text5.concordance("and")
	#Ambiguedad
	text7.concordance("arms")	
	text7.concordance("arm")
	#Mismo significado
	text1.concordance("terrible")
	text2.concordance("terrible")
	text5.concordance("terrible")
	text6.concordance("terrible")
	#Similar
	text1.similar("monstrous")
	text1.similar("terrible")
	text2.similar("terrible")
	text5.similar("terrible")
	text6.similar("terrible")
	#Common_contexts
	#List: x = []
	text2.common_contexts(["monstrous", "very"])
	#type(x) tipo de dato
	#Grafica
	text4.dispersion_plot(["citizens", "democracy", "freedom"])
	len(text3) #Cantidad de tokens
	type(text3)
	set(text3) #Vocabulario
	len(set(text3)) #Cantidad de tokens unicos
	sorted(set(text3))
	type(sorted(set(text3))) #Ya es una lista
		
	#31-01-20
	fdist1 = FreqDist(text1) #Es un diccionario
	type(fdist1)
	vocabulary1 = fdist1.keys()
	vocabulary1[:50] #50 palabras más frecuentes
	tuples = FreqDist.items() #Devuelve lista de tuplas
	val = FreqDist.values() #Devuelve las frecuencias
	

	#
	long_words = [w.lower() for w in vocabulary if len(w) > 15]
	print(long_words, '\n')

	#Conditional Freq. Dist
	#Measure of the lexical richness
	def lexical_diversity(text):
	return len(text) / len(set(text))

	#How often a word occurs in a text
	#percentage(text4.count('a'), len(text4))
	def percentage(count, total):
	return 100 * count / total


