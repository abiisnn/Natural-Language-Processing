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










