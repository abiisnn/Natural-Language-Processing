'''
Book: Dipanjan Sarkar-Text Analytics with Python
Chapter 3
'''

##################################
#		Stemming
##################################
# Snowball Stemmer
from nltk.stem import SnowballStemmer
ss = SnowballStemmer("spanish")
ss.stem('hola')