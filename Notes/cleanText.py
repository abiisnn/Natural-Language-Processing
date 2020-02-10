
#--------------
import nltk
import re
from bs4 import BeautifulSoup

def get_text_string(fname):
	f = open(fname, econdign='utf-8')
	text_string = f.read()
	f.close()

	soup = BeautifulSoup(text_string, 'lxml')
	text_string = 