import nltk
import re
import math
import numpy as np
import mord
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn import metrics


def flatten_corpus(corpus):
	return ' '.join([document.strip() for document in corpus])

def compute_ngrams(sequence, n):
	return zip(*[sequence[index:] for index in range(n)])

def get_top_ngrams(corpus, ngram_val=1, limit=5):
	corpus = flatten_corpus(corpus)
	tokens = nltk.word_tokenize(corpus)
	ngrams = compute_ngrams(tokens, ngram_val)
	ngrams_freq_dist = nltk.FreqDist(ngrams)
	sorted_ngrams_fd = sorted(ngrams_freq_dist.items(), key=itemgetter(1), reverse=True)
	sorted_ngrams = sorted_ngrams_fd[0:limit]
	sorted_ngrams = [(' '.join(text), freq) for text, freq in sorted_ngrams]
	return sorted_ngrams

text = "hello hello hello hello hello"
aux = get_top_ngrams(text)

