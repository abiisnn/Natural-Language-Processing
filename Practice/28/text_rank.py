from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds
from nltk.corpus import stopwords
import pandas as pd
import nltk
import numpy as np
import re
DOCUMENT = """
Los elefantes o elefántidos (Elephantidae) 
son una familia de mamíferos placentarios del orden Proboscidea. 
Antiguamente se clasificaban, junto con otros mamíferos de piel gruesa, 
en el orden, ahora inválido, de los paquidermos (Pachydermata). 
Existen hoy en día tres especies y diversas subespecies. 
Entre los géneros extintos de esta familia destacan los mamuts.
Elefante salvaje indio del bosque de Marayoor, Munnar, Kerala
Cría de elefante africano de sabana (Loxodonta africana), 
parque nacional Kruger, Sudáfrica
Los elefantes son los animales terrestres más grandes que 
existen en la actualidad. El periodo de gestación es de veintidós meses, 
el más largo en cualquier animal terrestre. 
El peso al nacer usualmente es 118 kg. 
Normalmente viven de cincuenta a setenta años, 
pero registros antiguos documentan edades máximas de ochenta y dos años. 
El elefante más grande que se ha cazado, de los que se tiene registro, 
pesó alrededor de 11 000 kg (Angola, 1956), 
alcanzando una altura en la cruz de 3,96 m, un metro más alto que el elefante 
africano promedio. El elefante más pequeño, de alrededor del tamaño de una cría o un 
cerdo grande, es una especie prehistórica que existió en la isla de Creta, 
Elephas creticus, durante el Pleistoceno.
Con un peso de 5 kg, el cerebro del elefante es el más grande de los 
animales terrestres. Se le atribuyen una gran variedad de comportamientos 
asociados a la inteligencia como el duelo, altruismo, adopción, juego, 
uso de herramientas, compasión y autorreconocimiento. 
Los elefantes pueden estar a la par con otras especies inteligentes como los cetáceos
y algunos primates. Las áreas más grandes en su cerebro están encargadas de la audición, 
el gusto y la movilidad.
"""
sentences = nltk.sent_tokenize(DOCUMENT)
len(sentences)

stop_words = nltk.corpus.stopwords.words('english')

def normalize_document(doc):
    # lower case and remove special characters\whitespaces

    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = nltk.word_tokenize(doc)
    clean = []
    for token in tokens:
    	t = []
    	for char in token:
    		if re.match(r'[a-záéíóúñüA-ZÁÉÍÓÚÜÑ]', char):
    			t.append(char)
    	letterToken = ''.join(t)
    	if letterToken != '':
    		clean.append(letterToken.lower())

    # filter stopwords out of document
    sw = stopwords.words('spanish')
    filtered_tokens = [token for token in tokens if token not in sw]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc

normalize_corpus = np.vectorize(normalize_document)
norm_sentences = normalize_corpus(sentences)

norm_sentences[:3]


#############
tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
dt_matrix = tv.fit_transform(norm_sentences)
dt_matrix = dt_matrix.toarray()

vocab = tv.get_feature_names()
td_matrix = dt_matrix.T


############ TEXT RANK
similarity_matrix = np.matmul(dt_matrix, dt_matrix.T)
similarity_graph = networkx.from_numpy_array(similarity_matrix)

scores = networkx.pagerank(similarity_graph)
ranked_sentences = sorted(((score, index) for index, score in scores.items()), reverse=True)
ranked_sentences[:10]

top_sentence_indices = [ranked_sentences[index][1] 
                        for index in range(num_sentences)]
top_sentence_indices.sort()
print('\n'.join(np.array(sentences)[top_sentence_indices]))
