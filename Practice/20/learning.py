from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

X, y = load_iris(return_X_y = True)
clf = LogisticRegression(random_state = 0).fit(X, y)
print("X: ", type(X))
print("y: ", type(y))

print("X:", len(X), "x", len(X[0]))
print("y:", len(y))

clf.predict(X)
clf.predict_proba(X[:2, :])


corpus = ['This is the first document.', 'This document is the second document.', 'And this is the third one.', 'Is this the first document?']

vectorizer = CountVectorizer()
X = vectorizer.fit_transfrom(corpus)
matrix_X = X.toarray()
vectorizer = TfidfVectorizer()
vectorizer.fit(text)
aux = vectorizer.idf_
print(aux)
type(aux)