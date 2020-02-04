from nltk.book import text3

vocabulary = sorted(set(text3))
print(len(vocabulary))

f = open('vocabulary_text3.txt', 'w') #Object of type FILE
for word in vocabulary:
	f.write(word + '\n')
f.close()
