import operator
sorted_fdist = sorted(fdist.items(), key = operator.itemgetter(1), reverse = True)
print("The type of dictionary items after sorting", type(sorted_fdist))
print("Dictionary items after sorting:")
print(sorted_fdist[:50], '\n')

most_frequent = []
for i in range(len(sorted_fdist)):
	most_frequent.append(sorted_fdist[i][0])

firs_most_frequent = most_frequent[:50]
print("First most frequent words")
print(firs_most_frequent, '\n')

fdist.plot(50)
fdist.plot(50, cumulative=True)

long_words = [w for w in vocabulary if len(w) > 15]
print(long_words, '\n')