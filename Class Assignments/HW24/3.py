import gensim.downloader as api
word_vectors = api.load("glove-wiki-gigaword-100")

result = word_vectors.most_similar(
    positive=['woman', 'king'], negative=['man'], topn=1)
print('king - man + woman =', result[0])

result = word_vectors.most_similar(
    positive=['mom', 'man'], negative=['woman'], topn=1)
print('mom - woman + man =', result[0])

result = word_vectors.most_similar(
    positive=['son', 'girl'], negative=['boy'], topn=1)
print('son - boy + girl =', result[0])

result = word_vectors.most_similar(
    positive=['ran', 'tomorrow'], negative=['yesterday'], topn=1)
print('ran - yesterday + tomorrow =', result[0])
