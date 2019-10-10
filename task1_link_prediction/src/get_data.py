import os 
import numpy as np 

from numpy.random import seed
from gensim.models import Word2Vec
# from sklearn.feature_extraction.text import CountVectorizer  
# from sklearn.feature_extraction.text import TfidfTransformer  
# from test_fasttext import load_vectors

seed(17)


def get_doc(only_title=False, only_abstract=False, remove_stopwords=True):
	# read all doc into lists
	path = 'data/t2-doc'
	file_list = [filename for filename in os.listdir(path) if filename.endswith('.xml')]
	file_list.sort(key=lambda x: int(x.split('.')[0]))
	num = len(file_list)
	print("num of doc =", num)
	
	docs = [[] for i in range(num)]
	for i, filename in enumerate(file_list):
		f = open(os.path.join(path, filename), 'r')
		f.readline()
		while True: # get title
			line = f.readline()
			if line == "</title>\n": 
				break
			line = line.split()
			docs[i] += line
		if only_abstract: 
			docs[i] = []
		if not only_title:
			for line in f: # get abstract
				if line == "</abstract>\n": 
					break
				line = line.split()
				docs[i] += line
			
		docs[i] = preprocess_doc(docs[i], remove_stopwords)
		if i % 5000 == 0: 
			print(i)
	return docs


def get_stopwords():
	f = open('stopwords.txt', 'r')
	stop_words = []
	for line in f: 
		stop_words.append(line[:-1])
	return stop_words


def preprocess_doc(s, remove_stopwords):
	s = ' '.join(s)
	s = s.lower()

	# Remove punctuation.
	remove_list = ['"', "$", '\\', '', '.', ',', '?', '(', ')', "'s", "'", ":", ";", "!", "_", "<", ">", "/", "{", "}", "`"]
	for punctuation in remove_list:
		s = s.replace(punctuation, '')  # Replace with None.
	for punctuation in ['-', '+']:
		s = s.replace(punctuation, ' ')  # Replace with one space.
	# Remove stopword.
	s = s.split()
	if remove_stopwords: 
		s = [w for w in s if not w in stop_words]  # Remove stop words
	s = [w for w in s if len(w) >= 2]  # Remove strange vocabulary
	s = [stem(w) for w in s]
	return s
	
	
def stem(word): 
	for suffix in ['ing', 'ly', 'ed', 'ious', 'ies', 'ive', 'es', 's', 'ment']:
		if word.endswith(suffix): 
			return word[:-len(suffix)]
	return word


def train_wordModel(doc, word_dim):
	print('train word 2 vec model...')
	wordModel = Word2Vec(doc, size=word_dim, window=5, min_count=1, workers=4)
	
	for word1 in ['quantum', 'quark', 'energy', 'messenger']:
		for word2 in ['quantum', 'quark', 'according', 'cat']:
			print(word1, word2, wordModel.similarity(word1, word2))
	return wordModel


def wordEmbedding(doc, wordModel, word_dim):
	print("Word Embedding")
	
	emb = np.zeros([len(doc), word_dim])
	doc_length = np.zeros([len(doc)])  # for doc length normalization
	allzero = 0
	keyerror_count = 0
	for i in range(len(doc)):
		for j in range(len(doc[i])):
			try:
				emb[i] += wordModel[doc[i][j]]
				doc_length[i] += 1.0
			except KeyError: 
				keyerror_count += 1
				continue
		if doc_length[i] != 0: 
			emb[i] = emb[i] / doc_length[i] 
		else: 
			# if no word of the title in the training data 
			allzero += 1
			print('all zero in word embedding', i)
		if i % 5000 == 0: 
			print(i)	
	print("Num (all zero in word embedding) =", allzero)
	print('valid length =', np.sum(doc_length))
	print('keyerror length =', keyerror_count)
	return emb


def embedding_fasttext(doc, data):
	emb = np.zeros([len(doc), 300])
	doc_length = np.zeros([len(doc)]) # for doc length normalization
	allzero = 0
	keyerror_count = 0
	for i in range(len(doc)):
		for j in range(len(doc[i])):
			try:
				emb[i] += data[doc[i][j]]
				doc_length[i] += 1.0
			except KeyError: 
				print(doc[i][j])
				keyerror_count += 1
				continue
		if doc_length[i] != 0: 
			emb[i] = emb[i] / doc_length[i] 
		else: 
			# if no word of the title in the training data 
			allzero += 1
			print('all zero in word embedding', i)
		if i % 5000 == 0: 
			print(i)	
	print("Num (all zero in word embedding) =", allzero)
	print('valid length =', np.sum(doc_length))
	print('keyerror length =', keyerror_count)
	return emb


def embedding_tfidf(doc):
	vectorizer = TfidfTransformer()  
	X = vectorizer.fit_transform(doc)  
	word = vectorizer.get_feature_names()  
	X = X.toarray()
	print(word[:20])
	print(X[:10], X.shape)
	return X


def save(emb):
	np.save('data/emb_word2vec_300.npy', emb)
	print('emb =', emb.shape)
	

def word2vec():
	#doc = get_doc(remove_stopwords=False)
	doc = get_doc()
	print('doc =', len(doc), doc[10])
	wordModel = train_wordModel(doc, 300)

	#title = get_doc(only_title=True)
	#abstract = get_doc(only_abstract=True)
	#emb1 = wordEmbedding(title, wordModel, 300)
	#emb2 = wordEmbedding(abstract, wordModel, 300)
	#print('emb =', emb1.shape, emb2.shape)
	#emb = np.concatenate([emb1, emb2], axis=1)
	emb = wordEmbedding(doc, wordModel, 300)
	save(emb)

	
def fasttext():
	title = get_doc(only_title=True)
	abstract = get_doc(only_abstract=True)
	# doc = get_doc()
	data = load_vectors('wiki-news-300d-1M.vec')
	# emb = embedding_fasttext(doc, data)
	emb1 = embedding_fasttext(title, data)
	emb2 = embedding_fasttext(abstract, data)
	print('emb =', emb1.shape, emb2.shape)
	emb = np.concatenate([emb1, emb2], axis = 1)
	save(emb)
	
	
def tfidf():
	doc = get_doc()
	emb = embedding_tfidf(doc)
	save(emb)

	
def main():
	stop_words = get_stopwords()
	word2vec()
	

if __name__ == '__main__':
	main()

	
