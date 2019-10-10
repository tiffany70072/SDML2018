import numpy as np 
from numpy.random import seed
seed(1)
from sklearn.utils import shuffle
import sys
sys.path.insert(0, '../')
import get_data_task1 as get_graph_data
from scipy.spatial.distance import cosine

def get_source_destin_relation(link, num_node):
	idx_source = [[] for i in range(num_node)]
	idx_destin = [[] for i in range(num_node)]
	for i in range(len(link)):
		s = link[i][0]
		d = link[i][1] # one link for s --> d
		idx_source[d].append(s)
		idx_destin[s].append(d)
	return idx_source, idx_destin

def negative_sampling_strategy(paper_idx_to_idx, num_node, pos_link, num, use_all = False):
	print('\nnegative sampling strategy ...\n')
	# select 2 steps, 3 steps
	# only sample negative link from index in first 80 data 
	
	idx_source, idx_destin = get_source_destin_relation(pos_link, num_node)
	neg_link = []
	# 2 steps
	for a in range(len(idx_destin)):
		for b in idx_destin[a]:
			for c in idx_destin[b]:
				if c not in idx_destin[a]: neg_link.append([a, c])
		#if a % 5000 == 0: print('a =', a)
	print('len (negative sample) =', len(neg_link), a, '/', len(idx_destin))
	neg_link = shuffle(np.array(neg_link), random_state = 1)
	if use_all == True: return neg_link
	else: return neg_link[:num]

def get_retrofitting_loss(original_emb, emb, link, alpha, beta):
	loss1, loss2 = 0, 0
	for i in range(emb.shape[0]):
		loss1 = loss1 + np.sum(np.power(original_emb[i] - emb[i], 2))
	for e in range(link.shape[0]):
		i = link[e][0]
		j = link[e][1]
		loss2 = loss2 + np.sum(np.power(emb[i] - emb[j], 2))
	loss = alpha * loss1 + beta * loss2
	print('loss = %.3f' % loss)
	#loss = alpha*np.sum((original_emb - emb)**2) + beta*np.sum(emb[link[i][0]] - emb[link[i][1]])

def update_emb(original_emb, link):
	alpha = 1
	beta = 0.1
	ratio = alpha/float(beta)
	print('retro fitting: alpha = %.1f, beta = %.3f, ratio = %.2f' %(alpha, beta, ratio))
	emb = np.copy(original_emb)
	idx_source, idx_destin = get_source_destin_relation(link, 17500)
	print('cosine = %.3f' % (1 - cosine(original_emb[1], original_emb[idx_source[1][0]])))
	#print('cosine = %.3f' % (1 - cosine(original_emb[5], original_emb[idx_source[5][0]])))
	#print('cosine = %.3f' % (1 - cosine(original_emb[3], original_emb[idx_source[3][0]])))
	
	get_retrofitting_loss(original_emb, emb, link, alpha, beta)
	for ite in range(10):
		for idx in range(emb.shape[0]):
			num_neighbors = len(idx_source[idx]) + len(idx_destin[idx])
			if num_neighbors == 0:
				#print('num neighbors = 0', idx)
				continue
			newVec = num_neighbors * original_emb[idx] * ratio
			for neigh in idx_source[idx]: newVec += emb[neigh]
			for neigh in idx_destin[idx]: newVec += emb[neigh]
			emb[idx] = newVec/((ratio+1)*num_neighbors)
		get_retrofitting_loss(original_emb, emb, link, alpha, beta)
	print('cosine = %.3f' % (1 - cosine(emb[1], emb[idx_source[1][0]])))
	#print('cosine = %.3f' % (1 - cosine(emb[5], emb[idx_source[5][0]])))
	#print('cosine = %.3f' % (1 - cosine(emb[3], emb[idx_source[3][0]])))
	#exit()
	return emb
'''
newWordVecs = deepcopy(wordVecs)
wvVocab = set(newWordVecs.keys())
loopVocab = wvVocab.intersection(set(lexicon.keys()))
for it in range(numIters):
	# loop through every node also in ontology (else just use data estimate)
	for word in loopVocab:
		wordNeighbours = set(lexicon[word]).intersection(wvVocab)
		numNeighbours = len(wordNeighbours)
		#no neighbours, pass - use data estimate
		if numNeighbours == 0:
			continue
		# the weight of the data estimate if the number of neighbours
		newVec = numNeighbours * wordVecs[word]
		# loop over neighbours and add to new vector (currently with weight 1)
		for ppWord in wordNeighbours:
			newVec += newWordVecs[ppWord]
		newWordVecs[word] = newVec/(2*numNeighbours)
'''

def main():
	emb = np.load('data/emb_word2vec_300.npy')
	link = get_graph_data.get_links('data/t2-train.txt', [i-1 for i in range(0, 17501)])
	update_emb(emb, link)

if __name__ == '__main__':
	main()

