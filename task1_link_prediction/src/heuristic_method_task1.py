from get_data import *
import pickle
from sklearn.utils import shuffle
from scipy.spatial.distance import cosine

def compute_source_destination(dim = 128, embedding_method = 'prune'):
	# consider combining train + test_seen
	# select each source/destin idx
	filename = 'data/t1-all-train.txt'
	idx_to_paper_idx, paper_idx_to_idx, num_node = reset_idx(filename)
	links = get_links(filename, paper_idx_to_idx)
	print('links =', len(links), links[0])
	idx_source = [[] for i in range(num_node)]
	idx_destin = [[] for i in range(num_node)]
	for i in range(len(links)):
		s = links[i][0]
		d = links[i][1] # one link for s --> d
		idx_source[d].append(s)
		idx_destin[s].append(d)
	for i in range(5):
		print('for node', i, ': source =', idx_source[i], ', destin =', idx_destin[i])
	#exit()
	#filehandler = open('data/idx_source', 'wb')
	#pickle.dump(idx_source, filehandler)
	#filehandler = open('data/idx_destin', 'wb')
	#pickle.dump(idx_destin, filehandler)

	# compute average of these source/destin idx
	if embedding_method == 'prune':
		emb = get_prune_embedding("data/prune-t1-all-train.emb", paper_idx_to_idx)
		#emb = get_prune_embedding(filename, paper_idx_to_idx)
	elif embedding_method == 'node2vec':
		emb = get_node2vec_embedding("data/node2vec-t1-all-train.emb", paper_idx_to_idx)
	elif embedding_method == 'deepwalk':
		#emb = get_node2vec_embedding("data/deepwalk-t1-all-train.emb", paper_idx_to_idx)
		#dim = 64
		emb = get_node2vec_embedding("data/deepwalk-t1-all-train-256.emb", paper_idx_to_idx)
		dim = 256
	ave_source = np.zeros([num_node, dim], dtype = float) # ave of id --> id
	ave_destin = np.zeros([num_node, dim], dtype = float) # id --> ave of id

	for i in range(num_node):
		for j in range(len(idx_source[i])):
			#print(i, j, ', idx_source =', idx_source[i][j])
			ave_source += emb[idx_source[i][j]]
		#print('ave_source =', ave_source[0][:3])
		if len(idx_source[i]) != 0: ave_source = ave_source/float(len(idx_source[i]))
		#print('ave_source =', ave_source[0][:3])
		
		for j in range(len(idx_destin[i])):
			#print(i, j, ', idx_destin =', idx_destin[i][j])
			ave_destin += emb[idx_destin[i][j]]
		#print('ave_destin =', ave_destin[0][:3])
		if len(idx_destin[i]) != 0: ave_destin = ave_destin/float(len(idx_destin[i]))
		#print('ave_destin =', ave_destin[0][:3])
		#if i > 0: break

	np.save('data/ave_source_deepwalk_256.npy', ave_source)
	np.save('data/ave_destin_deepwalk_256.npy', ave_destin)

def count_degree():
	filename = 'data/t1-all-train.txt'
	idx_to_paper_idx, paper_idx_to_idx, num_node = reset_idx(filename)
	indegree = np.zeros([num_node], dtype = int)
	outdegree = np.zeros([num_node], dtype = int)

	def count_one_file(filename, const, indegree, outdegree):
		links = get_links(filename, paper_idx_to_idx)
		for i in range(len(links)):
			s = links[i][0]
			d = links[i][1] # one link for s --> d
			indegree[d] += const
			outdegree[s] += const
		return indegree, outdegree

	indegree, outdegree = count_one_file('data/t1-train.txt', 1, indegree, outdegree)
	print('indegree =', indegree[:3], '\noutdegree =', outdegree[:3])
	indegree, outdegree = count_one_file('data/t1-test-seen.txt', 2, indegree, outdegree)
	print('indegree =', indegree[:3], '\noutdegree =', outdegree[:3])

	return indegree, outdegree
#count_degree()

def count_neighbor():
	handle = open('data/idx_source', 'rb')
	idx_source = pickle.load(handle)
	handle = open('data/idx_destin', 'rb')
	idx_destin = pickle.load(handle)

def negative_sampling_strategy1(paper_idx_to_idx, num_node, positive_links, num):
	set_train = get_index('data/t1-train.txt', paper_idx_to_idx)
	import random
	random.seed(14)
	links = [[] for i in range(num_node+1)]
	for i in range(positive_links.shape[0]):
		links[positive_links[i][0]].append(positive_links[i][1]) 
		# build an adjacency list, containing all pos and neg links, to avoid sample the same link
	neg_links = []
	print('sample negative links...')
	while len(neg_links) < num:
		s = randint(0, len(set_train)-1)
		t = randint(0, len(set_train)-1)
		s = set_train[s]
		t = set_train[t]
		if t not in links[s] and s != t: # and s in set_train and t in set_train:
			
			neg_links.append([s, t])
			#neg_links[len(neg_links)-1].append(s)
			#neg_links[len(neg_links)-1].append(t)
			links[s].append(t)
		if len(neg_links) % 100000 == 0: print(len(neg_links))
	print("negative links =", len(neg_links), neg_links[:5])
	neg_links = np.array(neg_links)
	print("negative links =", neg_links.shape, neg_links[:5])
	return neg_links

def get_test_negative_sample_link(paper_idx_to_idx):
	first80 = get_index('data/t1-train.txt', paper_idx_to_idx) # list
	last20_seen = get_index('data/t1-test-seen.txt', paper_idx_to_idx)
	last20_unseen = get_index('data/t1-test.txt', paper_idx_to_idx)
	last20 = list(set(last20_seen).union(set(last20_unseen)).difference(set(first80)))
	print('first80, last20 =', len(first80), len(last20), len(last20_seen), len(last20_unseen))
	# negative sample from last 20
	link_seen = get_links('data/t1-test-seen.txt', paper_idx_to_idx)
	#link_unseen, error = get_testing_links('data/t1-test.txt', paper_idx_to_idx)
	link_unseen = get_links('data/t1-test.txt', paper_idx_to_idx)
	print('link_seen, unseen =', link_seen.shape, link_unseen.shape)
	link = np.concatenate([link_seen, link_unseen], axis = 0)
	
	#for i in range(10000, 10005): print('for node', i, ': source =', idx_source[i], ', destin =', idx_destin[i])
	return link

def get_train_negative_sample_link(paper_idx_to_idx):
	link = get_links('data/t1-train.txt', paper_idx_to_idx)
	return link

def get_source_destin(link, num_node):
	idx_source = [[] for i in range(num_node)]
	idx_destin = [[] for i in range(num_node)]
	for i in range(len(link)):
		s = link[i][0]
		d = link[i][1] # one link for s --> d
		idx_source[d].append(s)
		idx_destin[s].append(d)
	return idx_source, idx_destin

def negative_sampling_strategy2(paper_idx_to_idx, num_node, num, emb = None, use_all = False, use_sim = False, use_train = False):
	print('\nnegative sampling strategy 2...\n')
	# combine train and idx in train data, only sample from index in first 80 data 
	# select 2 steps, 3 steps
	
	if use_train == False: link = get_test_negative_sample_link(paper_idx_to_idx)
	else: link = get_train_negative_sample_link(paper_idx_to_idx)
	idx_source, idx_destin = get_source_destin(link, num_node)

	neg_link = []
	# 2 steps
	if use_train == False: origin = [i for i in range(len(idx_source))]
	else: origin = [5*i for i in range(int(len(idx_source)/5))]	
	for a in range(len(idx_destin)):
		for b in idx_destin[a]:
			#print('\ni, j =', i, j, 'k =', end = ' ')
			for c in idx_destin[b]:
				if c not in idx_destin[a]: neg_link.append([a, c])
				#else: print('a, b, c =', a, b, c)
		#if len(neg_link) > num: break
		if a % 5000 == 0: print('a =', a)
	print('len (negative sample) =', len(neg_link), a, '/', len(idx_destin))
	neg_link = np.array(neg_link)
	
	if use_sim == False:
		neg_link = shuffle(neg_link, random_state = 17)
		if use_all == True: return neg_link
		else: return neg_link[:num]
	else: # select negative example by similarity
		lower_bound = 0.13 #0.51
		upper_bound = 0.94 #0.81
		link = []
		for i in range(neg_link.shape[0]):
			sim = 1 - cosine(emb[neg_link[i][0]], emb[neg_link[i][1]])
			if sim >= lower_bound and sim <= upper_bound:
				link.append([neg_link[i][0], neg_link[i][1]])
		print('bounded link =', len(link), len(neg_link))
		link = np.array(link)
		neg_link = shuffle(link)
		if use_all == True: return neg_link
		else: return neg_link[:num]

		return neg_link[:num]
	# negative sample from first 80

def negative_sampling_strategy3(paper_idx_to_idx, num_node, num, emb = None, use_all = False, use_sim = False, use_train = False):
	# num: return how many negative sample
	# emb: consider the similarity of embedding to select negative example
	# use_all: use all sample or only use "num" sample
	# use_train" use train/test+test_seen to sample
	# 3 steps

	if use_train == False: link = get_test_negative_sample_link(paper_idx_to_idx)
	else: link = get_train_negative_sample_link(paper_idx_to_idx)
	idx_source, idx_destin = get_source_destin(link, num_node)

	neg_link = [] # 3 steps, a to b, b to c, a to x, but x not to c
	for a in range(len(idx_destin)):
		for b in idx_destin[a]:
			for c in idx_destin[b]:
				for x in idx_destin[a]:
					if x != b and x not in idx_source[c]: neg_link.append([x, c])
				for x in idx_source[c]:
				 	if x != b and x not in idx_destin[a]: neg_link.append([a, x])
				if c not in idx_destin[a]: neg_link.append([a, c])
				#else: print('a, b, c =', a, b, c)
		#if len(neg_link) > num: break
		if a % 5000 == 0: print('a =', a)
	print('len (negative sample) =', len(neg_link), a, '/', len(idx_destin))
	neg_link = shuffle(np.array(neg_link))
	return neg_link[:num]

def get_similarity_percentage_link(filename, embedding_method): # get link first
	idx_to_paper_idx, paper_idx_to_idx, num_node = reset_idx('data/t1-all-train.txt')
	link = get_links(filename, paper_idx_to_idx)
	if embedding_method == 'node2vec':
		emb = get_node2vec_embedding("data/node2vec-t1-all-train.emb", paper_idx_to_idx)
	elif embedding_method == 'prune':
		emb = get_prune_embedding("data/prune-t1-all-train.emb", paper_idx_to_idx)
	elif embedding_method == 'deepwalk':
		emb = get_node2vec_embedding("data/deepwalk-t1-all-train-256.emb", paper_idx_to_idx)
	print('filename =', filename)
	get_similarity_percentage(emb, link)

def get_similarity_percentage(emb, link):
	sim_list = []
	for i in range(len(link)):
		sim_list.append(1 - cosine(emb[link[i][0]], emb[link[i][1]]))
		if i % 10000 == 0: print(i)
	sim_list = np.array(sim_list)
	print('dist. =', np.percentile(sim_list, [0, 1, 25, 50, 75, 99, 100]), np.mean(sim_list))

def main():
	a = 3
	#compute_source_destination(embedding_method='deepwalk')
	#get_similarity_percentage_link('data/t1-train.txt', 'deepwalk')
	#get_similarity_percentage_link('data/t1-test-seen.txt', 'deepwalk')
	#get_similarity_percentage_link('data/t1-test.txt', 'deepwalk')

if __name__ == '__main__':
	main()




