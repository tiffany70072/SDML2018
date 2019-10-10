import numpy as np 

def get_node2vec_embedding(filename, paper_idx_to_idx):
	#f = open('node2vec-master/emb/t1-train.emd', 'r')
	#f = open('node2vec-master/emb/test_output.emd', 'r')
	f = open(filename)
	header = f.readline().split()

	num_node = int(header[0])
	dim = int(header[1])
	print("num node = %d, emb dim = %d" %(num_node, dim))

	emb = np.empty([num_node, dim], dtype = float)
	#idx = np.empty([num_node], dtype = int)
	for i, line in enumerate(f):
		line = line.split()
		paper_idx = int(line[0])
		idx = paper_idx_to_idx[paper_idx]
		#idx[i] = int(line[0])
		for d in range(dim):
			#emb[i][d] = float(line[d+1])
			emb[idx][d] = float(line[d+1])
	print('emb (node2vec) =', emb.shape)
	#return idx, emb
	return emb
		
def get_prune_embedding(filename, paper_idx_to_idx):
	num_node = len(paper_idx_to_idx) # need to reset
	print('num_node =', 16372, len(paper_idx_to_idx))
	dim = 128
	emb = np.empty([num_node, dim], dtype = float)
	f = open(filename, 'r')
	for i, line in enumerate(f):
		try:
			#print(i, paper_idx_to_idx[i])
			line = line.split(',')
			for d in range(dim):
				emb[paper_idx_to_idx[i]][d] = float(line[d])
		except KeyError:
			continue
	return emb
#get_prune_embedding('data_old/prune-t1-all-train.emb')

def get_links(filename, paper_idx_to_idx): # get positive links
	f = open(filename, 'r')
	links = []
	for i, line in enumerate(f):
		line = line.split()
		try:
			s = paper_idx_to_idx[int(line[0])]
			d = paper_idx_to_idx[int(line[1])]
			links.append([s, d])
			#links[i].append(paper_idx_to_idx[int(line[0])])
			#links[i].append(paper_idx_to_idx[int(line[1])])
		except KeyError:
			continue
	links = np.array(links)
	#print("positive links =", links.shape, links[:5])
	return links

def reset_idx(filename):
	# turn paper index to new index from 1 to x
	# 1, 6, 2, 3 --> 0, 1, 2, 3
	f = open(filename, 'r')
	s = [] # set
	for i, line in enumerate(f):
		line = line.split()
		s.append(int(line[0]))
		s.append(int(line[1]))
	print('# (nodes) in links =', len(s))
	s = set(s)
	print('# (nodes) in links with unique id =', len(s))
	s = list(s)
	s.sort()
	#print('s =', s)
	idx_to_paper_idx = {}
	paper_idx_to_idx = {}
	for i in range(len(s)):
		idx_to_paper_idx[i] = s[i]
		paper_idx_to_idx[s[i]] = i	
		#print("s =", i, s[i])
	return idx_to_paper_idx, paper_idx_to_idx, len(s)
#reset_idx('node2vec-master/graph/test_input.txt')

def get_testing_links(filename, paper_idx_to_idx):
	# get links and reset index in t1-test.txt
	f = open(filename, 'r')
	links = []
	error = []
	for i, line in enumerate(f):
		line = line.split()
		links.append([])
		try:
			links[i].append(paper_idx_to_idx[int(line[0])])
			links[i].append(paper_idx_to_idx[int(line[1])])
		except KeyError:
			links[i].append(0)
			links[i].append(0)
			error.append(i)
			#print('err idx =', line[0], line[1])
	#print("test links =", len(links), links[:10])
	links = np.array(links)
	print("test links =", links.shape, links[:10])
	print("test error (not in training data) =", len(error))
	return links, error

def get_index(filename, paper_idx_to_idx):
	# for get negative link
	# only sample negative link from index in first 80 data 
	f = open(filename, 'r')
	s = [] # set
	for i, line in enumerate(f):
		line = line.split()
		s.append(int(line[0]))
		s.append(int(line[1]))
	print('# (nodes) in links =', len(s))
	s = set(s)
	print('# (nodes) in links with unique id =', len(s))
	s = list(s)
	s.sort()
	
	set_train = []
	for i in range(len(s)):
		try:
			set_train.append(paper_idx_to_idx[s[i]])
		except KeyError:
			continue
	print('set_train =', len(set_train))
	return set_train 



