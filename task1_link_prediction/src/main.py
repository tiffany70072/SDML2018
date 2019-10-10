import numpy as np
from random import randint

import sys
sys.path.insert(0, '../')
import pickle
import get_data
import heuristic_method
import build_model
import get_data_task1 as get_graph_data
import heuristic_method_task1

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import tensorflow as tf 
from sklearn.utils import shuffle
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean

from keras.utils import np_utils
import keras.backend as K
from keras import optimizers

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
sess = tf.Session(config=config)
K.set_session(sess)

# CUDA_VISIBLE_DEVICES=0 python main.py 

class LinkPredictionModel(object):
	def __init__(self):
		#self.embedding_type = sys.argv[1] # 'node2vec', 'prune', 'deepwalk'
		self.classifier_type = sys.argv[2] # 'nn', 'svm', 'regcl', 'similarity', 'rf'
		self.dim = 300 # dimension for node embedding
		self.n_ratio = 2.0
		self.use_link_emb = True
		self.use_element_wise = False
		self.use_cos_sim = True
		
		filename = 'data/t2-all.txt'
		#self.idx_to_paper_idx, self.paper_idx_to_idx, self.num_node = get_graph_data.reset_idx(filename)
		_, _, self.num_node = get_graph_data.reset_idx(filename)
		self.paper_idx_to_idx = [i-1 for i in range(0, 17501)] # paper id = 1, id = 0
		# (nodes) in links = 173364
		# (nodes) in links with unique id = 12489
		# number of nodes = 12489

		# use new word2vec model
		self.emb = np.load('data/emb_word2vec_300.npy')
		# use fixed random seed word2vec model
		#self.emb = np.load('data/emb_word2vec_300_randomseed.npy')

		print('number of nodes =', self.num_node, len(self.paper_idx_to_idx))
		print('self.emb =', self.emb.shape, self.emb[0][:10], self.emb[30][-10:])
		#print('doc_length =', self.doc_length[:10])
		#exit()

	def load_data(self):
		if self.classifier_type != 'similarity':
			filename = 'data/t2-train.txt'
			self.positive_links = get_graph_data.get_links(filename, self.paper_idx_to_idx)
			test_links = get_graph_data.get_links('data/t2-test.txt', self.paper_idx_to_idx)
			self.semi_positive = np.concatenate([self.positive_links, test_links], axis = 0)
			link1 = heuristic_method.negative_sampling_strategy(self.paper_idx_to_idx, 17500, self.semi_positive, int(self.positive_links.shape[0]*self.n_ratio/2))
			#link1 = heuristic_method.negative_sampling_strategy2(self.paper_idx_to_idx, self.num_node, int(num/3), use_train = True)
			link2 = heuristic_method.negative_sampling_strategy(self.paper_idx_to_idx, 17500, test_links, int(self.positive_links.shape[0]*self.n_ratio/2))
			self.negative_links = np.concatenate([link1, link2], axis = 0)
			
			print('negative sample =', self.negative_links.shape[0])

		#self.emb = heuristic_method.update_emb(self.emb, self.positive_links) # retrofitting
		print('self.emb =', self.emb.shape, self.emb[0][:10], self.emb[30][-10:])
		#np.save('data/emb_word2vec_retrofitting_1:01.npy', self.emb)
		#exit()
	def set_training_data(self):
		links = np.concatenate([self.positive_links, self.negative_links], axis = 0)
		print('set training data...')
		self.x = self.set_features(links)

		# set label y
		print('Set y...')
		self.y = [1 for i in range(self.positive_links.shape[0])] + [0 for i in range(self.negative_links.shape[0])]
		self.y = np.array(self.y)
		if self.classifier_type == 'nn' or self.classifier_type == 'regcl' or self.classifier_type == 'activecl': 
			from keras.utils.np_utils import to_categorical
			self.y = to_categorical(self.y)

	def set_features(self, links):
		dim = 0 # dimension for one inference
		if self.use_link_emb == True: dim += self.dim * 2
		#if self.use_ave_source_destin == True: dim += self.dim * 2 #self.dim * 2
		#if self.use_further_source_destin == True: dim += self.dim * 2 #self.dim * 2
		if self.use_element_wise == True: dim += self.dim
		if self.use_cos_sim == True: dim += 1
		#if self.use_count_degree == True: dim += 6'''
		#if self.use_common_neigh == True:

		# initialize container for inference
		x = np.empty([links.shape[0], dim], dtype = float)
		base1 = 0
		if self.use_link_emb == True:
			for i in range(links.shape[0]):
				x[i][:self.dim] = self.emb[links[i][0]]
				x[i][self.dim:2*self.dim] = self.emb[links[i][1]]
			base1 += 2 * self.dim
		if self.use_element_wise == True:
			for i in range(links.shape[0]):
				#x[i][base1:base1+self.dim] = np.multiply(self.emb[links[i][0]], self.emb[links[i][1]])
				x[i][base1:base1+self.dim] = (self.emb[links[i][0]] - self.emb[links[i][1]])
			base1 += self.dim
		'''
		# add ave_source, ave_destin
		if self.use_ave_source_destin == True:
			print('add ave source/destin')
			for i in range(links.shape[0]):
				x[i][base1:base1+self.dim] = self.ave_destin[links[i][0]]
				x[i][base1+self.dim:base1+2*self.dim] = self.ave_source[links[i][1]]
			base1 += 2 * self.dim
		if self.use_further_source_destin == True:
			print('add further source/destin')
			for i in range(links.shape[0]):
				x[i][base1:base1+self.dim] = self.ave_source[links[i][0]]
				x[i][base1+self.dim:base1+2*self.dim] = self.ave_destin[links[i][1]]
			base1 += 2 * self.dim'''

		# add cosine similarity
		if self.use_cos_sim == True:
			print('add cosine similarity')
			for i in range(links.shape[0]): 
				x[i][base1] = 1 - cosine(self.emb[links[i][0]], self.emb[links[i][1]])
				#x[i][base1+1] = euclidean(self.emb[links[i][0]], self.emb[links[i][1]])
				#x[i][base1+2] = np.corrcoef(self.emb[links[i][0]], self.emb[links[i][1]])[0, 1]
	
			base1 += 3

		return x

	def train_validation_split(self):
		print('train validation split...')
		n_ratio = self.n_ratio
		total_num = self.train_num+self.test_seen_num
		split_num = int(self.test_seen_num/2)
		# only use real last 20% positive links as positive example, with random sampled negative link
		#train_idx = [i for i in range(self.train_num+split_num)] + [i for i in range(total_num, total_num+(self.train_num+split_num)*n_ratio)]
		#valid_idx = [i for i in range(self.train_num+split_num, total_num)] + [i for i in range(total_num+(self.train_num+split_num)*n_ratio, total_num*(1+n_ratio))]
		
		# only use real last 20% positive links as validation data
		train_idx = [i for i in range(self.train_num+split_num)] + [i for i in range(total_num, int(total_num*(1+n_ratio)))]
		valid_idx = [i for i in range(self.train_num+split_num, total_num)] 
		
		self.x_train, self.x_valid = self.x[train_idx], self.x[valid_idx]
		self.y_train, self.y_valid = self.y[train_idx], self.y[valid_idx]
		
		print('x_train =', self.x_train.shape, ', x_valid = ', self.x_valid.shape, ', x = ', self.x.shape)
		print('pos =', self.positive_links.shape[0], ', neg =', self.negative_links.shape[0])
		print('y_train =', self.y_train.shape, ', y_valid = ', self.y_valid.shape, ', y = ', self.y.shape)
		print('valid_num =', self.train_num+split_num, total_num)
		print('y_valid =', self.y_valid[:5], self.y_valid[-5:])
		#del self.x, self.y

	def get_classifier(self):
		if self.classifier_type == 'nn' or self.classifier_type == 'regcl' or self.classifier_type == 'activecl':
			self.classifier = build_model.MLP(self.x.shape[1])
		elif self.classifier_type == 'svm': self.classifier = build_model.SVM()
		elif self.classifier_type == 'rf': self.classifier = build_model.randomforest()

	def train_classifier(self):
		adam = optimizers.Adam(lr = 0.001)
		self.classifier.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
		self.x, self.y = shuffle(self.x, self.y)
		self.classifier.fit(self.x, self.y, epochs = int(sys.argv[4]), validation_split = 0.1, batch_size = 128)

	def count_correct(self, x, name, ispos = False, isfeatures = False):
		if isfeatures == False: x = self.set_features(x)
		y_pred = self.classifier.predict(x)
		#print('y_pred', y_pred[:3])
		count = 0
		if ispos == False:
			for i in range(y_pred.shape[0]):
				if y_pred[i][1] < 0.5: count += 1
		else: 
			for i in range(y_pred.shape[0]):
				if y_pred[i][1] > 0.5: count += 1
		print(name, ', count correct =', count)

	def train_active_classifier(self):
		adam = optimizers.Adam(lr = 0.001)
		self.classifier.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
		self.x, self.y = shuffle(self.x, self.y)
		self.classifier.fit(self.x, self.y, epochs = 6, validation_split = 0.1, batch_size = 128)
		
		# different after the first time
		x_pos = self.set_features(self.positive_links)
		y = [1 for i in range(self.positive_links.shape[0])] + [0 for i in range(self.negative_links.shape[0])]
		from keras.utils.np_utils import to_categorical
		y = to_categorical(np.array(y))
		x_neg_com = self.set_features(self.negative_links)
		#for i in range(sys.argv[4]):
		for i in range(8):
			self.count_correct(x_neg_com, isfeatures = True, name = 'neg_com')
			y_pred = self.classifier.predict(x_neg_com)
			x_neg_new = heuristic_method.negative_sampling_strategy(self.paper_idx_to_idx, 17500, self.semi_positive, int(self.positive_links.shape[0]*self.n_ratio))
			x_neg_new = self.set_features(x_neg_new)
			#x_neg_com = np.copy(x_neg_ori)
			for i in range(y_pred.shape[0]): 
				if y_pred[i][1] < 0.5: x_neg_com[i] = x_neg_new[i]
					
			self.count_correct(x_neg_new, isfeatures = True, name = 'neg_new')
			self.count_correct(x_neg_com, isfeatures = True, name = 'neg_select')
			self.count_correct(self.positive_links, ispos = True, name = 'pos')

			y_test = self.classifier.predict(self.x_test)
			average_prob = np.median(y_test[:, 0])
			print('average_prob =', average_prob)

			x_train, y_train = shuffle(np.concatenate([x_pos, x_neg_com], axis = 0), y) 
			self.classifier.fit(x_train, y_train, epochs = 3, validation_split = 0.1, batch_size = 128, verbose = 2)

		y_test = self.classifier.predict(self.x_test)
		average_prob = np.median(y_test[:, 0])
		print('average_prob =', average_prob)
		#exit()

	def set_testing_data(self):
		x_test_link, self.error = get_graph_data.get_testing_links('data/t2-test.txt', self.paper_idx_to_idx)
		print('set testing data')
		self.x_test = self.set_features(x_test_link)
		print("x_test =", self.x_test.shape)
		print(x_test_link[:5])

	def predict_nn(self):
		y_test = self.classifier.predict(self.x_test)
		print("y_test =", y_test.shape, y_test[:10])
		y_test = np.argmax(y_test, axis = 1)
		print("y_test =", y_test.shape, '\n', y_test[:10])

		fout = open('results/' + sys.argv[1] + '_' + sys.argv[2] + '_' + sys.argv[3] + '.txt', 'w')
		for i in range(y_test.shape[0]): fout.write("%d\n" %y_test[i])
		print('one in test prediction =', np.sum(y_test), np.sum(y_test)/float(y_test.shape[0]))

		np.save('ensemble/' + sys.argv[1] + '_' + sys.argv[2] + '_' + sys.argv[3] + '.npy', np.array(y_test))

	def predict_regression_classifier(self):
		y_test = self.classifier.predict(self.x_test)
		average_prob = np.median(y_test[:, 0]) # 0
		print('average_prob =', average_prob)
		fout = open('results/' + sys.argv[1] + '_' + sys.argv[2] + '_' + sys.argv[3] + '.txt', 'w')
		count = 0
		for i in range(y_test.shape[0]): 
			if y_test[i][0] > average_prob: fout.write("%d\n" %1) # 0
			else: 
				fout.write("%d\n" %0)
				count += 1
		print('one in test prediction =', count/float(y_test.shape[0]))

		#np.save('ensemble/' + sys.argv[1] + '_' + sys.argv[2] + '_' + sys.argv[3] + '.npy', y_test)

	def train_svm(self):
		print('svm fitting...')
		self.classifier.fit(self.x_train, self.y_train)
		print('svm score =')
		print('train =', self.classifier.score(self.x_train, self.y_train))
		print('valid =', self.classifier.score(self.x_valid, self.y_valid))
		
		y_test = self.classifier.predict(self.x_test)
		fout = open('results/' + sys.argv[1] + '_' + sys.argv[2] + '_' + sys.argv[3] + '.txt', 'w')
		for i in range(y_test.shape[0]): fout.write("%d\n" %y_test[i])
		print("y_test =", len(y_test))

		np.save('ensemble/' + sys.argv[1] + '_' + sys.argv[2] + '_' + sys.argv[3] + '.npy', np.array(y_test))

	def similarity(self):
		x_test_link, self.error = get_graph_data.get_testing_links('data/t2-test.txt', self.paper_idx_to_idx)
		
		sim = np.zeros([x_test_link.shape[0]], dtype = float)
		for i in range(x_test_link.shape[0]):
			sim[i] = 1 - cosine(self.emb[x_test_link[i][0]], self.emb[x_test_link[i][1]])
		threshold = np.median(sim)
		print("threshold =", threshold)
		print('sim =', sim[:10])
		
		one = 0
		fout = open('results/' + sys.argv[1] + '_' + sys.argv[2] + '_' + sys.argv[3] + '_' + str(threshold)[:4] + '.txt', 'w')
		for i in range(x_test_link.shape[0]): 
			if sim[i] > threshold: 
				fout.write("%d\n" %1)
				one += 1
			else: fout.write("%d\n" %0)
		print('one in y_test =', one, len(x_test_link))
		np.save('ensemble/' + sys.argv[1] + '_' + sys.argv[2] + '_' + sys.argv[3] + '_' + str(threshold)[:4] + '.npy', np.array(sim))
	
	def train_rf(self):
		print('random forest fitting...')
		self.classifier.fit(self.x, self.y)
		print('features importance =', self.classifier.feature_importances_[-10:])
		print('random forest score =')
		print(self.classifier.score(self.x, self.y))
		#print(self.classifier.score(self.x_valid, self.y_valid))
		
		y_test = self.classifier.predict(self.x_test)
		fout = open('results/' + sys.argv[1] + '_' + sys.argv[2] + '_' + sys.argv[3] + '.txt', 'w')
		for err in self.error: y_test[err] = 1
		for i in range(y_test.shape[0]): fout.write("%d\n" %y_test[i])
		print("y_test =", len(y_test))
		
		np.save('ensemble/' + sys.argv[1] + '_' + sys.argv[2] + '_' + sys.argv[3] + '.npy', np.array(y_test))

def main():
	#get_data.get_doc()
	model = LinkPredictionModel()
	model.load_data()
	if model.classifier_type != 'similarity':
		model.set_training_data()
		model.get_classifier()
		model.set_testing_data()

	if model.classifier_type == 'nn' or model.classifier_type == 'regcl':
		model.train_classifier()
		if model.classifier_type == 'nn': model.predict_nn()
		else: model.predict_regression_classifier()
	elif model.classifier_type == 'activecl': model.train_active_classifier()
	elif model.classifier_type == 'svm': model.train_svm()
	elif model.classifier_type == 'similarity': model.similarity()
	elif model.classifier_type == 'rf': model.train_rf()
	else: 
		print('no this classifier')
		exit()	
	
if __name__ == "__main__":
	main()


		

