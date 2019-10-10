import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import read_data
#import build_model
import task2_rule_based
import utils
import sys
import pickle

import tensorflow as tf 
from sklearn.utils import shuffle
from sklearn.preprocessing import MultiLabelBinarizer
'''>>> mlb = MultiLabelBinarizer()
>>> mlb.fit_transform([(1, 2), (3,)])
array([[1, 1, 0],
	   [0, 0, 1]])
>>> mlb.classes_
array([1, 2, 3])'''

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

from keras.utils import np_utils
import keras.backend as K

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
sess = tf.Session(config=config)
K.set_session(sess)

# CUDA_VISIBLE_DEVICES=0 python main.py 

class FoodRecommenderSystem(object):
	def __init__(self):
		self.food_num = 5532 # (40, 2039)
		self.user_num = 2608
		
		self.interval = 10
		self.num_class = 50
		#self.latent_dim = 64
		self.use_predefined_embedding = False
		self.use_popular_embedding = 0 # 1
		self.use_concat = 0 #1
		#self.is_check_valid = 0 # split training and validation set
		#self.only_predict_popular_food = 1 # default: 1
		
		#self.model_type = 'mlp' # with food_features = 0
		#self.model_type = 'mf'

		self.epoch = 5

	def load_data(self):
		handle = open('data/rating_user_time_item', 'rb')
		self.table_list = pickle.load(handle) # user, count_date, item
		
		self.foodid_eachrank, self.eachrank_foodid = task2_rule_based.prepare_ranking_foodid(self.num_class)

	def set_training_data(self):
		print('set training data')
		train_num_each_user = 30 # training data = 30 * 2600
		train_num = train_num_each_user * self.user_num
		self.user_train = np.zeros([train_num]) # each with one user id
		self.ts_train = [[] for i in range(train_num)] # each with 10 days, with list(food->ranking id) --> with n-hot 300 dim vectors
		self.label_train = [[] for i in range(train_num)] # each with one list(food->rank)
		count = 0
		count_error = 0
		count_error_label = 0
		#print('table_list =', self.table_list[1][-3:])
		#exit()
		for i in range(self.user_num):
			first_start_day = max(len(self.table_list[i])-self.interval-1-train_num_each_user, 0)
			#for j in range(train_num_each_user):
			for start_day in range(first_start_day, min(first_start_day+train_num_each_user, len(self.table_list[i])-self.interval-1)):
				self.user_train[count] = i
				for day in range(self.interval):
					#if i == 1 and start_day == first_start_day + train_num_each_user - 1: print('---', self.table_list[i][start_day + day])
					try:
						self.ts_train[count].append(self.table_list[i][start_day + day].copy())
					except IndexError:
						print(count, len(self.ts_train))
						print(i, len(self.table_list))
						print(start_day, day, len(self.table_list[i]))
						print(first_start_day, train_num_each_user)
						exit()
					for j in range(len(self.ts_train[count][day])):
						try: self.ts_train[count][day][j] = self.foodid_eachrank[i][self.ts_train[count][day][j]]
						except KeyError: 
							#print(i, count, day, j, self.ts_train[count][day][j])
							self.ts_train[count][day][j] = self.num_class-1
							count_error += 1
							
					#print(ts_train[count][day])
				#print(self.table_list[i][start_day:start_day + self.interval])
				self.label_train[count] = self.table_list[i][start_day + self.interval].copy()
				for j in range(len(self.label_train[count])):
					try: self.label_train[count][j] = self.foodid_eachrank[i][self.label_train[count][j]]
					except KeyError: 
						self.label_train[count][j] = self.num_class-1
						count_error_label += 1
				
				count += 1
			
			if i % 200 == 0: print(i)
		if count != train_num:
			self.user_train = self.user_train[:count]
			self.ts_train = self.ts_train[:count]
			self.label_train = self.label_train[:count]
			print('not enough, count =', count, ", train_num =", train_num)
		print('count_error =', count_error)
		print('count_error_label =', count_error_label)
		'''print(self.ts_train[0][0])
		print(self.ts_train[0][1])
		print(self.ts_train[0][2])
		print(self.ts_train[6600][0])
		print(self.ts_train[6600][1])
		print(self.ts_train[6600][2])
		#exit()
		print('ts_train =', self.ts_train[0])
		print('ts_train =', self.ts_train[130])'''
		self.user_train, self.ts_train, self.label_train = shuffle(self.user_train, self.ts_train, self.label_train)
		#print('ts_train =', self.ts_train[0])
		#print('ts_train =', self.ts_train[130])
		count_error = 0
		label_array = np.zeros([self.user_train.shape[0], self.num_class], dtype = np.int8)
		for i in range(self.user_train.shape[0]):
			for k in range(len(self.label_train[i])):
				try: label_array[i][self.label_train[i][k]] = 1
				except IndexError: count_error += 1
		print('count_error =', count_error)
		self.label_train = label_array
	
	def build_model(self):
		from keras.models import Model, Sequential
		from keras.layers import Input, Dense, Lambda, Dropout, Concatenate, Flatten, Embedding
		from keras.layers import GRU, SimpleRNN
		
		if self.use_predefined_embedding or (self.use_popular_embedding and self.use_concat == 0): 
			user_input = Input(shape=[self.food_num], name='User')
			#user_vec = Dense(64, name='User-Emb')(user_input)
			user_vec = user_input
		elif self.use_popular_embedding and self.use_concat:
			pop_input = Input(shape=[self.food_num], name='User_pop')
			user_input = Input(shape=[1], name='User')
			user_embedding = Embedding(self.user_num + 1, 64, name='User-Emb')(user_input)
			user_vec = Flatten(name='FlattenUsers')(user_embedding)

		else: 
			user_input = Input(shape=[1], name='User')
			user_embedding = Embedding(self.user_num + 1, 64, name='User-Emb')(user_input)
			user_vec = Flatten(name='FlattenUsers')(user_embedding)
		
		history_input = Input(shape = [self.interval, self.num_class], name = 'History')
		#history_embedding = GRU(128, input_shape = (self.interval, self.num_class), return_sequences = False, name = "GRU")(history_input)
		history_embedding = SimpleRNN(128, input_shape = (self.interval, self.num_class), return_sequences = False, name = "GRU")(history_input)

		if self.use_popular_embedding and self.use_concat:
			x = Concatenate(axis = -1)([user_vec, pop_input, history_embedding])
		else: x = Concatenate(axis = -1)([user_vec, history_embedding])
		x = Dense(self.num_class, activation = 'sigmoid', name = "Dense")(x)
		if self.use_popular_embedding and self.use_concat:
			self.rec_model = Model([user_input, pop_input, history_input], x)
		else: self.rec_model = Model([user_input, history_input], x)
		self.rec_model.compile('adam', loss = "binary_crossentropy", metrics = ["accuracy"])
		#class_weights = self.calculating_class_weights(self.label_train)
		#self.rec_model.compile('adam', loss = self.get_weighted_loss(class_weights), metrics = ["accuracy"])
		
		self.rec_model.summary()
	
	def change_user_emb(self, user_id):
		user_emb = task2_rule_based.get_user_embedding()
		tmp = np.empty([user_id.shape[0], self.food_num], dtype = np.int8)
		for i in range(user_id.shape[0]):
			tmp[i] = np.copy(user_emb[int(user_id[i])])
		return tmp

	def change_user_emb_pop(self, user_id):
		user_emb = task2_rule_based.get_user_embedding_pop(self.num_class)
		tmp = np.empty([user_id.shape[0], self.food_num], dtype = np.int8)
		for i in range(user_id.shape[0]):
			tmp[i] = np.copy(user_emb[int(user_id[i])])
		if self.use_concat == False: return tmp
		else: return [user_id, tmp]

	def calculating_class_weights(self, y_true):
		from sklearn.utils.class_weight import compute_class_weight
		#print('y_true =', y_true.shape)
		number_dim = np.shape(y_true)[1]
		weights = np.empty([number_dim, 2])
		for i in range(number_dim):
			#weights[i] = compute_class_weight('balanced', [0.,1.], y_true[:, i])
			weights[i] = [1, 10]
		print('weights =', weights)
		return weights

	def get_weighted_loss(self, weights):
		def weighted_loss(y_true, y_pred):
			return K.mean((weights[:,0]**(1-y_true))*(weights[:,1]**(y_true))*K.binary_crossentropy(y_true, y_pred), axis=-1)
		return weighted_loss

	def train(self):
		#mlb = MultiLabelBinarizer()
		print('ts_train =', self.ts_train[0])
		print('ts_train =', self.ts_train[130])
		#self.ts_train = mlb.fit_transform(self.ts_train)
		#print(self.ts_train[0])
		#print(self.ts_train.shape)

		ts_array = np.zeros([self.user_train.shape[0], self.interval, self.num_class], dtype = np.int8)
		print('big =', ts_array.shape)
		count_error = 0
		for i in range(self.user_train.shape[0]):
			for j in range(self.interval):
				for k in range(len(self.ts_train[i][j])):
					tmp = self.ts_train[i][j][k]
					try: ts_array[i][j][tmp] = 1
					except IndexError: 
						print(i, j, tmp, ts_array.shape)
						count_error += 1
		print('count_error =', count_error)
		print('count key error =', np.sum(ts_array[:, :, self.num_class-1]))
		
		count_one = np.sum(self.label_train)
		print('count_one =', count_one, self.label_train.shape[0]*self.label_train.shape[1])
		
		if self.use_predefined_embedding: self.user_train = self.change_user_emb(self.user_train)
		if self.use_popular_embedding: self.user_train = self.change_user_emb_pop(self.user_train)
		if self.use_popular_embedding and self.use_concat:
			history = self.rec_model.fit([self.user_train[0], self.user_train[1], ts_array], self.label_train, epochs = self.epoch, validation_split = 0.2)
		else: history = self.rec_model.fit([self.user_train, ts_array], self.label_train, epochs = self.epoch, validation_split = 0.2)
								#class_weight = {0:1, 1:10})
		
		if self.use_popular_embedding and self.use_concat:
			pred = self.rec_model.predict([self.user_train[0][:1000], self.user_train[1][:1000], ts_array[:1000]])
		else: pred = self.rec_model.predict([self.user_train[:1000], ts_array[:1000]])
		#print('pred =', pred.shape)
		#print('round =', pred[0][:3], np.round(pred[0][:3]))
		count_one = np.sum(np.round(pred))
		print('count_one =', count_one, pred.shape[0]*pred.shape[1])
		pred = self.return_prob(pred)
		print('pred =', pred[0])
		print('real =', self.label_train[0])
		#exit()
		
		#else: x = [self.user_train, self.food_train]
		
		#history = model.fit([train.user_id, train.item_id], train.rating, epochs=100, verbose=0)
		pd.Series(history.history['loss']).plot(logy=True)
		plt.xlabel("Epoch")
		plt.ylabel("Train Error")
		plt.savefig('loss.png')

	def return_prob(self, pred):
		answer = np.empty([pred.shape[0], 20], dtype = int)

		for i in range(pred.shape[0]):
			pred[i][self.num_class-1] = 0
			tmp = np.argsort(-pred[i])
			#if i < 3:
			#	print(pred[i][tmp[0]], pred[i][tmp[1]], pred[i][tmp[2]], pred[i][tmp[99]])
			answer[i] = tmp[:20]
		print('return prob =', answer[0])
		print('return prob =', pred[0][:20])
		return answer

	def return_food_id(self, pred):
		new_pred = np.copy(pred)
		print('return food id, new_pred =', new_pred.shape)
		for i in range(pred.shape[0]):
			for j in range(20):
				new_pred[i][j] = self.eachrank_foodid[i][pred[i][j]]
		return new_pred

	def predict(self):
		print('predict and write results')
		user_test = np.array([i for i in range(self.user_num)])
		if self.use_predefined_embedding: user_test = self.change_user_emb(user_test)
		if self.use_popular_embedding: user_test = self.change_user_emb_pop(user_test)
		ts_test = np.zeros([self.user_num, self.interval, self.num_class])
		count_error = 0
		for i in range(self.user_num):
			start_day = len(self.table_list[i])-self.interval
			for j in range(self.interval):
				for k in range(len(self.table_list[i][start_day + j])):
					try:
						ts_test[i][j][self.table_list[i][start_day+j][k]] = 1
					except IndexError:
						count_error += 1
						ts_test[i][j][0] = 1
		print('predict, count_error =', count_error)
		if self.use_popular_embedding and self.use_concat:
			pred = self.rec_model.predict([user_test[0], user_test[1], ts_test])
		else: pred = self.rec_model.predict([user_test, ts_test])
		pred = self.return_prob(pred)
		pred = self.return_food_id(pred)
		print("pred =", pred.shape)

		#task2_rule_based.write_results(pred, 'gru_nb=200_units=128_interval=20', task = 2)
		task2_rule_based.write_results(pred, 'simpleRNN_nb=50_interval=10', task = 2)
	
def main():
	model = FoodRecommenderSystem()
	model.load_data()
	
	model.set_training_data()
	model.build_model()
	model.train()
	model.predict()
	
	


if __name__ == "__main__":
	main()


		
