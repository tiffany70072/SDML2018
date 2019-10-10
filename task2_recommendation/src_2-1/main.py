import keras.backend as K
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import sys
import tensorflow as tf 

import baseline
import build_model
import read_data
import utils

from keras.utils import np_utils
from numpy.random import seed
from sklearn.utils import shuffle
from sklearn.preprocessing import minmax_scale
from tensorflow import set_random_seed
seed(1)
set_random_seed(2)
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
sess = tf.Session(config=config)
K.set_session(sess)

# Usage: CUDA_VISIBLE_DEVICES=0 python main.py 


class FoodRecommenderSystem(object):
	def __init__(self):
		self.food_num = 5532 # (40, 2039)
		self.user_num = 2608
		self.food_features_dim = 1118
		self.latent_dim = 64
		self.is_check_valid = 0 # split training and validation set
		self.only_train_popular_food = 0
		self.only_predict_popular_food = 1 # default: 1
		self.use_log1p = 0 # default: 0
		self.use_food_bias = 0 # default: 0
		self.use_food_features = 0 # turn to factorization machine, default: 0
		self.use_food_normal = 0
		self.scale_rating = 0
		#self.model_type = 'mlp' # with food_features = 0
		#self.model_type = 'rank_label' # with food_features = 0
		self.model_type = 'mf'

		self.epochs = 1
		self.task = 2
		#self.n_ratio = 1

	def load_data(self):
		#self.table = read_data.build_mf_table()
		handle = open('data/rating_user_time_item', 'rb')
		self.table_list = pickle.load(handle) # user, count_date, item
		if self.use_food_features == 1:
			self.food_features = read_data.read_food_features()

	def remove_eaten(self, task=1): # set eaten food in table_valid into zero
		if task == 1:
			target = np.copy(self.table_valid)
			for i in range(self.user_num):
				for j in range(self.food_num):
					if self.table_train[i][j] != 0: 
						target[i][j] = 0
		return target

	def is_eaten(self, task=1):
		is_eaten_table = np.zeros(self.table_train.shape) # 1: not eaten in table_train
		for i in range(self.user_num):
			for j in range(self.food_num):
				if self.table_train[i][j] == 0: 
					is_eaten_table[i][j] = 1
		return is_eaten_table

	def get_valid_answer(self, task = 1): # return the id which needs to be predict
		#rank_list = [[] for i in range(user_num)] # each pair (user_id: (sorted item id))
		rank_array = self.remove_eaten()
		rank_array = -rank_array
		rank_argsort = np.argsort(rank_array)
		rank_sort = np.sort(rank_array)
		return rank_argsort[:, :20]

	def set_training_validation_table(self):
		self.table_train = np.zeros([self.user_num, self.food_num]) # user: array, item: array --> count item
		if self.is_check_valid: 
			self.table_valid = np.zeros([self.user_num, self.food_num])
		for i in range(len(self.table_list)):
			for j in range(len(self.table_list[i])):
				for k in range(len(self.table_list[i][j])):
					if not self.is_check_valid or j < len(self.table_list[i]) * 0.8: 
						self.table_train[i][self.table_list[i][j][k]] = 1
					else: 
						self.table_valid[i][self.table_list[i][j][k]] += 1

		#np.save('rating_train.npy', self.table_train)
		#np.save('rating_valid.npy', self.table_valid)
		del self.table_list
		if self.scale_rating:
			print(self.table_train[2][:10])
			self.table_train = minmax_scale(self.table_train, axis=1)
		print('table_train = (first three row)')
		for i in range(3): 
			print(self.table_train[i][:10])
		print('table_train =', np.sum(self.table_train))
		if self.is_check_valid: 
			print("table_valid =", np.sum(self.table_valid))
		
	def get_training_num(self):
		count = 0
		for i in range(self.table_train.shape[0]):
			for j in range(self.table_train.shape[1]):
				if self.table_train[i][j] != 0: 
					count += 1
		print('valid ratio =', count, count / float(self.table_train.shape[0] * self.table_train.shape[1]))
		return count

	def select_data_by_popular_food(self):
		popular_food_id = utils.select_popular_food(self.table_train)
		'''self.table_train = self.table_train[:, valid_food_id]
		self.table_valid = self.table_valid[:, valid_food_id]
		self.food_num = len(valid_food_id)
		print('table_train (popular food) =', self.table_train.shape)'''
		
		count = 0
		print('user_train =', self.user_train.shape)
		for i in range(self.table_train.shape[0]):
			for j in range(self.table_train.shape[1]):
				if self.table_train[i][j] != 0 and j in popular_food_id:
					self.user_train[count] = i
					self.food_train[count] = j
					self.target[count] = self.table_train[i][j]
					count += 1
		self.user_train, self.food_train, self.target = self.user_train[:count], self.food_train[:count], self.target[:count]
		print("user_train =", self.user_train.shape)

	def set_training_data(self):
		train_num = self.get_training_num()
		#user_train = np.zeros([train_num, self.user_num])
		#food_train = np.zeros([train_num, self.food_num])
		self.user_train = np.zeros([train_num])
		self.food_train = np.zeros([train_num], dtype=int)
		self.target = np.zeros([train_num])
		if self.only_train_popular_food:
			print('user_train =', self.user_train.shape)
			self.select_data_by_popular_food()
		else:
			count = 0
			for i in range(self.table_train.shape[0]):
				for j in range(self.table_train.shape[1]):
					if self.table_train[i][j] != 0:
						self.user_train[count] = i
						self.food_train[count] = j
						self.target[count] = self.table_train[i][j]
						count += 1
		if self.use_log1p: 
			self.target = np.log1p(self.target)
		if self.use_food_features:
			self.set_food_features(train_num)
			self.user_train, self.food_train, self.food_features_train, self.target = shuffle(
				self.user_train, self.food_train, self.food_features_train, self.target)
		else:
			self.user_train, self.food_train, self.target = shuffle(
				self.user_train, self.food_train, self.target)
	
	def set_food_features(self, train_num):
		self.food_features_train = np.zeros([train_num, self.food_features_dim])
		for i in range(train_num):
			self.food_features_train[i] = self.food_features[self.food_train[i]]

	def get_model(self):
		if self.use_food_features:
			self.mf = build_model.factorization_machine(self.user_num, self.food_num, self.food_features_dim, self.latent_dim)
		elif self.model_type == 'mlp': 
			self.mf = build_model.MLP(self.user_num, self.food_num, self.latent_dim)
		elif self.model_type == 'rank_label': 
			self.mf = build_model.mf_ranking_label_loss(self.user_num, self.food_num, self.latent_dim)
		else:
			self.mf = build_model.matrix_factorization(self.user_num, self.food_num, self.latent_dim, use_food_bias=self.use_food_bias)
			
	def train_mf(self):
		if self.use_food_features:
			x = [self.user_train, self.food_train, self.food_features_train]
		else: 
			x = [self.user_train, self.food_train]
		history = self.mf.fit(x, self.target, epochs=self.epochs, validation_split=0.2)
		pd.Series(history.history['loss']).plot(logy=True)
		plt.xlabel("Epoch")
		plt.ylabel("Train Error")
		plt.savefig('loss.png')
	
	def select_pred_by_popular_food(self, pred, user_id):
		for i in range(pred.shape[0]):
			if i not in self.popular_food_id: 
				pred[i] = 0
		return pred

	def evaluate_mf(self):
		if self.is_check_valid: 
			rank_argsort = self.get_valid_answer()
		if self.only_predict_popular_food:
			self.popular_food_id = utils.select_popular_food(self.table_train)

		is_eaten_table = self.is_eaten()
		user_valid = np.zeros([self.food_num])
		food_valid = np.array([i for i in range(self.food_num)])
		if self.use_food_features:
			food_features_valid = np.copy(self.food_features)
		score = 0
		all_pred = np.empty([self.user_num, 20])
		for i in range(self.user_num):
			if self.use_food_features:
				pred = self.mf.predict([user_valid, food_valid, food_features_valid])
			else: 
				pred = self.mf.predict([user_valid, food_valid])
			user_valid += 1
			pred = pred[:, 0]
			if self.task == 1: 
				pred = pred * is_eaten_table[i]
			if self.only_predict_popular_food: 
				pred = self.select_pred_by_popular_food(pred, i)
			pred_argsort = np.argsort(-pred)

			if self.is_check_valid: 
				score += utils.AP(rank_argsort[i, :20], pred_argsort[:20])
			else: 
				all_pred[i] = pred_argsort[:20]
		if self.is_check_valid: 
			print('MAP@20 = %.4f' % (score/float(self.user_num) / 20.0))
		elif self.task == 1: 
			baseline.write_results(all_pred, 'mf(minmax_scale)(pred:pop=100)')
		else: 
			baseline.write_results(all_pred, 'mf(pred:pop=100)_by_people', task=2)
	
	
def main():
	model = FoodRecommenderSystem()
	model.load_data()
	model.set_training_validation_table()
	model.get_model()

	if model.model_type == 'rank_label':
		print('rank_label')
		model.set_paired_training_data()
		model.train_mf_rank()
		model.evaluate_mf_rank()
	else:
		model.set_training_data()
		model.train_mf()
		model.evaluate_mf()
	

if __name__ == "__main__":
	main()	

