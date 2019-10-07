"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import keras
from keras import backend
import tensorflow as tf
from sklearn.utils import shuffle

from model import Model
from timeit import default_timer as timer
from datetime import datetime
import random

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
sess = tf.Session(config=config)
keras.backend.set_session(sess)

def naive_mlp_model(input_shape, units = 256): 
	from keras.models import Model, Sequential
	from keras.layers import Input, Dense, Conv1D
	from keras.layers import Lambda, Dropout, Concatenate, Flatten, Activation
	from keras.layers import LSTM, GRU, SimpleRNN
	# 2 layers mlp
	model = Sequential()
	model.add(Dense(units, W_regularizer = 'l2', activation = "relu", input_shape=(input_shape, )))
	model.add(Dropout(0.5))
	model.add(Dense(units, W_regularizer = 'l2', activation = "relu"))
	model.add(Dropout(0.5))
	model.add(Dense(2, W_regularizer = 'l2', activation = "softmax"))
	return model

def rolling(x, y):
	icu_num = int(np.sum(y))
	print('icu_num =', icu_num, np.mean(y))
	icu = x[:icu_num]
	print('original size =', icu.shape[0])
	for i in range(5):
		icu = np.concatenate([icu, icu], axis = 0) # 32
	icu = icu[:icu_num*32]
	x = np.concatenate([icu, x[icu_num:]])
	y = np.ones(x.shape[0])
	y[:icu.shape[0]] *= 0
	print('rolling size =', icu.shape[0], x.shape[0])
	return x, y
	#exit()

def initialize_uninitialized_global_variables(sess):
	"""
	Only initializes the variables of a TensorFlow session that were not
	already initialized.
	:param sess: the TensorFlow session
	:return:
	"""
	# List all global variables
	global_vars = tf.global_variables()
	# Find initialized status for all variables
	is_var_init = [tf.is_variable_initialized(var) for var in global_vars]
	is_initialized = sess.run(is_var_init)
	# List all variables that were not initialized previously
	not_initialized_vars = [var for (var, init) in zip(global_vars, is_initialized) if not init]
	# Initialize all uninitialized variables found, if any
	if len(not_initialized_vars):
		sess.run(tf.variables_initializer(not_initialized_vars))

class FGSMAttack:
	def __init__(self, model, epsilon):
		"""Attack parameter initialization. The attack performs k steps of
			size a, while always staying within epsilon from the initial
			point."""
		self.model = model
		self.epsilon = epsilon

		loss = model.xent
		self.grad = tf.gradients(loss, model.x_input)[0]

	def perturb(self, x_nat, y, sess):
		"""Given a set of examples (x_nat, y), returns a set of adversarial
			examples within epsilon of x_nat in l_infinity norm."""

		x = np.copy(x_nat)
		#x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
		grad = sess.run(self.grad, feed_dict={self.model.x_input: x, self.model.y_input: y})
		x += self.epsilon * np.sign(grad)
		#x = np.clip(x, -1., 1.) # ensure valid pixel range
		return x

if __name__ == '__main__':
	import json
	import sys
	import math

	###################### get_data #####################
	from read_data import read_saved_data
	from baseline import preprocess_training_data
	from keras.utils.np_utils import to_categorical
	
	#exit()

	###################### other preprocessing #####################
	# Use label smoothing
	#label_smooth = .1
	#y_train = y_train.clip(label_smooth / 9., 1. - label_smooth)
	
	# Define input TF placeholder
	window_size = 8
	input_shape = window_size*5+4
	#x = tf.placeholder(tf.float32, shape=(None, input_shape))
	#y = tf.placeholder(tf.float32, shape=(None, 2))
	#model = naive_mlp_model(input_shape)

	config = {'model_dir': 'model_0106_test'}
	
	def train():
		#with open('config.json') as config_file: config = json.load(config_file)
		# Setting up training parameters
		tf.set_random_seed(12345678)

		max_num_training_steps = 100
		num_output_steps = 1
		num_summary_steps = 10
		num_checkpoint_steps = 30
		batch_size = 1024

		model_type = 'mlp'
		window_size = 8
		basicInf_train, history_train, y_train = read_saved_data(is_train = 1, window_size = window_size)
		x_train = preprocess_training_data(basicInf_train, history_train, model_type, window_size)

		x_train, y_train = rolling(x_train, y_train)
		print(np.percentile(x_train, [0, 1, 50, 99, 100]))

		#for idx in [0, 1, 100, 1000, 10000]: print(x_train[idx])
		#exit()
		x_train, y_train = shuffle(x_train, y_train)

		# Setting up the data and the model
		global_step = tf.contrib.framework.get_or_create_global_step()
		model = Model()
		# Setting up the optimizer
		train_step = tf.train.AdamOptimizer(1e-3).minimize(model.xent, global_step=global_step)
		# Set up adversary
		#attack = LinfPGDAttack(model, config['epsilon'], config['k'], config['a'], config['random_start'], config['loss_func'])
		attack = FGSMAttack(model, epsilon = 0.1)
		# Setting up the Tensorboard and checkpoint outputs
		model_dir = config['model_dir']
		import os
		if not os.path.exists(model_dir): os.makedirs(model_dir)

		saver = tf.train.Saver(max_to_keep=3)
		tf.summary.scalar('accuracy adv train', model.accuracy)
		tf.summary.scalar('accuracy adv', model.accuracy)
		tf.summary.scalar('xent adv train', model.xent / batch_size)
		tf.summary.scalar('xent adv', model.xent / batch_size)
		#tf.summary.image('images adv train', model.x_image)
		merged_summaries = tf.summary.merge_all()

		#shutil.copy('config.json', model_dir)
		print('training data =', x_train.shape, y_train.shape)
		print('check labels =', np.mean(y_train[:]))

		with tf.Session() as sess:
			# Initialize the summary writer, global variables, and our time counter.
			summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
			sess.run(tf.global_variables_initializer())
			training_time = 0.0
			num_examples = x_train.shape[0]
			batch_size = 1024
			num_batches = int(math.ceil(num_examples / batch_size))

			print('Iterating over {} batches'.format(num_batches))

			# Main training loop
			for ii  in range(max_num_training_steps):
				bstart = (ii%num_batches) * batch_size
				bend = min(bstart + batch_size, num_examples)

				x_batch = x_train[bstart:bend, :]
				y_batch = y_train[bstart:bend]

				# Compute Adversarial Perturbations
				start = timer()
				x_batch_adv = attack.perturb(x_batch, y_batch, sess)
				end = timer()
				training_time += end - start

				nat_dict = {model.x_input: x_batch, model.y_input: y_batch}
				adv_dict = {model.x_input: x_batch_adv, model.y_input: y_batch}

				# Output to stdout
				if ii % num_output_steps == 0:
					nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
					adv_acc = sess.run(model.accuracy, feed_dict=adv_dict)
					#print('Step {}:    ({})'.format(ii, datetime.now()))
					#print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
					#print('    training adv accuracy {:.4}%'.format(adv_acc * 100))
					print('Step {}: '.format(ii), end = '')
					print('train acc - nat {:.4}%'.format(nat_acc * 100), end = '')
					print(' - adv {:.4}%'.format(adv_acc * 100), end = '')
					if ii != 0:
						print('    {} examples per second'.format(num_output_steps * batch_size / training_time))
						training_time = 0.0
				# Tensorboard summaries
				if ii % num_summary_steps == 0:
					summary = sess.run(merged_summaries, feed_dict=adv_dict)
					summary_writer.add_summary(summary, global_step.eval(sess))

				# Write a checkpoint
				if ii % num_checkpoint_steps == 0:
			  		saver.save(sess, os.path.join(model_dir, 'checkpoint'), global_step=global_step)

				# Actual training step
				start = timer()
				sess.run(train_step, feed_dict=adv_dict)
				end = timer()
				training_time += end - start


	def attack():
		model_type = 'mlp'
		window_size = 8
		basicInf_train, history_train, y_train = read_saved_data(is_train = 1, window_size = window_size)
		x_train = preprocess_training_data(basicInf_train, history_train, model_type, window_size)

		icu_num = int(np.sum(y_train))
		x_train = x_train[:icu_num]
		#y_train = to_categorical(y_train, 2)
		#y_valid = to_categorical(y_valid, 2)
		print(np.percentile(x_train, [0, 1, 50, 99, 100]))
		model_file = tf.train.latest_checkpoint(config['model_dir'])
		if model_file is None:
			print('No model found')
			sys.exit()    

		model = Model()        
		saver = tf.train.Saver()

		with tf.Session() as sess:
			# Iterate over the samples batch-by-batch
			saver.restore(sess, model_file)
			initialize_uninitialized_global_variables(sess)
			num_eval_examples = x_train.shape[0]
			eval_batch_size = 1024
			num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
			x_adv = [] # adv accumulator

			print('Iterating over {} batches'.format(num_batches))

			for i in range(1):
				attack = FGSMAttack(model, epsilon = 0.1)#*random.random()) 
				for ibatch in range(num_batches):

					bstart = ibatch * eval_batch_size
					bend = min(bstart + eval_batch_size, num_eval_examples)
					print('batch size: {}'.format(bend - bstart))

					x_batch = x_train[bstart:bend, :]
					y_batch = y_train[bstart:bend]
					x_batch_adv = attack.perturb(x_batch, y_batch, sess)

					x_adv.append(x_batch_adv)

			print('Storing examples')
			#path = config['store_adv_path']
			x_adv = np.concatenate(x_adv, axis=0)
			print('x_adv =', x_adv.shape)
			#np.save('adv_fgsm/adv_fgsm_' + sys.argv[2], x_adv)
			np.save('adv_fgsm_no_resample/adv_fgsm_' + sys.argv[2], x_adv)
			print('save at: ', sys.argv[2])
			#print('Examples stored in {}'.format(path))

	if sys.argv[1] == '1': train()
	if sys.argv[1] == '2': attack()

