import numpy as np 
import sys
from read_data import read_saved_data
import utils

from sklearn.utils import shuffle


from keras.utils.np_utils import to_categorical
from keras import backend as K
from keras import optimizers
from keras.callbacks import EarlyStopping

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
session = tf.Session(config=config)
K.set_session(session)

def preprocess_training_data(basicInf, history, basicInf_nan, history_nan, model_type, window_size):
	if model_type == 'mlp':
		x = np.concatenate([basicInf, np.reshape(history, [-1, window_size*5])], axis = 1)
		print('x =', x.shape)
	elif model_type == 'rnn': # only use history data (without basicInf)
		x = np.concatenate([history, history_nan], axis = 2)
		print('x =', x.shape)
	elif model_type == 'mlp+rnn': 
		history = np.concatenate([history, history_nan], axis = 2)
		x = [basicInf, history]
	elif model_type == 'multi':
		x = np.concatenate([basicInf, np.reshape(history, [-1, window_size*5])], axis = 1)
		history = np.concatenate([history, history_nan], axis = 2)
		x = [x, history]
	return x

# usage: time python baseline.py rnn
def main():
	model_type = sys.argv[1] # one of 'mlp', 'rnn', 'mlp+rnn'
	window_size = 8 # Max length of time series data for each person
	cw = 1.

	# read data
	basicInf_train, history_train, y_train, basicInf_nan_train, history_nan_train = read_saved_data(is_train = 1, window_size = window_size, with_nan = 1)
	basicInf_valid, history_valid, y_valid, basicInf_nan_valid, history_nan_valid = read_saved_data(is_train = 0, window_size = window_size, with_nan = 1)
	basicInf_train, history_train, y_train, basicInf_nan_train, history_nan_train = shuffle(basicInf_train, history_train, y_train, basicInf_nan_train, history_nan_train)
	
	# basicInf: first 4 col in icuxx.npy
	# history: 5 time series col (without and preprocessing and scaling) in icuxx.py
	# labels: 0 (no ICU) or 1 (ICU)
	x_train = preprocess_training_data(basicInf_train, history_train, basicInf_nan_train, history_nan_train, model_type, window_size)
	x_valid = preprocess_training_data(basicInf_valid, history_valid, basicInf_nan_valid, history_nan_valid, model_type, window_size)
	y_train = to_categorical(y_train, 2)
	y_valid = to_categorical(y_valid, 2)
	print('check labels =', np.sum(y_train), np.mean(y_train))

	
	# training
	model = utils.get_model(model_type, window_size, features_length = 10)
	class_weight = {0: 1., 1: cw}
	earlyStopping = EarlyStopping(monitor = 'val_loss', patience = 3)
	model.fit(x_train, y_train, epochs = 100, batch_size = 512, validation_data = (x_valid, y_valid), 
		shuffle = False, class_weight = class_weight, callbacks = [earlyStopping])

	# evaluation
	utils.evaluate(model, x_train, y_train)
	utils.evaluate(model, x_valid, y_valid)
	print('class weight = %.d', cw)

if __name__ == '__main__':
	main()


