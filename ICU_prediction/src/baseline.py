"""Some baseline methods for this problem. Usage: time python baseline.py rnn"""


import numpy as np 
import sys
import tensorflow as tf
import utils

from keras import backend as K
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from read_data import read_saved_data
from sklearn.utils import shuffle

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
session = tf.Session(config=config)
K.set_session(session)


def preprocess_training_data(basicInf, history, model_type, window_size):
	if model_type == 'mlp':
		x = np.concatenate([basicInf, np.reshape(history, [-1, window_size*5])], axis=1)
		print('x =', x.shape)
	elif model_type == 'rnn': # only use history data (without basicInf)
		x = np.copy(history)
	elif model_type == 'mlp+rnn': 
		x = [basicInf, history]
	elif model_type == 'multi' or model_type == 'cnn':
		x = np.concatenate([basicInf, np.reshape(history, [-1, window_size*5])], axis=1)
		x = [x, history]
	return x
	

def main():
	model_type = sys.argv[1]  # One of 'mlp', 'rnn', 'mlp+rnn', 'multi', 'cnn'.
	window_size = 8  # Max length of time series data for each person.
	cw = 30.
  
	# Read data
	basicInf_train, history_train, y_train = read_saved_data(is_train=1, window_size=window_size)
	basicInf_valid, history_valid, y_valid = read_saved_data(is_train=0, window_size=window_size)
	basicInf_train, history_train, y_train = shuffle(basicInf_train, history_train, y_train)
	
	# basicInf: first 4 col in icuxx.npy
	# history: 5 time series col (without and preprocessing and scaling) in icuxx.py
	# labels: 0 (no ICU) or 1 (ICU)
	x_train = preprocess_training_data(basicInf_train, history_train, model_type, window_size)
	x_valid = preprocess_training_data(basicInf_valid, history_valid, model_type, window_size)
	y_train = to_categorical(y_train, 2)
	y_valid = to_categorical(y_valid, 2)
	print('check labels =', np.sum(y_train), np.mean(y_train))
	
	# Training.
	model = utils.get_model(model_type, window_size)
	#class_weight = {0: 1., 1: 50.}
	class_weight = {0: 1., 1: cw}
	earlyStopping = EarlyStopping(monitor='val_loss', patience=3)
	model.fit(x_train, y_train, validation_data = (x_valid, y_valid), 
		  epochs=50, batch_size=256, shuffle=0, 
		  class_weight=class_weight, 
		  callbacks=[earlyStopping])

	# Evaluation.
	utils.evaluate(model, x_train, y_train)
	utils.evaluate(model, x_valid, y_valid)
	print('class weight = %.d', cw)


if __name__ == '__main__':
	main()
