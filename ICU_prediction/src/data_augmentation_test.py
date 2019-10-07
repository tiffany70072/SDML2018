import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np 
import sys
import tensorflow as tf
import utils

from keras import backend as K
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from read_data import read_saved_data
from sklearn.decomposition import PCA
from sklearn.utils import shuffle

from baseline import preprocess_training_data
from read_data import read_saved_data


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
session = tf.Session(config=config)
K.set_session(session)


def plot_pca():
	model_type = 'mlp'
	window_size = 8
	basicInf_train, history_train, y_train = read_saved_data(is_train=1, window_size=window_size)
	x_train = preprocess_training_data(basicInf_train, history_train, model_type, window_size)
	x_adv = np.load('adv_fgsm_0106_test.npy')
	print('x_adv =', x_adv.shape)
	icu_num = int(np.sum(y_train))

	print('x =', x_train.shape, x_adv.shape)
	def get_dim_from_pca(pred):
		pca = PCA(n_components=2)
		pca.fit(pred)
		PCA(copy=True, iterated_power='auto', n_components=2, random_state=None, svd_solver='auto', tol=0.0, whiten=False)
		print('ratio =', pca.explained_variance_ratio_)  
		y = pca.transform(pred)
		print('y =', y.shape) 
		return y

	x = np.concatenate([x_train, x_adv], axis = 0)
	colors = ['C0', 'C9', 'C2', 'C8', 'C1', 'C3']
	x = get_dim_from_pca(x)

	for j in range(0, x_train.shape[0], 100):
		if j < icu_num: 
			color = 'r'  # idx = 0
		elif j < x_train.shape[0]: 
			color = 'g'  # idx = 1
		elif j < x_train.shape[0] + x_adv.shape[0]: 
			color = 'b'  # idx = 2
		
		plt.plot(x[j][0], x[j][1], color=color, marker='o') 
		if j < icu_num: 
			plt.plot(x[j+1][0], x[j+1][1], color=color, marker='o') 
			plt.plot(x[j+2][0], x[j+2][1], color=color, marker='o') 
		plt.show() 
		plt.savefig('plt/0106_only_train')

		if j % 10000 == 0: 
			print(j)

			
def train():
	model_type = 'mlp+rnn' 
	window_size = 8
	basicInf_train, history_train, y_train = read_saved_data(is_train=1, window_size=window_size)
	x_train = preprocess_training_data(basicInf_train, history_train, model_type, window_size)
	basicInf_valid, history_valid, y_valid = read_saved_data(is_train=0, window_size=window_size)
	x_valid = preprocess_training_data(basicInf_valid, history_valid, model_type, window_size)

	for i in range(1, 11):
		if i == 1: 
			x_adv = np.load('adv_fgsm/adv_fgsm_' + str(i) + '.npy')
		else: 
			x_adv = np.r_[x_adv, np.load('adv_fgsm/adv_fgsm_' + str(i) + '.npy')]
	y_adv = np.ones([x_adv.shape[0]])
	x_adv = preprocess_training_data(x_adv[:, :4], x_adv[:, 4:].reshape([-1, window_size, 5]), model_type, window_size)
	print('x_adv =', x_adv[0].shape, x_train[0].shape)

	#x_train = np.r_[x_train, x_adv]
	x_train = [np.r_[x_train[0], x_adv[0]], np.r_[x_train[1], x_adv[1]]]
	y_train = np.r_[y_train, y_adv]
	print('check labels =', np.sum(y_train), np.mean(y_train))
	y_train = to_categorical(y_train, 2)
	y_valid = to_categorical(y_valid, 2)

	# Training
	model = utils.get_model(model_type, window_size)
	class_weight = {0: 1 - np.mean(y_train[:, 0]), 1: 1.}
	earlyStopping = EarlyStopping(monitor='val_loss', patience=3)
	model.fit(x_train, y_train, epochs=100, batch_size=512, validation_data=(x_valid, y_valid), 
		shuffle=1, class_weight=class_weight, callbacks=[earlyStopping])

	# Evaluation
	utils.evaluate(model, x_train, y_train)
	utils.evaluate(model, x_valid, y_valid)
	print('cw =', 1-np.mean(y_train[:, 0]))

	
if sys.argv[1] == '1': 
	train()
elif sys.argv[1] == '2': 
	plot_pca()


	
