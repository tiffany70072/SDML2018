from read_data import read_saved_data
from baseline import preprocess_training_data
import numpy as np
from sklearn.utils import shuffle

'''import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
sess = tf.Session(config=config)
keras.backend.set_session(sess)'''

window_size = 8
model_type = 'mlp'
basicInf_train, history_train, y_train = read_saved_data(is_train = 1, window_size = window_size)
basicInf_valid, history_valid, y_valid = read_saved_data(is_train = 0, window_size = window_size)

x_train = preprocess_training_data(basicInf_train, history_train, model_type, window_size)
x_valid = preprocess_training_data(basicInf_valid, history_valid, model_type, window_size)

icu_num = int(np.sum(y_train))
icu = x_train[:icu_num]
noicu = x_train[icu_num:2*icu_num]

print(np.sum(y_train[:icu_num]))
print(np.sum(y_train[icu_num:]))
for i in range(4):
	value = []
	for j in range(100, 200):
		for k in range(100, 200):
			value.append(abs(icu[j][i] - noicu[k][i]))
	print(i, "%.3f" %np.mean(np.array(value)))

for i in range(4, 44, 8):
	value = []
	for j in range(100):
		for k in range(100):
			value.append(abs(icu[j][i] - noicu[k][i]))
	print(i, "%.3f" %np.mean(np.array(value)))

'''
0 1.072
1 0.500
2 0.125
3 0.301
4 0.636
12 0.250
20 0.719
28 0.945
36 0.836
'''
