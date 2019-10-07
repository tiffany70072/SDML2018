from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import roc_auc_score
import numpy as np 

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv1D
from keras.layers import Lambda, Dropout, Concatenate, Flatten, Activation
from keras.layers import LSTM, GRU, SimpleRNN
#from dGRU import GRU as dGRU

def naive_rnn_model(max_length = 64, features_length = 5, cell_type = 'gru', units = 256):
	# one rnn layer + one mlp layer
	inputs = Input(shape = (None, features_length,), name = "input")
	if cell_type == 'lstm': x = LSTM(units, input_shape = (max_length, features_length), return_sequences = 0)(inputs)
	elif cell_type == 'gru': x = GRU(units, return_sequences = 0)(inputs)
	elif cell_type == 'rnn': x = SimpleRNN(units, return_sequences = 0)(inputs)
	x = Dense(units, W_regularizer = 'l2', activation = "relu")(x)
	x = Dropout(0.5)(x)
	x = Dense(2, W_regularizer = 'l2', activation = "softmax")(x)
	model = Model(input = inputs, output = x)
	return model

def naive_mlp_model(input_shape = 64*5+4, units = 256): 
	# 2 layers mlp
	print('input_shape =', input_shape)
	inputs = Input(shape = (input_shape,), name = "input")
	x = Dense(units, W_regularizer = 'l2', activation = "relu")(inputs)
	x = Dropout(0.5)(x)
	x = Dense(units, W_regularizer = 'l2', activation = "relu")(x)
	x = Dropout(0.5)(x)
	x = Dense(2, W_regularizer = 'l2', activation = "softmax")(x)
	model = Model(input = inputs, output = x)
	return model

def mlp_rnn_model(input_mlp_shape = 4, max_length = 64, features_length = 5, units = 256): 
	# concat[1 layer mlp, 1 layer rnn] --> 1 mlp
	inputs1 = Input(shape = (input_mlp_shape,), name = "mlp_input")
	x1 = Dense(units, kernel_regularizer = 'l2', activation = "relu")(inputs1)
	
	inputs2 = Input(shape = (max_length, features_length,), name = "rnn_input")
	if features_length == 5:
		x2 = LSTM(units, input_shape = (max_length, features_length), return_sequences = 0)(inputs2)
		#x2 = GRU(units, input_shape = (max_length, features_length), return_sequences = 0)(inputs2)
	elif features_length == 10:
		print('dgru')
		#exit()
		x2 = dGRU(units, input_shape = (max_length, features_length), return_sequences = 0, name = 'dgru')(inputs2)
		
	x = Concatenate()([x1, x2])
	#x = Dropout(0.5)(x)
	x = Dense(units, kernel_regularizer = 'l2', activation = "relu")(x)
	x = Dropout(0.5)(x)
	x = Dense(2, kernel_regularizer = 'l2', activation = "softmax")(x)
	model = Model(input = [inputs1, inputs2], output = x)
	return model

def multi_mlp_rnn_model(max_length, input_mlp_shape = 4, features_length = 5, units = 256): 
	# concat[1 layer mlp, 1 layer rnn] --> 1 mlp
	inputs1 = Input(shape = (input_mlp_shape,), name = "mlp_input")
	x1 = Dense(units, W_regularizer = 'l2', activation = "relu")(inputs1)
	
	inputs2 = Input(shape = (max_length, features_length,), name = "rnn_input")
	if features_length == 5:
		x2 = GRU(units, input_shape = (max_length, features_length), return_sequences = 0)(inputs2)
	elif features_length == 10:
		x2 = dGRU(units, input_shape = (max_length, features_length), return_sequences = 0, name='dgru')(inputs2)

	x = Concatenate()([x1, x2])
	#x = Dropout(0.5)(x)
	x = Dense(units, W_regularizer = 'l2', activation = "relu")(x)
	x = Dropout(0.5)(x)
	x = Dense(2, W_regularizer = 'l2', activation = "softmax")(x)
	model = Model(input = [inputs1, inputs2], output = x)
	return model

def mlp_cnn_model(input_mlp_shape = 4, max_length = 8, features_length = 5, units = 128):
	# concat[1 layer mlp, 1 layer 1-d cnn] --> 1 mlp
	inputs1 = Input(shape = (input_mlp_shape,), name = "mlp_input")
	x1 = Dense(units, W_regularizer = 'l2', activation = "relu")(inputs1)
	
	# input_shape = (nb_features, 3)
	inputs2 = Input(shape = (max_length, features_length,), name = "cnn_input")
	x2 = Conv1D(filters = units, kernel_size = 3)(inputs2)
	x2 = Activation('relu')(x2)
	x2 = Flatten()(x2)

	x = Concatenate()([x1, x2])
	#x = Dropout(0.5)(x)
	x = Dense(units, W_regularizer = 'l2', activation = "relu")(x)
	x = Dropout(0.5)(x)
	x = Dense(2, W_regularizer = 'l2', activation = "softmax")(x)
	model = Model(input = [inputs1, inputs2], output = x)
	return model

def get_model(model_type, window_size, features_length = 5):
	if model_type == 'mlp': model = naive_mlp_model(input_shape = window_size*5+4)
	elif model_type == 'rnn': model = naive_rnn_model(max_length = window_size, features_length = features_length)
	elif model_type == 'mlp+rnn': model = mlp_rnn_model(max_length = window_size, features_length = features_length)
	elif model_type == 'multi': model = multi_mlp_rnn_model(max_length = window_size, input_mlp_shape = window_size*5+4, features_length = features_length)
	elif model_type == 'cnn': model = mlp_cnn_model(input_mlp_shape = window_size*5+4, max_length = window_size)
	else:
		print('no this model')
		exit()
	model.summary()
	#model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['acc'])
	from keras.optimizers import Adam
	adam = Adam(lr = 0.001)
	model.compile(optimizer = adam, loss='categorical_crossentropy', metrics=['acc'])
	return model

def evaluate(model, x, real):
	real = np.argmax(real, axis = 1)
	pred = model.predict(x)
	pred_score = pred[:, 1]
	pred = np.where(pred[:, 0]>0.5, 0, 1)
	#print('pred =', pred[:10])
	print('check labels = %.2f, %.2f' %(np.mean(real), np.mean(pred)))
	print('real =', real.shape, real[:10], "%.2f" %np.mean(real))
	print('pred =', pred.shape, pred[:10], "%.2f" %np.mean(pred))

	# AUC
	#fpr, tpr, thresholds = metrics.roc_curve(real, pred, pos_label=2)
	#print('AUC =', metrics.auc(fpr, tpr))
	#print('fpr, tpr, thresholds =', fpr, tpr, thresholds)
	print('AUC = %.3f,' %roc_auc_score(real, pred_score),)

	# self defined
	cm = confusion_matrix(real, pred)
	err_noicu = cm[0, 1]/float(cm[0, 0] + cm[0, 1])
	err_icu = cm[1, 0]/float(cm[1, 1] + cm[1, 0])
	my_score = (err_noicu + err_icu)*0.5
	print('ave = %.3f, err(noicu) = %.2f, err(icu) = %.2f' %(my_score, err_noicu, err_icu))
	print(confusion_matrix(real, pred))
	return

	# f1 score
	print('f1 =', precision_recall_fscore_support(real, pred, average='macro'))
	print('f1 =', precision_recall_fscore_support(real, pred))