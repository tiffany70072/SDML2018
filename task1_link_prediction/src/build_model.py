from keras import regularizers
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from keras.utils import np_utils

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC


def MLP(input_dim, units=512):
	model = Sequential([
		Dense(units, input_dim=input_dim), 
		Activation('sigmoid'), 
		Dropout(0.5), 
		Dense(units), 
		Activation('sigmoid'), 
		Dropout(0.5), 
		Dense(units), 
		Activation('sigmoid'), 
		Dropout(0.5),
		Dense(2), 
		Activation('softmax'),])
	
	model.summary()
	return model


def SVM():
	return LinearSVC(random_state=0, tol=1e-5)


def randomforest():
	return RandomForestClassifier(n_estimators=1000, max_depth=3, random_state=0)
