from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import regularizers

from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

def MLP(input_dim):
	#model = Sequential([
	#	Dense(128, input_dim=input_dim), Activation('relu'),
	#	Dense(128), Activation('relu'),
	#	Dense(2), Activation('softmax'),])
	units = 512
	model = Sequential([
		Dense(units, input_dim=input_dim), Activation('sigmoid'), Dropout(0.5), 
		Dense(units), Activation('sigmoid'), Dropout(0.5), Dense(units), Activation('sigmoid'), Dropout(0.5),
		Dense(2), Activation('softmax'),])
	#model = Sequential([
	#	Dense(units, input_dim=input_dim, kernel_regularizer=regularizers.l2(0.01)), Activation('sigmoid'),
	#	Dense(units, kernel_regularizer=regularizers.l2(0.01)), Activation('sigmoid'),
	#	Dense(2), Activation('softmax'),])
	model.summary()
	return model

def SVM():
	#return svm.SVC(gamma='auto')
	#return svm.SVC(kernel='linear')
	return LinearSVC(random_state=0, tol=1e-5)

def randomforest():
	return RandomForestClassifier(n_estimators=1000, max_depth=3, random_state=0)

	