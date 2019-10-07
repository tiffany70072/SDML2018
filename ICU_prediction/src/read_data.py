"""Data reader."""

import numpy as np 
import pandas as pd
import pdb

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def read_pd(filename):
	df = pd.read_csv('../data/' + filename + '.npy')
	if debug: 
		print('data.shape', tmp.shape)
	return np.array(tmp)

def read_npy(filename, length = None):
	filename = '../data/' + filename + '.npy'
	arr = np.load(filename, fix_imports=1, encoding='latin1')
	print(filename, '=', arr.shape)
	if length == None: 
		return arr
	else: 
		return arr[:length]


def check_length():
	arr = read_npy('noicu_64')
	length_list = []
	for i in range(arr.shape[0]):
		for j in range(4, 9):
			length_list.append(len(arr[i][j]))
	length_list = np.array(length_list)
	print('length_list =', length_list.shape, length_list[:100])
	print('percentile =', np.percentile(length_list, [0, 1, 10, 25, 50, 75, 90, 99, 100]))
	

def get_basicInf(icu, noicu):
	basicInf = np.concatenate([icu[:, :4], noicu[:, :4]], axis = 0)
	basicInf = basicInf.astype(np.float64)
	basicInf_nan = np.isnan(basicInf)
	for j in range(4):
		basicInf[:, j] = np.where(np.isnan(basicInf[:, j]), np.nanmean(basicInf[:, j]), basicInf[:, j])
	return basicInf, basicInf_nan


def get_history(icu, noicu, window_size):
	data_num = icu.shape[0]+noicu.shape[0]
	whole = np.concatenate([icu[:, 4:9], noicu[:, 4:9]], axis = 0)
	
	history = np.zeros([data_num, 5, window_size])
	count = 0
	for i in range(data_num):
		if len(whole[i][0]) < 8: 
			count += 1
		for j in range(5):
			if window_size > len(whole[i][j]):
				history[i][j][window_size-len(whole[i][j]):] = np.copy(np.array(whole[i][j]))
				history[i][j][:window_size-len(whole[i][j])] = np.nan
			else:
				history[i][j] = np.copy(np.array(whole[i][j][-window_size:]))
	print('error =', count / float(data_num))
	print('nanmean =', np.nanmean(history))
	
	history = np.transpose(history, [0, 2, 1])
	history_nan = np.isnan(history)
	for j in range(5):	
		history[:, :, j] = np.where(np.isnan(history[:, :, j]), np.nanmean(history[:, :, j]), history[:, :, j])
	return history, history_nan


def get_labels(icu, noicu):
	data_num = icu.shape[0] + noicu.shape[0]
	labels = np.ones([data_num])
	labels[icu.shape[0]:] *= 0
	print('labels =', np.mean(labels)) 
	return labels


def read_saved_data(is_train, window_size=8, with_nan=False):
	if is_train: 
		tail = 'train'
	else: 
		tail = 'test'
		
	icu = read_npy('icu_8_period_8_' + tail) 
	noicu = read_npy('noicu_8_period_8_' + tail)
	basicInf, basicInf_nan = get_basicInf(icu, noicu)
	history, history_nan = get_history(icu, noicu, window_size)
	labels = get_labels(icu, noicu)
	print(tail + ' - check shape, basicInf =', basicInf.shape, ', history =', history.shape, ', labels =', labels.shape)
	
	if with_nan == 0: 
		return basicInf, history, labels
	else: 
		return basicInf, history, labels, basicInf_nan, history_nan
	

def split_data(icu, noicu):
	icu_train, icu_valid = train_test_split(icu, test_size=0.2, random_state=42)
	noicu_train, noicu_valid = train_test_split(noicu, test_size=0.2, random_state=42)
	print('shape =', icu_train.shape, icu_valid.shape, noicu_train.shape, noicu_valid.shape)
	return icu_train, icu_valid, noicu_train, noicu_valid


def main():
	read_npy('icu_64')
	check_length()
	read_saved_data(1, window_size=8)
	

if __name__ == '__main__':
	main()
