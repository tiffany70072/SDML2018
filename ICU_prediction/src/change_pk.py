import numpy as np 
import pandas as pd


def read_npy(filename, length = None):
	arr = np.load('../data/' + filename + '.npy')
	print(filename, '=', arr.shape)#, arr[0])
	np.save
	if length == None: return arr
	else: return arr[:length]

icu = read_npy('icu_8') 
noicu = read_npy('noicu_8')
