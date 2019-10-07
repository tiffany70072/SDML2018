import numpy as np 
from read_data import read_npy


def preprocessing():
	icu = read_npy('icu_8') 
	noicu = read_npy('noicu_8') 
	basicInf = np.concatenate([icu[:, :4], noicu[:, :4]], axis = 0)
	basicInf = basicInf.astype(np.float64)
	#basicInf = np.where(np.isnan(basicInf), 0, basicInf)

	data_num = icu.shape[0]+noicu.shape[0]
	whole = np.concatenate([icu[:, 4:9], noicu[:, 4:9]], axis = 0)
	
	window_size = 64
	history = np.zeros([data_num, 5, window_size])
	for i in range(data_num):
		for j in range(5):
			if window_size > len(whole[i][j]):
				history[i][j][window_size-len(whole[i][j]):] = np.copy(np.array(whole[i][j]))
			else:
				history[i][j] = np.copy(np.array(whole[i][j][-window_size:]))
	history = np.transpose(history, [0, 2, 1])
	#history = np.where(np.isnan(history), 0, history)

	return basicInf, history

def calculate_nan_ratio(arr):
	if len(arr.shape) == 2:
		for i in range(arr.shape[1]):
			n = np.count_nonzero(np.isnan(arr[:, i]))
			print(i, ', num =', n, n/float(arr.shape[0]))
	elif len(arr.shape) == 3:
		print('dim 1')
		for i in range(arr.shape[1]):
			n = np.count_nonzero(np.isnan(arr[:, i]))
			print '%.1f\t' %(100.0*n/float(arr.shape[0]*arr.shape[2])),
			if i % 10 == 9: print ""
		print('\ndim 2')
		for i in range(arr.shape[2]):
			n = np.count_nonzero(np.isnan(arr[:, :, i]))
			print(i, 'num =', n, n/float(arr.shape[0]*arr.shape[1]))


def main():
	basicInf, history = preprocessing()
	calculate_nan_ratio(basicInf)
	calculate_nan_ratio(history)
	'''
	(0, ', num =', 0, 0.0)
	(1, ', num =', 0, 0.0)
	(2, ', num =', 3304, 0.08796826326579515)
	(3, ', num =', 2280, 0.06070449159988285)
	dim 1
	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	
	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	
	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	
	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	
	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	
	0.0	0.0	0.0	0.0	0.0	0.0	0.4	5.1	7.9	7.8	
	7.8	7.9	8.3	7.8	
	dim 2
	(0, 'num =', 12480, 0.005191831518411033)
	(1, 'num =', 12503, 0.0052013997976516945)
	(2, 'num =', 27432, 0.011412045049122714)
	(3, 'num =', 28440, 0.01183138528714822)
	(4, 'num =', 18703, 0.007780675071221279)

	only icu
	(0, ', num =', 0, 0.0)
	(1, ', num =', 0, 0.0)
	(2, ', num =', 130, 0.07946210268948656)
	(3, ', num =', 46, 0.028117359413202935)
	dim 1
	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	
	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	
	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	
	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	
	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	
	0.0	0.0	0.0	0.0	0.0	0.0	7.0	8.6	8.2	9.0	
	9.7	10.1	7.8	4.1	
	dim 2
	(0, 'num =', 618, 0.005902353300733496)
	(1, 'num =', 618, 0.005902353300733496)
	(2, 'num =', 1752, 0.01673288508557457)
	(3, 'num =', 1390, 0.0132755195599022)
	(4, 'num =', 890, 0.008500152811735941)

	basic idea
	0, const
	mean, median
	mean, median of group
	ts prediction
	'''



if __name__ == '__main__':
	main()
