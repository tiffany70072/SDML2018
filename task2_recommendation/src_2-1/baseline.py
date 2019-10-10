"""Guess the most common one (for task1 and task2)."""

import numpy as np 
import pickle 
import read_data

from scipy.spatial.distance import cosine, euclidean



def output_task1(rank_array, eaten_array):
	# rank_array[user] = [2, 4, 3], prediction for each food
	# eaten_array[user] = [1, 3, 3], history count for each food for this user
	pred = np.empty([user_num, 20])
	for i in range(user_num):
		for j in range(food_num):
			if eaten_array[i][j] != 0: 
				rank_array[i][j] = 0
		rank_array[i] = -rank_array[i]
		rank_argsort = np.argsort(rank_array[i])[:20]
		rank_sort = np.sort(rank_array[i])[:20]
		pred[i] = np.copy(rank_argsort)
	return pred
	
	
def write_results(pred, filename='test', task=1):
	if task == 1: 
		fout = open('result/' + filename + '.csv', 'w')
	else: 
		fout = open('result-task2/' + filename + '.csv', 'w') 
		print('task =', task)
	fout.write('userid,foodid\n')
	handle = open('data/id_userID', 'rb')
	id_userID = pickle.load(handle)
	for i in range(pred.shape[0]):
		fout.write(str(id_userID[i]) + ',')
		for j in range(20): 
			fout.write(str(int(pred[i][j])) + " ")
		fout.write("\n")

		
def count_highest_food(task=1):
	table_array = read_data.build_mf_table(ratio=0.95) 
	#table_array = read_data.build_mf_table_by_people(ratio = 0.5) # one person at most eat one time
	rank_array = np.copy(table_array)
	tmp = np.sum(table_array, axis=0)
	print('rank_array =', tmp[:10])

	if task == 1: 
		pred = output_task1(rank_array, table_array)
		write_results(pred, 'count_most_(by_people)_remove_eaten')
	if task == 2:
		print('task =', task)
		pred = np.argsort(-rank_array)[:, :20]
		write_results(pred, 'count_most_(self, by_count)_last95%', task=2)

		
def compute_from_mf():
	user_embed = np.load('../all/user_embed_mf.npy')
	similarity = np.zeros([user_num, user_num]) 
	for i in range(user_num):
		for j in range(i, user_num):
			one_sim = 1 - cosine(user_embed[i],user_embed[j])
			similarity[i][j] = one_sim
			similarity[j][i] = one_sim
	return similarity


def select_similar_people(num_people=100, task=1):
	table_array = read_data.build_mf_table_by_people(ratio=0.0) # one person at most eat one time
	similarity = np.zeros([user_num, user_num]) 
	for i in range(user_num):
		for j in range(i, user_num):
			one_sim = 1 - cosine(table_array[i], table_array[j])
			similarity[i][j] = one_sim
			similarity[j][i] = one_sim
	
	rank_array = np.zeros(table_array.shape)
	for i in range(user_num):
		most_sim = np.argsort(-similarity[i]) # euclidean no need negative sign
		most_sim_value = np.sort(-similarity[i])
		for j in range(1, 1+num_people):
			rank_array[i] += similarity[i][most_sim[j]] * table_array[most_sim[j]]
	if task == 1: 
		pred = output_task1(rank_array, table_array)
		write_results(pred, 'sim_first100_cosine')
	

def predict_last_day():
	handle = open('data/rating_user_time_item', 'rb')
	table_list = pickle.load(handle)


food_num = 5532
user_num = 2608
def main():
	#count_highest_food(task=2) # only use this to reproduce baseline
	#get_eaten()

	userID_id = read_data.reset_user_id()
	read_data.read_rating_train(userID_id)
	select_similar_people()


if __name__ == '__main__':
	main()

	
'''
table_array =
[35. 13. 89. 24.  3. 12. 12. 16.  1.  9.]
[ 0. 17.  0.  0.  6.  1.  0.  0. 18.  0.]
[4. 2. 0. 0. 0. 0. 0. 0. 0. 0.]
rank_array = [ 5233. 12750. 55144.  3240. 10040.  4660.  2241.  4194.  3991.   145.]
pred = [ 39  34  18   2  19 139  42 230  25  81 102 110 107  69  61  41 530  67
 192  80 384 212   1 604 167 194 577 125 178 117 309 395  83 435   4 166
  47 224 382 396 376 776 133  22  73 168 308  38 272 802] 
 [-272695.  -63830.  -56424.  -55144.  -47333.  -40426.  -36113.  -34843.
  -31913.  -29557.  -26758.  -24829.  -22611.  -18979.  -18946.  -18064.
  -16915.  -15521.  -15368.  -14485.  -13628.  -13168.  -12750.  -12203.
  -12046.  -11863.  -11694.  -11121.  -11088.  -11074.  -11004.  -10846.
  -10380.  -10236.  -10040.   -9517.   -9441.   -9258.   -8648.   -8547.
   -8363.   -8165.   -8118.   -7963.   -7866.   -7788.   -7484.   -7351.
   -7322.   -7135.]
'''
