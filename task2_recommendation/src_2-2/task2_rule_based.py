# guess the highest one (for task1 and task2)
import numpy as np 
import pickle 
import read_data
from scipy.spatial.distance import cosine, euclidean
	
def write_results(pred, filename = 'test', task = 1):
	if task == 1: fout = open('result/' + filename + '.csv', 'w')
	else: 
		fout = open('result-task2/' + filename + '.csv', 'w') 
		print('task =', task)
	fout.write('userid,foodid\n')
	handle = open('data/id_userID', 'rb')
	id_userID = pickle.load(handle)
	for i in range(pred.shape[0]):
		fout.write(str(id_userID[i]) + ',')
		for j in range(20): fout.write(str(int(pred[i][j])) + " ")
		fout.write("\n")

def count_foods_num(): # count how many kinds of foods does one user eat
	table_array = read_data.build_mf_table_by_people()
	print('table_array =', table_array.shape)
	print(table_array[:4][:5])
	kinds = np.sum(table_array, axis = 1)
	print("kinds =", np.percentile(kinds, [0, 1, 25, 50, 75, 99, 100]), np.mean(kinds))
	#kinds = [ 99.   118. 155.   187.   236.   391.   469.  ] 201.

def count_record_day():
	handle = open('data/rating_user_time_item', 'rb')
	table_list = pickle.load(handle)
	count = []
	for i in range(user_num):
		count.append(len(table_list[i]))
	count = np.array(count)
	print("count days =", np.percentile(count, [0, 1, 25, 50, 75, 99, 100]), np.mean(count))
	# [ 23.  34.  66.  93. 128. 151. 160.] 95, 
	# 1028 entries for each user

def predict_last_day():
	answer = np.zeros([user_num, 20], dtype = int)
	handle = open('data/rating_user_time_item', 'rb')
	table_list = pickle.load(handle)

	for i in range(user_num):
		one_answer = []
		full = 0
		for j in range(len(table_list[i])-1, -1, -1):
			for k in range(len(table_list[i][j])):
				if table_list[i][j][k] not in one_answer:
					one_answer.append(table_list[i][j][k])
					#print(table_list[i][j][k])
					if len(one_answer) == 20: 
						full = 1
						break
			if full == 1: break
		#print("one_answer =", one_answer)
		answer[i] = np.array(one_answer)
		#break
	print('answer =', answer[0])
	write_results(answer, 'predict_last_few_days', task = 2)

def predict_last_day_2times():
	answer = np.zeros([user_num, 20], dtype = int)
	handle = open('data/rating_user_time_item', 'rb')
	table_list = pickle.load(handle)

	for i in range(user_num):
		answer_one = []
		answer_two = []
		full = 0
		for j in range(len(table_list[i])-1, -1, -1):
			for k in range(len(table_list[i][j])):
				if table_list[i][j][k] not in answer_one: answer_one.append(table_list[i][j][k])
				elif table_list[i][j][k] not in answer_two:
					answer_two.append(table_list[i][j][k])
					if len(answer_two) == 20: 
						full = 1
						break
			if full == 1: break
		answer[i] = np.array(answer_two)
	print('answer =', answer[0])
	write_results(answer, 'predict_last_few_days_2times', task = 2)

def prepare_ranking_foodid(num_class = 300):
	table_array = read_data.build_mf_table()
	print(table_array[:10][:10])

	# store each foodid (of each user) by its ranking, less than 300 --> label -1
	foodid_rank_each_user = np.zeros([user_num, num_class], dtype = int) 
	foodid_rank_each_user -= 1
	for i in range(user_num): # user_num
		rank_array = -table_array[i]
		#print('rank_array =', rank_array)
		rank_argsort = np.argsort(rank_array)[:num_class]
		rank_sort = (np.sort(rank_array)[:num_class]) * (-1)
		#print('argsort =', rank_argsort)
		#print('sort =', rank_sort)
		for j in range(num_class):
			if rank_sort[j] > 0: foodid_rank_each_user[i][j] = rank_argsort[j]
			else: break
		#if i > 50: break
	#print('foodid_rank_each_user', foodid_rank_each_user[:3, :10])
	print('foodid_rank_each_user', foodid_rank_each_user[1, :])
	#for j in range(10): print(table_array[3][foodid_rank_each_user[3][j]])
	 
	# map each foodid to user's ranking
	foodid_to_each_ranking = [{} for i in range(user_num)]
	#each_ranking_to_foodid = [{} for i in range(user_num)]
	
	#for i in range(1, 2):
	for i in range(user_num):
		for j in range(num_class):
			if foodid_rank_each_user[i][j] == -1: break
			foodid_to_each_ranking[i][foodid_rank_each_user[i][j]] = j
	#print('foo')
	print('foodid_to_each_ranking =', foodid_to_each_ranking[1])
	return foodid_to_each_ranking, foodid_rank_each_user

def check_table_list():
	handle = open('data/rating_user_time_item', 'rb')
	table_list = pickle.load(handle)
	i = 1
	print(table_list[i][-3:])

def get_user_embedding():
	eaten_table = read_data.build_mf_table_by_people() # [0, 1, 0, 0, ...]
	return eaten_table

def get_user_embedding_pop(num_class):
	rank_table = read_data.build_mf_table()
	eaten_table = np.zeros([user_num, food_num])
	for i in range(user_num):
		rank = np.argsort(-rank_table[i])[:num_class]
		for pop_food in rank:
			eaten_table[i][pop_food] = 1
	return eaten_table

def weighted_count_highest_food(weight = 0.9):
	handle = open('data/rating_user_time_item', 'rb')
	table_list = pickle.load(handle)

	count_table = np.zeros([user_num, food_num], dtype = float)
	for i in range(user_num):
		currrent_weight = 1
		for day in range(len(table_list[i])-1, -1, -1):
			for j in range(len(table_list[i][day])):
				count_table[i][table_list[i][day][j]] += currrent_weight
			currrent_weight *= weight
			if currrent_weight < 0.01: break

	print('task =', 2)
	print('count =', count_table[0][:10])
	pred = np.argsort(-count_table)[:, :20]
	write_results(pred, 'count_most_weight=0.9', task = 2)


food_num = 5532
user_num = 2608
def main():
	#count_record_day()
	#count_foods_num()
	#predict_last_day() 
	#predict_last_day_2times()

	#prepare_ranking_foodid()
	#check_table_list()
	userID_id = read_data.reset_user_id()
	read_data.read_rating_train(userID_id)
	weighted_count_highest_food()

	


if __name__ == '__main__':
	main()

