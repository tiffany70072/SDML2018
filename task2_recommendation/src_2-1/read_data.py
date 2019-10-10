import numpy as np 
import pandas as pd
import pickle 
import json

'''
date,userid,foodid
2014-09-15,6,0
2014-09-16,6,0
2014-09-16,6,1
2014-09-16,6,2
'''

# all list: user(array?) - time(not equal) - food(array?)
def read_rating_train(userID_id):
	df = pd.read_csv('../all/rating_train.csv')
	print('data.shape', df.shape)
	data = np.array(df)
	del df
	table = [[] for i in range(len(userID_id))] # user: list, time: list, item: list
	# [[[0], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [2, 0, 3, 11, 12, 13, 14, 15, 16, 17]...
	#zero_array = np.zeros([])
	
	currentUser = -1
	currentDate = '0'
	for i in range(data.shape[0]):
		if currentUser != userID_id[data[i, 1]]:
			currentUser = userID_id[data[i, 1]]
			countDate = -1
		if currentDate != data[i, 0]:
			currentDate = data[i, 0]
			countDate += 1
			table[userID_id[data[i, 1]]].append([])
			table[userID_id[data[i, 1]]][countDate].append(data[i, 2])
		else:
			table[userID_id[data[i, 1]]][countDate].append(data[i, 2])
		#if i > 100: break
		if i % 100000 == 0: print(i)
	print(table[0][:10])
	
	filehandler = open('data/rating_user_time_item', 'wb')
	pickle.dump(table, filehandler)

def build_mf_table(ratio = 0):
	handle = open('data/rating_user_time_item', 'rb')
	table_list = pickle.load(handle)
	table_array = np.zeros([user_num, food_num]) # user: array, item: array --> count item
	for i in range(len(table_list)):
		for j in range(int(len(table_list[i])*ratio), len(table_list[i])):
			for k in range(len(table_list[i][j])):
				table_array[i][table_list[i][j][k]] += 1
	print('table_array =')
	#for i in range(3): print(table_array[i][:10])
	return table_array

def build_mf_table_by_people(ratio = 0):
	handle = open('data/rating_user_time_item', 'rb')
	table_list = pickle.load(handle)
	table_array = np.zeros([user_num, food_num]) # user: array, item: array --> count item
	for i in range(len(table_list)):
		for j in range(int(len(table_list[i])*ratio), len(table_list[i])):
			for k in range(len(table_list[i][j])):
				table_array[i][table_list[i][j][k]] = 1
		#print(int(len(table_list[i])*ratio))
	print('table_array =')
	for i in range(3): print(table_array[i][:10])
	return table_array
	
	
def reset_user_id():
	df = pd.read_csv('../all/rating_train.csv')
	print('data.shape', df.shape)
	data = np.array(df)[:, 1]
	userID_id = {}
	id_userID = {}
	#userID_set = set()
	count = 0
	for i in range(data.shape[0]):
		try: _ = userID_id[data[i]]
		except KeyError: 
			userID_id[data[i]] = count
			id_userID[count] = data[i]
			if count < 10: print(data[i], count)
			count += 1
	
	filehandler = open('data/userID_id', 'wb')
	pickle.dump(userID_id, filehandler)
	filehandler = open('data/id_userID', 'wb')
	pickle.dump(id_userID, filehandler)

	print('userID_id =', len(userID_id))
	return userID_id # = 2608

def reset_food_id():
	df = pd.read_csv('../all/rating_train.csv')
	print('data.shape', df.shape)
	data = np.array(df)[:, 2]
	foodID_id = {}
	#userID_set = set()
	count = 0
	for i in range(data.shape[0]):
		try: _ = foodID_id[data[i]]
		except KeyError: 
			foodID_id[data[i]] = count
			if count < 10: print(data[i], count)
			count += 1
	print('foodID_id =', len(foodID_id))
	return foodID_id # = 5532

def read_user():
	#with open('../all/user.csv' , 'r') as reader: jf = json.loads(reader.read())
	#print(jf['data']['hi_info']['login_ip'])
	df = pd.read_csv('../all/user.csv')
	#col = ['userid','username','age','gender','location','city','state','title','about_me','reasons','inspirations','friends_count']
	col = ['userid','age','gender','state','friends_count']
	df = df[col]
	return np.array(df)

def read_food_features():
	food_features = np.load("../all/foodMatrix.npy")
	print("food_features =", food_features.shape)
	return food_features

food_num = 5532
user_num = 2608

def main():
	#userID_id = reset_user_id()
	#foodID_id = reset_food_id()
	#read_rating_train(userID_id)
	read_food_features()

if __name__ == '__main__':
	main()

