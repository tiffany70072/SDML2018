'''
Data format:
date,userid,foodid
2014-09-15,6,0
2014-09-16,6,0
2014-09-16,6,1
2014-09-16,6,2
'''

import json
import numpy as np 
import pandas as pd
import pickle 


def read_rating_train(userID_id):
	# all list: user(array?) - time(not equal) - food(array?)
	df = pd.read_csv('../all/rating_train.csv')
	print('data.shape', df.shape)
	data = np.array(df)
	del df
	table = [[] for i in range(len(userID_id))]  # user: list, time: list, item: list
	
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
	return table_array


def build_mf_table_by_people(ratio=0):
	handle = open('data/rating_user_time_item', 'rb')
	table_list = pickle.load(handle)
	table_array = np.zeros([user_num, food_num])  # user: array, item: array --> count item
	for i in range(len(table_list)):
		for j in range(int(len(table_list[i])*ratio), len(table_list[i])):
			for k in range(len(table_list[i][j])):
				table_array[i][table_list[i][j][k]] = 1
	return table_array
	
	
def reset_user_id():
	df = pd.read_csv('../all/rating_train.csv')
	print('data.shape', df.shape)
	data = np.array(df)[:, 1]
	userID_id = {}
	id_userID = {}
	count = 0
	for i in range(data.shape[0]):
		try: _ = userID_id[data[i]]
		except KeyError: 
			userID_id[data[i]] = count
			id_userID[count] = data[i]
			if count < 10: 
				print(data[i], count)
			count += 1
	
	filehandler = open('data/userID_id', 'wb')
	pickle.dump(userID_id, filehandler)
	filehandler = open('data/id_userID', 'wb')
	pickle.dump(id_userID, filehandler)
	return userID_id # = 2608


def reset_food_id():
	df = pd.read_csv('../all/rating_train.csv')
	data = np.array(df)[:, 2]
	foodID_id = {}
	count = 0
	for i in range(data.shape[0]):
		try: _ = foodID_id[data[i]]
		except KeyError: 
			foodID_id[data[i]] = count
			if count < 10: 
				print(data[i], count)
			count += 1
	print('foodID_id =', len(foodID_id))
	return foodID_id # = 5532


def read_user():
	df = pd.read_csv('../all/user.csv')
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
	userID_id = reset_user_id()
	foodID_id = reset_food_id()
	read_rating_train(userID_id)
	read_food_features()

	
if __name__ == '__main__':
	main()
