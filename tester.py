#!/usr/bin/python3

# Written by Diwei Chen
# Email Address: diwei77chen@gmail.com

import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import KFold
# from sklearn import cross_validation
# from sklearn.metrics import f1_score
# import matplotlib.pyplot as plt
# from sklearn.cross_validation import train_test_split
path_to_testing_data = sys.argv[1]
path_to_classifier = sys.argv[2]
path_to_results = sys.argv[3]

# load testing data
with open(path_to_testing_data, 'rb') as f:
	testing_set = pickle.load(f)
# load classifier
regr = LogisticRegression()
with open(path_to_classifier, 'rb') as f:
	regr = pickle.load(f)

# load data
bag_of_tokens = set()
bag_of_tags = set()
with open('bag_of_tokens.dat', 'rb') as f:
	bag_of_tokens = pickle.load(f)
with open('bag_of_tags.dat', 'rb') as f:
	bag_of_tags = pickle.load(f)

feature_vec = []
result = []
predict_vec = []
test_tmp = []

test_target_vector = []

# construct feature vector
def ftvector(test):
	global bag_of_tags, bag_of_tokens, feature_vec, test_tmp, test_target_vector
	for index0 in range(len(test)):
		for index1 in range(len(test[index0])):
			
			index_back1 = index1 + 1
			index_back2 = index1 + 2
			feature_vec_tmp = []
			# Feature1: if the token is in bag_of_tokens
			if test[index0][index1][0] in bag_of_tokens:
				feature_vec_tmp.append(1)
			else:
				feature_vec_tmp.append(0)
			# Feature2: if the tag of the token is in bag_of_tags
			if test[index0][index1][1] in bag_of_tags:
				feature_vec_tmp.append(1)
			else:
				feature_vec_tmp.append(0)
			# Feature3: if the token satisfies condition: NN/NNP of NN/NNP
			if test[index0][index1][1] == 'NN' or test[index0][index1][1] == 'NNP':
				if index_back2 < len(test[index0]):
					if test[index0][index_back1][0] == 'of' and (test[index0][index_back2][1] == 'NN' or test[index0][index_back2] == 'NNS'):
						feature_vec_tmp.append(1)
					else:
						feature_vec_tmp.append(0)
				else:
					feature_vec_tmp.append(0)
			else:
				feature_vec_tmp.append(0)
			# Feature4: if the token satisfies the condition: NNP NNP NNP
			if test[index0][index1][1] == 'NNP':
				if index_back2 < len(test[index0]):
					if test[index0][index_back1][1] == 'NNP' and test[index0][index_back2][1] == 'NNP':
						feature_vec_tmp.append(1)
					else:
						feature_vec_tmp.append(0)
				else:
					feature_vec_tmp.append(0)
			else:
				feature_vec_tmp.append(0)
			test_tmp.append(test[index0][index1][0])
			feature_vec.append(feature_vec_tmp)
			test_target_vector.append(test[index0][index1][2])

# orgnise result list
def orgnise(test):
	global test_tmp, predict_vec, result
	index_front = 0
	index_back = 0
	for index0 in range(len(test)):
		result_sentence = []
		index1 = len(test[index0])
		index_back += index1
		# print(index_front)
		# print(index_back)
		# break
		for index in range(index_front, index_back):
			result_tuple = []
			result_tuple.append(test_tmp[index])
			result_tuple.append(predict_vec[index])
			result_sentence.append(result_tuple)
		# print(result_sentence)
		result.append(result_sentence)
		index_front += index1


ftvector(testing_set)
predict_vec = regr.predict(feature_vec)
orgnise(testing_set)
with open(path_to_results, 'wb') as f:
	pickle.dump(result, f)




with open('predict.dat', 'wb') as f:
	pickle.dump(predict_vec, f)
with open('target.dat', 'wb') as f:
	pickle.dump(test_target_vector, f)
# print(f1_score(feature_vec, predict_vec, average='micro'))
# print(len(testing_set))
# print(len(result))
# print(result)
# print(len(test_tmp))
# print(len(predict_vec))
# print(result[0])
# print(len(testing_set[0]))


# print(bag_of_tokens)
# print(bag_of_tags)
# print(regr.coef_)