#!/usr/bin/python3

# Created:  17 Oct 2016
# Modified: 20 Nov 2016
# Author:   Diwei Chen
# Email:    diwei77chen@gmail.com

# Taking advantage of logistic regression, the goal of this project is to locate segments
# of text from input document and classify them in one of the pre-defined categories(e.g., 
# PERSON, LOCATION and ORGNIZATION). In this project, we only perform NER for a single
# category of TITLE. We define a TITLE as an appellation associated with a person by virtue
# of occupation, office, birth or as an honorific. For example, in the following sentence,
# both Prime Minister and MP are TITLES:
#				Prime Minister Malcolm Turnbull MP visited Germany yesterday.

# Usage: 1st argument: input the path of testing data set
#		 2nd argument: input the path of a classifier
#		 3nd argument: input the path for writing testing results

import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

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

# construct feature vector for testing data set
def ftvector(test):
	global bag_of_tags, bag_of_tokens, feature_vec, test_tmp, test_target_vector
	for index0 in range(len(test)):
		for index1 in range(len(test[index0])):
			
			index_back1 = index1 + 1
			index_back2 = index1 + 2
			feature_vec_tmp = []
			# Feature0: intercept
			feature_vec_tmp.append(1)
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

# orgnise the result list to match original data format
def orgnise(test):
	global test_tmp, predict_vec, result
	index_front = 0
	index_back = 0
	for index0 in range(len(test)):
		result_sentence = []
		index1 = len(test[index0])
		index_back += index1
		for index in range(index_front, index_back):
			result_tuple = []
			result_tuple.append(test_tmp[index])
			result_tuple.append(predict_vec[index])
			result_sentence.append(result_tuple)
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

# showing the f1 score for the classifier
# print(f1_score(feature_vec, predict_vec, average='micro'))


