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

# Usage: 1st argument: input the path of training data set
#		 2nd argument: input the path for writting the LG classifier to disk

import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

path_to_training_data = sys.argv[1]
path_to_classifier = sys.argv[2]

with open(path_to_training_data, 'rb') as f:
	training_set = pickle.load(f)

# bag of tokens which are classified as TITLE
bag_of_tokens = set()
# bag of tags which are classified as TITLE
bag_of_tags = set()
bag_of_tags.add('NN')
bag_of_tags.add('NNS')
bag_of_tags.add('NNP')

# feature vector:
# feature0: intercept, for tuning purpose
# feature1: if this token is in bag_of_tokens
#				Bernoulli model: 	YES: 1; NO: 0		
# feature2: if this tag is in bag_of_tags
#				Bernoulli model:	YES: 1; NO: 0
# feature3: if the token satisfies NN/NNP of NN/NNP
# feature4: if the token satisfies NNP NNP NNP

feature_vec = []
# target vector:
# the same size as feature vector
target_vec = []

# preprocessing - extracting bag_of_tokens, constructing target vector
def preprocessing(train):
	global size_tmp, bag_of_tokens, bag_of_tags
	bag_of_tokens.clear()
	for index0 in range(len(train)):
		for index1 in range(len(train[index0])):
			if train[index0][index1][2] == 'TITLE':
				bag_of_tokens.add(train[index0][index1][0])
				

# construct feature vector for training data set
def ftvector(train):
	global size_tmp, bag_of_tokens, bag_of_tags, feature_vec, target_vec
	del feature_vec[:]
	del target_vec[:]
	for index0 in range(len(train)):
		for index1 in range(len(train[index0])):
			index_back1 = index1 + 1
			index_back2 = index1 + 2
			feature_vec_tmp = []
			# Feature0: intercept
			feature_vec_tmp.append(1)
			# Feature1: if the token is in Bag of Tokens
			if train[index0][index1][0] in bag_of_tokens:
				feature_vec_tmp.append(1)
			else:
				feature_vec_tmp.append(0)
			# Feature2: if the tag of the token is in Bag of Tags
			if train[index0][index1][1] in bag_of_tags:
				feature_vec_tmp.append(1)
			else:
				feature_vec_tmp.append(0)
			# Feature3: if the token satisfies NN/NNP of NN/NNP
			if train[index0][index1][1] == 'NN' or train[index0][index1][1] == 'NNP':
				if index_back2 < len(train[index0]):
					if train[index0][index_back1][0] == 'of' and (train[index0][index_back2][1] == 'NN' or train[index0][index_back2][1] == 'NNP'):
						feature_vec_tmp.append(1)
					else:
						feature_vec_tmp.append(0)
				else:
					feature_vec_tmp.append(0)
			else:
				feature_vec_tmp.append(0)
			# Feature4: if the token satisfies NNP NNP NNP
			if train[index0][index1][1] == 'NNP':
				if index_back2 < len(train[index0]):
					if train[index0][index_back1][1] == 'NNP' and train[index0][index_back2][1] == 'NNP':
						feature_vec_tmp.append(1)
					else:
						feature_vec_tmp.append(0)
				else:
					feature_vec_tmp.append(0)
			else:
				feature_vec_tmp.append(0)
			# Feature5: if the token satisfies NN/NNP NN/NNP
			# if train[index0][index1][1] == 'NN' or train[index0][index1][1] == 'NNP':
			# 	if index_back1 < len(train[index0]):
			# 		if train[index0][index_back1][1] == 'NN' or train[index0][index_back1][1] == 'NNP':
			# 			feature_vec_tmp.append(1)
			# 		else:
			# 			feature_vec_tmp.append(0)
			# 	else:
			# 		feature_vec_tmp.append(0)
			# else:
			# 	feature_vec_tmp.append(0)
			feature_vec.append(feature_vec_tmp)
			target_vec.append(train[index0][index1][2])

preprocessing(training_set)

# store bag_of_tokens on disk for resuable purpose
with open('bag_of_tokens.dat', 'wb') as f:
	pickle.dump(bag_of_tokens, f)
# store bag_of_tags on disk for reusable purpose
with open('bag_of_tags.dat', 'wb') as f:
	pickle.dump(bag_of_tags, f)

ftvector(training_set)

regr = LogisticRegression()
regr.fit(feature_vec, target_vec)

# write the classifier to disk
with open(path_to_classifier, 'wb') as f:
	pickle.dump(regr, f)
