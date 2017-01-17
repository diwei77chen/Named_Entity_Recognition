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

path_to_training_data = sys.argv[1]
# *.pkl
path_to_classifier = sys.argv[2]

with open(path_to_training_data, 'rb') as f:
	training_set = pickle.load(f)
# with open('/Users/diwei/Desktop/comp9318/project1/test.dat', 'rb') as f:
# 	testing_set = pickle.load(f)
# the ratio of 'NNP', 'NN' and 'NNS' in training_set is : 0.3648541936939478

# bag of tokens which are classified as TITLE
bag_of_tokens = set()
# bag of tags which are classified as TITLE
bag_of_tags = set()
bag_of_tags.add('NN')
bag_of_tags.add('NNS')
bag_of_tags.add('NNP')
# bag_of_tags.add('VBG')
# bag_of_tags.add('VBZ')
# print(training_set[0])

# print(testing_set)
# feature vector:
# feature1: if this token is in bag_of_tokens
#				Bernoulli model: 	YES: 1; NO: 0		
# feature2: if this tag is in bag_of_tags
#				Bernoulli model:	YES: 1; NO: 0
# 
feature_vec = []
# target vector:
# same size of feature vector
target_vec = []

# size of training_set 1977

# size_tmp = len(testing_set)
# print(testing_set[0])
# size_tmp = 10

# this processing would remove sentence feature
# print(count)
# for index0 in range(len(training_set)):
# print(training_set[0])

# preprocessing - extracting bag_of_tokens and bag_of_tags, constructing target vector
def preprocessing(train):
	global size_tmp, bag_of_tokens, bag_of_tags
	bag_of_tokens.clear()
	# bag_of_tags.clear()
	for index0 in range(len(train)):
		for index1 in range(len(train[index0])):
			if train[index0][index1][2] == 'TITLE':
				bag_of_tokens.add(train[index0][index1][0])
				# bag_of_tags.add(train[index0][index1][1])
				

# construct feature vector
def ftvector(train):
	global size_tmp, bag_of_tokens, bag_of_tags, feature_vec, target_vec
	del feature_vec[:]
	del target_vec[:]
	for index0 in range(len(train)):
		for index1 in range(len(train[index0])):
			index_back1 = index1 + 1
			index_back2 = index1 + 2
			feature_vec_tmp = []
			# feature_vec_tmp.append(1)

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

# def cvpreprocessing(train):
# 	global bag_of_tokens, bag_of_tags, training_set_partial
# 	bag_of_tokens.clear()
# 	# bag_of_tags.clear()
# 	for index0 in range(len(train)):
# 		if training_set_partial[train[index0]][2] == 'TITLE':
# 			bag_of_tokens.add(training_set_partial[train[index0]][0])
# 			# bag_of_tags.add(training_set_partial[train[index0]][1])


# def cvftvector(train):
# 	global bag_of_tokens, bag_of_tags, feature_vec, target_vec, training_set_partial
# 	del feature_vec[:]
# 	del target_vec[:]
# 	for index0 in range(len(train)):
# 		# index_back1 = train[inde]
# 		feature_vec_tmp = []
# 		# Feature1
# 		if training_set_partial[train[index0]][0] in bag_of_tokens:
# 			feature_vec_tmp.append(1)
# 		else:
# 			feature_vec_tmp.append(0)
# 		# Feature2
# 		if training_set_partial[train[index0]][1] in bag_of_tags:
# 			feature_vec_tmp.append(1)
# 		else:
# 			feature_vec_tmp.append(0)
# 		feature_vec.append(feature_vec_tmp)
# 		target_vec.append(training_set_partial[train[index0]][2])

# train
# train = training_set
preprocessing(training_set)
with open('bag_of_tokens.dat', 'wb') as f:
	pickle.dump(bag_of_tokens, f)
with open('bag_of_tags.dat', 'wb') as f:
	pickle.dump(bag_of_tags, f)

ftvector(training_set)

regr = LogisticRegression()
regr.fit(feature_vec, target_vec)
# print(regr.coef_)
# ftvector(testing_set)
# a = regr.predict(feature_vec)
print(regr.score(feature_vec, target_vec))
# print(f1_score(target_vec, a, average='micro'))
with open(path_to_classifier, 'wb') as f:
	pickle.dump(regr, f)



# with open('target.dat', 'wb') as f:
# 	pickle.dump(target_vec, f)

print(regr.coef_)
# print(bag_of_tags)
# print(bag_of_tokens)
# feature_vec = np.array(feature_vec).reshape(-1,1)

# print(feature_vec)
# print(target_vec)
# a = regr.predict(feature_vec)
# print(regr.get_params(deep=True))

# for i in a:
# 	if i == 'TITLE':
# 		print(i)

# print(a)
# k = []
# for i in a:
# 	k.append(i)
# print(k)
	# if i == 'TITLE':
		# print(i)
# print(regr.score(feature_vec, target_vec))

# training_set_partial = []
# for index0 in range(len(training_set)):
# 		for index1 in range(len(training_set[index0])):
# 			training_set_partial.append(training_set[index0][index1])

# using 10-fold cross validation for evaluation
# kf_total = cross_validation.KFold(len(training_set_partial), n_folds=2, shuffle=True, random_state=4)
# counter = 0
# mk = [[]]
# for train, test in kf_total:
# 	cvpreprocessing(train)
# 	cvftvector(train)
# 	regr = LogisticRegression()
# 	regr.fit(feature_vec, target_vec)
# 	cvpreprocessing(test)
# 	cvftvector(test)
# 	# print(regr.coef_)
# 	# print(feature_vec)
# 	# print(regr.score(feature_vec, target_vec))
# 	a = regr.predict(feature_vec)
# 	print(f1_score(target_vec, a, average='micro'))
# 	mk_tmp = []
# 	for i in a :
# 		if i == 'TITLE':
# 			mk_tmp.append('TITLE')
# 	mk.append(mk_tmp)
# print(feature_vec)
# for i in mk:
# 	print(len(i))
# print(bag_of_tags)

	# print("%s %s" % (train, test))
#

# feature_vec=[[i] for i in feature_vec]
# print(len(feature_vec), len(target_vec))
# print(feature_vec)


# using 10-fold cross validation for evaluation
# kf = KFold(n_splits=10)
# for train, test in kf.split(training_set):
# 	print("%s %s" % (train, test))


