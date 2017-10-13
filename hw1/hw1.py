#!/usr/bin/env python3
#encoding='big5'
'''
ML hw1 PM2.5 predicted
B04701232 joannechen1223
hw1
'''
import pandas as pd
import numpy as np
import sys

f = pd.read_csv(sys.argv[1], encoding = 'big5', header=None)
f = f.replace('NR', 0)

test_data = pd.DataFrame()

for i in range(0,4320,18):
	temp = f.ix[i,2:10]
	for j in range(1,18):
		temp = temp.append(f.ix[i+j,2:10], ignore_index=True)

	temp.reset_index(drop=True, inplace=True)
	test_data = test_data.append(temp, ignore_index=True)

test_data.reset_index(drop=True, inplace=True)
np_test = test_data.as_matrix()

np_test = np_test.astype(float)

relate = [75, 76, 77, 78, 79, 80, 83, 84, 85, 86, 87, 88, 89]
sqrelate = [88, 89]
test_x = []
for i in relate:
	test_x.append(np_test[:,i])
for i in sqrelate:
	test_x.append(np_test[:,i]**2)

np_test_x = np.array(test_x).transpose()
np_test_x = np.concatenate((np.ones((np_test_x.shape[0], 1)), np_test_x), axis=1)

w = np.load('model_1.npy')

output = [['id', 'value']]
y = np.dot(np_test_x,w)

for i in range(240):
	temp=["id_"+repr(i), y[i]]
	output.append(temp)

df = pd.DataFrame(output)
df.to_csv(sys.argv[2], index=False, header=False)