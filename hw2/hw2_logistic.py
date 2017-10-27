#!/usr/bin/env python3
#encoding='big5'
import pandas as pd
import numpy as np
import sys
import math

def load_data():
	train = pd.read_csv(sys.argv[1])
	np_x = train.as_matrix()
	test = pd.read_csv(sys.argv[2])
	np_test = test.as_matrix()
	np_test = np_test.astype(float)
	return np_x, np_test


def feature_chose(X_all, X_test):
	X_all = np.concatenate((X_all, X_all[:,0:1]**2), axis=1)
	X_all = np.concatenate((X_all, X_all[:,3:6]**2), axis=1)
	X_all = np.concatenate((X_all, X_all[:,0:1]**3), axis=1)
	X_all = np.concatenate((X_all, X_all[:,3:6]**3), axis=1)
	X_all = np.concatenate((X_all[:,0:1], X_all[:,2:]), axis=1)

	X_test = np.concatenate((X_test, X_test[:,0:1]**2), axis=1)
	X_test = np.concatenate((X_test, X_test[:,3:6]**2), axis=1)
	X_test = np.concatenate((X_test, X_test[:,0:1]**3), axis=1)
	X_test = np.concatenate((X_test, X_test[:,3:6]**3), axis=1)
	X_test = np.concatenate((X_test[:,0:1], X_test[:,2:]), axis=1)
	return X_all, X_test


def sigmoid(x,w):
	z = np.dot(x, w)
	return np.clip(1/(1+np.exp(-z)), 0.000000000000001, 0.999999999999999)

def normalize(a,size):
	m = np.mean(a, axis=0)
	s = np.std(a, axis=0)
	for i in range(size):
		a[i,:] = (a[i,:]-m)/s;
	return a

def predict(np_test, w):
	output = [['id', 'label']]
	s_test = sigmoid(np_test,w)
	y_ = np.around(s_test)
	for i in range(np_test.shape[0]):
		output.append([i+1,int(y_[i])])
	df = pd.DataFrame(output)
	df.to_csv(sys.argv[3], index=False, header=False)
	return



def main():
	np_x, np_test = load_data()
	np_x, np_test = feature_chose(np_x, np_test)
	
	np_all = np.concatenate((np_x,np_test), axis=0)
	np_all= normalize(np_all,np_all.shape[0])
	np_x = np_all[0:np_x.shape[0]]
	np_test = np_all[np_x.shape[0]:]
	
	np_test = np.concatenate((np.ones((np_test.shape[0], 1)), np_test), axis=1)

	w = np.load('logistic.npy')
	predict(np_test, w)
	return 

if __name__ == '__main__':
	main()