#!/usr/bin/env python3
#encoding='big5'
import pandas as pd
import numpy as np
import sys
import math

def load_data():
	train = pd.read_csv(sys.argv[1])
	label = pd.read_csv(sys.argv[2])
	np_x = train.as_matrix()
	np_y = label.as_matrix()
	np_y = np_y.ravel()
	test = pd.read_csv(sys.argv[3])
	np_test = test.as_matrix()
	np_test = np_test.astype(float)
	return np_x, np_y, np_test


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


def sigmoid(z):
	return np.clip(1/(1+np.exp(-z)), 0.000000000000001, 0.999999999999999)

def normalize(a,size):
	m = np.mean(a, axis=0)
	s = np.std(a, axis=0)
	for i in range(size):
		a[i,:] = (a[i,:]-m)/s;
	return a

def train(np_x, np_y):
	# Gaussian distribution parameters
	train_data_size = np_x.shape[0]
	feature_size = np_x.shape[1]
	cnt1 = 0
	cnt2 = 0
	# Mean
	mu1 = np.zeros((feature_size,))
	mu2 = np.zeros((feature_size,))
	for i in range(train_data_size):
		if np_y[i] == 1:
			mu1 += np_x[i]
			cnt1 += 1
		else:
			mu2 += np_x[i]
			cnt2 += 1
	mu1 /= cnt1
	mu2 /= cnt2
	# Shared Covariance Matrix
	sigma1 = np.zeros((feature_size,feature_size))
	sigma2 = np.zeros((feature_size,feature_size))
	for i in range(train_data_size):
		if np_y[i] == 1:
			sigma1 += np.dot(np.transpose([np_x[i] - mu1]), [(np_x[i] - mu1)])
		else:
			sigma2 += np.dot(np.transpose([np_x[i] - mu2]), [(np_x[i] - mu2)])
	sigma1 /= cnt1
	sigma2 /= cnt2
	shared_sigma = (float(cnt1) / train_data_size) * sigma1 + (float(cnt2) / train_data_size) * sigma2
	
	return mu1, mu2, shared_sigma, cnt1, cnt2
    

def predict(np_test, mu1, mu2, shared_sigma, N1, N2):
	output = [['id', 'label']]
	sigma_inverse = np.linalg.inv(shared_sigma)
	w = np.dot((mu1-mu2), sigma_inverse)
	x = np_test.T
	b = (-0.5) * np.dot(np.dot([mu1], sigma_inverse), mu1) + (0.5) * np.dot(np.dot([mu2], sigma_inverse), mu2) + np.log(float(N1)/N2)
	a = np.dot(w, x) + b
	y = sigmoid(a)
	y_ = np.around(y)
	for i in range(np_test.shape[0]):
		output.append([i+1,int(y_[i])])
	df = pd.DataFrame(output)
	df.to_csv(sys.argv[4], index=False, header=False)
	return

def main():
	np_x, np_y, np_test = load_data()
	np_x, np_test = feature_chose(np_x, np_test)
	
	np_all = np.concatenate((np_x,np_test), axis=0)
	np_all= normalize(np_all,np_all.shape[0])
	np_x = np_all[0:np_x.shape[0]]
	np_test = np_all[np_x.shape[0]:]
	
	mu1, mu2, shared_sigma, N1, N2 = train(np_x, np_y)
	predict(np_test, mu1, mu2, shared_sigma, N1, N2)
	return 

if __name__ == '__main__':
	main()