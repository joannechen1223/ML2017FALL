import sys
import numpy as np
import pandas as pd
import keras as keras
from keras.models import Sequential, load_model

def load_data():
	dfX_test = pd.read_csv(sys.argv[1])
	X_test = dfX_test['feature'].values.tolist()
	for i in range(len(X_test)):
		X_test[i] = X_test[i].split(' ')
	X_test = np.asarray(X_test, dtype=np.float32)
	X_test = X_test/255
	return X_test

def print_result(result):
	output = [['id', 'label']]
	for i in range(result.shape[0]):
		output.append([i, np.argmax(result[i])])
	return output

def predict(X_test):
	best_model = load_model('model-00546-0.69220.h5')
	result = best_model.predict(X_test)
	Y_test = print_result(result)
	df = pd.DataFrame(Y_test)
	df.to_csv(sys.argv[2], index=False, header=False)
	return

def main():
	X_test = load_data()
	X_test = np.reshape(X_test, (X_test.shape[0], 48, 48, 1))
	predict(X_test)

if __name__=="__main__":
	main()
