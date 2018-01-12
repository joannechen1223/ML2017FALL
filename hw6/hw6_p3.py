'''
https://blog.keras.io/building-autoencoders-in-keras.html
'''
import sys
import pandas as pd
import numpy as np

def load_data():
	X_test = pd.read_csv(sys.argv[1])
	X_test = np.asarray(X_test)
	return X_test

def main():
	X_test = load_data()
	label = np.load("label.npy")
	output = [["ID", "Ans"]]
	n_test = np.shape(X_test)
	for i in range(n_test[0]):
		if label[X_test[i,1]]==label[X_test[i,2]]:
			output.append([i, 1])
		else:
			output.append([i, 0])
	df = pd.DataFrame(output)
	df.to_csv(sys.argv[2], index=False, header=False)
	return


if __name__=="__main__":
	main()