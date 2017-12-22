import sys
import numpy as np
import pandas as pd
import keras as keras
from keras.models import load_model


def load_data():
	df_test = pd.read_csv(sys.argv[1])
	test = np.asarray(df_test)
	user_test = test[:,1]
	movie_test = test[:,2]
	return user_test, movie_test

def predict_result(user_test, movie_test):
	model = load_model("best_model.h5")
	result = model.predict([user_test, movie_test], batch_size=512)
	output = [['TestDataID', 'Rating']]
	for i in range(result.shape[0]):
		output.append([i+1, float(result[i])])
	df = pd.DataFrame(output)
	df.to_csv(sys.argv[2], index=False, header=False)
	return

def main():
	user_test, movie_test = load_data()
	predict_result(user_test, movie_test)
	return

if __name__=="__main__":
	main()
