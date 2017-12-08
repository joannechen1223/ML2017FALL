'''
B04701232 ML HW4 text_sentment
hw4_test.py
without unlabeled data
'''
import sys
import pickle
import numpy as np
import pandas as pd
import keras as keras
from keras.models import load_model
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

MAX_SEQUENCE_LENGTH = 32
EMBEDDING_DIM = 100

def load_data():
	#loading
	df_test = pd.read_csv(sys.argv[1], sep="\n")
	list_test = df_test.values.tolist()
	list_test = [x[0] for x in list_test]
	X_test = []
	for i in range(10):
		X_test.append(list_test[i][2:])
	for i in range(10,100):
		X_test.append(list_test[i][3:])
	for i in range(100,1000):
		X_test.append(list_test[i][4:])
	for i in range(1000,10000):
		X_test.append(list_test[i][5:])
	for i in range(10000,100000):
		X_test.append(list_test[i][6:])
	for i in range(100000,200000):
		X_test.append(list_test[i][7:])
	#preprocessing
	with open('tokenizer.pickle', 'rb') as handle:
		tokenizer = pickle.load(handle)
	sequences = tokenizer.texts_to_sequences(X_test)
	word_index = tokenizer.word_index
	X_test = pad_sequences(sequences, maxlen = MAX_SEQUENCE_LENGTH)
	X_test = np.asarray(X_test, dtype=np.int)
	return X_test


def predict_result(X_test):
	model = load_model("best_model.h5")
	result = model.predict(X_test, batch_size=512)
	output = [['id', 'label']]
	for i in range(result.shape[0]):
		output.append([i, int(np.around(result[i]))])
	df = pd.DataFrame(output)
	df.to_csv(sys.argv[2], index=False, header=False)
	return

def main():
	X_test = load_data()
	predict_result(X_test)

if __name__=="__main__":
	main()
