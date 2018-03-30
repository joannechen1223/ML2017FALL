import sys
import pickle
import numpy as np
import pandas as pd
import keras as keras
from keras.models import load_model
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

MAX_SEQUENCE_LENGTH = 5
EMBEDDING_DIM = 64

def load_data():
	#loading
	test_Q=[]
	with open('test_Q_cut.txt','r') as f_Q:
		for line in f_Q:
			line = line.replace("\n", "")
			test_Q.append(line)
	#print(len(test_Q))
	#print(test_Q)
	test_A=[]
	with open('test_A_cut.txt','r') as f_A:
		for line in f_A:
			line = line.replace("\n", "")
			test_A.append(line)
	#print(len(test_A))

	
	#preprocessing
	with open('tokenizer.pickle', 'rb') as handle:
		tokenizer = pickle.load(handle)
	sequencesQ = tokenizer.texts_to_sequences(test_Q)
	sequencesA = tokenizer.texts_to_sequences(test_A)
	word_index = tokenizer.word_index
	X_test_Q = pad_sequences(sequencesQ, maxlen = MAX_SEQUENCE_LENGTH)
	X_test_A = pad_sequences(sequencesA, maxlen = MAX_SEQUENCE_LENGTH)
	X_test_Q = np.asarray(X_test_Q, dtype=np.int)
	X_test_A = np.asarray(X_test_A, dtype=np.int)
	#print(np.shape(X_test_Q))
	#print(np.shape(X_test_A))
	return X_test_Q, X_test_A
	



def predict_result(X_test_Q, X_test_A):
	model = load_model(sys.argv[1])
	model.summary()
	result = model.predict([X_test_Q, X_test_A], batch_size=512)
	print(result)
	print(np.shape(result))
	
	output = [['id', 'ans']]
	cnt = 1
	maxj = 0
	max_options = -1
	for i in range(result.shape[0]):
		if(result[i]>maxj):
			maxj = result[i]
			max_options = i%6
		if(i%6==5):
			output.append([cnt, max_options])
			cnt = cnt+1
			maxj = 0
			max_options = -1
	df = pd.DataFrame(output)
	df.to_csv(sys.argv[2], index=False, header=False)
	
	return


def main():
	X_test_Q, X_test_A = load_data()
	print(X_test_Q)
	print(X_test_A)
	predict_result(X_test_Q, X_test_A)

if __name__=="__main__":
	main()
