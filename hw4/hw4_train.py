'''
B04701232 ML HW4 text_sentment
hw4_train.py
without unlabeled data
'''
import sys
import pickle
import numpy as np
import pandas as pd
import keras as keras
from keras import regularizers
from keras.layers import Embedding, LSTM, Input, Bidirectional, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential, load_model, Model
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam, Adagrad
from keras.callbacks import EarlyStopping, ModelCheckpoint
from gensim.models import Word2Vec

MAX_NUM_WORDS = 100000
MAX_SEQUENCE_LENGTH = 32
EMBEDDING_DIM = 100
DROPOUT = 0.4
EPOCHS = 15
BATCH = 128

def load_data():
	df_train = pd.read_csv(sys.argv[1], sep="\n", header=None)
	list_train = df_train.values.tolist()
	list_train = [x[0] for x in list_train]
	Y_train = [s[:1] for s in list_train]
	X_train = [s[10:] for s in list_train]
	return X_train, Y_train

def data_preprocess(X_train, Y_train):
	with open('tokenizer.pickle', 'rb') as handle:
		tokenizer = pickle.load(handle)
	sequences = tokenizer.texts_to_sequences(X_train)
	word_index = tokenizer.word_index
	X_train = pad_sequences(sequences, maxlen = MAX_SEQUENCE_LENGTH)	
	X_train = np.asarray(X_train, dtype=np.int)
	Y_train = np.asarray(Y_train, dtype=np.int)
	#split validation data
	X_train, X_valid = X_train[:-20000], X_train[-20000:]
	Y_train, Y_valid = Y_train[:-20000], Y_train[-20000:]
	return X_train, Y_train, X_valid, Y_valid, word_index


def main():
	#Data loading and Preprocessing
	X_train, Y_train = load_data()
	X_train, Y_train, X_valid, Y_valid, word_index = data_preprocess(X_train, Y_train)
	
	#Pretrain Word2Vec
	embedding_matrix = np.load("w2v_embedding_matrix.npy")
	
	#Embedding layer
	embedding_layer = Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False)
	sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
	embedded_sequences = embedding_layer(sequence_input)
	
	#RNN Layer(LSTM)
	RNN_output = Bidirectional(LSTM(256, return_sequences=True, dropout = DROPOUT))(embedded_sequences)
	RNN_output = Bidirectional(LSTM(256, return_sequences=True, dropout = DROPOUT))(RNN_output)
	RNN_output = Bidirectional(LSTM(128, return_sequences=True, dropout = DROPOUT))(RNN_output)
	RNN_output = Bidirectional(LSTM(128, return_sequences=False, dropout = DROPOUT))(RNN_output)
	
	#Dense Layer
	output = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.1))(RNN_output)
	output = Dropout(DROPOUT)(output)	
	output = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.1))(output)
	output = Dropout(DROPOUT)(output)
	pred = Dense(1, activation='sigmoid')(output)

	#Build Model
	model = Model(sequence_input, pred)
	model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

	#Early Stop
	callbacks = []
	callbacks.append(ModelCheckpoint('./model/{epoch:05d}-{val_acc:.5f}.h5', monitor='val_acc', save_best_only=False, period=1))
	model.summary()
	history = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), epochs=EPOCHS, batch_size=BATCH, callbacks=callbacks)
	
if __name__=="__main__":
	main()