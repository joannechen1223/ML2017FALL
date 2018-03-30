import sys
import pickle
import numpy as np
import pandas as pd
import keras as keras
from keras import regularizers
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Input, Bidirectional, GRU, Dot, Flatten
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential, load_model, Model
from gensim.models import Word2Vec
from keras.callbacks import EarlyStopping, ModelCheckpoint



EMBEDDING_DIM = 64
MAX_NUM_WORDS = 10000
MAX_SEQUENCE_LENGTH = 5
DROPOUT = 0.4
EPOCHS = 15
BATCH = 128


def load_data():
	df_train_1 = pd.read_csv(sys.argv[1], sep="\n", header=None)
	df_train_2 = pd.read_csv(sys.argv[2], sep="\n", header=None)
	df_train_3 = pd.read_csv(sys.argv[3], sep="\n", header=None)
	df_train_4 = pd.read_csv(sys.argv[4], sep="\n", header=None)
	df_train_5 = pd.read_csv(sys.argv[5], sep="\n", header=None)
	list_train_1 = df_train_1.values.tolist()
	list_train_2 = df_train_2.values.tolist()
	list_train_3 = df_train_3.values.tolist()
	list_train_4 = df_train_4.values.tolist()
	list_train_5 = df_train_5.values.tolist()
	X_train_Q = []
	X_train_A = []
	for line in range(len(list_train_1)-1):
		for word in list_train_1[line]:
			X_train_Q.append(word)
	for line in range(len(list_train_2)-1):
		for word in list_train_2[line]:
			X_train_Q.append(word)
	for line in range(len(list_train_3)-1):
		for word in list_train_3[line]:
			X_train_Q.append(word)
	for line in range(len(list_train_4)-1):
		for word in list_train_4[line]:
			X_train_Q.append(word)
	for line in range(len(list_train_5)-1):
		for word in list_train_5[line]:
			X_train_Q.append(word)
	for line in range(1,len(list_train_1)):
		for word in list_train_1[line]:
			X_train_A.append(word)
	for line in range(1,len(list_train_2)):
		for word in list_train_2[line]:
			X_train_A.append(word)
	for line in range(1,len(list_train_3)):
		for word in list_train_3[line]:
			X_train_A.append(word)
	for line in range(1,len(list_train_4)):
		for word in list_train_4[line]:
			X_train_A.append(word)
	for line in range(1,len(list_train_5)):
		for word in list_train_5[line]:
			X_train_A.append(word)
	Y_train = []
	for i in range(len(X_train_Q)):
		Y_train.append(1)
	#for j in range(10):
	for i in range(len(X_train_Q)):
		Y_train.append(0)
	
	X_train_Q = X_train_Q + X_train_Q 
	X_train_A = X_train_A + X_train_A[-1000:] + X_train_A[:-1000]
	'''
	print(len(X_train_Q))
	print(len(X_train_A))
	print(len(Y_train))
	'''
	return X_train_Q, X_train_A, Y_train

def w2v(sentences, word_index):
	#list_sentences = sentences.tolist()
	l=len(sentences)
	wordseq = []
	for i in range(l):
		wordseq.append(text_to_word_sequence(sentences[i]))
	pretrain_model = Word2Vec(wordseq, size=EMBEDDING_DIM, min_count=3, sg=1)
	embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
	for word, i in word_index.items():
		if word in pretrain_model.wv.vocab:
			embedding_matrix[i] = pretrain_model.wv[word]
	return embedding_matrix


def tokenize(X_train_Q, X_train_A):
	tokenizer = Tokenizer(num_words = MAX_NUM_WORDS, lower=False)
	tokenizer.fit_on_texts(X_train_Q)
	sequences_Q = tokenizer.texts_to_sequences(X_train_Q)
	sequences_A = tokenizer.texts_to_sequences(X_train_A)
	word_index = tokenizer.word_index
	print(word_index)
	length = len(word_index)
	embedding_matrix = w2v(X_train_Q, word_index)
	#print(word_index)
	'''
	with open('tokenizer.pickle', 'wb') as handle:
		pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
	'''
	X_train_Q = pad_sequences(sequences_Q, maxlen = MAX_SEQUENCE_LENGTH)
	X_train_A = pad_sequences(sequences_A, maxlen = MAX_SEQUENCE_LENGTH)
	
	#X_train, X_valid = data[:-len(train_1)], data[-len(train_1):]
	#print(np.shape(X_train_Q))
	#print(np.shape(X_train_A))
	with open('tokenizer.pickle', 'wb') as handle:
		pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

	np.save("w2v_embedding_matrix.npy", embedding_matrix)

	return X_train_Q, X_train_A, length, embedding_matrix




def main():
	X_train_Q, X_train_A, Y_train = load_data()
	X_train_Q, X_train_A, length, embedding_matrix = tokenize(X_train_Q, X_train_A)
	train = np.concatenate((X_train_Q, X_train_A), axis=1)
	Y_train = np.asarray(Y_train)
	len = np.shape(Y_train)
	Y_train = np.reshape(Y_train, (len[0],1))
	train = np.concatenate((train, Y_train), axis=1)
	np.random.shuffle(train)
	print(np.shape(train))
	X_train_Q = train[:,:MAX_SEQUENCE_LENGTH]
	X_train_A = train[:,MAX_SEQUENCE_LENGTH:MAX_SEQUENCE_LENGTH*2]
	Y_train = train[:,MAX_SEQUENCE_LENGTH*2]
	X_train_Q, X_valid_Q = X_train_Q[:-100000], X_train_Q[-100000:]
	X_train_A, X_valid_A = X_train_A[:-100000], X_train_A[-100000:]
	Y_train, Y_valid = Y_train[:-100000], Y_train[-100000:]
	print(np.shape(embedding_matrix))
	print(length)
	embedding_layer = Embedding(length + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False)

	sequenceQ_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
	embedded_sequences_Q = embedding_layer(sequenceQ_input)
	sequenceA_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
	embedded_sequences_A = embedding_layer(sequenceA_input)
	Q = Bidirectional(GRU(64, return_sequences=False, dropout = DROPOUT))(embedded_sequences_Q)
	A = Bidirectional(GRU(64, return_sequences=False, dropout = DROPOUT))(embedded_sequences_A)
	#Q = Flatten()(Q)
	#A = Flatten()(A)
	output = Dot(1)([Q, A])
	pred = Dense(1, activation='sigmoid')(output)
	
	model = Model([sequenceQ_input, sequenceA_input], pred)
	model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	

	callbacks = []
	callbacks.append(ModelCheckpoint('./model_try10/{epoch:05d}-{val_acc:.5f}.h5', monitor='val_acc', save_best_only=False, period=1))
	model.summary()
	history = model.fit([X_train_Q, X_train_A], Y_train, validation_data=([X_valid_Q, X_valid_A], Y_valid), epochs=EPOCHS, batch_size=BATCH, callbacks=callbacks)


if __name__=="__main__":
	main()
