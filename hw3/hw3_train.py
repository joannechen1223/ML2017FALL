'''
2017MLfall
author:b04701232
reference:
1.https://github.com/WindQAQ/ML2017/blob/master/hw3/train.py
2.discuss with b04902008, 
'''
import sys
import numpy as np
import pandas as pd
import keras as keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import Adam, Adagrad
from keras.layers import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint

def load_data():
	#training data
	df_train = pd.read_csv(sys.argv[1])
	Y_train = df_train['label'].values.tolist()
	Y_train = np.asarray(Y_train, dtype=np.float32)
	X_train = df_train['feature'].values.tolist()
	for i in range(len(X_train)):
		X_train[i] = X_train[i].split(' ')
	X_train = np.asarray(X_train, dtype=np.float32)
	Y_train = keras.utils.to_categorical(Y_train, 7)
	X_train = X_train/255
	return X_train, Y_train

def main():
	epochs = 600
	batch = 64
	X, Y = load_data()
	X = np.reshape(X, (X.shape[0], 48, 48, 1))

	#validation split
	X_train, X_valid = X[:-5000], X[-5000:]
	Y_train, Y_valid = Y[:-5000], Y[-5000:]

	#horizontal flip
	X_train = np.concatenate((X_train, X_train[:, :, ::-1]), axis=0)
	Y_train = np.concatenate((Y_train, Y_train), axis=0)


	#data preprocess
	datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)

	#model construct
	model = Sequential()
	model.add(Conv2D(input_shape=(48, 48, 1), filters= 32,kernel_size=(3, 3), strides=(1,1), padding="same", use_bias=True))
	model.add(LeakyReLU(alpha=0.1))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2,2), padding="same"))
	model.add(Dropout(0.4))	
	
	model.add(Conv2D(filters= 64,kernel_size=(3, 3), strides=(1,1), padding="same", use_bias=True))
	model.add(LeakyReLU(alpha=0.1))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2,2), padding="same"))
	model.add(Dropout(0.2))	
	
	model.add(Conv2D(filters= 128,kernel_size=(3, 3), strides=(1,1), padding="same", use_bias=True))
	model.add(LeakyReLU(alpha=0.1))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2,2), padding="same"))
	model.add(Dropout(0.25))	
	
	model.add(Conv2D(filters= 128,kernel_size=(3, 3), strides=(1,1), padding="same", use_bias=True))
	model.add(LeakyReLU(alpha=0.1))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2,2), padding="same"))
	model.add(Dropout(0.3))
	
	model.add(Conv2D(filters= 256,kernel_size=(3, 3), strides=(1,1), padding="same", use_bias=True))
	model.add(LeakyReLU(alpha=0.1))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2,2), padding="same"))
	model.add(Dropout(0.35))
	
	model.add(Conv2D(filters= 256,kernel_size=(3, 3), strides=(1,1), padding="same", use_bias=True))
	model.add(LeakyReLU(alpha=0.1))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2,2), padding="same"))
	model.add(Dropout(0.4))

	model.add(Flatten())
	
	model.add(Dense(units=256))
	model.add(LeakyReLU(alpha=.001))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))

	model.add(Dense(units=256))
	model.add(LeakyReLU(alpha=.001))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))

	model.add(Dense(units=256))
	model.add(LeakyReLU(alpha=.001))
	model.add(BatchNormalization())

	model.add(Dense(units=7, activation='softmax'))
	
	#early stop
	callbacks = []
	callbacks.append(ModelCheckpoint('./model-{epoch:05d}-{val_acc:.5f}.h5', monitor='val_acc', save_best_only=True, period=1))
	
	model.summary()
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch),steps_per_epoch=len(X_train) // 32, epochs=epochs, validation_data=(X_valid, Y_valid), callbacks=callbacks)

if __name__=="__main__":
	main()
