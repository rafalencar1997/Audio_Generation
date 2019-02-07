import numpy as np
import tensorflow as tf

from keras import losses, models, optimizers
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import (Layer, Input, Flatten, Dropout, BatchNormalization, Reshape,
                          MaxPool1D, AveragePooling1D, GlobalAveragePooling1D, GlobalAveragePooling2D,
                          Conv2DTranspose, Conv1D, Dense, LeakyReLU, ReLU, Activation,
                          LSTM, SimpleRNNCell)

# Generator Model
def generator(NoiseDim, OutputShape):
    model = Sequential()
    model.add(Dense(2100, input_shape=(NoiseDim,)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Reshape((2100, 1)))

    model.add(Conv1D(32, 7, padding='same'))
    model.add(Activation('linear'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dropout(rate=0.1))

    model.add(Conv1D(64, 5, padding='same'))
    model.add(Activation('linear'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dropout(rate=0.1))

    model.add(Conv1D(128, 3, padding='same'))
    model.add(Activation('linear'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dropout(rate=0.1))
 
    model.add(Conv1D(420, 11, padding='same'))
    model.add(Activation('linear'))
    model.add(AveragePooling1D(20))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dropout(rate=0.1))
    model.add(Flatten())
    return model

# Discriminator Model
def discriminator(InputShape):
    model = Sequential()
    model.add(Reshape((InputShape, 1), input_shape=(InputShape,)))
    model.add(Conv1D(32, 11, strides=7, padding='valid'))
    model.add(Activation('linear'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dropout(rate=0.1))

    model.add(Conv1D(16, 7, strides=5, padding='valid'))
    model.add(Activation('linear'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dropout(rate=0.1))

    model.add(Conv1D(8, 5, strides=3, padding='valid'))
    model.add(Activation('linear'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dropout(rate=0.1))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dense(1, activation='tanh'))
    return model

# Stacked Generator and Discriminator
def stacked_G_D(Generator, Discriminator):
    model = Sequential()
    model.add(Generator)
    model.add(Discriminator)
    return model

# Encoder
def encoder(InputShape, EncodeSize):
    model = Sequential()
    model.add(Reshape((InputShape, 1), input_shape=(InputShape,)))
    model.add(Conv1D(32, 11, strides=7, padding='valid'))
    model.add(Activation('linear'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dropout(rate=0.1))

    model.add(Conv1D(16, 7, strides=5, padding='valid'))
    model.add(Activation('linear'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dropout(rate=0.1))

    model.add(Conv1D(4, 5, strides=3, padding='valid'))
    model.add(Activation('linear'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dropout(rate=0.1))
    model.add(Flatten())
    model.add(Dense(EncodeSize))
    model.add(LeakyReLU(alpha=0.01))
    return model

# AutoEndoder
def autoEncoder(Encoder, Generator):
    model = Sequential()
    model.add(Encoder)
    model.add(Generator)
    return model

def discriminator2():
    model = Sequential()
    model.add(Reshape((SHAPE, 1), input_shape=(SHAPE,)))
    model.add(Conv1D(32, 5, strides=4, padding='valid'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPool1D(pool_size=4, strides=4, padding='valid'))
    
    model.add(Conv1D(64, 5, strides=34, padding='valid'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPool1D(pool_size=4, strides=4, padding='valid'))
    
    model.add(Flatten())
    model.add(Dense(NOISE_DIM))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(1))
    model.add(LeakyReLU(alpha=0.01))
    return model

def generator2():
    model = Sequential()
    model.add(Dense(DENSE_SIZE, input_shape=(NOISE_DIM,)))
    model.add(Activation('linear'))
    #model.add(ReLU())
    model.add(BatchNormalization(momentum=0.9))
    
    model.add(Reshape((DENSE_SIZE,1,1)))
    
    model.add(Conv2DTranspose(64, kernel_size=(4,1), strides=(3, 1), padding='same'))
    model.add(Activation('linear'))
    #model.add(ReLU())
    model.add(BatchNormalization(momentum=0.9))
    
    model.add(Conv2DTranspose(1, kernel_size=(4,1), strides=(7, 1), padding='same'))
    model.add(Activation('linear'))
    
    #model.add(GlobalAveragePooling2D())
    
    model.add(Flatten())
    return model
