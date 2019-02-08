import numpy as np
import tensorflow as tf

from keras import losses, models, optimizers
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import (Layer, Input, Flatten, Dropout, BatchNormalization, Reshape,
                          MaxPool1D, AveragePooling1D, AveragePooling2D, GlobalAveragePooling1D, GlobalAveragePooling2D,
                          Conv2DTranspose, Conv1D, Dense, LeakyReLU, ReLU, Activation,
                          LSTM, SimpleRNNCell)

# Generator Model
def generator(NoiseDim, OutputShape):
    model = Sequential()
    model.add(Dense(1000, input_shape=(NoiseDim,)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Reshape((1000, 1)))

    model.add(Conv1D(16, 20, padding='same'))
    model.add(ReLU())
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dropout(rate=0.1))

    model.add(Conv1D(32, 25, padding='same'))
    model.add(ReLU())
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dropout(rate=0.1))

    model.add(Conv1D(64, 50, padding='same'))
    model.add(ReLU())
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dropout(rate=0.1))
 
    model.add(Conv1D(64, 100, padding='same'))
    model.add(Dropout(rate=0.3))
    model.add(Flatten())
    return model

# Discriminator Model
def discriminator(InputShape):
    model = Sequential()
    
    model.add(Reshape((InputShape, 1), input_shape=(InputShape,)))
    model.add(Conv1D(32, 100, strides=7, padding='valid'))
    model.add(ReLU())
    model.add(AveragePooling1D(4))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dropout(rate=0.1))

    model.add(Conv1D(16, 50, strides=5, padding='valid'))
    model.add(ReLU())
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dropout(rate=0.1))

    model.add(Conv1D(8, 25, strides=3, padding='valid'))
    model.add(ReLU())
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dropout(rate=0.1))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dense(1, activation='sigmoid'))
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
    model.add(Conv1D(32, 100, strides=7, padding='valid'))
    model.add(ReLU())
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dropout(rate=0.1))
    
    model.add(Conv1D(16, 50, strides=5, padding='valid'))
    model.add(ReLU())
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dropout(rate=0.1))

    model.add(Conv1D(8, 25, strides=3, padding='valid'))
    model.add(ReLU())
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
