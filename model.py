"""
We are using simple neural network
input -> linear -> activation -> output
"""
import numpy as np

from keras import Sequential, optimizers
from keras.layers import Dense, Activation

class NeuralNet:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = Sequential([
            Dense(64, input_shape=input_shape,
            kernel_initializer='random_normal'),
            Activation('relu'),
            Dense(4),
            Activation('softmax')
        ])

    def compile(self, lr=0.001):
        opt = optimizers.Adam(lr = lr)
        self.model.compile(optimizer=opt,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        return self.model
        
    def train(self, X=None, y=None, batch_size=32, epochs=1):
        self.model.fit(x=X,
                       y=y,
                       batch_size=batch_size,
                       epochs=epochs)
    
    def predict(self, x, batch_size=None):
        return np.argmax(self.model.predict(x, batch_size), axis=1)