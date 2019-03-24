"""
Here we are predicting output for fizzbuzz game
with our neural network
"""
import numpy as np

from model import NeuralNet
from generate_data import inputs, labels

# Min is 1 and max is 1024
first = 101
last = 1024 
X = inputs(first, last)
y = labels(first, last)
model = NeuralNet(input_shape=(10,))
model.compile(lr = 0.001)
model.train(X,y,batch_size=32,epochs=1000)

first_test = 1
last_test = 100
X_test = inputs(first_test, last_test)
y_pred = model.predict(X_test)
iter = range(first_test, last_test+1)
for i in range(last_test-first_test+1):
    if y_pred[i] == 0:
        pred = i+1 
    elif y_pred[i] == 1:
        pred = "fizz"
    elif y_pred[i] == 2:
        pred = "buzz"
    else:
        pred = "fizzbuzz"
    print("{}: {}".format(i+1, pred))