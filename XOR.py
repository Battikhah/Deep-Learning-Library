"""
XOR cannot be solved by a single layer perceptron, 
but can be solved by a multi-layer perceptron.
"""

from net.train import train
from net.nn import NeuralNets
from net.layers import Linear, Tanh

import numpy as np

# Define the XOR dataset
XOR_inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
XOR_targets = np.array([
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]
])

""""
# Create the neural network (Single Linear Layer)
net = NeuralNets([
    Linear(input_size=2, output_size=2),
])
train(net, XOR_inputs, XOR_targets, num_epochs=5000)

for x,y, in zip(XOR_inputs, XOR_targets):
    prediction = net.forward(x)
    print(f"Input: {x}, Target: {y}, Prediction: {prediction[0]}")
"""

# Create the neural network (Multi-Layer Perceptron)
net = NeuralNets([
    Linear(input_size=2, output_size=2),
    Tanh(),
    Linear(input_size=2, output_size=2),
])

train(net, XOR_inputs, XOR_targets, num_epochs=5000)

for x, y in zip(XOR_inputs, XOR_targets):
    prediction = net.forward(x)
    print(f"Input: {x}, Target: {y}, Prediction: {prediction[0]}")