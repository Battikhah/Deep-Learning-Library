"""
What is FizzBuzz?

FizzBuzz is a classic programming problem, often used in coding interviews and as a beginner's exercise. The task is as follows:

- Print the numbers from 1 to a given number `n`.
- For multiples of 3, print "Fizz" instead of the number.
- For multiples of 5, print "Buzz" instead of the number.
- For multiples of both 3 and 5 (15), print "FizzBuzz" instead of the number.
"""

import numpy as np
from typing import List

from net.train import train
from net.nn import NeuralNets
from net.layers import Linear, Tanh
from net.optim import SGD, Adam
from net.acc import check_accuracy, check_precision, check_recall, check_f1_score

def fizzbuzz_encode(x:int) -> List[int]:
    if x % 15 == 0:
        return [0, 0, 0, 1]  # FizzBuzz
    elif x % 3 == 0:
        return [0, 0, 1, 0]  # Fizz
    elif x % 5 == 0:
        return [0, 1, 0, 0]  # Buzz
    else:
        return [1, 0, 0, 0]  # Neither

def binary_encode(x: int) -> List[int]:
    return [int(bit) for bit in np.binary_repr(x, width=10)]

inputs = np.array([
    binary_encode(i) 
    for i in range(101, 1024)
])

targets = np.array([
    fizzbuzz_encode(i)
    for i in range(101, 1024)
])

net = NeuralNets([
    Linear(input_size=10, output_size=20),
    Tanh(),
    Linear(input_size=20, output_size=4)
])

train(net, inputs, targets, num_epochs=5000, optimizer=Adam(learning_rate=0.001))

for x in range(1, 101):
    prediction = net.forward(np.array([binary_encode(x)]))
    fizzbuzz_output = np.argmax(prediction[0])
    actual_output = np.argmax(fizzbuzz_encode(x))

    print(f"Input: {x}, "
          f"Predicted: {['Fizz', 'Buzz', 'FizzBuzz', 'Neither'][fizzbuzz_output]}, "
          f"Actual: {['Fizz', 'Buzz', 'FizzBuzz', 'Neither'][actual_output]}")

print("Checking model performance...")
print("Accuracy:", check_accuracy(net, inputs, targets))
print("Precision:", check_precision(net, inputs, targets))
print("Recall:", check_recall(net, inputs, targets))
print("F1 Score:", check_f1_score(net, inputs, targets))
print("Training complete.")

