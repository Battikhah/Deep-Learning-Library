# Deep Learning Library

A minimal deep learning library built from scratch in Python using NumPy. This library provides fundamental building blocks for creating and training neural networks.

## Features

- **Neural Network Architecture**: Modular design with customizable layers
- **Layer Types**:
  - Linear (fully connected) layers
  - Activation layers (Tanh, with more coming soon)
- **Loss Functions**: Mean Squared Error (MSE)
- **Optimizers**: Stochastic Gradient Descent (SGD)
- **Data Handling**: Batch processing with shuffling support
- **Training Pipeline**: Complete training loop with forward/backward propagation

## Project Structure

```
net/
├── __init__.py       # Package initialization
├── tensor.py         # Tensor type definitions
├── layers.py         # Neural network layers
├── nn.py            # Neural network container
├── loss.py          # Loss functions
├── optim.py         # Optimizers
├── data.py          # Data iterators and batch processing
└── train.py         # Training loop implementation
```

## Examples

### XOR Problem

Classic example demonstrating that XOR cannot be solved with a single layer but requires a multi-layer perceptron:

```python
from net.train import train
from net.nn import NeuralNets
from net.layers import Linear, Tanh
import numpy as np

# XOR dataset
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

# Multi-layer perceptron
net = NeuaralNets([
    Linear(input_size=2, output_size=2),
    Tanh(),
    Linear(input_size=2, output_size=2),
])

train(net, inputs, targets, num_epochs=5000)
```

### FizzBuzz Neural Network

A creative approach to solving FizzBuzz using neural networks:

```python
from net.train import train
from net.nn import NeuaralNets
from net.layers import Linear, Tanh
from net.optim import SGD

net = NeuaralNets([
    Linear(input_size=10, output_size=50),
    Tanh(),
    Linear(input_size=50, output_size=4),
])

train(net, inputs, targets, num_epochs=5000, optimizer=SGD(learning_rate=0.001))
```

## Usage

1. **Define your network architecture**:

```python
from net.nn import NeuaralNets
from net.layers import Linear, Tanh

net = NeuaralNets([
    Linear(input_size=10, output_size=20),
    Tanh(),
    Linear(input_size=20, output_size=1),
])
```

2. **Train your network**:

```python
from net.train import train
from net.optim import SGD

train(net, inputs, targets,
      num_epochs=1000,
      optimizer=SGD(learning_rate=0.01))
```

3. **Make predictions**:

```python
predictions = net.forward(test_inputs)
```

## Core Components

- **[`NeuaralNets`](net/nn.py)**: Container for layers that handles forward/backward propagation
- **[`Layer`](net/layers.py)**: Base class for all network layers
- **[`Linear`](net/layers.py)**: Fully connected layer with weights and biases
- **[`Tanh`](net/layers.py)**: Hyperbolic tangent activation function
- **[`Relu`](net/layers.py)**: Rectified linear activation function
- **[`Sigmoid`](net/layers.py)**: Maps input to (0, 1) range.
- **[`SGD`](net/optim.py)**: Stochastic gradient descent optimizer
- **[`MSE`](net/loss.py)**: Mean squared error loss function

## Future Updates

This repository is actively being developed and will be updated with:

- **Additional Layer Types**:

  - Convolutional layers
  - LSTM/GRU layers
  - Dropout layers
  - Batch normalization

- **More Loss Functions**:

  - Cross-entropy loss
  - Binary cross-entropy
  - Huber loss

- **Model Evaluation & Metrics**:

  - ✅ Accuracy calculation
  - ✅ Precision and Recall metrics
  - ✅ F1-Score calculation
  - ✅ Confusion matrix visualization
  - ROC curves and AUC metrics

- **Advanced Features**:

  - Model serialization/loading
  - GPU support
  - Regularization techniques (L1/L2)
  - Learning rate scheduling
  - Early stopping
  - Model checkpointing

- **Optimization Improvements**:

  - ✅ Adam optimizer
  - RMSprop
  - AdaGrad
  - Learning rate decay
  - Momentum variants

- **Data Processing**:

  - Data preprocessing utilities
  - Data augmentation
  - Train/validation/test splits
  - Cross-validation support

- **Documentation & Examples**:
  - More comprehensive tutorials
  - Performance benchmarks
  - Comparison with other frameworks
  - Advanced usage examples

## Recent Additions ✨

- **Adam Optimizer**: Adaptive learning rate optimization
- **Additional Activation Functions**: ReLU and Sigmoid
- **Model Evaluation Suite**: Comprehensive metrics including accuracy, precision, recall, and F1-score
- **Enhanced FizzBuzz Example**: Complete with performance evaluation

## Requirements

- Python 3.x
- NumPy

## License

This project is open source and available under the MIT License.
