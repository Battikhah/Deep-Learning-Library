# Deep Learning Library

A comprehensive deep learning library built from scratch in Python using NumPy. This library provides fundamental building blocks for creating, training, and evaluating neural networks with modern optimization techniques.

## Features

- **Neural Network Architecture**: Modular design with customizable layers
- **Layer Types**:
  - Linear (fully connected) layers with multiple initialization methods
  - Activation layers (Tanh, ReLU, Sigmoid, Softmax)
  - Dropout layer for regularization
- **Loss Functions**: Mean Squared Error (MSE)
- **Optimizers**:
  - Stochastic Gradient Descent (SGD) with learning rate decay
  - Adam optimizer with adaptive learning rates
- **Data Handling**:
  - Batch processing with shuffling support
  - Train/test splitting functionality
- **Model Evaluation**: Comprehensive metrics suite
- **Training Pipeline**: Complete training loop with forward/backward propagation
- **Model Persistence**: Save and load trained models

## Project Structure

```
net/
├── __init__.py       # Package initialization
├── tensor.py         # Tensor type definitions
├── layers.py         # Neural network layers (Linear, Activation, Dropout)
├── nn.py            # Neural network container with save/load functionality
├── loss.py          # Loss functions (MSE)
├── optim.py         # Optimizers (SGD, Adam)
├── data.py          # Data iterators and batch processing
├── train.py         # Training loop implementation
├── train_test.py    # Train/test splitting utilities
└── eval.py          # Comprehensive evaluation metrics
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
net = NeuralNets([
    Linear(input_size=2, output_size=2),
    Tanh(),
    Linear(input_size=2, output_size=2),
])

train(net, inputs, targets, num_epochs=5000)
```

### FizzBuzz Neural Network

A creative approach to solving FizzBuzz using neural networks with comprehensive evaluation:

```python
from net.train import train
from net.nn import NeuralNets
from net.layers import Linear, Tanh
from net.optim import Adam
from net.eval import check_accuracy, check_precision, check_recall, check_f1_score

# Create network with Xavier initialization
net = NeuralNets([
    Linear(input_size=10, output_size=64, init_type="xavier"),
    Tanh(),
    Linear(input_size=64, output_size=32, init_type="xavier"),
    Tanh(),
    Linear(input_size=32, output_size=4, init_type="xavier")
])

# Train with Adam optimizer
train(net, inputs, targets, num_epochs=5000, optimizer=Adam(learning_rate=0.001))

# Evaluate model performance
print("Accuracy:", check_accuracy(net, inputs, targets))
print("Precision:", check_precision(net, inputs, targets))
print("Recall:", check_recall(net, inputs, targets))
print("F1 Score:", check_f1_score(net, inputs, targets))
```

## Usage

### 1. Define your network architecture

```python
from net.nn import NeuralNets
from net.layers import Linear, Tanh, Relu, Dropout

net = NeuralNets([
    Linear(input_size=10, output_size=20, init_type="xavier"),
    Relu(),
    Dropout(dropout_rate=0.3),
    Linear(input_size=20, output_size=10, init_type="he"),
    Tanh(),
    Linear(input_size=10, output_size=1, init_type="xavier")
])
```

### 2. Split your data

```python
from net.train_test import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    inputs, targets, test_size=0.2, random_state=42
)
```

### 3. Train your network

```python
from net.train import train
from net.optim import Adam, SGD

# Option 1: Adam optimizer
train(net, X_train, y_train,
      num_epochs=1000,
      optimizer=Adam(learning_rate=0.001, learning_decay=0.01))

# Option 2: SGD with learning rate decay
train(net, X_train, y_train,
      num_epochs=1000,
      optimizer=SGD(learning_rate=0.01, learning_decay=0.001))
```

### 4. Evaluate your model

```python
from net.eval import check_accuracy, confusion_matrix_seaborn

# Basic metrics
accuracy = check_accuracy(net, X_test, y_test)
precision = check_precision(net, X_test, y_test)
recall = check_recall(net, X_test, y_test)
f1 = check_f1_score(net, X_test, y_test)

# Visualize confusion matrix
confusion_matrix_seaborn(net, X_test, y_test)
```

### 5. Save and load models

```python
# Save trained model
net.save('my_model.pkl')

# Load model
net = NeuralNets([])  # Create empty network
net = net.load('my_model.pkl')
```

## Core Components

### Network Architecture

- **[`NeuralNets`](net/nn.py)**: Container for layers with forward/backward propagation and model persistence

### Layers

- **[`Layer`](net/layers.py)**: Base class for all network layers
- **[`Linear`](net/layers.py)**: Fully connected layer with Xavier/He initialization
- **[`Tanh`](net/layers.py)**: Hyperbolic tangent activation function
- **[`Relu`](net/layers.py)**: Rectified linear activation function
- **[`Sigmoid`](net/layers.py)**: Sigmoid activation function
- **[`Softmax`](net/layers.py)**: Softmax activation function
- **[`Dropout`](net/layers.py)**: Dropout layer for regularization

### Optimizers

- **[`SGD`](net/optim.py)**: Stochastic gradient descent with learning rate decay
- **[`Adam`](net/optim.py)**: Adam optimizer with adaptive learning rates

### Loss Functions

- **[`MSE`](net/loss.py)**: Mean squared error loss function

### Evaluation Metrics

- **[`check_accuracy`](net/eval.py)**: Calculate model accuracy
- **[`check_precision`](net/eval.py)**: Calculate macro-averaged precision
- **[`check_recall`](net/eval.py)**: Calculate macro-averaged recall
- **[`check_f1_score`](net/eval.py)**: Calculate macro-averaged F1-score
- **[`confusion_matrix`](net/eval.py)**: Generate confusion matrix
- **[`confusion_matrix_seaborn`](net/eval.py)**: Visualize confusion matrix

### Data Utilities

- **[`train_test_split`](net/train_test.py)**: Split data into training and testing sets
- **[`BatchIterator`](net/data.py)**: Batch processing with shuffle support

## Recent Features ✅

- **✅ Adam Optimizer**: Adaptive learning rate optimization with bias correction
- **✅ Multiple Activation Functions**: ReLU, Sigmoid, Softmax in addition to Tanh
- **✅ Dropout Layer**: Regularization to prevent overfitting
- **✅ Weight Initialization**: Xavier and He initialization methods
- **✅ Learning Rate Decay**: For both SGD and Adam optimizers
- **✅ Model Evaluation Suite**: Accuracy, precision, recall, F1-score, and confusion matrix
- **✅ Train/Test Splitting**: Utilities for data splitting with shuffle support
- **✅ Model Persistence**: Save and load trained models
- **✅ Enhanced Examples**: FizzBuzz with comprehensive evaluation

## Future Enhancements

### Planned Layer Types

- Convolutional layers (Conv2D, MaxPool2D)
- Recurrent layers (LSTM, GRU)
- Batch normalization
- Layer normalization

### Additional Loss Functions

- Cross-entropy loss
- Binary cross-entropy
- Huber loss
- Custom loss function support

### Advanced Optimizers

- RMSprop
- AdaGrad
- Learning rate scheduling
- Gradient clipping

### Enhanced Features

- GPU support via CuPy
- Automatic differentiation
- Model visualization
- Tensorboard-like logging
- Cross-validation utilities
- Early stopping mechanisms

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

**Dependencies:**

- Python 3.7+
- NumPy >= 1.21.0
- Matplotlib >= 3.5.0 (for visualization)
- Seaborn >= 0.11.0 (for confusion matrix plotting)

## Getting Started

1. **Clone the repository**
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Run the XOR example**: `python XOR.py`
4. **Run the FizzBuzz example**: `python fizzbuss.py`

## Examples in Action

Both [XOR.py](XOR.py) and [fizzbuss.py](fizzbuss.py) provide complete working examples that demonstrate:

- Network architecture design
- Training with different optimizers
- Model evaluation with multiple metrics
- Confusion matrix visualization

## License

This project is open source and available under the MIT License.
