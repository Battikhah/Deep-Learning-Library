"""
Neural network layers for the model.

"""
from typing import Dict, Callable
from net.tensor import Tensor
import numpy as np

class Layer:
    """
    Base class for all layers in the neural network.
    This class defines the interface for layers, which includes methods
    for forward and backward propagation.
    """

    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {} # Dictionary to hold layer parameters (weights, biases, etc.)
        self.grads: Dict[str, Tensor] = {}


    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass through the layer.
        This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def backward(self, grad_output: Tensor) -> Tensor:
        """
        Backward pass through the layer.
        This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
class Linear(Layer):
    """
    Linear layer (fully connected layer).
    This layer applies a linear transformation to the input.
    """

    def __init__(self, input_size: int, output_size: int, init_type="xavier") -> None:
        super().__init__()
        # Initialize weights and biases
        if init_type == "xavier":
            self.params["w"] = np.random.randn(input_size, output_size) * np.sqrt(2. / (input_size + output_size))
        elif init_type == "he":
            self.params["w"] = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        else:
            self.params["w"] = np.random.randn(input_size, output_size) * 0.01 

        self.params["b"] = np.random.randn(output_size)

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass through the linear layer.
        Applies the transformation: output = input @ weights + biases.
        """
        self.input = input
        return input @ self.params["w"] + self.params["b"]

    def backward(self, grad_output: Tensor) -> Tensor:
        """
        if y = f(x) and x = a @ b + c,
        then dy/da = f'(x) * b,
        dy/db = f'(x) * a,
        dy/dc = f'(x) * 1,
        where f'(x) is the derivative of the activation function.

        which lead to dy/da = f'(x) @ b.T
        and dy/db = f'(x) @ a.T
        and dy/dc = f'(x) * 1
        """

        self.grads["b"] = np.sum(grad_output, axis=0)
        self.grads["w"] = self.input.T @ grad_output
        return grad_output @ self.params["w"].T

class Dropout(Layer):
    """
    Dropout Layer.
    This layer randomly sets a fraction of the input units to zero during training,
    which helps prevent overfitting.
    """
    def __init__(self, dropout_rate: float = 0.3) -> None:
        super().__init__()
        self.dropout_rate = dropout_rate
        self.mask = None
        self.training = True  
    def forward(self, input: Tensor) -> Tensor:
        if self.training:
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, size=input.shape)
            return input * self.mask / (1 - self.dropout_rate)  
        return input  
    
    def backward(self, grad_output: Tensor) -> Tensor:
        if self.training:
            return grad_output * self.mask / (1 - self.dropout_rate)
        return grad_output 

# Type alias for activation functions
F = Callable[[Tensor], Tensor]
class Activation(Layer):
    """
    Base class for activation layers.
    This class defines the interface for activation layers, which includes methods
    for forward and backward propagation.
    """

    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass through the activation layer.
        Applies the activation function to the input.
        """
        self.input = input
        return self.f(input)
    
    def backward(self, grad_output: Tensor) -> Tensor:
        """
        Backward pass through the activation layer.
        Applies the derivative of the activation function to the gradient.
        """
        return self.f_prime(self.input) * grad_output

# Hyperbolic tangent activation function and its derivative
def tanh(x: Tensor) -> Tensor:
    """
    Hyperbolic tangent activation function.
    """
    return np.tanh(x)

def tanh_prime(x: Tensor) -> Tensor:
    """
    Derivative of the hyperbolic tangent activation function.
    """
    return 1 - np.tanh(x) ** 2

class Tanh(Activation):
    def __init__(self) -> None:
        super().__init__(f=tanh, f_prime=tanh_prime)

# ReLU activation function and its derivative
def relu(x: Tensor) -> Tensor:
    return np.maximum(0, x)

def relu_prime(x: Tensor) -> Tensor:
    return (x > 0).astype(float)

class Relu(Activation):
    def __init__(self) -> None:
        super().__init__(f=relu, f_prime=relu_prime)

# Sigmoid activation function and its derivative
def sigmoid(x: Tensor) -> Tensor:
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x: Tensor) -> Tensor:
    sig = sigmoid(x)
    return sig * (1 - sig)

class Sigmoid(Activation):
    def __init__(self) -> None:
        super().__init__(f=sigmoid, f_prime=sigmoid_prime)

# Softmax activation function and its derivative
def softmax(x: Tensor) -> Tensor:
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def softmax_prime(x: Tensor) -> Tensor:
    s = softmax(x)
    return s * (1 - s)

class Softmax(Activation):
    def __init__(self) -> None:
        super().__init__(f=softmax, f_prime=softmax_prime)
