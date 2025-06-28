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

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        # Initialize weights and biases
        self.params["w"] = np.random.randn(input_size, output_size)
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

F = Callable[[Tensor], Tensor]

class Activiation(Layer):
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

class Tanh(Activiation):
    def __init__(self) -> None:
        super().__init__(f=tanh, f_prime=tanh_prime)