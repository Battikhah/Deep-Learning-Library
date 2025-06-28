"""
A nueral net is a collection of layers.
It is a container for layers that allows for forward and 
backward propagation through the network.
"""
from typing import Sequence, Iterator, Tuple
from net.tensor import Tensor
from net.layers import Layer

class NeuaralNets:
    def __init__(self, layers: Sequence[Layer]) -> None:
        """
        Initialize the neural network with a sequence of layers.
        :param layers: A sequence of Layer objects.
        """
        self.layers = layers
    
    def forward(self, input: Tensor) -> Tensor:
        """
        Perform a forward pass through the network.
        :param input: Input tensor to the network.
        :return: Output tensor after passing through all layers.
        """
        for layer in self.layers:
            input = layer.forward(input)
        return input
    
    def backward(self, grad_output: Tensor) -> Tensor:
        """
        Perform a backward pass through the network.
        :param grad_output: Gradient tensor from the loss function.
        :return: Gradient tensor after passing through all layers.
        """
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output
    
    def params_and_grads(self) -> Iterator[Tuple[Tensor, Tensor]]:
        """
        Iterate over the parameters and their gradients in the network.
        :return: An iterator of tuples containing (parameter, gradient).
        """
        for layer in self.layers:
            for param_name, param in layer.params.items():
                yield param, layer.grads[param_name]