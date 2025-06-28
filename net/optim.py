"""
Optimizer adjusts the parameters of a model based 
on the gradients computed during backpropagation.
"""

from net.nn import NeuaralNets

class Optimizer:
    def step(self, net: NeuaralNets) -> None:
        """
        Update the model parameters based on the gradients.
        This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
class SGD(Optimizer):
    def __init__(self, learning_rate: float = 0.01) -> None:
        self.learning_rate = learning_rate
    
    def step(self, net: NeuaralNets) -> None:
        # Update the model parameters using Stochastic Gradient Descent.
        for param, grad in net.params_and_grads():
            param-= self.learning_rate * grad
