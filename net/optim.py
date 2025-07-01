"""
Optimizer adjusts the parameters of a model based 
on the gradients computed during backpropagation.
"""
import numpy as np
from net.nn import NeuralNets

class Optimizer:
    def step(self, net: NeuralNets) -> None:
        """
        Update the model parameters based on the gradients.
        This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
class SGD(Optimizer):
    def __init__(self, learning_rate: float = 0.01) -> None:
        self.learning_rate = learning_rate
    
    def step(self, net: NeuralNets) -> None:
        # Update the model parameters using Stochastic Gradient Descent.
        for param, grad in net.params_and_grads():
            param-= self.learning_rate * grad

class Adam(Optimizer):
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8) -> None:
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0
    
    def step(self, net: NeuralNets) -> None:
        self.t += 1
        for param, grad in net.params_and_grads():
            param_id = id(param)
            # Ensure grad is a numpy array for copy and math operations
            grad = np.array(grad)
            if param_id not in self.m:
                self.m[param_id] = np.zeros_like(grad)
                self.v[param_id] = np.zeros_like(grad)
            self.m[param_id] = self.beta1 * self.m[param_id] + (1 - self.beta1) * grad
            self.v[param_id] = self.beta2 * self.v[param_id] + (1 - self.beta2) * (grad ** 2)
            
            m_hat = self.m[param_id] / (1 - self.beta1 ** self.t)
            v_hat = self.v[param_id] / (1 - self.beta2 ** self.t)
            
            update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            param -= update