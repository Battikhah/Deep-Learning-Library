"""
Optimizer adjusts the parameters of a model based 
on the gradients computed during backpropagation.
"""

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
            if id(param) not in self.m:
                self.m[id(param)] = grad.copy()
                self.v[id(param)] = grad.copy()
            else:
                self.m[id(param)] = self.beta1 * self.m[id(param)] + (1 - self.beta1) * grad
                self.v[id(param)] = self.beta2 * self.v[id(param)] + (1 - self.beta2) * (grad ** 2)
            
            m_hat = self.m[id(param)] / (1 - self.beta1 ** self.t)
            v_hat = self.v[id(param)] / (1 - self.beta2 ** self.t)
            
            param -= self.learning_rate * m_hat / (v_hat ** 0.5 + self.epsilon)
