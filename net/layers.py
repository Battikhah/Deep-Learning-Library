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

class RNN(Layer):
    def __init__(self, input_size: int, hidden_size: int, activation: str = "tanh"):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation
        
        # Initialize weights
        self.params = {
            "W_xh": np.random.randn(input_size, hidden_size) * 0.01,
            "W_hh": np.random.randn(hidden_size, hidden_size) * 0.01,
            "b_h": np.zeros((1, hidden_size))
        }
        
        self.grads = {key: np.zeros_like(val) for key, val in self.params.items()}
        
        # Store states for backpropagation
        self.hidden_states = []
        self.inputs = []
        
    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass for RNN
        inputs shape: (sequence_length, batch_size, input_size)
        """
        seq_len, batch_size, _ = input.shape
        
        # Initialize hidden state
        h = np.zeros((batch_size, self.hidden_size))
        outputs = []
        
        self.hidden_states = [h]
        self.inputs = []
        
        for t in range(seq_len):
            x_t = input[t]  # (batch_size, input_size)
            self.inputs.append(x_t)
            
            # h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
            h = np.tanh(
                np.dot(x_t, self.params["W_xh"]) + 
                np.dot(h, self.params["W_hh"]) + 
                self.params["b_h"]
            )
            
            self.hidden_states.append(h)
            outputs.append(h)
        
        return np.array(outputs)  # (seq_len, batch_size, hidden_size)
    
    def backward(self, grad_output: Tensor) -> Tensor:
        """
        Backward pass for RNN using Backpropagation Through Time (BPTT)
        """
        seq_len, batch_size, _ = grad_output.shape
        
        # Initialize gradients
        self.grads = {key: np.zeros_like(val) for key, val in self.params.items()}
        input_grads = np.zeros((seq_len, batch_size, self.input_size))
        
        # Initialize gradient flowing back through hidden states
        dh_next = np.zeros((batch_size, self.hidden_size))
        
        # Backpropagate through time
        for t in reversed(range(seq_len)):
            # Current gradients
            dh = grad_output[t] + dh_next
            
            # Gradient through tanh activation
            h_t = self.hidden_states[t + 1]
            dtanh = dh * (1 - h_t * h_t)  # derivative of tanh
            
            # Gradients for weights and biases
            self.grads["W_xh"] += np.dot(self.inputs[t].T, dtanh)
            self.grads["W_hh"] += np.dot(self.hidden_states[t].T, dtanh)
            self.grads["b_h"] += np.sum(dtanh, axis=0, keepdims=True)
            
            # Gradients for inputs and previous hidden state
            input_grads[t] = np.dot(dtanh, self.params["W_xh"].T)
            dh_next = np.dot(dtanh, self.params["W_hh"].T)
        
        return input_grads

class LSTM(Layer):
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize all gates' weights
        concat_size = input_size + hidden_size
        
        self.params = {
            # Forget gate
            "W_f": np.random.randn(concat_size, hidden_size) * 0.01,
            "b_f": np.zeros((1, hidden_size)),
            
            # Input gate
            "W_i": np.random.randn(concat_size, hidden_size) * 0.01,
            "b_i": np.zeros((1, hidden_size)),
            
            # Candidate values
            "W_C": np.random.randn(concat_size, hidden_size) * 0.01,
            "b_C": np.zeros((1, hidden_size)),
            
            # Output gate
            "W_o": np.random.randn(concat_size, hidden_size) * 0.01,
            "b_o": np.zeros((1, hidden_size))
        }
        
        self.grads = {key: np.zeros_like(val) for key, val in self.params.items()}
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, input: Tensor) -> Tensor:
        seq_len, batch_size, _ = input.shape

        # Initialize states
        h = np.zeros((batch_size, self.hidden_size))
        C = np.zeros((batch_size, self.hidden_size))
        
        outputs = []
        self.cache = []
        
        for t in range(seq_len):
            x_t = input[t]

            # Concatenate input and previous hidden state
            concat = np.column_stack([h, x_t])
            
            # Gates
            f_t = self.sigmoid(np.dot(concat, self.params["W_f"]) + self.params["b_f"])
            i_t = self.sigmoid(np.dot(concat, self.params["W_i"]) + self.params["b_i"])
            C_tilde = np.tanh(np.dot(concat, self.params["W_C"]) + self.params["b_C"])
            o_t = self.sigmoid(np.dot(concat, self.params["W_o"]) + self.params["b_o"])
            
            # Update cell state and hidden state
            C = f_t * C + i_t * C_tilde
            h = o_t * np.tanh(C)
            
            outputs.append(h)
            
            # Cache for backward pass
            self.cache.append({
                'h_prev': self.cache[-1]['h'] if self.cache else np.zeros_like(h),
                'C_prev': self.cache[-1]['C'] if self.cache else np.zeros_like(C),
                'h': h, 'C': C, 'f_t': f_t, 'i_t': i_t, 
                'C_tilde': C_tilde, 'o_t': o_t, 'concat': concat
            })
        
        return np.array(outputs)
    
    def backward(self, grad_output: Tensor) -> Tensor:
        seq_len, batch_size, _ = grad_output.shape
        input_grads = np.zeros((seq_len, batch_size, self.input_size))
        dh_next = np.zeros((batch_size, self.hidden_size))
        dC_next = np.zeros((batch_size, self.hidden_size))
        self.grads = {key: np.zeros_like(val) for key, val in self.params.items()}
        for t in reversed(range(seq_len)):
            # Current gradients
            h = self.cache[t]['h']
            C = self.cache[t]['C']
            f_t = self.cache[t]['f_t']
            i_t = self.cache[t]['i_t']
            C_tilde = self.cache[t]['C_tilde']
            o_t = self.cache[t]['o_t']
            concat = self.cache[t]['concat']
            
            dh = grad_output[t] + dh_next
            dC = dC_next + dh * o_t * (1 - np.tanh(C) ** 2)
            
            # Gradients for gates
            do_t = dh * np.tanh(C)
            dC_prev = dC * f_t
            df_t = dC * C * f_t * (1 - f_t)
            di_t = dC * C_tilde * i_t * (1 - i_t)
            dC_tilde = dC * i_t * (1 - C_tilde ** 2)
            
            # Gradients for weights and biases
            self.grads["W_f"] += np.dot(concat.T, df_t)
            self.grads["b_f"] += np.sum(df_t, axis=0, keepdims=True)
            
            self.grads["W_i"] += np.dot(concat.T, di_t)
            self.grads["b_i"] += np.sum(di_t, axis=0, keepdims=True)
            
            self.grads["W_C"] += np.dot(concat.T, dC_tilde)
            self.grads["b_C"] += np.sum(dC_tilde, axis=0, keepdims=True)
            
            self.grads["W_o"] += np.dot(concat.T, do_t)
            self.grads["b_o"] += np.sum(do_t, axis=0, keepdims=True)
            
            # Gradients for inputs and previous hidden state
            input_grads[t] = (
                df_t @ self.params["W_f"].T +
                di_t @ self.params["W_i"].T +
                dC_tilde @ self.params["W_C"].T +
                do_t @ self.params["W_o"].T
            )[:, :self.input_size]
            
            dh_next = (
                df_t @ self.params["W_f"].T +
                di_t @ self.params["W_i"].T +
                dC_tilde @ self.params["W_C"].T +
                do_t @ self.params["W_o"].T
            )[:, self.input_size:]
            dC_next = dC * f_t
        return input_grads

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
