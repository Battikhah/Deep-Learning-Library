"""
Function that trains a neural network model.
"""

from net.tensor import Tensor
from net.nn import NeuaralNets
from net.loss import Loss, MSE
from net.optim import Optimizer, SGD
from net.data import DataIterator, BatchIterator

def train(net: NeuaralNets, 
          inputs: Tensor, 
          targets: Tensor, 
          num_epochs: int = 10,
          iterator: DataIterator = BatchIterator(),
          loss: Loss = MSE(),
          optimizer: Optimizer = SGD()) -> None:
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in iterator(inputs, targets):
            # Forward pass
            predictions = net.forward(batch.inputs)
            # Calculate loss
            epoch_loss += loss.loss(predictions, batch.targets)
            # Gradiant calculation
            grad = loss.grad(predictions, batch.targets)
            # Backward pass
            net.backward(grad)
            # Update parameters
            optimizer.step(net)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(inputs)}")


