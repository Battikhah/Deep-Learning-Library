import numpy as np
from net.tensor import Tensor

class Loss:
    """
    Base class for loss functions.
    This class defines the interface for loss functions, which includes
    methods to calculate the loss and its gradient.
    """
    
    def loss(self, y_predictions: Tensor, y_true: Tensor) -> float:
        """
        Calculate the loss between predictions and true values.
        This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def grad(self, y_predictions: Tensor, y_true: Tensor) -> Tensor:
        """
        Calculate the gradient of the loss with respect to predictions.
        This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
class MSE(Loss):
    """
    Mean Squared Error (MSE) loss function.
    This class implements the MSE loss and its gradient.
    """

    def loss(self, y_predictions: Tensor, y_true: Tensor) -> float:
        """
        Calculate the Mean Squared Error loss.
        """
        return float(np.mean((y_predictions - y_true) ** 2))

    def grad(self, y_predictions: Tensor, y_true: Tensor) -> Tensor:
        """
        Calculate the gradient of the Mean Squared Error loss.
        """
        return 2 * (y_predictions - y_true) / y_predictions.size