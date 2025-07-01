"""
This file will contain the functions that check certain parameters of the model.
"""

import numpy as np
from net.tensor import Tensor
from net.nn import NeuralNets

def check_accuracy(net: NeuralNets, inputs: Tensor, targets: Tensor) -> float:
    """
    Calculate the accuracy of the model on the given dataset.
    
    Args:
        net: The neural network model
        inputs: Input data
        targets: True target labels (one-hot encoded)
    
    Returns:
        Accuracy as a float between 0 and 1
    """
    predictions = net.forward(inputs)
    
    # Convert predictions and targets to class indices
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(targets, axis=1)
    
    # Calculate accuracy
    correct_predictions = np.sum(predicted_classes == true_classes)
    total_predictions = len(true_classes)
    
    return correct_predictions / total_predictions

def check_precision(net: NeuralNets, inputs: Tensor, targets: Tensor) -> float:
    """
    Calculate the macro-averaged precision of the model.
    
    Args:
        net: The neural network model
        inputs: Input data
        targets: True target labels (one-hot encoded)
    
    Returns:
        Macro-averaged precision as a float between 0 and 1
    """
    predictions = net.forward(inputs)
    
    # Convert predictions and targets to class indices
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(targets, axis=1)
    
    # Get unique classes
    classes = np.unique(true_classes)
    precisions = []
    
    for cls in classes:
        # True positives: correctly predicted as this class
        true_positives = np.sum((predicted_classes == cls) & (true_classes == cls))
        
        # False positives: incorrectly predicted as this class
        false_positives = np.sum((predicted_classes == cls) & (true_classes != cls))
        
        # Calculate precision for this class
        if true_positives + false_positives > 0:
            precision = true_positives / (true_positives + false_positives)
        else:
            precision = 0.0
        
        precisions.append(precision)
    
    # Return macro-averaged precision
    return float(np.mean(precisions))

def check_recall(net: NeuralNets, inputs: Tensor, targets: Tensor) -> float:
    """
    Calculate the macro-averaged recall of the model.
    
    Args:
        net: The neural network model
        inputs: Input data
        targets: True target labels (one-hot encoded)
    
    Returns:
        Macro-averaged recall as a float between 0 and 1
    """
    predictions = net.forward(inputs)
    
    # Convert predictions and targets to class indices
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(targets, axis=1)
    
    # Get unique classes
    classes = np.unique(true_classes)
    recalls = []
    
    for cls in classes:
        # True positives: correctly predicted as this class
        true_positives = np.sum((predicted_classes == cls) & (true_classes == cls))
        
        # False negatives: incorrectly predicted as not this class
        false_negatives = np.sum((predicted_classes != cls) & (true_classes == cls))
        
        # Calculate recall for this class
        if true_positives + false_negatives > 0:
            recall = true_positives / (true_positives + false_negatives)
        else:
            recall = 0.0
        
        recalls.append(recall)
    
    # Return macro-averaged recall
    return float(np.mean(recalls))

def check_f1_score(net: NeuralNets, inputs: Tensor, targets: Tensor) -> float:
    """
    Calculate the macro-averaged F1-score of the model.
    
    Args:
        net: The neural network model
        inputs: Input data
        targets: True target labels (one-hot encoded)
    
    Returns:
        Macro-averaged F1-score as a float between 0 and 1
    """
    precision = check_precision(net, inputs, targets)
    recall = check_recall(net, inputs, targets)
    
    if precision + recall > 0:
        return 2 * (precision * recall) / (precision + recall)
    else:
        return 0.0