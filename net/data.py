"""
Feeds inputs into our network via batches
"""
import pandas as pd
from net.tensor import Tensor
import numpy as np
from typing import Iterator, NamedTuple

Batch = NamedTuple('Batch', [('inputs', Tensor), ('targets', Tensor)])

class DataIterator:
    """
    An iterator that yields batches of data.
    This class is used to feed inputs into the network in batches.
    """

    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator[Batch]:
        raise NotImplementedError("Subclasses should implement this method.")
    
class BatchIterator(DataIterator):
    def __init__(self, batch_size: int= 32, shuffle: bool=True) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator[Batch]:
        starts = np.arange(0, len(inputs), self.batch_size)

        if self.shuffle:
            np.random.shuffle(starts)

        for start in starts:
            end = start + self.batch_size
            batch_inputs = inputs[start:end]
            batch_targets = targets[start:end]
            yield Batch(batch_inputs, batch_targets)

def label_encode(labels: np.ndarray) -> tuple:
    """
    Convert categorical labels to integer encoding.
    
    Args:
        labels: Array of categorical values
    
    Returns:
        tuple: (encoded_values, unique_labels)
    """
    unique_labels = np.unique(labels)
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    encoded = np.array([label_to_index[label] for label in labels], dtype=np.int32)
    
    return encoded, unique_labels

def standard_scaler(data: np.ndarray) -> tuple:
    """
    Standardize features by removing the mean and scaling to unit variance.
    
    Args:
        data: Feature matrix
    
    Returns:
        tuple: (scaled_data, means, stds)
    """
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    
    # Avoid division by zero
    stds = np.where(stds == 0, 1.0, stds)
    
    scaled_data = (data - means) / stds
    
    return scaled_data, means, stds