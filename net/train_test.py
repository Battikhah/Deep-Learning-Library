from net.tensor import Tensor
import numpy as np

def train_test_split(inputs: Tensor, 
                    targets: Tensor, 
                    test_size: float = 0.2, 
                    random_state: int = 0,
                    shuffle: bool = True) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Split arrays into random train and test subsets.
    
    Parameters:
    -----------
    inputs : Tensor
        Input data to split
    targets : Tensor
        Target data to split
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split (0.0 to 1.0)
    random_state : int, optional
        Random seed for reproducible results
    shuffle : bool, default=True
        Whether to shuffle the data before splitting
    
    Returns:
    --------
    tuple[Tensor, Tensor, Tensor, Tensor]
        X_train, X_test, y_train, y_test
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0.0 and 1.0")
    
    if len(inputs) != len(targets):
        raise ValueError("inputs and targets must have the same length")
    
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(inputs)
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    test_size_absolute = int(n_samples * test_size)
    train_size_absolute = n_samples - test_size_absolute
    
    train_indices = indices[:train_size_absolute]
    test_indices = indices[train_size_absolute:]
    
    X_train = inputs[train_indices]
    X_test = inputs[test_indices]
    y_train = targets[train_indices]
    y_test = targets[test_indices]
    
    return X_train, X_test, y_train, y_test