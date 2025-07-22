"""
Insurance Dataset Neural Network Example

This example demonstrates using the deep learning library on a real-world dataset
to predict insurance charges based on various features like age, sex, BMI, etc.
"""

import pandas as pd
import numpy as np
from net.nn import NeuralNets
from net.layers import Linear, Tanh, Relu, Dropout
from net.train import train
from net.optim import Adam, SGD
from net.eval import check_accuracy, check_precision, check_recall, check_f1_score, confusion_matrix_seaborn
from net.train_test import train_test_split
from net.loss import MSE
import matplotlib.pyplot as plt


def label_encode(column):
    """
    Simple label encoder implementation using NumPy.
    
    Args:
        column: Array-like column to encode
    
    Returns:
        tuple: (encoded_values, unique_labels)
    """
    unique_values = np.unique(column)
    mapping = {val: idx for idx, val in enumerate(unique_values)}
    encoded = np.array([mapping[val] for val in column])
    return encoded, unique_values


def standard_scaler(X):
    """
    Standard scaler implementation using NumPy.
    Standardizes features by removing the mean and scaling to unit variance.
    
    Args:
        X: Feature matrix
    
    Returns:
        tuple: (scaled_X, mean, std)
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    # Avoid division by zero
    std = np.where(std == 0, 1, std)
    scaled_X = (X - mean) / std
    return scaled_X, mean, std


def load_and_preprocess_data():
    """
    Load and preprocess the insurance dataset without sklearn.
    
    Returns:
        tuple: (X, y) where X is features and y is target (binned charges)
    """
    # Load the dataset
    df = pd.read_csv('Datasets/insurance.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Dataset columns: {list(df.columns)}")
    print("\nFirst few rows:")
    print(df.head())
    print("\nDataset info:")
    print(df.info())
    print("\nCharges statistics:")
    print(df['charges'].describe())
    
    # Encode categorical variables manually
    df['sex_encoded'], sex_labels = label_encode(df['sex'].values)
    df['smoker_encoded'], smoker_labels = label_encode(df['smoker'].values)
    df['region_encoded'], region_labels = label_encode(df['region'].values)
    
    print(f"\nEncoding mappings:")
    print(f"Sex: {dict(zip(sex_labels, range(len(sex_labels))))}")
    print(f"Smoker: {dict(zip(smoker_labels, range(len(smoker_labels))))}")
    print(f"Region: {dict(zip(region_labels, range(len(region_labels))))}")
    
    # Select features
    features = ['age', 'sex_encoded', 'bmi', 'children', 'smoker_encoded', 'region_encoded']
    X = df[features].values.astype(np.float32)
    
    # Create target variable (classification task)
    # Convert continuous charges to categorical bins
    charges = df['charges'].values
    charge_percentiles = np.percentile(charges.astype(np.float64), [25, 50, 75])
    print(f"\nCharge percentiles (25%, 50%, 75%): {charge_percentiles}")
    
    # Create 4 categories: Low, Medium-Low, Medium-High, High
    y_classes = np.zeros(len(charges), dtype=int)
    y_classes[charges <= charge_percentiles[0]] = 0  # Low
    y_classes[(charges > charge_percentiles[0]) & (charges <= charge_percentiles[1])] = 1  # Medium-Low
    y_classes[(charges > charge_percentiles[1]) & (charges <= charge_percentiles[2])] = 2  # Medium-High
    y_classes[charges > charge_percentiles[2]] = 3  # High
    
    print(f"\nClass distribution:")
    unique, counts = np.unique(y_classes, return_counts=True)
    for cls, count in zip(unique, counts):
        class_names = ['Low', 'Medium-Low', 'Medium-High', 'High']
        print(f"Class {cls} ({class_names[cls]}): {count} samples ({count/len(y_classes)*100:.1f}%)")
    
    # Convert to one-hot encoding for neural network
    num_classes = 4
    y_onehot = np.zeros((len(y_classes), num_classes))
    y_onehot[np.arange(len(y_classes)), y_classes] = 1
    
    # Normalize features using custom standard scaler
    X_normalized, feature_means, feature_stds = standard_scaler(X)
    
    print(f"\nFeatures shape: {X_normalized.shape}")
    print(f"Target shape: {y_onehot.shape}")
    print(f"Feature means: {feature_means}")
    print(f"Feature stds: {feature_stds}")
    
    return X_normalized, y_onehot, feature_means, feature_stds, sex_labels, smoker_labels, region_labels


def create_insurance_model(input_size: int, num_classes: int) -> NeuralNets:
    """
    Create a neural network model for insurance charge prediction.
    
    Args:
        input_size: Number of input features
        num_classes: Number of output classes
    
    Returns:
        NeuralNets: The created neural network
    """
    return NeuralNets([
        Linear(input_size=input_size, output_size=128, init_type="xavier"),
        Relu(),
        Dropout(dropout_rate=0.3),
        Linear(input_size=128, output_size=64, init_type="xavier"),
        Tanh(),
        Dropout(dropout_rate=0.2),
        Linear(input_size=64, output_size=32, init_type="xavier"),
        Relu(),
        Linear(input_size=32, output_size=num_classes, init_type="xavier")
    ])


def evaluate_model_performance(net: NeuralNets, X_test: np.ndarray, y_test: np.ndarray):
    """
    Evaluate and display comprehensive model performance metrics.
    
    Args:
        net: Trained neural network
        X_test: Test features
        y_test: Test targets (one-hot encoded)
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    
    # Calculate metrics
    accuracy = check_accuracy(net, X_test, y_test)
    precision = check_precision(net, X_test, y_test)
    recall = check_recall(net, X_test, y_test)
    f1 = check_f1_score(net, X_test, y_test)
    
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Show predictions vs actual for first 20 test samples
    print("\n" + "-"*50)
    print("SAMPLE PREDICTIONS")
    print("-"*50)
    print(f"{'Index':<6} {'Predicted':<15} {'Actual':<15} {'Correct':<8}")
    print("-"*50)
    
    predictions = net.forward(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    
    class_names = ['Low', 'Medium-Low', 'Medium-High', 'High']
    
    for i in range(min(20, len(X_test))):
        pred_name = class_names[predicted_classes[i]]
        true_name = class_names[true_classes[i]]
        is_correct = "✓" if predicted_classes[i] == true_classes[i] else "✗"
        print(f"{i:<6} {pred_name:<15} {true_name:<15} {is_correct:<8}")
    
    # Display confusion matrix
    print("\n" + "-"*50)
    print("CONFUSION MATRIX")
    print("-"*50)
    try:
        confusion_matrix_seaborn(net, X_test, y_test)
    except Exception as e:
        print(f"Could not display confusion matrix: {e}")
    
    return accuracy, precision, recall, f1


def demonstrate_feature_importance(X: np.ndarray, feature_names: list):
    """
    Demonstrate feature statistics and importance.
    
    Args:
        X: Feature matrix
        feature_names: List of feature names
    """
    print("\n" + "="*50)
    print("FEATURE STATISTICS")
    print("="*50)
    
    for i, name in enumerate(feature_names):
        feature_values = X[:, i]
        print(f"{name}:")
        print(f"  Mean: {np.mean(feature_values):.3f}")
        print(f"  Std:  {np.std(feature_values):.3f}")
        print(f"  Min:  {np.min(feature_values):.3f}")
        print(f"  Max:  {np.max(feature_values):.3f}")
        print()


def predict_single_sample(net: NeuralNets, sample_data: dict, feature_means: np.ndarray, 
                         feature_stds: np.ndarray, sex_labels: np.ndarray, 
                         smoker_labels: np.ndarray, region_labels: np.ndarray):
    """
    Make a prediction for a single insurance sample.
    
    Args:
        net: Trained neural network
        sample_data: Dictionary with sample features
        feature_means: Feature means for normalization
        feature_stds: Feature stds for normalization
        sex_labels, smoker_labels, region_labels: Label encoding mappings
    
    Returns:
        str: Predicted insurance charge category
    """
    # Encode categorical variables
    sex_mapping = {label: idx for idx, label in enumerate(sex_labels)}
    smoker_mapping = {label: idx for idx, label in enumerate(smoker_labels)}
    region_mapping = {label: idx for idx, label in enumerate(region_labels)}
    
    # Create feature vector
    features = np.array([
        sample_data['age'],
        sex_mapping[sample_data['sex']],
        sample_data['bmi'],
        sample_data['children'],
        smoker_mapping[sample_data['smoker']],
        region_mapping[sample_data['region']]
    ], dtype=np.float32).reshape(1, -1)
    
    # Normalize features
    normalized_features = (features - feature_means) / feature_stds
    
    # Make prediction
    prediction = net.forward(normalized_features)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    class_names = ['Low', 'Medium-Low', 'Medium-High', 'High']
    return class_names[predicted_class]


def main():
    """
    Main function to run the insurance dataset experiment.
    """
    print("Insurance Dataset Neural Network Analysis")
    print("="*60)
    
    # Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    X, y, feature_means, feature_stds, sex_labels, smoker_labels, region_labels = load_and_preprocess_data()
    
    # Feature names for reference
    feature_names = ['age', 'sex_encoded', 'bmi', 'children', 'smoker_encoded', 'region_encoded']
    
    # Show feature statistics
    demonstrate_feature_importance(X, feature_names)
    
    # Split data into train and test sets
    print("\n2. Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    
    # Create model
    print("\n3. Creating neural network model...")
    input_size = X.shape[1]
    num_classes = y.shape[1]
    net = create_insurance_model(input_size, num_classes)
    
    print(f"Model architecture:")
    print(f"  Input size: {input_size}")
    print(f"  Output size: {num_classes}")
    print(f"  Total layers: {len(net.layers)}")
    
    # Train model with Adam optimizer
    print("\n4. Training model...")
    print("Using Adam optimizer with learning rate decay...")
    
    adam_optimizer = Adam(learning_rate=0.001, learning_decay=0.01)
    
    train(net, X_train, y_train, 
          num_epochs=1000, 
          optimizer=adam_optimizer)
    
    # Evaluate model
    print("\n5. Evaluating model performance...")
    accuracy, precision, recall, f1 = evaluate_model_performance(net, X_test, y_test)
    
    # Save the model
    print("\n6. Saving trained model...")
    model_filename = 'insurance_model.pkl'
    net.save(model_filename)
    print(f"Model saved as: {model_filename}")
    
    # Demonstrate model loading
    print("\n7. Demonstrating model loading...")
    loaded_net = NeuralNets([])
    loaded_net = loaded_net.load(model_filename)
    
    # Verify loaded model works
    loaded_accuracy = check_accuracy(loaded_net, X_test, y_test)
    print(f"Loaded model accuracy: {loaded_accuracy:.4f}")
    print(f"Accuracy matches: {'✓' if abs(accuracy - loaded_accuracy) < 1e-6 else '✗'}")
    
    # Alternative training with SGD
    print("\n8. Training alternative model with SGD...")
    net_sgd = create_insurance_model(input_size, num_classes)
    
    sgd_optimizer = SGD(learning_rate=0.01, learning_decay=0.001)
    train(net_sgd, X_train, y_train, 
          num_epochs=500, 
          optimizer=sgd_optimizer)
    
    sgd_accuracy = check_accuracy(net_sgd, X_test, y_test)
    print(f"SGD model accuracy: {sgd_accuracy:.4f}")
    
    # Compare models
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    print(f"Adam Optimizer:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"SGD Optimizer:   {sgd_accuracy:.4f} ({sgd_accuracy*100:.2f}%)")
    print(f"Best performer:  {'Adam' if accuracy > sgd_accuracy else 'SGD'}")
    
    # Demonstrate single prediction
    print("\n9. Demonstrating single prediction...")
    sample_person = {
        'age': 35,
        'sex': 'male',
        'bmi': 28.5,
        'children': 2,
        'smoker': 'no',
        'region': 'northwest'
    }
    
    prediction = predict_single_sample(
        net, sample_person, feature_means, feature_stds,
        sex_labels, smoker_labels, region_labels
    )
    
    print(f"Sample person: {sample_person}")
    print(f"Predicted insurance category: {prediction}")
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    main()