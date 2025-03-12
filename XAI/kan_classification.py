# https://www.kaggle.com/code/seyidcemkarakas/kan-tabular-data-binary-classification
import os
import pandas as pd
import json
import logging
import numpy as np
import torch
from torch import nn
from kan import KAN  # Import KAN correctly
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    auc
)

import argparse
from tqdm import tqdm
from utils.helper import (
    setup_logging,
    preprocess_combined_data,
    clean_feature_names
)

import warnings
warnings.filterwarnings("ignore")

LABEL = "Loan Status"  # Global variable for the target column


def ks_score(true_labels, predicted_probabilities):
    """Calculate the KS score."""
    fpr, tpr, _ = roc_curve(true_labels, predicted_probabilities)
    return max(tpr - fpr)


def calculate_metrics(true_labels, predicted_probabilities, model_name, best_hyperparameters):
    """
    Calculate and return evaluation metrics for a model.
    Metrics include accuracy, ROC AUC, PR AUC, F1 score, KS score, and more.
    """
    predicted_labels = [1 if probability > 0.5 else 0 for probability in predicted_probabilities]
    
    precision, recall, _ = precision_recall_curve(true_labels, predicted_probabilities)
    pr_auc = auc(recall, precision)

    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(true_labels, predicted_labels),
        'ROC_AUC': roc_auc_score(true_labels, predicted_probabilities),
        'PR_AUC': pr_auc,
        'F1_score': f1_score(true_labels, predicted_labels),
        'KS_score': ks_score(true_labels, predicted_probabilities),
        'num': len(true_labels),
        'best_hyperparameters': best_hyperparameters
    }
    return metrics


def train_and_evaluate_kan(
    training_data, 
    testing_data, 
    feature_columns, 
    hyperparameters, 
    logger
):
    """
    Train and evaluate the KAN model using the provided hyperparameters.
    """
    logger.info("Starting training for KAN model")
    
    # Extract hyperparameters
    grid_size = hyperparameters.get('grid_size', 10)
    k_value = hyperparameters.get('k', 3)
    optimizer = hyperparameters.get('optimizer', 'LBFGS')
    steps = hyperparameters.get('steps', 100)
    validation_split = hyperparameters.get('validation_split', 0.2)
    
    # Prepare feature inputs and target
    X = training_data[feature_columns]
    y = training_data[LABEL]
    
    # Split training data to create a validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_split, random_state=42
    )
    
    # Convert data to PyTorch tensors
    train_input = torch.tensor(X_train.values, dtype=torch.float32)
    train_label = torch.tensor(y_train.values, dtype=torch.long)
    val_input = torch.tensor(X_val.values, dtype=torch.float32)
    val_label = torch.tensor(y_val.values, dtype=torch.long)
    test_input = torch.tensor(testing_data[feature_columns].values, dtype=torch.float32)
    test_label = torch.tensor(testing_data[LABEL].values, dtype=torch.long)
    
    # Define KAN model
    model = KAN(width=[len(feature_columns), 2*(len(feature_columns))+1, 2], grid=grid_size, k=k_value)
    model.auto_save
    
    # Define accuracy metric functions for training
    def train_acc():
        preds = torch.argmax(model(train_input), dim=1)
        return torch.mean((preds == train_label).float())

    def val_acc():
        preds = torch.argmax(model(val_input), dim=1)
        return torch.mean((preds == val_label).float())
    
    # Train the model
    logger.info(f"Training KAN model with grid={grid_size}, k={k_value}, optimizer={optimizer}, steps={steps}")
    dataset = {
        'train_input': train_input, 
        'train_label': train_label, 
        'test_input': val_input, 
        'test_label': val_label
        }
    
    results = model.fit(
        dataset,
        metrics=(train_acc, val_acc),
        opt=optimizer,
        steps=steps,
        loss_fn=torch.nn.CrossEntropyLoss()
        # lamb=0.01, 
        # lamb_entropy=10.
    )
    
    # Log training results
    logger.info(f"Training completed. Final train accuracy: {sum(results['train_acc'])/len(results['train_acc'])}, "
                f"validation accuracy: {sum(results['val_acc'])/len(results['val_acc'])}")
    
    # Make predictions on the test set
    logger.info("Making predictions with KAN model on test set")
    with torch.no_grad():
        test_outputs = model(test_input)
        # Get probabilities for the positive class (class 1)
        predicted_probabilities = torch.softmax(test_outputs, dim=1)[:, 1].numpy()
    
    # Calculate metrics
    metrics = calculate_metrics(
        true_labels=testing_data[LABEL].values, 
        predicted_probabilities=predicted_probabilities, 
        model_name="KAN", 
        best_hyperparameters=hyperparameters
    )
    logger.info(f"Metrics for KAN model: {metrics}")
    return metrics, model


def main(args):
    """Main function to load data, train KAN model, and save evaluation metrics."""
    # Set up logging
    setup_logging()
    logging.info("Starting script")
    training_data_path = args.training_data_path
    testing_data_path = args.testing_data_path
    metrics_output_path = args.metrics_output_path

    # Load data
    logging.info("Loading training and test datasets")
    training_data = pd.read_parquet(training_data_path).dropna()
    testing_data = pd.read_parquet(testing_data_path).dropna()
    
    # Preprocess data together to ensure they have the same columns after encoding
    training_data, testing_data = preprocess_combined_data(
        training_data, 
        testing_data, 
        threshold=args.encoding_threshold
    )
    training_data, testing_data = clean_feature_names(training_data, testing_data)
    
    # Convert any boolean columns to integers (0 or 1)
    for column in training_data.columns:
        if training_data[column].dtype == bool:
            training_data[column] = training_data[column].astype(int)
            testing_data[column] = testing_data[column].astype(int)
            
    # Check for any remaining non-numeric columns and convert them
    for column in training_data.columns:
        if not pd.api.types.is_numeric_dtype(training_data[column]):
            logging.warning(f"Column {column} has non-numeric type {training_data[column].dtype}. Attempting to convert to numeric.")
            training_data[column] = pd.to_numeric(training_data[column], errors='coerce')
            testing_data[column] = pd.to_numeric(testing_data[column], errors='coerce')
    
    
    if args.few_shot:
        logging.info(f"Sampling {args.few_shot} shot from training data")
        training_data = training_data.sample(n=args.few_shot, random_state=42)
        metrics_output_path = metrics_output_path.replace(".json", f"_{args.few_shot}_shot.json")
        
    logging.info(f"Training dataset shape: {training_data.shape}")
    logging.info(f"Test dataset shape: {testing_data.shape}")
    
    # Define feature columns (excluding the target column)
    feature_columns = [col for col in training_data.columns if col != LABEL]

    # Define KAN hyperparameters
    kan_hyperparameters = {
        'grid_size': args.grid_size,
        'k': args.k_value,
        'optimizer': args.optimizer,
        'steps': args.steps,
        'validation_split': args.validation_split
    }
    
    # Train and evaluate KAN model
    try:
        metrics, _ = train_and_evaluate_kan(
            training_data=training_data, 
            testing_data=testing_data, 
            feature_columns=feature_columns, 
            hyperparameters=kan_hyperparameters, 
            logger=logging
        )
        
        # Save metrics to the output path
        logging.info(f"Saving metrics to {metrics_output_path}")
        os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
        with open(metrics_output_path, 'w') as output_file:
            json.dump([metrics], output_file, indent=4)

        logging.info(f"Metrics successfully saved to {metrics_output_path}")
    
    except Exception as error:
        logging.error(f"Error training KAN model: {error}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train KAN model and calculate metrics for loan status prediction."
    )
    parser.add_argument(
        '--training_data_path', 
        type=str, 
        required=True, 
        help="Path to training dataset in Parquet format"
    )
    parser.add_argument(
        '--testing_data_path', 
        type=str, 
        required=True, 
        help="Path to testing dataset in Parquet format"
    )
    parser.add_argument(
        '--encoding_threshold', 
        type=int, 
        required=True, 
        help="Threshold for using one-hot encoding, categorical \
        columns that have unique values higher than this threshold \
        will be encoded using label encoder, otherwise one-hot"
    )
    parser.add_argument(
        '--metrics_output_path', 
        type=str, 
        required=True, 
        help="Path to save the evaluation metrics in JSON format"
    )
    parser.add_argument(
        '--few_shot', 
        type=int, 
        required=False, 
        default=None,
        help="n-shot to use for training if e.g., 8 shot, experts are trained with 8 training samples"
    )
    # KAN-specific hyperparameters
    parser.add_argument(
        '--grid_size', 
        type=int, 
        default=10, 
        help="Grid size for KAN model"
    )
    parser.add_argument(
        '--k_value', 
        type=int, 
        default=3, 
        help="Maximum degree of basic functions (k parameter)"
    )
    parser.add_argument(
        '--optimizer', 
        type=str, 
        default='LBFGS',
        choices=['LBFGS', 'Adam'],
        help="Optimizer to use for training"
    )
    parser.add_argument(
        '--steps', 
        type=int, 
        default=100, 
        help="Number of training steps"
    )
    parser.add_argument(
        '--validation_split', 
        type=float, 
        default=0.2, 
        help="Fraction of training data to use for validation"
    )
    
    args = parser.parse_args()

    main(args)