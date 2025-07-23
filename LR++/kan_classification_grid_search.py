# https://www.kaggle.com/code/seyidcemkarakas/kan-tabular-data-binary-classification
import os
import pandas as pd
import json
import logging
import numpy as np
import torch
from torch import nn
from kan import KAN
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score, 
    precision_recall_curve,
    roc_curve,
    f1_score,
    auc
)
import itertools
from tqdm import tqdm

import argparse
from utils.helper import (
    setup_logging,
    preprocess_combined_data,
    clean_feature_names
)

import warnings
warnings.filterwarnings("ignore")

LABEL = "Loan Status"

def ks_score(true_labels, predicted_probabilities):
    """Calculate the KS score."""
    fpr, tpr, _ = roc_curve(true_labels, predicted_probabilities)
    return max(tpr - fpr)

def calculate_metrics(true_labels, predicted_probabilities, model_name, hyperparameters):
    """Calculate evaluation metrics"""
    predicted_labels = (predicted_probabilities > 0.5).astype(int)
    
    precision, recall, _ = precision_recall_curve(true_labels, predicted_probabilities)
    pr_auc = auc(recall, precision)

    return {
        'model': model_name,
        'accuracy': accuracy_score(true_labels, predicted_labels),
        'ROC_AUC': roc_auc_score(true_labels, predicted_probabilities),
        'PR_AUC': pr_auc,
        'F1_score': f1_score(true_labels, predicted_labels),
        'KS_score': ks_score(true_labels, predicted_probabilities),
        'num': len(true_labels),
        'hyperparameters': hyperparameters
    }

def generate_width_options(n_features, max_layers):
    """Generate all possible width configurations up to max_layers"""
    widths = []
    
    # Always include base case (1 layer)
    # base_width = [n_features, 2]
    # widths.append(base_width)
    
    # if max_layers >= 2:
    #     second_layer = [n_features, 2*n_features + 1, 2]
    #     widths.append(second_layer)
        
    if max_layers >= 3:
        third_layer = [
            n_features, 
            2*n_features + 1, 
            2*(2*n_features + 1) + 1, 
            2
        ]
        widths.append(third_layer)
    
    if max_layers >= 4:
        logging.warning("Grid search for number of KAN layers only supports up to 3 layers for now.")    
    return widths

def _train_single_kan(training_data, testing_data, feature_columns, 
                     width, grid_size, k_value, optimizer, steps, 
                     validation_split, logger):
    """Train and evaluate single KAN model"""
    hyperparameters = {
        'width': width,
        'grid_size': grid_size,
        'k': k_value,
        'optimizer': optimizer,
        'steps': steps,
        'validation_split': validation_split
    }
    
    # Prepare data
    X = training_data[feature_columns]
    y = training_data[LABEL]
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_split, random_state=42
    )
    
    # Convert to tensors
    train_input = torch.tensor(X_train.values, dtype=torch.float32)
    train_label = torch.tensor(y_train.values, dtype=torch.long)
    val_input = torch.tensor(X_val.values, dtype=torch.float32)
    val_label = torch.tensor(y_val.values, dtype=torch.long)
    test_input = torch.tensor(testing_data[feature_columns].values, dtype=torch.float32)
    test_label = torch.tensor(testing_data[LABEL].values, dtype=torch.long)
    
    # Initialize model
    logger.info(f"Training KAN model with width={width}, grid={grid_size}, k={k_value}")
    model = KAN(width=width, grid=grid_size, k=k_value)
    
    # Define accuracy functions
    def train_acc():
        preds = torch.argmax(model(train_input), dim=1)
        return (preds == train_label).float().mean()

    def val_acc():
        preds = torch.argmax(model(val_input), dim=1)
        return (preds == val_label).float().mean()
    
    # Train model
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
        loss_fn=nn.CrossEntropyLoss()
    )
    
    # Evaluate on test set
    with torch.no_grad():
        test_outputs = model(test_input)
        predicted_probabilities = torch.softmax(test_outputs, dim=1)[:, 1].numpy()
    
    return calculate_metrics(
        true_labels=testing_data[LABEL].values,
        predicted_probabilities=predicted_probabilities,
        model_name="KAN",
        hyperparameters=hyperparameters
    ), model

def train_and_evaluate_kan(training_data, testing_data, feature_columns, 
                          grid_size_range, k_range, max_layers, metric, 
                          optimizer, steps, validation_split, logger):
    """Perform full grid search across architecture, grid size, and k"""
    n_features = len(feature_columns)
    width_options = generate_width_options(n_features, max_layers)
    grid_sizes = list(range(grid_size_range[0], grid_size_range[1]+1))
    k_values = list(range(k_range[0], k_range[1]+1))
    
    # Generate all combinations
    param_combinations = list(itertools.product(
        width_options,
        grid_sizes,
        k_values
    ))
    
    total_combinations = len(param_combinations)
    logger.info(f"Starting grid search with {total_combinations} combinations")
    logger.info(f"Width options: {width_options}")
    logger.info(f"Grid sizes: {grid_sizes}")
    logger.info(f"k values: {k_values}")

    best_metric = -np.inf
    best_metrics = None
    best_model = None
    all_metrics = []

    # Perform grid search with progress bar
    for params in tqdm(param_combinations, desc="Grid Search Progress"):
        width, grid_size, k_value = params
        
        metrics, model = _train_single_kan(
            training_data=training_data,
            testing_data=testing_data,
            feature_columns=feature_columns,
            width=width,
            grid_size=grid_size,
            k_value=k_value,
            optimizer=optimizer,
            steps=steps,
            validation_split=validation_split,
            logger=logger
        )
        
        all_metrics.append(metrics)
        logging.info(f"Current metrics: {metrics}")
        
        # Update best model
        if metrics[metric] > best_metric:
            best_metric = metrics[metric]
            best_metrics = metrics
            best_model = model

    logger.info(f"Best hyperparameters found: {best_metrics['hyperparameters']}")
    return best_metrics, best_model, all_metrics

def main(args):
    """Main execution function"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load and preprocess data
    training_data = pd.read_parquet(args.training_data_path).dropna()
    testing_data = pd.read_parquet(args.testing_data_path).dropna()
    
    training_data, testing_data = preprocess_combined_data(
        training_data, testing_data, threshold=args.encoding_threshold
    )
    training_data, testing_data = clean_feature_names(training_data, testing_data)
    
    # Convert boolean columns
    for col in training_data.columns:
        if training_data[col].dtype == bool:
            training_data[col] = training_data[col].astype(int)
            testing_data[col] = testing_data[col].astype(int)
    
    # Handle few-shot learning
    if args.few_shot:
        logger.info(f"Using {args.few_shot}-shot learning")
        training_data = training_data.sample(n=args.few_shot, random_state=42)
        args.best_metrics_output_path = args.best_metrics_output_path.replace(
            ".json", f"_{args.few_shot}_shot.json"
        )
    
    feature_columns = [col for col in training_data.columns if col != LABEL]

    # Perform grid search
    best_metrics, _, all_metrics = train_and_evaluate_kan(
        training_data=training_data,
        testing_data=testing_data,
        feature_columns=feature_columns,
        grid_size_range=(args.grid_size_lower, args.grid_size_upper),
        k_range=(args.k_lower, args.k_upper),
        max_layers=args.kan_search_layer,
        metric=args.metric,
        optimizer=args.optimizer,
        steps=args.steps,
        validation_split=args.validation_split,
        logger=logger
    )
    
    # Save results
    with open(args.best_metrics_output_path, 'w') as f:
        json.dump([best_metrics], f, indent=4)
    logger.info(f"Best metrics saved to {args.best_metrics_output_path}")
    
    with open(args.all_metrics_output_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    logger.info(f"All metrics saved to {args.all_metrics_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KAN Model Training with Architecture Search")
    parser.add_argument('--training_data_path', type=str, required=True)
    parser.add_argument('--testing_data_path', type=str, required=True)
    parser.add_argument('--encoding_threshold', type=int, required=True)
    parser.add_argument('--best_metrics_output_path', type=str, required=True)
    parser.add_argument('--all_metrics_output_path', type=str, required=True)
    parser.add_argument('--few_shot', type=int, default=None)
    
    # Architecture search parameters
    parser.add_argument('--kan_search_layer', type=int, required=True,
                      help="Max number of layers to search (1, 2, or 3)")
    
    # Hyperparameter ranges
    parser.add_argument('--grid_size_lower', type=int, default=5)
    parser.add_argument('--grid_size_upper', type=int, default=100)
    parser.add_argument('--k_lower', type=int, default=1)
    parser.add_argument('--k_upper', type=int, default=10)
    
    # Other parameters
    parser.add_argument('--metric', type=str, default='KS_score')
    parser.add_argument('--optimizer', type=str, default='LBFGS', choices=['LBFGS', 'Adam'])
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--validation_split', type=float, default=0.2)
    
    args = parser.parse_args()
    main(args)