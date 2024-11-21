import pandas as pd
import json
import logging
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
    auc
)
import numpy as np
import argparse
from tqdm import tqdm

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import setup_logging

import warnings
warnings.filterwarnings("ignore")

LABEL = "Loan Status"  # Global variable for the target column


def ks_score(true_labels, predicted_probabilities):
    """Calculate the KS score."""
    fpr, tpr, thresholds = precision_recall_curve(true_labels, predicted_probabilities)
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


def train_and_evaluate_model(
    training_data, 
    testing_data, 
    feature_columns, 
    model_name, 
    model, 
    hyperparameter_grid, 
    logger
):
    """
    Train and evaluate a model. Perform hyperparameter tuning if a grid is provided.
    Return the evaluation metrics and the trained model.
    """
    logger.info(f"Starting training for {model_name}")

    if hyperparameter_grid:
        # Perform grid search for models with hyperparameters
        grid_search = GridSearchCV(
            model,
            hyperparameter_grid,
            scoring='roc_auc',
            cv=3,
            n_jobs=-1
        )
        grid_search.fit(training_data[feature_columns], training_data[LABEL])
        best_model = grid_search.best_estimator_
        best_hyperparameters = grid_search.best_params_
        logger.info(f"Best hyperparameters for {model_name}: {best_hyperparameters}")
    else:
        # Train without grid search (e.g., for Naive Bayes)
        model.fit(training_data[feature_columns], training_data[LABEL])
        best_model = model
        best_hyperparameters = {}
        logger.info(f"Completed training for {model_name} without hyperparameter tuning.")

    # Predict on the test set
    logger.info(f"Making predictions with {model_name}")
    predicted_probabilities = best_model.predict_proba(testing_data[feature_columns])[:, 1]

    # Calculate metrics
    metrics = calculate_metrics(
        true_labels=testing_data[LABEL], 
        predicted_probabilities=predicted_probabilities, 
        model_name=model_name, 
        best_hyperparameters=best_hyperparameters
    )
    logger.info(f"Metrics for {model_name}: {metrics}")
    return metrics, best_model


def preprocess_combined_data(train_data, test_data, threshold=5):
    """
    Preprocess training and testing datasets by concatenating them,
    encoding categorical features, and then splitting them back.
    
    Parameters:
        train_data (pd.DataFrame): Training dataset.
        test_data (pd.DataFrame): Testing dataset.
        threshold (int): Max unique values for one-hot encoding. Otherwise, label encoding is used.
    
    Returns:
        pd.DataFrame, pd.DataFrame: Processed training and testing datasets.
    """
    # Add a temporary column to identify dataset type
    train_data = train_data.copy()
    test_data = test_data.copy()
    
    train_data['__dataset_type'] = 'train'
    test_data['__dataset_type'] = 'test'

    # Concatenate train and test datasets
    combined_data = pd.concat([train_data, test_data], axis=0)

    # Encode the target column (Loan Status) into binary values
    if LABEL in combined_data:
        combined_data[LABEL] = combined_data[LABEL].map({
            'Fully Paid': 1,
            'Charged Off': 0
        })
        if combined_data[LABEL].isnull().any():
            raise ValueError(f"Unexpected values in {LABEL}. Ensure it only contains 'Fully Paid' or 'Charged Off'.")

    # Encode categorical features
    categorical_columns = [
        column for column in combined_data.select_dtypes(include=['object']).columns.tolist()
        if column != LABEL and column != '__dataset_type'  # Exclude the LABEL and __dataset_type columns
    ]

    for column in categorical_columns:
        unique_values = combined_data[column].nunique()

        if unique_values <= threshold:
            # Apply one-hot encoding
            one_hot = pd.get_dummies(
                combined_data[column],
                prefix=column,
                drop_first=False  # Keep all categories
            )
            combined_data = pd.concat([combined_data, one_hot], axis=1)
            combined_data.drop(column, axis=1, inplace=True)
        else:
            # Apply label encoding
            label_encoder = LabelEncoder()
            combined_data[column] = label_encoder.fit_transform(combined_data[column])

    # Split back into train and test datasets
    train_data_processed = combined_data[combined_data['__dataset_type'] == 'train'].drop('__dataset_type', axis=1)
    test_data_processed = combined_data[combined_data['__dataset_type'] == 'test'].drop('__dataset_type', axis=1)

    return train_data_processed, test_data_processed


def clean_feature_names(training_data, testing_data):
    """
    Ensure all column names in the DataFrame are strings and free of invalid characters.
    """
    
    def clean(dataframe):
        dataframe.columns = [
            str(column).replace("[", "_")
                    .replace("]", "_")
                    .replace("<", "_")
                    .replace(">", "_")
            for column in dataframe.columns
        ]
        return dataframe
    return clean(training_data), clean(testing_data)


def main(args):
    """Main function to load data, train models, and save evaluation metrics."""
    # Set up logging
    setup_logging()
    logging.info("Starting script")
    training_data_path=args.training_data_path
    testing_data_path=args.testing_data_path
    metrics_output_path=args.metrics_output_path

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
    if args.few_shot:
        logging.info(f"Sampling {args.few_shot} shot from training data")
        training_data = training_data.sample(n=args.few_shot, random_state=42)
        metrics_output_path = metrics_output_path.replace(".json", f"_{args.few_shot}_shot.json")
        
    logging.info(f"Training dataset shape: {training_data.shape}")
    logging.info(f"Test dataset shape: {testing_data.shape}")
    
    # Define feature columns (excluding the target column)
    feature_columns = [col for col in training_data.columns if col != LABEL]

    # Define models and their hyperparameter grids
    models_and_hyperparameter_grids = {
        "Logistic Regression": (
            LogisticRegression(max_iter=5000),
            {
                "C": [0.01, 0.1, 0.3, 1, 10],
                "penalty": ["l1", "l2"],
                "solver": ["saga", 'liblinear'],
                "n_jobs":[-1]
            }
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=0),
            {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 20],
                "min_samples_split": [2, 5, 10]
            }
        ),
        "XGBoost": (
            xgb.XGBClassifier(
                nthread=10, 
                random_state=0, 
                use_label_encoder=False, 
                eval_metric="logloss"
            ),
            {
                "max_depth": [2, 3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "n_estimators": [50, 100, 200],
                "scale_pos_weight": [1, 5, 7, 10],  
                "min_child_weight": [350],          
                "subsample": [1.0]                  
            }
        ),
        "LightGBM": (
            lgb.LGBMClassifier(random_state=0),
            {
                "num_leaves": [31, 50, 70],
                "learning_rate": [0.05, 0.1, 0.2],
                "n_estimators": [50, 100, 200],
                "min_child_samples": [10, 20, 30]
            }
        ),
        "Naive Bayes": (
            GaussianNB(),
            None  # No hyperparameters for Naive Bayes
        )
    }

    # Train and evaluate each model
    all_metrics = []
    for model_name, (model, hyperparameter_grid) in tqdm(
        models_and_hyperparameter_grids.items(), 
        desc="Training Models"
    ):
        try:
            metrics, _ = train_and_evaluate_model(
                training_data=training_data, 
                testing_data=testing_data, 
                feature_columns=feature_columns, 
                model_name=model_name, 
                model=model, 
                hyperparameter_grid=hyperparameter_grid, 
                logger=logging
            )
            all_metrics.append(metrics)
        except Exception as error:
            logging.error(f"Error training {model_name}: {error}", exc_info=True)
    
    # Save metrics to the output path
    logging.info(f"Saving metrics to {metrics_output_path}")
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    with open(metrics_output_path, 'w') as output_file:
        json.dump(all_metrics, output_file, indent=4)

    logging.info(f"Metrics successfully saved to {metrics_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train multiple models and calculate metrics."
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
        will be encoded using lable encoder, otherwise one-hot"
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
    args = parser.parse_args()

    main(args)
