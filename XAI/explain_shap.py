#!/usr/bin/env python
import os
import json
import argparse
import logging
import numpy as np
import pandas as pd

from utils.helper import preprocess_combined_data

# Machine Learning libraries
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# SHAP for interpretability
import shap

# Sklearn utilities
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Function to select the best model based on the given metric
# ---------------------------------------------------------------------------
def select_best_model(expert_systems_results, metric):
    """
    Given a list of dictionaries (expert_systems_results) where each dictionary
    has at least 'model' and <metric> as keys, select the entry with the highest value of <metric>.
    
    Args:
        expert_systems_results (list[dict]): Each dictionary contains model metrics and hyperparameters.
        metric (str): The name of the metric to select the best model by (e.g., 'ROC_AUC').

    Returns:
        dict: The dictionary (model metrics) corresponding to the best model.
    """
    best_model = None
    best_metric_value = -np.inf
    for result in expert_systems_results:
        if result[metric] > best_metric_value:
            best_metric_value = result[metric]
            best_model = result
    return best_model


# ---------------------------------------------------------------------------
# Function to load data
# ---------------------------------------------------------------------------
def load_data(training_path, testing_path):
    """
    Load training and testing data from parquet files.

    Args:
        training_path (str): Path to the training data (parquet).
        testing_path (str): Path to the testing data (parquet).

    Returns:
        training_data (pd.DataFrame), testing_data (pd.DataFrame)
    """
    logging.info("Loading training data from: %s", training_path)
    training_data = pd.read_parquet(training_path).dropna()

    logging.info("Loading testing data from: %s", testing_path)
    testing_data = pd.read_parquet(testing_path).dropna()

    return training_data, testing_data


# ---------------------------------------------------------------------------
# Function to define the model dictionary
# ---------------------------------------------------------------------------
def get_models():
    """
    Define a dictionary for the possible models we used in the main script.
    
    Returns:
        dict: {model_name: (model_class, default_params)}
    """
    model_dict = {
        "Logistic Regression": (LogisticRegression, {"max_iter": 5000}),
        "Random Forest": (RandomForestClassifier, {"random_state": 0}),
        "XGBoost": (xgb.XGBClassifier, {
            "nthread": 10,
            "random_state": 0,
            "use_label_encoder": False,
            "eval_metric": "logloss"
        }),
        "LightGBM": (lgb.LGBMClassifier, {"random_state": 0}),
        "Naive Bayes": (GaussianNB, {})
    }
    return model_dict


# ---------------------------------------------------------------------------
# Function to train (or retrain) the best model on the entire training data
# ---------------------------------------------------------------------------
def retrain_best_model(best_model_info, training_data, label_col="Loan Status"):
    """
    Retrain the best model with its best hyperparameters on the entire training data.
    
    Args:
        best_model_info (dict): Dictionary containing 'model', 'best_hyperparameters', etc.
        training_data (pd.DataFrame): Full training data.
        label_col (str): Name of the target column.

    Returns:
        trained_model: The model trained on the entire training dataset.
        list[str]: The feature columns used for training.
    """
    # Extract the best model name and hyperparameters
    best_model_name = best_model_info["model"]
    best_hparams = best_model_info.get("best_hyperparameters") or {}

    # Get the model class and default params from our dictionary
    model_dict = get_models()
    model_class, default_params = model_dict[best_model_name]

    # Merge default params with best hyperparameters found
    final_params = {**default_params, **best_hparams}

    # Create model instance with those parameters
    trained_model = model_class(**final_params)

    # Define feature columns
    feature_columns = [col for col in training_data.columns if col != label_col]

    # Fit the model on the entire training set
    X_train = training_data[feature_columns]
    y_train = training_data[label_col]
    trained_model.fit(X_train, y_train)

    logging.info(f"Retrained {best_model_name} with best hyperparameters: {best_hparams}")
    return trained_model, feature_columns


# ---------------------------------------------------------------------------
# Function to demonstrate SHAP interpretability
# ---------------------------------------------------------------------------
def demonstrate_shap(
    model,
    training_data,
    testing_data,
    feature_columns,
    label_col="Loan Status",
    num_samples_to_explain=5
):
    """
    Demonstrate SHAP by explaining a random subset of the test data.

    Args:
        model: A trained (scikit-learn compatible) model that supports .predict()/.predict_proba().
        training_data (pd.DataFrame): The full training dataset (for reference, if needed).
        testing_data (pd.DataFrame): The test dataset.
        feature_columns (list[str]): The feature names used in training.
        label_col (str): The name of the target column.
        num_samples_to_explain (int): How many test samples to use for SHAP explanations.
    """
    import random

    # Extract feature matrix and labels from test set
    X_test = testing_data[feature_columns]
    y_test = testing_data[label_col]

    # We'll select a random subset from the test set
    num_samples_to_explain = min(num_samples_to_explain, len(X_test))
    subset_indices = random.sample(range(len(X_test)), num_samples_to_explain)
    X_subset = X_test.iloc[subset_indices]

    # Initialize SHAP
    # - If your model is tree-based (RandomForest, XGBoost, LightGBM), you can often use TreeExplainer.
    # - Otherwise, use the generic Explainer or KernelExplainer.
    model_name = type(model).__name__.lower()
    if "xgb" in model_name or "lgbm" in model_name or "forest" in model_name:
        explainer = shap.TreeExplainer(model)
        logging.info("Using TreeExplainer for a tree-based model.")
    else:
        explainer = shap.Explainer(model, X_test)  # for linear or generic models
        logging.info("Using generic Explainer for a non-tree model.")

    # Compute SHAP values for the subset
    shap_values = explainer(X_subset)

    # For classification tasks, SHAP may produce multiple columns in shap_values.values
    # We'll show a summary plot for the 'most likely' or 'first' class, or you can specify an index.
    # * shap_values
