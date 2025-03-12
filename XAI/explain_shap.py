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
    draw_dependence=False
):
    """
    Demonstrate SHAP by explaining the entire test dataset.
    
    Args:
        model: A trained (scikit-learn compatible) model that supports .predict()/.predict_proba().
        training_data (pd.DataFrame): The full training dataset (for reference, if needed).
        testing_data (pd.DataFrame): The test dataset.
        feature_columns (list[str]): The feature names used in training.
        label_col (str): The name of the target column.
        draw_dependence (bool): Whether to draw the dependence plot for the most impactful feature.
    """
    # Extract feature matrix and labels from test set
    X_test = testing_data[feature_columns]
    y_test = testing_data[label_col]

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

    # Compute SHAP values for the entire test set
    shap_values = explainer(X_test)

    # 1) Summary Plot
    logging.info("Generating SHAP summary plot for the test dataset.")
    shap.summary_plot(shap_values, X_test)

    # 2) If draw_dependence is True, plot the dependence plot for the most impactful feature
    if draw_dependence:
        # Get the feature with the largest mean absolute SHAP value
        mean_abs_shap_values = np.abs(shap_values.values).mean(axis=0)
        most_impactful_feature_idx = np.argmax(mean_abs_shap_values)
        most_impactful_feature = feature_columns[most_impactful_feature_idx]
        
        logging.info(f"Most impactful feature: {most_impactful_feature}")
        shap.dependence_plot(most_impactful_feature, shap_values.values, X_test)


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------
def main():
    """
    1. Parse arguments
    2. Load data
    3. Load JSON of model results
    4. Select best model
    5. Retrain best model
    6. Demonstrate SHAP interpretability for the test dataset
    """
    parser = argparse.ArgumentParser(description="Demonstrate SHAP on the best model.")
    parser.add_argument("--training_data_path", type=str, required=True,
                        help="Path to the training data (parquet)")
    parser.add_argument("--testing_data_path", type=str, required=True,
                        help="Path to the testing data (parquet)")
    parser.add_argument("--metrics_json_path", type=str, required=True,
                        help="Path to the JSON file containing metrics and hyperparams")
    parser.add_argument("--output_metric", type=str, default="ROC_AUC",
                        help="Metric to select the best model (e.g., ROC_AUC, accuracy, etc.)")
    parser.add_argument("--draw_dependence", type=bool, default=False,
                        help="Whether to draw dependence plot for the most impactful feature")

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    logging.info("Starting SHAP interpretability demonstration script")

    # 1. Load Data
    training_data, testing_data = load_data(args.training_data_path, args.testing_data_path)
    # Preprocess or encode the data the same way as your main training process
    training_data, testing_data, mappings = preprocess_combined_data(
        training_data,
        testing_data,
        threshold=0,
        return_mapping=True
    )

    # 2. Load JSON of model results
    logging.info("Loading model results from JSON: %s", args.metrics_json_path)
    with open(args.metrics_json_path, 'r') as f:
        expert_systems_results = json.load(f)

    # 3. Select the best model based on the chosen metric
    best_model_info = select_best_model(expert_systems_results, args.output_metric)
    logging.info(
        "Best model found: %s with %s=%.4f",
        best_model_info["model"],
        args.output_metric,
        best_model_info[args.output_metric]
    )

    # 4. Retrain the best model on the entire training data
    best_model, feature_columns = retrain_best_model(
        best_model_info,
        training_data,
        label_col="Loan Status"
    )

    # 5. Demonstrate SHAP interpretability on the test dataset
    demonstrate_shap(
        model=best_model,
        training_data=training_data,
        testing_data=testing_data,
        feature_columns=feature_columns,
        label_col="Loan Status",
        draw_dependence=args.draw_dependence
    )

    logging.info("SHAP interpretability demonstration completed.")


if __name__ == "__main__":
    main()
