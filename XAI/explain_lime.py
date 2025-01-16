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

# LIME for interpretability
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

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
# (Optional) Additional data preprocessing if needed
# (Here we'll assume your data is already clean and properly preprocessed)
# ---------------------------------------------------------------------------

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
    best_hparams = best_model_info["best_hyperparameters"] if best_model_info["best_hyperparameters"] else {}

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
# Function to demonstrate LIME interpretability on a single test sample
# ---------------------------------------------------------------------------
def demonstrate_lime(
    model, 
    training_data, 
    testing_data, 
    feature_columns, 
    label_col="Loan Status", 
    sample_index=0
):
    """
    Demonstrate LIME interpretability by explaining the prediction of a single test sample.
    
    Args:
        model: A trained (scikit-learn compatible) model that supports .predict_proba().
        training_data (pd.DataFrame): The full training dataset (for LIME's reference).
        testing_data (pd.DataFrame): The test dataset.
        feature_columns (list[str]): The feature names used in training.
        label_col (str): The name of the target column.
        sample_index (int): Index of the test sample to interpret.
    """
    # Extract the test instance
    X_test = testing_data[feature_columns]
    y_test = testing_data[label_col]

    # For classification tasks, define class names (optional)
    # If your data is binary (Loan Status in {0,1}), define them as needed:
    class_names = ['Fully Paid', 'Charged Off']  

    # Create a LimeTabularExplainer using the training data
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(training_data[feature_columns]),
        feature_names=feature_columns,
        class_names=class_names,
        mode='classification'
    )

    # Pick one sample to interpret
    sample = X_test.iloc[sample_index].values

    # Generate explanation
    exp = explainer.explain_instance(
        sample,
        model.predict_proba,  # The prediction function
        num_features=21  # Number of features to show in the explanation
    )

    # Print out the explanation in the console
    print("Explanation for test sample at index", sample_index)
    for feature_explanation in exp.as_list():
        print(feature_explanation)

    # Visualize the explanation as a figure
    fig = exp.as_pyplot_figure()
    plt.title(f"LIME Explanation for Sample {sample_index}")
    plt.tight_layout()
    plt.show()

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
    6. Demonstrate LIME interpretability on a single test sample
    """
    parser = argparse.ArgumentParser(description="Demonstrate LIME on the best model.")
    parser.add_argument("--training_data_path", type=str, required=True,
                        help="Path to the training data (parquet)")
    parser.add_argument("--testing_data_path", type=str, required=True,
                        help="Path to the testing data (parquet)")
    parser.add_argument("--metrics_json_path", type=str, required=True,
                        help="Path to the JSON file containing metrics and hyperparams")
    parser.add_argument("--output_metric", type=str, default="ROC_AUC",
                        help="Metric to select the best model (e.g., ROC_AUC, accuracy, etc.)")
    parser.add_argument("--sample_index", type=int, default=0,
                        help="Index of test sample to explain with LIME")
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    logging.info("Starting interpretability demonstration script")

    # 1. Load Data and encode the categorical variables
    training_data, testing_data = load_data(args.training_data_path, args.testing_data_path)
    training_data, testing_data, mappings = preprocess_combined_data(training_data, testing_data, threshold=0, return_mapping=True)

    # 2. Load JSON of model results
    logging.info("Loading model results from JSON: %s", args.metrics_json_path)
    with open(args.metrics_json_path, 'r') as f:
        expert_systems_results = json.load(f)

    # 3. Select the best model based on the chosen metric
    best_model_info = select_best_model(expert_systems_results, args.output_metric)
    logging.info(f"Best model found: {best_model_info['model']} with {args.output_metric}={best_model_info[args.output_metric]}")

    # 4. Retrain the best model on the entire training data
    best_model, feature_columns = retrain_best_model(best_model_info, training_data, label_col="Loan Status")

    # 5. Demonstrate LIME interpretability on a single test sample
    demonstrate_lime(
        model=best_model,
        training_data=training_data,
        testing_data=testing_data,
        feature_columns=feature_columns,
        label_col="Loan Status",
        sample_index=args.sample_index
    )

    logging.info("Interpretability demonstration completed.")


if __name__ == "__main__":
    main()
