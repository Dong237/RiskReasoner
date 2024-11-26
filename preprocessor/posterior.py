"""
This script processes data to append the predicted probabilities from the best expert system to input prompts, 
enhancing them for downstream usage by Large Language Models (LLMs). The script selects the best expert system 
based on a specified evaluation metric, trains the model, computes predictions, and integrates these probabilities 
into the LLM input queries.

Naming Convention:
- "prior.py": Refers to the initial script that preprocesses the data for expert systems and LLMs, where the inputs for LLMs
  are solely the selected features from the datasets.
- "posterior.py": Refers to this script, which incorporates posterior probabilities (predicted probabilities) into LLM prompts,
  aiming to enhance their inputs for better performance (or at least test this).

Functionality:
1. Load training and testing datasets prepared by the "prior" script.
2. Select the best expert system based on evaluation metrics (e.g., ROC_AUC, F1_score).
3. Train the selected expert system model on the training data.
4. Compute predicted probabilities for both training and test datasets.
5. Append the predicted probabilities to the input prompts for LLMs.
6. Save the resulting datasets with the appended probabilities as:
   - `train_posterior.json` (for training data).
   - `test_posterior.json` (for testing data).

Input Arguments:
- `--expert_systems_result` (str, required): Path to the JSON file containing expert system performance evaluation results.
- `--metric` (str, required): The metric used for selecting the best model (e.g., "accuracy", "ROC_AUC", "PR_AUC", "F1_score", "KS_score").
- `--data_path` (str, required): Path to the directory containing the expert system and LLMs data.
- `--encoding_threshold` (int, optional, default=20): Threshold for using one-hot encoding. Columns with more unique values than this threshold will use label encoding.

Outputs:
- Enhanced datasets saved in JSON format with posterior probabilities included in LLM prompts:
  - `train_posterior.json` (for training data).
  - `test_posterior.json` (for testing data).
"""


import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import lightgbm as lgb

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    jload,
    jdump,
    preprocess_combined_data,
    clean_feature_names,
    setup_logging
)
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    accuracy_score,
    f1_score,
    roc_curve
)

# Define folders
prior_folder = Path("prior")
llm_data_folder = prior_folder / "llms"
experts_data_folder = prior_folder / "experts"

posterior_llm_folder = Path("posterior") # posterior data is only for llms, no need for subfolders

LABEL = "Loan Status"

# Function to select the best model based on the given metric
def select_best_model(expert_systems_results, metric):
    best_model = None
    best_metric_value = -np.inf
    for result in expert_systems_results:
        if result[metric] > best_metric_value:
            best_metric_value = result[metric]
            best_model = result
    return best_model


# Function to train the model
def train_model(model_name, hyperparameters, X_train, y_train):
    if model_name == "Logistic Regression":
        model = LogisticRegression(**hyperparameters)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(**hyperparameters)
    elif model_name == "XGBosot":
        model = xgb.XGBClassifier(**hyperparameters)
    elif model_name == "LightGBM":
        model = lgb.LGBMClassifier(**hyperparameters)
    elif model_name == "Naive Bayes":
        model = GaussianNB(**hyperparameters)
    else:
        raise ValueError(f"Model {model_name} not supported.")
    
    model.fit(X_train, y_train)
    return model


# Function to calculate metrics
def compute_binary_metrics(labels, pos_probs, pred_labels):
    # Compute metrics using probabilities
    roc_auc = roc_auc_score(labels, pos_probs)
    precision, recall, _ = precision_recall_curve(labels, pos_probs)
    pr_auc = auc(recall, precision)
    fpr, tpr, _ = roc_curve(labels, pos_probs)
    ks_score = max(abs(tpr - fpr))
    accuracy = accuracy_score(labels, pred_labels)
    f1 = f1_score(labels, pred_labels)

    metrics = {
        'accuracy': accuracy,
        'F1_score': f1,
        'ROC_AUC': roc_auc,
        'PR_AUC': pr_auc,
        'KS_score': ks_score,
        'num_samples': len(labels)
    }

    return metrics


# Function to create the modified queries for LLMs
def create_llm_queries_with_predictions(data, model, model_name, X_train, y_train, X_test, y_test):
    # Get predictions (probabilities) from both train and test datasets
    train_prob_bad = model.predict_proba(X_train)[:, 0]  # Probability of being bad (class 0)
    test_prob_bad = model.predict_proba(X_test)[:, 0] 
    train_prob_good = model.predict_proba(X_train)[:, 1]
    test_prob_good = model.predict_proba(X_test)[:, 1]
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Log metrics
    train_metrics = compute_binary_metrics(y_train, train_prob_good, train_pred)
    test_metrics = compute_binary_metrics(y_test, test_prob_good, test_pred)

    logging.info(f"Training Metrics: {train_metrics}")
    logging.info(f"Testing Metrics: {test_metrics}")

    # Prepare the queries for the training dataset
    train_queries = []
    for i in range(len(X_train)):
        query = {
            "id": i,
            "query": f"Assess the client's loan status based on the following loan records from Lending Club. "
                     f"Respond with only 'good' or 'bad', and do not provide any additional information. "
                     f"For instance, 'The client has a stable income, no previous debts, and owns a property.' "
                     f"should be classified as 'good'. \nText: " + create_text(X_train.iloc[i]) +
                     f"As reference, the predicted probability of this client's loan status being good given by the machine learning model "
                     f"{model_name} is {train_prob_good[i] * 100:.2f}%, and the probability of it being bad is {train_prob_bad[i] * 100:.2f}%.",
            "answer": "good" if y_train.iloc[i] == 1 else "bad",
            "choices": ["good", "bad"],
            "gold": y_train.iloc[i],
            "text": create_text(X_train.iloc[i])
        }
        train_queries.append(query)

    # Prepare the queries for the testing dataset
    test_queries = []
    for i in range(len(X_test)):
        query = {
            "id": i,
            "query": f"Assess the client's loan status based on the following loan records from Lending Club. "
                     f"Respond with only 'good' or 'bad', and do not provide any additional information. "
                     f"For instance, 'The client has a stable income, no previous debts, and owns a property.' "
                     f"should be classified as 'good'. \nText: " + create_text(X_test.iloc[i]) +
                     f"As reference, the predicted probability of this client's loan status being good given by the machine learning model "
                     f"{model_name} is {test_prob_good[i] * 100:.2f}%, and the probability of it being bad is {test_prob_bad[i] * 100:.2f}%.",
            "answer": "good" if y_test.iloc[i] == 1 else "bad",
            "choices": ["good", "bad"],
            "gold": y_test.iloc[i],
            "text": create_text(X_test.iloc[i])
        }
        test_queries.append(query)

    return train_queries, test_queries


# Function to generate text description for each row
def create_text(row):
    features = [
        "Installment", "Loan Purpose", "Loan Application Type", "Interest Rate", "Last Payment Amount", 
        "Loan Amount", "Revolving Balance", "Delinquency In 2 years", "Inquiries In 6 Months", 
        "Mortgage Accounts", "Grade", "Open Accounts", "Revolving Utilization Rate", 
        "Total Accounts", "Fico Range Low", "Fico Range High", "Address State", "Employment Length", 
        "Home Ownership", "Verification Status", "Annual Income"
    ]
    text = ""
    for feature, value in zip(features, row):
        if feature in ["Interest Rate", "Revolving Utilization Rate"]:
            text += f"The {feature} is {value:.2f}%. "
        else:
            text += f"The {feature} is {value}. "
    return text


# Main function
def main():
    parser = argparse.ArgumentParser(description="Train model and generate queries for LLMs.")
    parser.add_argument('--expert_systems_result', type=str, help="Path to the JSON file with expert system performance evaluation.")
    parser.add_argument('--metric', type=str, choices=["accuracy", "ROC_AUC", "PR_AUC", "F1_score", "KS_score"], 
                        help="The metric to use for selecting the best model.")
    parser.add_argument('--data_path', type=str, help="Path to the directory containing the expert system and llms data.")
    parser.add_argument('--encoding_threshold', type=int, default=20, 
                        help="Threshold for using one-hot encoding. Columns with more unique values use label encoding.")

    args = parser.parse_args()

    # Set up logging
    setup_logging()

    logging.info("Loading expert system results")
    expert_systems_results = jload(args.expert_systems_result)
    best_model_result = select_best_model(expert_systems_results, args.metric)

    model_name = best_model_result["model"]
    hyperparameters = best_model_result["best_hyperparameters"]
    logging.info(f"Selected model: {model_name} with {args.metric} = {best_model_result[args.metric]}")

    # Load training and testing datasets
    logging.info("Loading datasets")
    data_path = Path(args.data_path)
    # Ensure posterior directory exists
    (data_path / experts_data_folder).mkdir(parents=True, exist_ok=True)
    training_data = pd.read_parquet(data_path / experts_data_folder / "train_expert.parquet")
    testing_data = pd.read_parquet(data_path / experts_data_folder / "test_expert.parquet")

    # Preprocess and clean data
    logging.info("Preprocessing datasets")
    training_data, testing_data = preprocess_combined_data(training_data, testing_data, threshold=args.encoding_threshold)
    training_data, testing_data = clean_feature_names(training_data, testing_data)

    # Split the target variable from features
    y_train = training_data[LABEL]
    X_train = training_data.drop(columns=[LABEL])

    y_test = testing_data[LABEL]
    X_test = testing_data.drop(columns=[LABEL])

    # Train the best model
    logging.info("Training the best model")
    model = train_model(model_name, hyperparameters, X_train, y_train)

    # Generate queries for LLMs and log metrics
    logging.info("Generating queries and calculating metrics")
    train_queries, test_queries = create_llm_queries_with_predictions(X_train, model, model_name, X_train, y_train, X_test, y_test)

    # Save the queries
    logging.info("Saving generated queries")
    train_output_path = Path(args.data_path) / posterior_llm_folder  / "train_posterior.json"
    test_output_path = Path(args.data_path) / posterior_llm_folder  / "test_posterior.json"
    os.makedirs(train_output_path.parent, exist_ok=True)

    jdump(train_queries, train_output_path, indent=4)
    jdump(test_queries, test_output_path, indent=4)

    logging.info(f"Training completed. Queries saved to {train_output_path} and {test_output_path}")


if __name__ == "__main__":
    main()
