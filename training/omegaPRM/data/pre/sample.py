"""
This script samples a balanced subset (half correct, half incorrect) of 
train.parquet (9k+ data) based on the predicted label from the LLM, results
in about 1.7K+ data in the end. The data further servere for OmegaPRM data 
generation
"""


import pandas as pd
from utils.helper import jload
from sklearn.utils import shuffle


# Global variables for file paths
INFERENCE_RESULTS = 'results/posterior/inference/Qwen2.5-7B-Instruct_train_posterior_cot.json'
TRAINING_DATA = 'datasets/posterior/train_posterior.parquet'
SAVE_PATH_PARQUET = 'datasets/posterior/train_posterior_sampled.parquet'
SAVE_PATH_JSON = 'datasets/posterior/train_posterior_sampled.json'


def compare_labels_and_store_ids(json_data):
    """Compare pred_label and label, store their ids in respective lists."""
    ids_incorrect = []
    ids_correct = []

    for entry in json_data:
        if entry['pred_label'] == "miss":
            # count miss as correct instrad of incorrect
            ids_correct.append(entry['id'])
            continue
        pred_label = int(entry['pred_label'])
        label = int(entry['label'])
        row_id = int(entry['id'])
        if pred_label == label:
            ids_correct.append(row_id)
        else:
            ids_incorrect.append(row_id)
    return ids_incorrect, ids_correct

def sample_train_data(train_correct, train_incorrect):
    """Sample an equal number of rows from train_correct as train_incorrect."""
    return train_correct.sample(n=len(train_incorrect), random_state=42)


def process_and_shuffle_data(train_correct_sampled, train_incorrect):
    """Concatenate and shuffle the train data."""
    final_df = pd.concat([train_correct_sampled, train_incorrect])
    final_df = shuffle(final_df).reset_index(drop=True)
    return final_df


def main():
    # Step 1: Load the JSON data
    json_data = jload(INFERENCE_RESULTS)
    
    # Step 2: Compare labels and store ids in respective lists
    ids_incorrect, ids_correct = compare_labels_and_store_ids(json_data)
    
    # Step 3: Load the Parquet data
    train_df = pd.read_parquet(TRAINING_DATA)
    
    # Step 4: Filter rows based on ids
    train_incorrect = train_df[train_df['id'].isin(ids_incorrect)]
    train_correct = train_df[train_df['id'].isin(ids_correct)]
    
    # Step 5: Sample data
    train_correct_sampled = sample_train_data(train_correct, train_incorrect)
    
    # Step 6: Process and shuffle the data
    final_df = process_and_shuffle_data(train_correct_sampled, train_incorrect)
    
    # Step 7: Save the final dataframe to a new parquet file
    final_df.to_parquet(SAVE_PATH_PARQUET, index=False)
    final_df.to_json(SAVE_PATH_JSON, orient='records', indent=4)
    print(f"Data saved to {SAVE_PATH_PARQUET} and {SAVE_PATH_JSON}")


if __name__ == '__main__':
    main()

