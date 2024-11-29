import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helper import jload

import json
import pandas as pd
from sklearn.utils import shuffle


# Global variables for file paths
INFERENCE_RESULTS = 'your_file.json'
TRAINING_DATA = 'train.parquet'
SAVE_PATH = 'train_sampled.parquet'


def jload(file_path):
    """Load JSON data from the given file path."""
    with open(file_path, 'r') as f:
        return json.load(f)


def compare_labels_and_store_ids(json_data):
    """Compare pred_label and label, store their ids in respective lists."""
    ids_incorrect = []
    ids_correct = []

    for entry in json_data:
        if entry['pred_label'] == entry['label']:
            ids_correct.append(entry['id'])
        else:
            ids_incorrect.append(entry['id'])
    
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
    final_df.to_parquet(SAVE_PATH, index=False)
    
    # Output the final dataframe
    return final_df


if __name__ == '__main__':
    main()

