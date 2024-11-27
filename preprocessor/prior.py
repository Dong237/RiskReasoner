# Script copied from "CALM" by The-FinAI
# URL: https://github.com/The-FinAI/CALM/blob/main/data/credit_scoring/Lending%20Club/prepocess.py
# Date: 2024-11-19
# License: MIT License


r"""
This document is about credit scoring task using a dataset named "Lending Club".
Description_based: clear semantic information about the features

Original data description：
2007 through current Lending Club accepted and rejected loan data.
\url{https://www.kaggle.com/datasets/wordsforthewise/lending-club}

Basic data information：

It is a tabular data. 
After being processed in Lending Club.docx, The data have 21 features, including 
Loan Information: Installment, Loan Purpose, Loan Application Type, Interest Rate, 
Last Payment Amount, Loan Amount, Revolving Balance; History Information: Delinquency In 2 years, 
Inquiries In 6 Months, Mortgage Accounts, Grade, Open Accounts, Revolving Utilization Rate, 
Total Accounts, Fico Range Low, Fico Range High; Soft Information: Address State, Employment Length, 
Home Ownership, Verification Status, Annual Income.

The number of the data is 53,812/1,345,310.  We split 0.7 for train, 0.1  for dev, and 
0.2  for test

When using, please add the "prompt" at the beginning: 
Assess the client's loan status based on the following loan records from Lending Club. 
Respond with only 'good' or 'bad', and do not provide any additional information. 
For instance, 'The client has a stable income, no previous debts, and owns a property.' 
should be classified as 'good'. 
Text: 
"""

import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import jdump


##### Config
data_folder = Path("datasets")
prior_folder = Path("prior")
llm_data_folder = prior_folder / "llms"
experts_data_folder = prior_folder / "experts"

# Ensure directories exist
llm_data_folder.mkdir(parents=True, exist_ok=True)
experts_data_folder.mkdir(parents=True, exist_ok=True)

name = "accepted_2007_to_2018Q4.csv"
feature_size = 21 + 1  # Target_index = -1
train_size, dev_size, test_size = 0.7, 0.1, 0.2

LABEL = "Loan Status"

if train_size + dev_size + test_size != 1:
    print("sample size wrong!!!")

mean_list = ['Installment', 'Loan Purpose', 'Loan Application Type', 'Interest Rate', 'Last Payment Amount',
             'Loan Amount', 'Revolving Balance',
             'Delinquency In 2 years', 'Inquiries In 6 Months', 'Mortgage Accounts', 'Grade', 'Open Accounts',
             'Revolving Utilization Rate', 'Total Accounts', 'Fico Range Low', 'Fico Range High',
             'Address State', 'Employment Length', 'Home Ownership', 'Verification Status', 'Annual Income',
             'Loan Status']


#####function
def process_table(data, mean_list):
    data_tmp = []
    prompt = 'Assess the client\'s loan status based on the following loan records from Lending Club. ' \
             'Respond with only \'good\' or \'bad\', and do not provide any additional information. For instance, ' \
             '\'The client has a stable income, no previous debts, and owns a property.\' ' \
             'should be classified as \'good\'. \nText: '

    for j in range(len(data)):
        text = 'The client has attributes as follows: '
        for i in range(len(data[0]) - 1):
            sp = '. ' if i != len(data[0]) - 2 else '.'
            if i == 3 or i == 12:
                text = text + f'The {mean_list[i]} is {str(data[j][i])}%' + sp  # state of
            else:
                text = text + f'The {mean_list[i]} is {str(data[j][i])}' + sp  # state of
        answer = 'good' if data[j][-1] == 'Fully Paid' else 'bad'
        gold = 0 if data[j][-1] == 'Fully Paid' else 1
        # 'Fully Paid' is good and 'Charged off' is bad
        data_tmp.append(
            {
                'id': j, 
                "query": f"{prompt}'{text}'" + ' \nAnswer:', 
                'answer': answer, 
                "choices": ["good", "bad"],
                "gold": gold, 'text': text
                }
            )        
    return data_tmp


def save_data(data, dataname, mean_list=mean_list):
    data_tmp = process_table(data, mean_list)
    ## save as parquet files used in baseline establishment
    df = pd.DataFrame(data_tmp)
    parquet_file_path = data_folder / llm_data_folder / f'{dataname}.parquet'
    df.to_parquet(parquet_file_path, index=False)
    ## Save as json files used in baseline establishment
    json_file_path = data_folder / llm_data_folder / f'{dataname}.json'
    df.to_json(json_file_path, orient='records', indent=4)
    return data_tmp


def get_num(data):
    data_con = np.array(data)
    check = np.unique(data_con[:, -1])
    check1 = (data_con[:, -1] == check[0]).sum()
    check2 = (data_con[:, -1] == check[1]).sum()
    return check2, check1


def get_data(name):
    selected_feature = ['installment', 'purpose', 'application_type', 'int_rate', 'last_pymnt_amnt', 'loan_amnt',
                        'revol_bal', 'delinq_2yrs', 'inq_last_6mths', 'mort_acc', 'grade', 'open_acc', 'revol_util',
                        'total_acc', 'fico_range_low', 'fico_range_high', 'addr_state', 'emp_length', 'home_ownership',
                        'verification_status', 'annual_inc', 'loan_status']
    data = pd.read_csv(name, sep=',', header=0, low_memory=False, usecols=selected_feature).reindex(
        columns=selected_feature)
    # only reserve Fully Paid and Charged Off
    for loan_st in data['loan_status'].unique().tolist():
        if loan_st != 'Fully Paid' and loan_st != 'Charged Off':
            data = data.drop(data[data['loan_status'] == loan_st].index)
    data.dropna(subset=['loan_status'], inplace=True)
    save_data, drop_data = train_test_split(data, test_size=0.99, stratify=data['loan_status'], random_state=100)
    return save_data


def save_expert_system_data(data, train_ind, dev_ind, test_ind):
    data = pd.DataFrame(data, columns=mean_list)
    df_train = data.iloc[train_ind]
    df_dev = data.iloc[dev_ind]
    df_test = data.iloc[test_ind].dropna()  # Drop NaN for test data

    if not os.path.exists(data_folder/experts_data_folder):
        os.makedirs(data_folder/experts_data_folder)

    df_train.to_parquet(data_folder/experts_data_folder/"train_expert.parquet")
    df_dev.to_parquet(data_folder/experts_data_folder/"dev_expert.parquet")
    df_test.to_parquet(data_folder/experts_data_folder/"test_expert.parquet")

    # Balanced sampling for test data
    LABEL = 'Loan Status'  # Replace with the column name for the labels
    df_test_balanced = balance_test_set(df_test, LABEL)
    df_test_balanced.to_parquet(data_folder/experts_data_folder/"test_expert_balanced.parquet")
    print("Save expert system data done")
    return df_test_balanced  # Return the balanced test data for further use


def balance_test_set(df_test, label_column):
    """
    Balance the test set to have 1000 rows with an equal number of each class.
    """
    # Count unique values in the label column
    counts = df_test[label_column].value_counts()
    print("Test data label counts before balancing:", counts)

    # Separate rows for each class
    charged_off_rows = df_test[df_test[label_column] == "Charged Off"]
    fully_paid_rows = df_test[df_test[label_column] == "Fully Paid"]

    k = len(charged_off_rows)  # Number of rows with "Charged Off"
    if k > 1000:
        raise ValueError("Too many 'Charged Off' rows to create a balanced dataset.")

    # Sample from the "Fully Paid" class to make up the rest
    fully_paid_sample = fully_paid_rows.sample(1000 - k, random_state=10086)

    # Combine the two sets
    balanced_df = pd.concat([charged_off_rows, fully_paid_sample])
    balanced_df = balanced_df.sample(frac=1, random_state=10086).reset_index(drop=True)  # Shuffle rows
    print("Test data label counts after balancing:", balanced_df[label_column].value_counts())
    return balanced_df


def drop_nan(data, columns=mean_list):
    """
    Drop NaN values from data and return cleaned data as a list of lists.
    """
    df = pd.DataFrame(data, columns=columns)
    df_clean = df.dropna()
    return df_clean.values.tolist()


#####process
if __name__ == '__main__':
    # Get data
    data = get_data(data_folder/name).values.tolist()
    check_num = get_num(data)
    random.seed(10086)

    # Split data into train, dev, and test
    train_ind = random.sample([i for i in range(len(data))], int(len(data) * train_size))
    train_data = [data[i] for i in train_ind]

    index_left = list(set(list(range(len(data)))) - set(train_ind))
    dev_ind = random.sample(index_left, int(len(data) * dev_size))
    dev_data = [data[i] for i in dev_ind]

    test_ind = list(set(index_left) - set(dev_ind))
    test_data = [data[i] for i in test_ind]

    # Drop NaN values from test data
    test_data_clean = drop_nan(test_data, mean_list)

    # Save expert system data and get balanced test data
    test_data_balanced = save_expert_system_data(data, train_ind, dev_ind, test_ind)

    # Use balanced test data for JSON saving
    test_prompt_data = save_data(test_data_clean, 'test')
    train_prompt_data = save_data(train_data, 'train')
    dev_prompt_data = save_data(dev_data, 'valid')

    # Save balanced test data for prompt-based system
    test_prompt_data_balanced = save_data(test_data_balanced.values.tolist(), 'test_balanced')
    print("Save llms data done")
