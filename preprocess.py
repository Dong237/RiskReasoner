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
import json

#####config
from sklearn.model_selection import train_test_split

data_folder = Path("datasets")
llm_data_folder = Path("llms")
experts_data_folder = Path("experts")

name = "accepted_2007_to_2018Q4.csv"
feature_size = 21 + 1  # Target_index = -1
train_size, dev_size, test_size = 0.7, 0.1, 0.2

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
                "gold": gold, 'text': text}
            )
    return data_tmp


def json_save(data, dataname, mean_list=mean_list, out_jsonl=False):
    data_tmp = process_table(data, mean_list)
    if out_jsonl:
        with open('{}.jsonl'.format(data_folder/dataname), 'w') as f:
            for i in data_tmp:
                json.dump(i, f)
                f.write('\n')
            print('-----------')
            print(f"{dataname}.jsonl write done")
        f.close()
    df = pd.DataFrame(data_tmp)
    # 保存为 Parquet 文件
    parquet_file_path = data_folder / llm_data_folder / f'{dataname}.parquet'
    df.to_parquet(parquet_file_path, index=False)
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
    df_test = data.iloc[test_ind]
    if not os.path.exists(data_folder/experts_data_folder):
        os.makedirs(data_folder/experts_data_folder)
    df_train.to_parquet(data_folder/experts_data_folder/"train_expert.parquet")
    df_dev.to_parquet(data_folder/experts_data_folder/"dev_expert.parquet")
    df_test.to_parquet(data_folder/experts_data_folder/"test_expert.parquet")
    print("Save expert system data done")


#####process
if __name__ == '__main__':
    data = get_data(data_folder/name).values.tolist()
    check_num = get_num(data)
    random.seed(10086)

    train_ind = random.sample([i for i in range(len(data))], int(len(data) * train_size))
    train_data = [data[i] for i in train_ind]

    index_left = list(set(list(range(len(data)))) - set(train_ind))
    dev_ind = random.sample(index_left, int(len(data) * dev_size))
    dev_data = [data[i] for i in dev_ind]

    test_ind = list(set(index_left) - set(dev_ind))
    test_data = [data[i] for i in test_ind]
    
    save_expert_system_data(data, train_ind, dev_ind, test_ind)

    test_prompt_data = json_save(test_data, 'test')
    train_prompt_data = json_save(train_data, 'train')
    dev_prompt_data = json_save(dev_data, 'valid')
    print("Save llms data done")
    
