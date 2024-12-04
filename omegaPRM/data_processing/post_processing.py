import json
from pathlib import Path
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.helper import jdump
from utils.constants import STEP_TAG

# Constants
INCORRECT_MC_BOUNDARY = 0.125  # 2/16
CORRECT_MC_BOUNDARY = 1.0
DATA_FOLDER = Path("datasets/omegaPRM/")


# Function to load data from multiple .jsonl files
def load_data_from_folders(folders):
    data_all = []
    for folder in folders:
        file_path = folder / "risk_reasoner.jsonl"
        with open(file_path, 'r') as f:
            # Read each line and parse it as a JSON object
            data = [json.loads(line.strip()) for line in f]
            data_all.extend(data)
    return data_all


# Function to process data and generate the selected data
def process_data(data_all):
    data_selected = []
    for data in data_all:
        for reasoning in data['reasoning_steps_text']:
            # Determine if the reasoning is incorrect or correct
            incorrect_flag = reasoning["mc_value"] <= INCORRECT_MC_BOUNDARY
            correct_flag = reasoning["mc_value"] >= CORRECT_MC_BOUNDARY
            
            if incorrect_flag or correct_flag:
                # Prepare the question and reasoning steps
                question = data["question"] + STEP_TAG
                reasoning_steps = reasoning["solution_prefix"].replace(question, "")
                
                # Create labels based on reasoning steps
                labels = [1] * len(reasoning_steps.split(STEP_TAG))
                labels[-1] = 0 if incorrect_flag else 1
                
                # Create selected data entry
                selected = {
                    "question": question,
                    "reasoning_steps": (
                        reasoning_steps+STEP_TAG 
                        if not reasoning_steps.endswith(STEP_TAG)  # we need this tag to create label while training the PRM
                        else reasoning_steps
                        ),
                    "label": labels,
                    "mc_value": reasoning["mc_value"]
                }
                data_selected.append(selected)
    
    return data_selected

# Main script
def main():
    # Define folder paths
    folders = [DATA_FOLDER/f"omegaPRM_part{i}" for i in range(1, 5)]
    
    # Load data from .jsonl files
    data_all = load_data_from_folders(folders)
    
    # Process data to select relevant items
    data_selected = process_data(data_all)
    
    # Save the final selected data to a JSON file
    jdump(data_selected, DATA_FOLDER/"risk_reasoner_v1.json")

if __name__ == "__main__":
    main()
