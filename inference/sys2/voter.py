import os
import random
import logging
from collections import Counter
from utils.helper import jload, jdump, compute_binary_metrics_from_results,setup_logging

def process_element(element):
    """
    Given one element from the JSON file, perform majority voting on the pred_labels
    in 'response_N' and then randomly select one result among those with the majority label.
    """
    responses = element.get("response_N", [])
    if not responses:
        return None

    # Extract the predicted labels from each response
    labels = [resp.get("pred_label") for resp in responses]
    
    # Count the occurrences of each label
    label_counts = Counter(labels)
    # Find the majority label (if a tie, Counter.most_common returns one of them)
    majority_label, _ = label_counts.most_common(1)[0]
    
    # Filter responses that have the majority label
    majority_responses = [resp for resp in responses if resp.get("pred_label") == majority_label]
    
    # Randomly choose one response among those with the majority label
    selected_response = random.choice(majority_responses)
    
    # align keys for metrics computation
    # selected_response["label"] = selected_response.pop("gold_label")
    return selected_response

def process_json(input_file):
    """
    Process the entire JSON file.
    For each element, select one result from 'response_N' using majority voting.
    Returns a list of selected responses.
    """

    data = jload(input_file)
    
    selected_results = []
    for element in data:
        selected = process_element(element)
        if selected is not None:
            selected_results.append(selected)
    return selected_results

def main():
    # Specify the input JSON file (change the filename as needed)
    input_filename = "datasets/generator/test_balanced_posterior_generator_cot_N_llama_r1_4096_2000.json"
    # Specify the output JSON file to store the selected responses
    output_filename = "datasets/verified/test_balanced_posterior_generator_cot_N_llama_r1_4096_2000_voted_response.json"
    
    setup_logging()
    
    if os.path.exists(output_filename):
        selected_results = jload(output_filename)
        logging.info(f"Loaded {len(selected_results)} selected responses from {output_filename}")
        
    else:
        # Process the JSON file
        selected_results = process_json(input_filename)
        # Save the list of selected responses to the output file
        jdump(selected_results, output_filename)
        logging.info(f"Processed {len(selected_results)} elements. Results saved to '{output_filename}'.")

    metrics = compute_binary_metrics_from_results(selected_results)
    logging.info(f"Evaluation on the voted results: {metrics}")
        
if __name__ == "__main__":
    main()
