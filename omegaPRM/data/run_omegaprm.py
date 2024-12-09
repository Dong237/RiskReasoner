import argparse
import json
import logging
import os
from tqdm import tqdm
from typing import Dict
import torch
torch.cuda.empty_cache()
from omegaprm import LanguageModel, OmegaPRM

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.constants import Prompts
from utils.helper import setup_logging, jload, jdump

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import torch
torch.cuda.empty_cache()
print("="*50)
print(torch.cuda.current_device())  # Prints the ID of the GPU being used
print("Device Name:", torch.cuda.get_device_name(torch.cuda.current_device())) 
print("="*50)

DS_NAME = "risk_reasoner"
QUESTION_KEY = "query_cot"
ANSWER_KEY = "answer"
INSTRUCTION = Prompts.INSTRUCTION_STEP_BY_STEP.value
INITIAL_BATCH_ANSWERS_SIZE = 16


# Filter a single question based on 32 rollouts
def should_process_question(question: Dict[str, str], llm: LanguageModel) -> bool:
    """
    Determine whether a question should be processed based on initial rollouts.

    Checks if the question produces both correct and incorrect answers in a batch
    of rollouts, indicating it is suitable for further processing.

    Parameters:
        question (Dict[str, str]): A dictionary containing the problem and final answer.
        llm (LanguageModel): An instance of the language model for generating rollouts.

    Returns:
        bool: True if the question should be processed, False otherwise.
    """

    prompt = INSTRUCTION + question[QUESTION_KEY] # NOTE query_cot must be combined with CoT specific instruction
    correct_answer = question[ANSWER_KEY]

    has_correct = False
    has_incorrect = False
    
    initial_batch_answers = llm.generate_rollout(prompt, INITIAL_BATCH_ANSWERS_SIZE)

    for answer in initial_batch_answers:
        if llm.evaluate_correctness(answer, correct_answer):
            has_correct = True
        else:
            has_incorrect = True

        if has_correct and has_incorrect:
            return True

    return False


# Run OmegaPRM on a question if it passes the filter
def process_question(omega_prm: OmegaPRM, question: Dict[str, str]):
    """
    Run OmegaPRM on a single question to collect reasoning steps.

    Parameters:
        omega_prm (OmegaPRM): An instance of the OmegaPRM algorithm.
        question (Dict[str, str]): A dictionary containing the problem and final answer.

    Returns:
        Dict: A dictionary containing the problem, final answer, and collected reasoning steps.
    """
    cot_question_with_instruction = INSTRUCTION + question[QUESTION_KEY]
    data_tree, data_text = omega_prm.run(
        question=cot_question_with_instruction,
        answer=question[ANSWER_KEY]
        )
    collected_data = {
        "question": cot_question_with_instruction,
        "final_answer": question[ANSWER_KEY],
        "reasoning_steps_tree": data_tree,
        "reasoning_steps_text": data_text
    }
    return collected_data


# Save collected data for each question
def save_question_data(collected_data: Dict, index: int, output_path: str):
    """
    Save the collected data for a processed question to a JSONL file.

    Parameters:
        collected_data (Dict): Data collected for the question, including reasoning steps.
        index (int): The index of the question being processed.
        output_path (str): Path to the output JSONL file.
    """

    collected_data["question_id"] = index
    with open(output_path, "a") as fd:
        line = json.dumps(collected_data)
        fd.write(f"{line}\n")
    logging.info(f"Question {index} is saved to {output_path}")


def main(args):
    """
    Main execution function to process questions with OmegaPRM.

    - Loads questions from a file.
    - Filters questions using initial rollouts.
    - Processes each filtered question using the OmegaPRM algorithm.
    - Saves collected data to the output file.

    Parameters:
        args (argparse.Namespace): Command-line arguments parsed by argparse.
    """

    setup_logging()
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{DS_NAME}.jsonl")
    # ensure output_file is empty since we are appending to it later
    with open(output_file, "w") as fd:
        fd.write("")

    logging.info("Starting OmegaPRM processing")
    logging.info(f"Using model: {args.model_name} on device: {args.device}")
    logging.info(f"Question file: {args.question_file}")

    questions = jload(args.question_file)

    llm = LanguageModel(
        model_name=args.model_name,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        model_type=args.model_type,
    )

    omega_prm = OmegaPRM(
        LM=llm,
        c_puct=args.c_puct,
        alpha=args.alpha,
        beta=args.beta,
        L=args.length_scale,
        k=args.num_rollouts,
        N=args.max_search_count,
        rollout_budget=args.rollout_budget,
        save_data_type=args.save_data_type,
    )

    processed_count = 0  # Counter for processed questions
    passed_question_ids = []  # List to store question IDs that passed the filter
    for idx, question in tqdm(enumerate(questions), desc="Targeting questions..."):
        logging.info(f"Testing whether question num {idx} should be processed...")
        if should_process_question(question, llm):
            logging.info(f"Question passed filter, processing question num {idx}...")
            passed_question_ids.append(idx)
            collected_data = process_question(omega_prm, question)
            save_question_data(collected_data, idx, output_file)
            processed_count += 1
        else:
             logging.info(f"Question did not pass filter, skipping question num: {idx}")
        
    # Log summary
    logging.info(
        f"Total questions processed by OmegaPRM: {processed_count}/{len(questions)}"
    )
    jdump(passed_question_ids, os.path.join(args.output_dir,"passed_question_ids.json"))
    logging.info("Finished processing questions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OmegaPRM on filtered questions")

    parser.add_argument(
        "--question_file",
        type=str,
        required=True,
        help="Path to the questions JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help=f"Directory to save the output file {DS_NAME}.jsonl",
    )
    parser.add_argument(
        "--log_file_prefix",
        type=str,
        default="omega_prm",
        help="Prefix for the log files",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="/root/.cache/modelscope/hub/Qwen/Qwen2___5-Math-7B-Instruct",
        help="Model name or path for the language model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the model on (e.g., 'cuda', 'cpu')",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=2048, help="Max tokens for LLM generation"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for LLM generation",
    )
    parser.add_argument(
        "--top_k", type=int, default=50, help="Top-K sampling for LLM generation"
    )
    parser.add_argument(
        "--top_p", type=float, default=1.0, help="Top-P sampling for LLM generation"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="hf",
        help="Model backend to use ('hf' for Hugging Face or 'vllm')",
    )

    # OmegaPRM parameters with provided defaults
    parser.add_argument(
        "--c_puct", type=float, default=0.125, help="Exploration constant for OmegaPRM"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="Weight for MC(s) in OmegaPRM"
    )
    parser.add_argument(
        "--beta", type=float, default=0.9, help="Length penalty for OmegaPRM"
    )
    parser.add_argument(
        "--length_scale", type=int, default=500, help="length scale in OmegaPRM"
    )
    parser.add_argument(
        "--num_rollouts",
        type=int,
        default=16,
        help="Number of rollouts for Monte Carlo estimation in OmegaPRM",
    )
    parser.add_argument(
        "--max_search_count", type=int, default=20, help="Max search count in OmegaPRM"
    )
    parser.add_argument(
        "--rollout_budget", type=int, default=200, help="Rollout budget for OmegaPRM"
    )
    parser.add_argument(
        "--save_data_type",
        type=str,
        choices=["tree", "text", "both"],
        default="both",
        help="Data tpye to save, either in tree structure for OmegaPRM or pure json, or both",
    )

    args = parser.parse_args()
    main(args)
