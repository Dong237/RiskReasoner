"""
The main script for finetuning an LLM with PPO.

Readers of interest can refer to this blog post: https://www.notion.so/swtheking/eb7b2d1891f44b3a84e7396d19d39e6f?v=01bcb084210149488d730064cbabc99f
for common tricks during PPO finetuning.
"""

#!/usr/bin/env python
import sys
import os
import logging
import numpy as np
from pathlib import Path
import torch
from utils.helper import setup_logging
sys.path.append("../../")
from training.rl.ppo.config import get_config
from training.rl.ppo.envs.env import RiskEnv
from training.rl.ppo.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
from training.rl.ppo.runner import RiskRunner
from training.rl.ppo.models.prm import ProcessRM


def parse_args(args, parser):
    parser.add_argument(
        '--dataset_name', 
        type=str, 
        default='prealgebra', 
        help="Which dataset to test on."
    )
    parser.add_argument(
        '--dataset_path', 
        type=str, 
        help="Path to the dataset file."
    )
    parser.add_argument(
        '--n_agents', 
        type=int, 
        default=1
    )
    parser.add_argument(
        '--model_name_or_path', 
        type=str, 
        help="Name of the agent model or path to the agent model checkpoint."
    )
    parser.add_argument(
        '--prm_model_name_or_path', 
        type=str, 
        default='', 
        help="Name of the model or path to the process reward model."
    )
    parser.add_argument(
        '--prm_lora_weights', 
        type=str, 
        default='', 
        help="Path to the process reward model lora checkpoint."
    )
    parser.add_argument(
        '--max_new_tokens', 
        type=int, 
        default=512, 
        help="max_new_tokens controls the length of the generated action."
    )
    parser.add_argument(
        '--model_max_length', 
        type=int, 
        default=2048, 
        help="model_max_length for the tokenizer, should be larger than the whole reasoning process."
    )
    parser.add_argument(
        '--vacab_size', 
        type=int, 
        default=151936
    )
    parser.add_argument(
        '--gradient_cp_steps', 
        type=int, 
        default=2
    )

    all_args = parser.parse_known_args(args)[0]
    return all_args
    
def make_vec_env(
    dataset_name, 
    dataset_path, 
    n_rollout_threads, 
    mode, 
    max_step, 
    ):
    def get_env_fn(rank):
        def init_env():
            env = RiskEnv(
                rank=rank, 
                dataset_name=dataset_name, 
                dataset_path=dataset_path, 
                mode=mode,
                max_step=max_step,
                )
            return env
        return init_env
    return ShareSubprocVecEnv(
        [get_env_fn(i) for i in range(n_rollout_threads)]
        )

def build_run_dir(all_args):
    # Construct the base run directory path
    root_folder = "logs/ppo/"
    run_dir = Path(root_folder + "results") / all_args.dataset_name / all_args.algorithm_name
    
    # Check if the run directory exists
    if not run_dir.exists():
        # Create the directory if it does not exist
        os.makedirs(str(run_dir))
        # Initialize the first run as 'run1'
        curr_run = 'run1'
    else:
        # If the directory exists, check for existing runs
        exst_run_nums = [
            int(str(folder.name).split('run')[1])  # Extract run numbers from folder names
            for folder in run_dir.iterdir()        # Iterate through subdirectories
            if str(folder.name).startswith('run') # Filter folders starting with 'run'
        ]
        # If no runs exist, initialize as 'run1'
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            # Otherwise, increment the highest run number for the new run
            curr_run = 'run%i' % (max(exst_run_nums) + 1)

    # Append the current run folder to the run directory path
    run_dir = run_dir / curr_run
    
    # Create the current run directory if it does not already exist
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    
    # Return the complete path to the current run directory
    return run_dir
    
def main(args):
    setup_logging()
    parser = get_config()
    all_args = parse_args(args, parser)
    run_dir = build_run_dir(all_args)

    envs = make_vec_env(
        all_args.dataset_name, 
        all_args.dataset_path, 
        all_args.n_rollout_threads, 
        "train", 
        all_args.episode_length,  # episodes will be forced to stop at this length
        )
    
    eval_envs = make_vec_env(
        all_args.dataset_name, 
        all_args.dataset_path, 
        all_args.n_eval_rollout_threads, 
        "test", 
        all_args.episode_length,
        )

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": envs.n_agents,
        "run_dir": run_dir
    }

    runner = RiskRunner(config)
    logging.info("Starting training")
    runner.run()
    logging.info("Finished training")
    
    logging.info("Writting TensorBoard logs")
    runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
    runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
