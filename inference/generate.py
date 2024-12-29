"""
Script for Generating Outputs Using Different Types of Generators.

This script provides a unified interface for generating outputs from three different types of generators:
- `Generator`: A standard generator for non-CoT (Chain of Thought) based outputs.
- `GeneratorCoT`: A generator designed for CoT-based outputs.
- `GeneratorCoTN`: A more advanced generator that supports CoT with a "best-of-N" strategy, 
generation from this type of generator can be further used for verification or majority voting.

Usage:
- To use the script, specify the generator type and any relevant arguments through the command line. For example:
  ```bash
  python script.py --generator_type Generator --data_path input.json --output_path output.json
"""


from sys1.generator import Generator
from sys1.generator_cot import GeneratorCoT
from sys2.generator_cot_N import GeneratorCoTN
from transformers import HfArgumentParser

import logging
from dataclasses import dataclass, asdict
from typing import Literal, Optional, Type, Dict, Any
from utils.helper import setup_logging

# Registry to manage generator types and their argument classes
GENERATOR_REGISTRY: Dict[str, Dict[str, Any]] = {}

def register_generator(generator_type: str, generator_class: Type, arg_class: Type):
    """Register a generator type with its corresponding class and argument dataclass."""
    GENERATOR_REGISTRY[generator_type] = {
        "class": generator_class,
        "args": arg_class
    }
    

@dataclass
class BaseGeneratorArguments:
    """
    Base arguments for all types of generators.
    """
    model_name_or_path: str = "Qwen2.5-7B-Instruct"
    max_new_tokens: int = 2048
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    model_max_length: int = 2048
    lora_weights: Optional[str] = None
   
@dataclass
class System1Arguments(BaseGeneratorArguments):
    """Arguments specific to Generator (non-CoT) and GeneratorCoT."""
    batch_size: int = 16
    generation_strategy: str = "greedy"

@dataclass
class System2Arguments(BaseGeneratorArguments):
    """Arguments specific to GeneratorCoTN."""
    N: int = 16
    cuda_visible_devices: Optional[Literal["0,1,2,3"]] = None
    generation_strategy: Literal["greedy", "sampling"] = "sampling"
    
@dataclass
class DataArguments:
    """Arguments for data handling."""
    data_path: str = "data.json"
    output_path: str = "output.json"


@dataclass
class GlobalArguments:
    """Arguments to select the generator type."""
    generator_type: Literal["Generator", "GeneratorCoT", "GeneratorCoTN"] = "Generator"


# Register generators with the factory
register_generator("Generator", Generator, System1Arguments)
register_generator("GeneratorCoT", GeneratorCoT, System1Arguments)
register_generator("GeneratorCoTN", GeneratorCoTN, System2Arguments)


def main():
    # Set up logging
    setup_logging()

    # Parse global arguments to determine generator type
    global_parser = HfArgumentParser(
        (GlobalArguments, DataArguments, System1Arguments, System2Arguments)
        )
    global_args, data_args, sys1_args, sys2_args= global_parser.parse_args_into_dataclasses()
    
    # Retrieve the appropriate generator class and arguments dataclass
    generator_class = GENERATOR_REGISTRY[global_args.generator_type]["class"]
    if global_args.generator_type in ["Generator", "GeneratorCoT"]:
        generator_args = sys1_args
    elif global_args.generator_type == "GeneratorCoTN":
        generator_args = sys2_args 
    else:
        raise ValueError(f"Unknown generator type: {global_args.generator_type}")
    
    logging.info(f"Selected generator type: {global_args.generator_type}")
    logging.info("Loading the dataset")
    dataset = generator_class.load(data_args.data_path)

    logging.info(f"Instantiating the {global_args.generator_type}")
    generator = generator_class(**asdict(generator_args))
    
    logging.info("Generating results with the generator")
    generation = generator(dataset)
    
    logging.info("Saving the results")
    generator.save(generation, data_args.output_path)
    logging.info(f"Results saved to {data_args.output_path}")


if __name__ == "__main__":
    main()
