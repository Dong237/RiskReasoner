from generator_cot_N import Generator
from transformers import HfArgumentParser

import logging
from dataclasses import dataclass, field, asdict
from typing import Literal, Optional, Union, List
from utils.helper import (
    jdump,
    jload,
    setup_logging, 
)


@dataclass
class GeneratorArguments:
    """
    Arguments for generating responses from the LLM.
    """
    model_name_or_path: str = "Qwen2.5-7B-Instruct"
    max_new_tokens: int = 2048
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    model_max_length: int = 2048
    N: int = field(
        default=16,
        metadata={
            "help": "The N from 'best-of-N', indicating the amout of alternative responses"
        }
    )
    lora_weights: Optional[str] = None
    cuda_visible_devices: Optional[Literal["0,1,2,3"]] = None


@dataclass
class DataArguments:
    data_path: str = "data.json"
    output_path: str = "output.json"


def main():
    # Set up logging
    setup_logging()

    # Parse arguments using HfArgumentParser
    parser = HfArgumentParser(
        (GeneratorArguments, DataArguments)
    )
    generator_args, data_args = parser.parse_args_into_dataclasses()
    
    logging.info("Loading the dataset")
    dataset = jload(data_args.data_path)

    logging.info("Instantiating the generator")
    generator = Generator(**asdict(generator_args))
    
    logging.info("Generating results with the generator")
    data_tb_verified = generator(dataset)
    
    logging.info("Saving the results")
    jdump(data_tb_verified, data_args.output_path)
    logging.info(f"Results saved to {data_args.output_path}")


if __name__ == "__main__":
    main()