import os
import torch
import threading
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.constants import Prompts
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List
# Import vllm if using vLLM backend
try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("vLLM not installed. Install it if you wish to use it as a model backend.")

# Set the environment variable for the endpoint
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"

SYSTEM_PROMPT = Prompts.SYSTEM_PROMPT_CREDIT_SCORING.value

class LLMService:
    """
    A class to manage a large language model (LLM) service using Hugging Face's transformers library.
    """

    def __init__(self, model_name: str = "Qwen2.5-Math-7B-Instruct",
                 device: str = "cuda", max_new_tokens: int = 2048,
                 temperature: float = 0.7, top_k: int = 30, top_p: float = 0.9, model_type: str="hf"):
        """
        Initialize the LLMService with model parameters and sampling settings.

        Parameters:
        - model_name (str): Path or Hugging Face hub model name.
        - device (str): Device for computation, e.g., 'cuda' or 'cpu'.
        - max_new_tokens (int): Maximum number of new tokens to generate.
        - temperature (float): Sampling temperature for response generation.
        - top_k (int): Top-K sampling parameter for response diversity.
        - top_p (float): Top-P sampling parameter for response diversity.
        """
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.model_type = model_type.lower()
        self.pipe = None
        self.llm = None
        self.model = None
        self.tokenizer = None
        self.load_lock = threading.Lock()


    def start_service(self):
        """
        Start the LLM service by loading the model into the chosen pipeline if it's not already loaded.
        Ensures thread-safe loading using a lock.
        """
        with self.load_lock:
            # tokneizer is used for both model types
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.model_type == "hf":
                if self.model is None or self.tokenizer is None:
                    print(f"Loading Hugging Face model '{self.model_name}' on device '{self.device}'...")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype="auto",
                        device_map="cuda"
                        )
                    print(f"Hugging Face model loaded successfully on GPU {self.model.device}.")
            elif self.model_type == "vllm":
                if self.llm is None:
                    print(f"Loading vLLM model '{self.model_name}' on device '{self.device}'...")
                    self.llm = LLM(self.model_name, tensor_parallel_size=1)
                    print("vLLM model loaded successfully.")
            else:
                raise ValueError("Unsupported model_type. Choose 'hf' for Hugging Face or 'vllm' for vLLM.")


    def generate_response(self, prompt: str, num_copies: int = 2) -> List[str]:
        """
        Generate responses from the model based on the provided prompt, duplicated to form a batch.

        Parameters:
        - prompt (str): The input prompt to generate responses for.
        - num_copies (int): The number of copies of the prompt to create for batch processing (default is 16).

        Returns:
        - List[str]: A list of generated responses, each corresponding to a duplicate of the input prompt.
        """
        if self.model_type == "hf":
            return self._generate_response_hf(prompt, num_copies)
        elif self.model_type == "vllm":
            return self._generate_response_vllm(prompt, num_copies)
        else:
            raise ValueError("Unsupported model_type. Choose 'hf' for Hugging Face or 'vllm'.")


    def _generate_response_hf(self, prompt: str, num_copies: int = 2) -> List[str]:
        """
        Generate responses from the model based on the provided prompt, duplicated to form a batch.

        Parameters:
        - prompt (str): The input prompt to generate responses for.
        - num_copies (int): The number of copies of the prompt to create for batch processing (default is 16).

        Returns:
        - List[str]: A list of generated responses, each corresponding to a duplicate of the input prompt.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("LLM service not started. Please call start_service() first.")

        # Create a batch of structured messages for each prompt
        messages = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},  # Common system message
                {"role": "user", "content": prompt}  # User's input prompt
            ]
            for _ in range(num_copies)
        ]

        # Apply the chat template (assumes the tokenizer has an apply_chat_template method)
        text_batch = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,  # We handle tokenization ourselves
            add_generation_prompt=True  # Add generation-specific prompt (like stop tokens, etc.)
        )

        # Tokenize the batch of inputs
        model_inputs = self.tokenizer(
            text_batch, return_tensors="pt", padding=True, truncation=True
            ).to(self.model.device)

        # Generate responses using the model (batch inference)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                do_sample=True,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p
            )

        # Process the generated output and decode the results
        generated_ids = [
            output_ids[len(input_ids):]  # Slice the generation to remove the input part
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        # Decode all generated outputs in batch
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


    def _generate_response_vllm(self, prompt: str, num_copies: int) -> List[str]:
        """
        Generate responses using vLLM.
        """
        if self.llm is None:
            raise ValueError("LLM service not started for vLLM model. Please call start_service() first.")

        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens
        )
        
        messages = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},  # Common system message
                {"role": "user", "content": prompt}  # User's input prompt
            ]
            for _ in range(num_copies)
        ]

        # Apply the chat template (assumes the tokenizer has an apply_chat_template method)
        prompts = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False, 
            add_generation_prompt=True  # Add generation-specific prompt (like stop tokens, etc.)
        )

        # prompts = [prompt] * num_copies
        responses = self.llm.generate(prompts, sampling_params=sampling_params)

        return [response.outputs[0].text for response in responses]


if __name__ == "__main__":
    # Initialize the service for vLLM
    llm_service = LLMService(model_type="hf")
    llm_service.start_service()

    prompt = "What is game theory?"
    responses = llm_service.generate_response(prompt, num_copies=3)

    print(responses)
