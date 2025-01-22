import os
import torch
import logging
import numpy as np
from peft import PeftModel
from safetensors.torch import save_file
from torch.distributions.categorical import Categorical
from transformers import AutoTokenizer, AutoModelForCausalLM
from training.rl.ppo.models.critic import APPOCritic, TPPOCritic
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from utils.constants import POSSIBLE_END_TOKENS


class QwenLoRAgent:
    """
    QwenLoRAgent encapsulates a language model fine-tuned with Low-Rank Adaptation (LoRA),
    acting as both an actor (policy network) and a critic (value network) for Reinforcement
    Learning algorithms such as APPO, TPPO, and GRPO.

    It provides methods for action generation, value estimation, token slicing, and
    integration with various RL training loops. The class also supports saving and loading
    LoRA adapters for parameter-efficient fine-tuning.
    """
    def __init__(self, model_name, max_new_tokens, model_max_length, algo, load_path=None):
        """
        Initializes the QwenLoRAgent with a base model and LoRA configuration.

        Args:
            model_name (str): Path or name of the pre-trained language model to load.
            max_new_tokens (int): Maximum number of new tokens to generate for actions.
            algo (str): The RL algorithm in use (e.g., "APPO", "TPPO", "GRPO").
            load_path (str, optional): If provided, loads the LoRA adapters from this path.
                                       Defaults to None.
        """
        self.device = "cuda"
        self.algo = algo
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            use_fast=True, 
            padding_side='left', 
            trust_remote_code=True,
            model_max_length=model_max_length
            )
        # TODO: figure out why use this pad_token_id
        self.tokenizer.pad_token_id = 151655 # "<|image_pad|>"
        self.possible_end_token_ids = [self.tokenizer.encode(token)[0] for token in POSSIBLE_END_TOKENS]
        
        logging.info(f"Loading base model for actor and critic...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
            )
        
        self.max_new_tokens = max_new_tokens
        
        if load_path is None:
            self.actor = self._init_actor().to(self.device)
            if self.algo != "GRPO":
                self.critic = self._init_critic().to(self.device)
        else:
            self.load(load_path)
        
    
    def _init_actor(self, lora_weights = None):
        """
        Initializes the actor model (policy network) with LoRA adapters.

        Args:
            lora_weights (str, optional): Path to pre-trained LoRA adapter weights. 
                                          If None, creates a fresh LoRA configuration. 
                                          Defaults to None.

        Returns:
            PeftModel or transformers.PreTrainedModel: 
                The initialized LoRA-adapted model (actor).
        """
        
        if lora_weights is None:
            logging.info("Initializing actor model with a fresh LoRA configuration.")
            config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj",],
                lora_dropout=0,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(self.base_model, config)

            model.print_trainable_parameters()
            
            # Override the model's state_dict to return only LoRA-specific parameters
            old_state_dict = model.state_dict
            model.state_dict = (
                lambda self, *_, **__: get_peft_model_state_dict(
                    self, old_state_dict()
                )
            ).__get__(model, type(model))
        else:
            logging.info(f"Initializing actor model with LoRA weights: {lora_weights}")
            model = PeftModel.from_pretrained(
                self.base_model,
                lora_weights,
                torch_dtype=torch.float16,
            )
            
        model.half()
        return model
    
    def _init_critic(self, critic_weights = None):
        """
        Initializes the critic model (value network) based on the specified RL algorithm.

        Args:
            critic_weights (str, optional): Path to pre-trained critic weights. 
                                            If None, initializes a fresh critic. 
                                            Defaults to None.

        Returns:
            nn.Module: The initialized critic network (APPOCritic or TPPOCritic).
        
        Raises:
            NotImplementedError: If the algorithm is neither APPO nor TPPO.
        """
        if self.algo == "APPO":
            critic = APPOCritic(self.actor, self.tokenizer)
        elif self.algo == "TPPO":
            critic = TPPOCritic(self.actor, self.tokenizer)
        else:
            raise NotImplementedError
        if critic_weights is not None:
            critic.v_head.load_state_dict(
                torch.load(critic_weights, map_location= "cpu")
                )
        logging.info(f"Initialized the {self.algo} critic model")
        return critic
    
    def get_actions(self, obs_input_ids, obs_attn_mask):
        """
        Generates actions (model outputs) for the given observations using the actor,
        returning both the actions (decoded strings) and their tokenized representations.

        Args:
            obs (np.ndarray or list): A batch of textual observations.

        Returns:
            tuple:
                - actions (np.ndarray[str]): Array of decoded action strings (response to input question).
                - action_tokens (torch.Tensor): Tensor of token IDs for each action, 
                  shape (batch_size, max_new_tokens).
        """        
        # Generate actions using sampling-based decoding (top_k, temperature, etc.)
        output = self.actor.generate(
            input_ids=obs_input_ids,
            attention_mask=obs_attn_mask,
            do_sample=True,
            top_k=50,
            temperature=0.8,
            max_new_tokens=self.max_new_tokens,
            # 1802: "и", 16748: "ки", 198: "\n", 624: ".\n", 715: " \n", 76325: " \n\n\n\n\n"
            eos_token_id=self.possible_end_token_ids + [
                self.tokenizer.eos_token_id, self.tokenizer.pad_token_id
                ],
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
        )
        sequences = output.sequences
        
        actions = []
        # Initialize action_tokens with PAD tokens
        action_tokens = torch.ones(
            (sequences.shape[0], self.max_new_tokens), 
            dtype=torch.int64).to("cuda") * self.tokenizer.pad_token_id
        # Extract the newly generated tokens for each sequence
        for i in range(sequences.shape[0]):
            action_token = sequences[i][obs_input_ids[i].shape[0]:]
            action_tokens[i, :action_token.shape[0]] = action_token
            action = self.tokenizer.decode(action_token, skip_special_tokens=True)
            actions.append(action)
        actions = np.array(actions, dtype=np.object_)
        
        return actions, action_tokens
        
    def get_action_values(self, obs_input_ids, obs_attn_mask):
        """
        Computes value function predictions for the given observations using the critic.

        Args:
            obs (np.ndarray or list): A batch of textual observations.
        
        Returns:
            torch.Tensor: Value predictions of shape (batch_size,).
        """        
        # Temporarily disable LoRA adapters for the actor while using the critic
        with self.actor.disable_adapter():
            values = self.critic(obs_input_ids, attention_mask=obs_attn_mask)
        return values
    
    def get_slice(self, logits, obs_full_lengths, act_real_lengths):
        """
        Extracts the relevant portion of logits that correspond to generated action tokens.

        Args:
            logits (torch.Tensor): Logits output from the model, shape (batch_size, seq_len, vocab_size).
            obs_full_lengths (int): Length of the observation tokens.
            act_real_lengths (torch.Tensor): Actual lengths of the action tokens for each sample.

        Returns:
            torch.Tensor: Sliced logits of shape (batch_size, max_new_tokens, vocab_size),
                          containing only the action logits.
        """
        action_slice = torch.zeros((logits.shape[0], self.max_new_tokens, logits.shape[-1])).to("cuda")
        for i in range(logits.shape[0]):
            start_idx = obs_full_lengths - 1
            end_idx = obs_full_lengths + act_real_lengths[i] - 1
            action_slice[i, :act_real_lengths[i]] = logits[i, start_idx:end_idx]
        return action_slice
    
    def get_token_values(self, obs, action_tokens):
        """
        Computes value function predictions for each action token (used in token-level methods like TPPO).

        Args:
            obs (np.ndarray or list): Batch of textual observations.
            action_tokens (torch.Tensor): Tensor of tokenized actions, shape (batch_size, max_new_tokens).

        Returns:
            torch.Tensor: Value predictions sliced to match the action token positions,
                          shape (batch_size, max_new_tokens).
        """
        obs_token_seq = self.tokenizer(obs.tolist(), return_tensors="pt", padding=True)
        obs_input_ids = obs_token_seq["input_ids"].to("cuda")
        obs_attn_mask = obs_token_seq["attention_mask"].to("cuda")
        obs_full_lengths = obs_input_ids.shape[1]
        
        act_attn_mask = (action_tokens != 0)
        act_real_lengths = act_attn_mask.sum(dim=1)
        
        # Concatenate observation and action tokens
        obs_act_ids = torch.cat([obs_input_ids, action_tokens], dim=1)
        obs_act_mask = torch.cat([obs_attn_mask, act_attn_mask], dim=1)
        
        with self.actor.disable_adapter():
            values = self.critic(obs_act_ids, attention_mask=obs_act_mask)
        values = self.get_slice(values, obs_full_lengths, act_real_lengths)
        return values
    
    def get_token_logits(self, obs_input_ids, obs_attn_mask, action_tokens, batch_infer_size=None):
        """
        Retrieves the logits for each token in the combined observation+action sequence.
        Can perform inference in batches if `batch_infer` is True.

        Args:
            obs (np.ndarray or list): Batch of textual observations.
            action_tokens (torch.Tensor): Token IDs for the actions, shape (batch_size, max_new_tokens).
            batch_infer (bool): If True, uses a batch inference method to handle large inputs. Defaults to False.

        Returns:
            tuple:
                - pi_logits (torch.Tensor): The logits for the current policy, shape (batch_size, max_new_tokens, vocab_size).
                - rho_logits (torch.Tensor): An additional set of logits (possibly for old policy or reference), 
                  shape (batch_size, max_new_tokens, vocab_size).
        """        
        obs_full_lengths = obs_input_ids.shape[1]
        
        act_attn_mask = (action_tokens != 0)
        act_real_lengths = act_attn_mask.sum(dim=1)
        
        obs_act_ids = torch.cat([obs_input_ids, action_tokens], dim=1)
        obs_act_mask = torch.cat([obs_attn_mask, act_attn_mask], dim=1)
        # Optionally perform batch inference to handle large inputs
        if batch_infer_size:
            # Get the logtis with respect to action tokens
            with self.actor.disable_adapter():
                # This is the reference logits
                rho_logits = self.batch_infer(self.actor, obs_act_ids, obs_act_mask, obs_full_lengths, act_real_lengths, batch_infer_size)
            # This is the logits correspond to the policy model (to be trained in RL)      
            pi_logits = self.batch_infer(self.actor, obs_act_ids, obs_act_mask, obs_full_lengths, act_real_lengths, batch_infer_size)
        else:
            with self.actor.disable_adapter():
                rho_outputs = self.actor(input_ids=obs_act_ids, attention_mask=obs_act_mask, return_dict=True)
                rho_logits = self.get_slice(rho_outputs.logits, obs_full_lengths, act_real_lengths)
                
            pi_outputs = self.actor(input_ids=obs_act_ids, attention_mask=obs_act_mask, return_dict=True)
            pi_logits = self.get_slice(pi_outputs.logits, obs_full_lengths, act_real_lengths)
        
        return pi_logits, rho_logits
    
    def batch_infer(self, model, input_ids, attn_mask, obs_full_lengths, act_real_lengths, infer_batch_size=16):     
        """
        Performs inference in smaller batches to handle large inputs without running out of memory.

        Args:
            model (transformers.PreTrainedModel): The actor model used for inference.
            input_ids (torch.Tensor): Concatenated observation+action token IDs, shape (batch_size, seq_len).
            attn_mask (torch.Tensor): Attention mask for the combined sequences, shape (batch_size, seq_len).
            obs_full_lengths (int): Length of the observation tokens for slicing.
            act_real_lengths (torch.Tensor): Actual lengths of the action tokens for each sample.
            infer_batch_size (int): Batch size for sub-batching inference. Defaults to 16.

        Returns:
            torch.Tensor: Concatenated logits for all sub-batches, shape (batch_size, max_new_tokens, vocab_size).
        """
        logits = []
        for i in range(0, input_ids.shape[0], infer_batch_size):
            input_ids_batch = input_ids[i:i+infer_batch_size, :]
            attn_mask_batch = attn_mask[i:i+infer_batch_size, :]
            outputs = model(input_ids=input_ids_batch, attention_mask=attn_mask_batch, return_dict=True)
            
            logits_batch = self.get_slice(outputs.logits, obs_full_lengths, act_real_lengths)
            logits.append(logits_batch.clone())
        logits = torch.cat(logits, dim=0)
        
        return logits
        
    def get_last_token_position(self, action_tokens):
        """
        Determines the position of the last non-padding token in a sequence of action tokens.

        Args:
            action_tokens (torch.Tensor): Token IDs for one generated action, shape (max_new_tokens,).

        Returns:
            int: The index of the last non-pad token in the action token sequence.
        """
        pos = len(action_tokens) - 1
        while action_tokens[pos] == self.tokenizer.pad_token_id:
            pos -= 1
        return pos

    def get_joint_action_log_probs(self, obs_input_ids, obs_attn_mask, action_tokens, batch_infer_size):
        """
        Computes the joint log probabilities of the actions (token sequences) and entropy for each sample.

        Args:
            obs (np.ndarray or list): Observations corresponding to each action.
            action_tokens (torch.Tensor): Token IDs for the actions, shape (batch_size, max_new_tokens).
            batch_infer (bool): If True, uses batch inference for large inputs. Defaults to False.

        Returns:
            tuple:
                - action_log_probs (torch.Tensor): Summed log probabilities of each token in the action, shape (batch_size,).
                - entropies (torch.Tensor): Mean token-level entropy for each sequence, shape (batch_size,).
        """
        pi_logits, _ = self.get_token_logits(obs_input_ids, obs_attn_mask, action_tokens, batch_infer_size)
        pi_log_softmax = torch.log_softmax(pi_logits, dim=-1)
        action_log_probs = []
        entropies = []
        for i in range(pi_logits.shape[0]):
            act_token_length = self.get_last_token_position(action_tokens[i]) + 1
            log_softmax_slice = pi_log_softmax[i, :act_token_length, :]
            action_token_slice = action_tokens[i, :act_token_length]
            # Gather log probabilities for each token in the action
            token_log_probs = torch.gather(
                log_softmax_slice, 
                -1, 
                action_token_slice.unsqueeze(-1)
                ).squeeze(-1)
            action_log_prob = token_log_probs.sum()  # sum over all tokens as the prob of this action
            action_log_probs.append(action_log_prob)
            
            # Calculate entropy
            entropy = Categorical(logits=pi_logits[i, :act_token_length, :]).entropy().mean()
            entropies.append(entropy)
        action_log_probs = torch.stack(action_log_probs)
        entropies = torch.stack(entropies)
        return action_log_probs, entropies
    
    def tokenize_obs(self, obs):
        obs_token_seq = self.tokenizer(obs.tolist(), return_tensors="pt", padding=True)
        obs_input_ids = obs_token_seq.input_ids.to(self.device)
        obs_attn_mask = obs_token_seq.attention_mask.to(self.device)
        return obs_input_ids, obs_attn_mask
    
    @torch.no_grad()
    def infer_for_rollout(self, obs, batch_infer_size=None):
        """
        Generates actions and computes their value estimates and log probabilities
        for rollout collection in a training loop.

        Args:
            obs (np.ndarray or list): A batch of textual observations.

        Returns:
            tuple:
                - actions (np.ndarray[str]): Decoded actions.
                - action_tokens (np.ndarray[int]): Tokenized actions.
                - values (np.ndarray[float]): Value estimates.
                - log_probs (np.ndarray[float]): Log probabilities of the actions.
        """
        obs_input_ids, obs_attn_mask = self.tokenize_obs(obs)
        actions, action_tokens = self.get_actions(obs_input_ids, obs_attn_mask)
        
        if self.algo == "APPO":
            # Action-level PPO
            values = self.get_action_values(obs_input_ids, obs_attn_mask)
            values = values.float().cpu().numpy()
            # Entropy is computed but not needed here
            action_log_probs, _ = self.get_joint_action_log_probs(
                obs_input_ids, 
                obs_attn_mask, 
                action_tokens, 
                batch_infer_size,
                )
            action_tokens = action_tokens.int().cpu().numpy()
            action_log_probs = action_log_probs.float().cpu().numpy()
            log_probs = action_log_probs
        elif self.algo == "TPPO":
            # Token-level PPO
            values = self.get_token_values(obs, action_tokens).squeeze(-1)
            # TODO adapt this if you use TPPO and want to use batch inference
            pi_logits, _ = self.get_token_logits(obs, action_tokens, batch_infer_size=None) 
            pi_log_softmax = torch.log_softmax(pi_logits, dim=-1)
            token_log_probs = torch.gather(pi_log_softmax, -1, action_tokens.unsqueeze(-1)).squeeze(-1)

            values = values.float().cpu().numpy()
            action_tokens = action_tokens.int().cpu().numpy()
            log_probs = token_log_probs.float().cpu().numpy()
        elif self.algo == "GRPO":
            # A simpler variant without a critic
            values = np.zeros((obs.shape[0],)) # fake values, grpo does not use critic
            action_log_probs, _ = self.get_joint_action_log_probs(
                obs_input_ids, 
                obs_attn_mask, 
                action_tokens, 
                batch_infer_size=None  # TODO adapt this if you use GRPO and want to use batch inference
                )
            action_tokens = action_tokens.int().cpu().numpy()
            log_probs = action_log_probs.float().cpu().numpy()
        else:
            raise NotImplementedError
        
        return actions, action_tokens, values, log_probs
    
    def get_next_tppo_values(self, obs): 
        """
        Retrieves the last token's value predictions for TPPO. 
        This is useful when computing bootstrapped returns at the end of an episode.

        Args:
            obs (np.ndarray or list): Observations for which values are needed.

        Returns:
            torch.Tensor: Value predictions for the last token, shape (batch_size,).
        """
        token_seq = self.tokenizer(obs.tolist(), return_tensors="pt", padding=True)
        input_ids = token_seq["input_ids"].to("cuda")
        attn_mask = token_seq["attention_mask"].to("cuda")
        
        # values
        with self.actor.disable_adapter():
            values = self.critic(input_ids, attention_mask=attn_mask)
            values = values[:, -1]
        return values
    
    @torch.no_grad()
    def get_next_values(self, obs):
        """
        Retrieves the value function predictions for the next state to bootstrap returns.

        Args:
            obs (np.ndarray or list): Observations for which the next state values are needed.

        Returns:
            np.ndarray: Value predictions in numpy format.
        """
        
        obs_token_seq = self.tokenizer(obs.tolist(), return_tensors="pt", padding=True)
        obs_input_ids = obs_token_seq.input_ids.to(self.device)
        obs_attn_mask = obs_token_seq.attention_mask.to(self.device)
        
        if self.algo == "APPO":
            values = self.get_action_values(obs_input_ids, obs_attn_mask)
            values = values.cpu().float().numpy()
        elif self.algo == "TPPO":
            values = self.get_next_tppo_values(obs).squeeze(-1)
            values = values.cpu().float().numpy()
        elif self.algo == "GRPO":
            values = np.zeros((obs.shape[0],)) # fake values, grpo does not use critic
        else: 
            raise NotImplementedError
        return values
        
    def infer_for_action_update(self, obs, action_tokens= None, batch_infer_size=None):
        """
        Computes log probabilities and entropies for a batch of (obs, action_tokens),
        typically used for policy gradient updates.

        Args:
            obs (np.ndarray or list): Batch of observations.
            action_tokens (torch.Tensor, optional): Batch of tokenized actions.

        Returns:
            tuple:
                - action_log_probs (torch.Tensor): Summed log probabilities per sequence.
                - entropies (torch.Tensor): Mean entropy per sequence.

        Raises:
            AssertionError: If action_tokens is None.
        """
        assert action_tokens is not None, "action_tokens could not be none"
        obs_input_ids, obs_attn_mask = self.tokenize_obs(obs)
        action_log_probs, entropies = self.get_joint_action_log_probs(
            obs_input_ids, 
            obs_attn_mask, 
            action_tokens,
            batch_infer_size,
            )
        return action_log_probs, entropies
    
    def infer_for_token_update(self, obs, action_tokens):
        """
        Retrieves token-level logits for both the current and reference policies, if needed.

        Args:
            obs (np.ndarray or list): Batch of observations.
            action_tokens (torch.Tensor): Batch of tokenized actions.

        Returns:
            tuple:
                - pi_logits (torch.Tensor): Logits for the current policy.
                - rho_logits (torch.Tensor): Additional logits, possibly for the old policy or reference.
        """
        pi_logits, rho_logits = self.get_token_logits(obs, action_tokens)
        return pi_logits, rho_logits
    
    def test_get_actions(self, obs):
        """
        Generates actions using a deterministic approach (no sampling) for testing and debugging.

        Args:
            obs (np.ndarray or list): Observations for action generation.

        Returns:
            np.ndarray[str]: Array of decoded action strings.
        """
        prompts = obs.tolist()
        token_seq = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = token_seq["input_ids"].to("cuda")
        attn_mask = token_seq["attention_mask"].to("cuda")
        
        output = self.actor.generate(
            input_ids,
            attention_mask=attn_mask,
            do_sample=False,
            # top_k=50,
            # temperature=0.5,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=[self.tokenizer.eos_token_id, self.tokenizer.pad_token_id, 198, 624, 715, 271, 76325], 
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
        )
        sequences = output.sequences
        
        actions = []
        for i in range(sequences.shape[0]):
            action_token = sequences[i][input_ids[i].shape[0]:]
            action = self.tokenizer.decode(action_token, skip_special_tokens=True)
            actions.append(action)
        actions = np.array(actions, dtype=np.object_)
        
        return actions

    def save(self, save_dir, episode):
        exp_path = os.path.join(save_dir, "episode_{:04d}".format(episode))

        os.makedirs(exp_path, exist_ok=True)
        # Save lora adapter_config.json
        self.actor.save_pretrained(exp_path, )

        # Save lora weights, since the above method leads to an empty file
        lora_weights = {}
        for name, param in self.actor.named_parameters():
            if "lora" in name:  
                lora_weights[name] = param.data

        # Save the LoRA weights to a file
        save_file(lora_weights, exp_path + "/adapter_model.safetensors")
        logging.info(f"Saved LoRA weights to {exp_path}/adapter_model.safetensors")

    def load(self, save_dir):
        """
        Loads the actor model from a saved checkpoint and optionally reinitializes the critic.

        Args:
            save_dir (str): Path to the directory containing the saved LoRA actor weights.
        """
        print("load model")
        self.actor = self._init_actor(save_dir).to(self.device)

    def train(self):
        """
        Sets the actor and critic networks to training mode, enabling dropout and other
        training-specific behaviors.
        """
        self.generator.train()
        self.critic.train()

    def eval(self):
        """
        Sets the actor and critic networks to evaluation mode, disabling dropout and other
        training-specific behaviors.
        """
        self.generator.eval()
        self.critic.eval()

