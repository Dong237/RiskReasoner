
import math
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from training.utils import get_grad_norm, huber_loss, mse_loss

class APPOTrainer:
    """
    APPOTrainer facilitates the training process for agents using the APPO (Actor-Proximal Policy Optimization) algorithm.
    
    It manages the optimization of both the actor (policy network) and the critic (value network) based on
    data sampled from the LanguageBuffer. The class supports various loss computations, gradient clipping,
    and handles multiple PPO epochs and minibatches to ensure stable and efficient training.

    Attributes:
        tpdv (dict): Dictionary specifying tensor data type and device.
        agent (Agent): The agent containing the actor and critic models.
        clip_param (float): Clipping parameter for PPO.
        ppo_epoch (int): Number of PPO epochs per training iteration.
        mini_batch_size (int): Mini batch size per PPO epoch.
        value_loss_coef (float): Coefficient for scaling the value loss.
        max_grad_norm (float): Maximum gradient norm for clipping.
        huber_delta (float): Delta parameter for the Huber loss.
        entropy_coef (float): Coefficient for the entropy regularization term.
        _use_max_grad_norm (bool): Flag indicating whether to use gradient clipping.
        _use_clipped_value_loss (bool): Flag indicating whether to use clipped value loss.
        _use_huber_loss (bool): Flag indicating whether to use Huber loss instead of MSE.
        lr (float): Learning rate for the policy optimizer.
        critic_lr (float): Learning rate for the critic optimizer.
        opti_eps (float): Epsilon value for the optimizers.
        gradient_cp_steps (int): Number of gradient checkpointing steps for policy updates.
        policy_optimizer (torch.optim.Optimizer): Optimizer for the actor network.
        critic_optimizer (torch.optim.Optimizer): Optimizer for the critic network.
    """
    
    def __init__(self, args, agent, num_agents):
        """
        Initializes the APPOTrainer with the provided configurations and agent.
        
        Args:
            args (argparse.Namespace): Configuration arguments containing hyperparameters and settings.
                Expected attributes include:
                    - clip_param (float)
                    - ppo_epoch (int)
                    - mini_batch_size (int)
                    - value_loss_coef (float)
                    - max_grad_norm (float)
                    - huber_delta (float)
                    - entropy_coef (float)
                    - use_max_grad_norm (bool)
                    - use_clipped_value_loss (bool)
                    - use_huber_loss (bool)
                    - lr (float)
                    - critic_lr (float)
                    - opti_eps (float)
                    - gradient_cp_steps (int)
            agent (Agent): The agent instance containing actor and critic models.
            num_agents (int): Number of agents interacting with the environment.
        """
        self.tpdv = dict(dtype=torch.float32, device=torch.device("cuda:0"))
        self.agent = agent

        # Hyperparameters and settings
        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.mini_batch_size = args.mini_batch_size
        self.value_loss_coef = args.value_loss_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta
        self.entropy_coef = args.entropy_coef
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.gradient_cp_steps = args.gradient_cp_steps

        # Initialize optimizers for actor and critic
        self.policy_optimizer = torch.optim.AdamW(
            # Only the parameters that require gradients are passed to the optimizer
            filter(lambda p: p.requires_grad, self.agent.actor.parameters()), 
            lr=self.lr, 
            # Constant added to the denominator to improve numerical stability, preventing division by zero
            eps=1e-5, 
            weight_decay=0
            )
        self.critic_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.agent.critic.parameters()), 
            lr=self.critic_lr, 
            eps=1e-5
            )

    def cal_policy_loss(self, log_prob_infer, log_prob_batch, advantages_batch, entropy):
        """
        Calculates the policy loss using the PPO clipped objective and computes an approximate KL divergence.
        
        Args:
            log_prob_infer (torch.Tensor): Log probabilities of actions under the current policy. Shape: (batch_size,).
            log_prob_batch (torch.Tensor): Log probabilities of actions under the old policy (from buffer). Shape: (batch_size,).
            advantages_batch (torch.Tensor): Advantage estimates. Shape: (batch_size,).
            entropy (torch.Tensor): Entropy of the policy distribution. Shape: (batch_size,).
        
        Returns:
            tuple:
                - policy_loss (torch.Tensor): Scalar tensor representing the policy loss.
                - approx_kl (torch.Tensor): Scalar tensor representing the approximate KL divergence.
        """
        # Compute the log ratio between new and old policies
        log_ratio = log_prob_infer - log_prob_batch
        imp_weights = torch.exp(log_ratio)
        
        # Approximate KL divergence
        approx_kl = ((imp_weights - 1) - log_ratio).mean()
        
        # Clipped surrogate objective
        surr1 = -torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_batch
        surr2 = -imp_weights * advantages_batch
        surr = torch.max(surr1, surr2)
        
        # Final policy loss with entropy regularization
        policy_loss = surr.mean() - self.entropy_coef * entropy.mean()
        return policy_loss, approx_kl
        
    
    def cal_value_loss(self, values_infer, value_preds_batch, return_batch):
        """
        Calculates the value loss using either Huber loss or Mean Squared Error (MSE),
        supporting PPO's value function clipping.
        
        Args:
            values_infer (torch.Tensor): Value predictions from the current critic. Shape: (batch_size,).
            value_preds_batch (torch.Tensor): Value predictions from the old critic (from buffer). Shape: (batch_size,).
            return_batch (torch.Tensor): Target returns. Shape: (batch_size,).
        
        Returns:
            torch.Tensor: Scaled value loss as a scalar tensor.
        """
        # Clipped value predictions
        value_pred_clipped = value_preds_batch + (values_infer - value_preds_batch).clamp(-self.clip_param, self.clip_param)
        
        # Errors for clipped and unclipped value predictions
        error_clipped = return_batch - value_pred_clipped
        error_unclipped = return_batch - values_infer
        
        # Choose loss function based on configuration
        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_unclipped = huber_loss(error_unclipped, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_unclipped = mse_loss(error_unclipped)
        
        # Take maximum of clipped and unclipped losses
        value_loss = torch.max(value_loss_clipped, value_loss_unclipped).mean()
        return value_loss * self.value_loss_coef  # Scale the value loss

    def ppo_update(self, sample):
        """
        Performs a single PPO update step, optimizing both the critic and the actor.
        
        Args:
            sample (tuple): A tuple containing the following elements:
                - obs_batch (np.ndarray): Observations batch. Shape: (batch_size, ...).
                - action_batch (np.ndarray): Actions batch. Shape: (batch_size, ...).
                - log_prob_batch (np.ndarray): Log probabilities of actions under the old policy. Shape: (batch_size,).
                - value_preds_batch (np.ndarray): Value predictions from the old critic. Shape: (batch_size,).
                - return_batch (np.ndarray): Target returns. Shape: (batch_size,).
                - advantages_batch (np.ndarray): Advantage estimates. Shape: (batch_size,).
                - action_tokens_batch (np.ndarray): Tokenized actions batch. Shape: (batch_size, max_new_tokens).
        
        Returns:
            tuple:
                - value_loss (float): Value loss for the critic.
                - critic_grad_norm (float): Gradient norm for the critic.
                - policy_loss (float): Policy loss for the actor.
                - policy_grad_norm (float): Gradient norm for the actor.
        """
        # Unpack the sample
        obs_batch, action_batch, log_prob_batch, \
            value_preds_batch, return_batch, advantages_batch, action_tokens_batch = sample

        # Convert numpy arrays to PyTorch tensors and move to CUDA
        log_prob_batch = torch.from_numpy(log_prob_batch).to("cuda")
        value_preds_batch = torch.from_numpy(value_preds_batch).to("cuda")
        return_batch = torch.from_numpy(return_batch).to("cuda")
        advantages_batch = torch.from_numpy(advantages_batch).to("cuda")
        action_tokens_batch = torch.from_numpy(action_tokens_batch).to("cuda")
        batch_size = obs_batch.shape[0]
        
        # ------------------------- Critic Update ------------------------- #
        # Obtain new value predictions from the critic
        # NOTE value_preds and value_infer could be a bit different even before critic model gets updated
        obs_input_ids, obs_attn_mask = self.agent.tokenize_obs(np.concatenate(obs_batch))
        values_infer = self.agent.get_action_values(obs_input_ids, obs_attn_mask)
        values_infer = values_infer.view(batch_size, -1)
        
        # Calculate value loss
        value_loss = self.cal_value_loss(values_infer, value_preds_batch, return_batch)
        
        # Backpropagate value loss
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        
        # Optionally clip gradients
        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.agent.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_grad_norm(self.agent.critic.parameters())
        
        # Step the critic optimizer
        self.critic_optimizer.step()
        
        # Extract scalar values
        value_loss = value_loss.item()
        critic_grad_norm = critic_grad_norm.item()
        
        # Reset gradients for the critic
        self.critic_optimizer.zero_grad()
        

        # ------------------------- Policy Update ------------------------- #
        # Zero gradients for the policy optimizer
        self.policy_optimizer.zero_grad()
        assert batch_size >= self.gradient_cp_steps, "Batch size must be greater than or equal to gradient checkpointing steps"
        cp_batch_size = max(int(batch_size // self.gradient_cp_steps), self.gradient_cp_steps)
        total_approx_kl = 0
        
        # Iterate over the batch in chunks for gradient checkpointing
        for start in range(0, batch_size, cp_batch_size):
            end = start + cp_batch_size
            
            # Obtain new log probabilities and entropy from the current policy
            log_prob_infer, entropy = self.agent.infer_for_action_update(
                np.concatenate(obs_batch[start:end]), 
                action_tokens_batch[start:end].view(-1, action_tokens_batch.shape[-1]),
                batch_infer_size=cp_batch_size
                )

            # Reshape log probabilities
            log_prob_infer = log_prob_infer.view(obs_batch[start:end].shape[0], -1)
            
            # Normalize advantages
            cp_adv_batch = advantages_batch[start:end]
            cp_adv_batch = (cp_adv_batch - cp_adv_batch.mean()) / (cp_adv_batch.std() + 1e-8)
            
            # Reshape entropy
            entropy = entropy.view(obs_batch[start:end].shape[0], -1)
            
            # Calculate policy loss and approximate KL divergence
            policy_loss, approx_kl = self.cal_policy_loss(
                log_prob_infer, 
                log_prob_batch[start:end], 
                cp_adv_batch,
                entropy
                )
            
            # Accumulate KL divergence
            total_approx_kl += approx_kl / self.gradient_cp_steps
            
            # Scale policy loss and backpropagate
            policy_loss /= self.gradient_cp_steps
            policy_loss.backward()
        
        # Early stopping if KL divergence is too high
        if total_approx_kl > 0.02:
            # TODO add logging of warning here
            self.policy_optimizer.zero_grad()
            return value_loss, critic_grad_norm, 0, 0
        
        # Clip gradients for the policy
        policy_grad_norm = nn.utils.clip_grad_norm_(
            self.agent.actor.parameters(), 
            self.max_grad_norm
            )
        
        # Step the policy optimizer
        self.policy_optimizer.step()
        
        # Extract scalar values
        policy_loss = policy_loss.item()
        policy_grad_norm = policy_grad_norm.item()
        
        # Reset gradients for the policy
        self.policy_optimizer.zero_grad()
    
        return value_loss, critic_grad_norm, policy_loss, policy_grad_norm

    def train(self, buffer):
        """
        Perform a training update using minibatch Gradient Descent over multiple PPO epochs.
        
        Args:
            buffer (LanguageBuffer): Buffer containing collected training data for APPO.
        
        Returns:
            dict: Contains averaged training metrics over all updates, including:
                - 'value_loss': Average value loss across all minibatches.
                - 'value_grad_norm': Average gradient norm for the critic across all minibatches.
                - 'policy_loss': Average policy loss across all minibatches.
                - 'policy_grad_norm': Average gradient norm for the policy across all minibatches.
        """
        train_info = {}
        train_info['value_loss'] = 0
        train_info['value_grad_norm'] = 0
        train_info['policy_loss'] = 0
        train_info['policy_grad_norm'] = 0

        update_time = 0
        for _ in tqdm(range(self.ppo_epoch), desc="Running PPO Epoch:"):
            
            # Obtain a generator (7 elements contained) for minibatches
            # obs_batch, action_batch, log_prob_batch, value_preds_batch, 
            # return_batch, advantages_batch, action_tokens_batch
            data_generator = buffer.appo_sampler(self.mini_batch_size)
            for i, sample in enumerate(data_generator):
                # Perform PPO update on the sampled minibatch
                value_loss, value_grad_norm, policy_loss, policy_grad_norm = self.ppo_update(sample)
                
                # Accumulate training metrics
                train_info['value_loss'] += value_loss
                train_info['value_grad_norm'] += value_grad_norm
                train_info['policy_loss'] += policy_loss
                train_info['policy_grad_norm'] += policy_grad_norm
                update_time += 1

        # Average the training metrics
        for k in train_info.keys():
            train_info[k] /= update_time
 
        return train_info

    def prep_training(self):
        """
        Set the actor and critic networks to training mode.
        
        Enables training-specific behaviors like dropout and batch normalization.
        """
        self.agent.actor().train()
        self.agent.critic().train()

    def prep_rollout(self):
        """
        Set the actor and critic networks to evaluation mode.
        
        Disables training-specific behaviors like dropout and batch normalization.
        """
        self.agent.actor().eval()
        self.agent.critic().eval()
