import os
import torch
import logging
import numpy as np
from tqdm import tqdm
from functools import reduce
from tensorboardX import SummaryWriter
from training.rl.ppo.models.prm import ProcessRM
from training.rl.ppo.agents.agent import QwenLoRAgent
from training.rl.ppo.data.language_buffer import LanguageBuffer
from training.rl.ppo.trainers.trainer_appo import APPOTrainer
from training.rl.ppo.trainers.trainer_tppo import TPPOTrainer
from training.rl.ppo.trainers.trainer_grpo import GRPOTrainer

def _t2n(x):
    """
    Converts a PyTorch tensor to a NumPy array.
    
    Args:
        x (torch.Tensor): Input tensor.
    
    Returns:
        np.ndarray: Converted NumPy array.
    """
    return x.detach().cpu().numpy()

class RiskRunner:
    """
    MathRunner orchestrates the training and evaluation of agents in a mathematical environment using RL algorithms.
    
    It integrates the environment, agent, reward model, buffer, and trainer to manage the end-to-end training pipeline.
    
    Attributes:
        num_agents (int): Number of agents interacting with the environment.
        all_args (argparse.Namespace): Configuration arguments containing hyperparameters and settings.
        n_eval_rollout_threads (int): Number of parallel evaluation rollout threads.
        num_env_steps (int): Total number of environment steps for training.
        episode_length (int): Number of timesteps per episode.
        n_rollout_threads (int): Number of parallel rollout threads (agents).
        log_interval (int): Frequency (in episodes) to log training metrics.
        eval_interval (int): Frequency (in episodes) to perform evaluation.
        save_interval (int): Frequency (in episodes) to save model checkpoints.
        algo (str): Name of the RL algorithm being used (e.g., "APPO", "TPPO", "GRPO").
        prm_type (str): Type of the reward model ("MS" or "Qwen").
        run_dir (Path): Directory path for the current run.
        log_dir (str): Directory path for logging.
        writter (SummaryWriter): TensorBoard writer for logging metrics.
        save_dir (str): Directory path for saving models.
        envs (ShareSubprocVecEnv): Vectorized training environments.
        eval_envs (ShareSubprocVecEnv): Vectorized evaluation environments.
        agent (QwenLoRAgent): The agent instance containing actor and critic networks.
        buffer (LanguageBuffer): Buffer for storing and managing training data.
        prm (nn.Module): Reward model instance (`MSProcessRM` or `QwenProcessRM`).
        trainer (APPOTrainer | TPPOTrainer | GRPOTrainer): Trainer instance based on the selected RL algorithm.
    """
    
    def __init__(self, config):
        """
        Initializes the MathRunner with the provided configuration.
        
        Args:
            config (dict): Dictionary containing configuration parameters and initialized components.
                Expected keys:
                    - 'all_args': argparse.Namespace with configuration settings.
                    - 'envs': ShareSubprocVecEnv instance for training environments.
                    - 'eval_envs': ShareSubprocVecEnv instance for evaluation environments.
                    - 'num_agents': int, number of agents.
                    - 'run_dir': Path, directory path for the current run.
        """
        self.num_agents = config['num_agents']
        self.all_args = config['all_args']
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.log_interval = self.all_args.log_interval
        self.eval_interval = self.all_args.eval_interval
        self.save_interval = self.all_args.save_interval
        self.algo = self.all_args.algorithm_name

        # Directory setup for logging and saving models
        self.run_dir = config["run_dir"]
        self.log_dir = str(self.run_dir / 'logs')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.save_dir = str(self.run_dir / 'models/')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # Tensorboard logging
        self.writter = SummaryWriter(self.log_dir)

        # Initialize environments
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        
        self.prm = ProcessRM(
            self.all_args.prm_model_name_or_path, 
            self.all_args.prm_lora_weights,
            self.all_args.model_max_length,
        )
        
        # Initialize agent
        self.agent = QwenLoRAgent(
            self.all_args.model_name_or_path, 
            self.all_args.max_new_tokens, 
            self.all_args.model_max_length,
            self.algo,
            )
        
        # Initialize buffer
        self.buffer = LanguageBuffer(
            self.all_args, 
            self.num_agents, 
            self.agent.tokenizer.pad_token_id
            )

        # Initialize trainer based on algorithm
        if self.algo == "APPO":
            self.trainer = APPOTrainer(self.all_args, self.agent, self.num_agents)
        elif self.algo == "TPPO":
            self.trainer = TPPOTrainer(self.all_args, self.agent, self.num_agents)
        elif self.algo == "GRPO":
            self.trainer = GRPOTrainer(self.all_args, self.agent, self.num_agents)
        else:
            raise NotImplementedError

    def run(self):
        """
        Executes the main training loop, managing episodes, data collection, training updates, logging, and model saving.
        
        The training loop performs the following steps for each episode:
            1. Resets the training environments.
            2. Iterates over each timestep in the episode:
                a. Collects actions from the agent.
                b. Computes rewards using the reward model.
                c. Steps the environment with the collected actions.
                d. Inserts the collected data into the buffer.
                e. Records episodic returns if an episode ends.
            3. After the episode:
                a. Processes the buffer to compute returns and advantages.
                b. Performs training updates using the trainer.
                c. Saves the model at specified intervals.
                d. Logs training metrics at specified intervals.
                e. Optionally evaluates the agent's performance.
        """
        # Reset environments and initialize buffer observations
        obs = self.envs.reset()
        self.buffer.obs[0] = obs.copy()

        # Calculate the number of episodes based on total environment steps
        episodes = 1 # int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        
        episodic_returns = []
        for episode in tqdm(range(episodes), desc='Rollout:'):
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads  
            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_tokens, log_probs = self.collect(step)
                
                # Output rewards
                rewards = self.prm.get_reward(obs, actions)
                
                # Step the environments
                obs, dones, infos = self.envs.step(actions)

                # Insert data into buffer
                self.insert((obs, rewards, dones, values, actions, action_tokens, log_probs))
                
                # TODO how to stop the episode for the certain env
                for i in range(self.n_rollout_threads):
                    if dones[i, 0]:
                        episodic_returns.append(rewards[i, 0])
                
            # compute return and update network
            self.before_update()
            train_infos = self.trainer.train(self.buffer)      
            self.buffer.after_update()
            
            # save model
            if (episode == episodes - 1 or episode % self.save_interval == 0):
                self.save(episode)

            # log information
            if episode % self.log_interval == 0:
                logging.info("total_num_steps: ", total_num_steps)
                logging.info("average_step_rewards: ", np.mean(self.buffer.rewards))
                train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
                train_infos["average_currect_rate"] = np.mean(episodic_returns)
                self.log_infos(train_infos, total_num_steps)
                episodic_returns = []

            # eval
            # if self.all_args.use_eval and episode % self.eval_interval == 0:
            #     self.eval(total_num_steps)
        

    @torch.no_grad()
    def collect(self, step):
        """
        Collects actions, action tokens, value estimates, and log probabilities from the agent for a given timestep.
        
        Args:
            step (int): Current timestep within the episode.
        
        Returns:
            tuple:
                - values (np.ndarray): Value estimates from the agent's critic. Shape: (n_rollout_threads, num_agents).
                - actions (np.ndarray): Actions sampled by the agent's policy. Shape: (n_rollout_threads, num_agents).
                - action_tokens (np.ndarray): Tokenized actions. Shape: (n_rollout_threads, num_agents, max_new_tokens).
                - log_probs (np.ndarray): Log probabilities of the sampled actions. Shape: (n_rollout_threads, num_agents).
        """
        # Obtain behavior data from the agent
        actions, action_tokens, values, log_probs = self.agent.infer_for_rollout(
            np.concatenate(self.buffer.obs[step])
            )
        
        # Split the data across rollout threads
        values = np.array(np.split(values, self.n_rollout_threads))
        actions = np.array(np.split(actions, self.n_rollout_threads))
        action_tokens = np.array(np.split(action_tokens, self.n_rollout_threads))
        log_probs = np.array(np.split(log_probs, self.n_rollout_threads))

        return values, actions, action_tokens, log_probs

    def insert(self,data):
        """
        Inserts collected data into the buffer using the appropriate method based on the RL algorithm.
        
        Args:
            data (tuple): A tuple containing the following elements:
                - obs (np.ndarray): Observations from the environment after stepping. Shape: (n_rollout_threads, num_agents, ...).
                - rewards (np.ndarray): Rewards received from the environment. Shape: (n_rollout_threads, num_agents).
                - dones (np.ndarray): Done flags indicating episode termination. Shape: (n_rollout_threads, num_agents).
                - values (np.ndarray): Value estimates from the agent's critic. Shape: (n_rollout_threads, num_agents).
                - actions (np.ndarray): Actions taken by the agents. Shape: (n_rollout_threads, num_agents).
                - action_tokens (np.ndarray): Tokenized actions. Shape: (n_rollout_threads, num_agents, max_new_tokens).
                - log_probs (np.ndarray): Log probabilities of the actions. Shape: (n_rollout_threads, num_agents).
        """
        obs, rewards, dones, values, actions, action_tokens, log_probs = data

        # Determine which environments have ended episodes
        dones_env = np.all(dones, axis=1)
        # Mask is used to mask out the value of terminal state when calculating TD error. 
        # A mask value of 1.0 indicates that the agent is active and its state should be maintained for further computations. 
        masks = np.ones((self.n_rollout_threads, self.num_agents), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents), dtype=np.float32)

        # Insert data into the buffer based on the algorithm
        if self.algo == "APPO" or self.algo == "GRPO":
            self.buffer.insert_appo(obs, actions, values, rewards, masks, action_tokens, log_probs)
        elif self.algo == "TPPO":
            self.buffer.insert_tppo(obs, actions, values, rewards, masks, action_tokens, log_probs)
        else:
            raise NotImplementedError

    @torch.no_grad()
    def before_update(self):
        """
        Prepares the buffer for the training update by computing returns and advantages.
        
        It retrieves the next value estimates from the agent's critic and processes the buffer
        to compute returns and advantages using methods specific to the RL algorithm in use.
        """
        # Obtain value estimates based on the last observation
        next_values = self.agent.get_next_values(np.concatenate(self.buffer.obs[-1]))
        next_values = np.array(np.split(next_values, self.n_rollout_threads))
        
        # Process the buffer to compute returns and advantages
        if self.algo == "APPO":
            self.buffer.batch_process_appo(next_values)
        elif self.algo == "TPPO":
            self.buffer.batch_process_tppo(next_values)
        elif self.algo == "GRPO":
            self.buffer.batch_process_grpo()
        else:
            raise NotImplementedError

    def log_infos(self, infos, total_num_steps):
        """
        Logs training metrics to TensorBoard.
        
        Args:
            infos (dict): Dictionary containing training metrics to log.
                Example keys: 'value_loss', 'policy_loss', etc.
            total_num_steps (int): Total number of environment steps taken so far.
        """
        for k, v in infos.items():
            self.writter.add_scalars(k, {k: v}, total_num_steps)
    
    @torch.no_grad()
    def eval(self, total_num_steps):
        """
        Evaluates the agent's performance on the evaluation environment.
        
        Args:
            total_num_steps (int): Total number of environment steps taken so far.
        
        Procedure:
            1. Resets the evaluation environments.
            2. Runs evaluation episodes until the specified number of evaluation episodes is reached.
            3. Collects and records episodic returns.
            4. Logs the average correctness rate to TensorBoard.
        """
        eval_episode = 0
        eval_episodic_returns = []

        # Reset evaluation environments
        eval_obs = self.eval_envs.reset()
        while True:
            # Obtain actions from the agent's policy without exploration
            eval_actions, _ = self.agent.get_actions(np.concatenate(eval_obs))
            eval_actions = np.array(np.split(eval_actions, self.n_eval_rollout_threads))
            
            # Step the evaluation environments with the sampled actions
            eval_obs, eval_rewards, eval_dones, _ = self.eval_envs.step(eval_actions)

            # Record episodic returns if any evaluation episode ends
            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones[eval_i, 0]:
                    eval_episode += 1
                    eval_episodic_returns.append(eval_rewards[eval_i])

            # Stop evaluation after reaching the desired number of episodes
            if eval_episode >= self.all_args.eval_episodes:
                eval_currect_rate = np.mean(eval_episodic_returns)
                env_infos = {'eval_currect_rate': eval_currect_rate}     
                print("total_num_steps: ", total_num_steps)
                print("eval_currect_rate is {}.".format(eval_currect_rate))           
                self.log_infos(env_infos, total_num_steps)
                break
                
    def save(self, episode):
        """
        Saves the agent's actor and critic networks to the designated save directory.
        
        Args:
            episode (int): The current episode number, used for naming the saved model.
        """
        self.agent.save(self.save_dir, episode)

    def restore(self, model_dir):
        """
        Restores the agent's networks from a saved model directory.
        
        Args:
            model_dir (str): Directory path from which to restore the agent's models.
        """
        self.agent.restore(model_dir)


