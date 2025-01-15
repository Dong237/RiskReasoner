import random
import numpy as np
from utils.helper import load_dataset 
from training.rl.ppo.envs.prompts import IN_CONTEXT_EXAMPLE


class RiskEnv:

    def __init__(self, rank, dataset_name, dataset_path, mode):
        
        self.rank = rank
        self.mode = mode
        self.dataset_name = dataset_name
        self.dataset = load_dataset(dataset_path)
        self.n_agents = 1
        self.max_step = 10
        self.step_count = 0
        
        if self.mode == "test":
            self.problem_idx = 0
        
        self.problem = None
        self.label = None
        self.step_tag = "ки"
        self.current_state = None

    def reset(self):
        ## FIXME the reset method needs adaptation
        random.seed(42)
        problem_answer_pair = random.choice(self.dataset)
            
        self.problem = problem_answer_pair["problem"]
        self.label = problem_answer_pair["final_answer"]
        
        print(f"\n\n\n\n======== new problem: {self.problem}, label: {self.label} ==========", )
        
        self.current_state = IN_CONTEXT_EXAMPLE + self.problem + "\n"
        obs = np.array([self.current_state], dtype=np.object_)
        self.step_count = 0
        return obs
    
    def step(self, action):
        self.step_count += 1
        # FIXME this whole function needs adaptations
        action = action[0].replace(self.step_tag, "").strip()
        print(f"action: {action}")
        self.current_state = self.current_state + action + " " + self.step_tag + "\n"
        # NOTE appending a_t to s_t leads to S_{t+1}
        next_obs = np.array([self.current_state], dtype=np.object_)
        # FIXME this might need adaptation for RiskReasoner
        score = 0.0
        if "step" in action.lower() or "answer" in action.lower():
            score = 1.0
        if "answer" in action.lower():
            dones = np.ones((self.n_agents), dtype=bool)
        elif self.step_count >= self.max_step:
            dones = np.ones((self.n_agents), dtype=bool)
        else:
            dones = np.zeros((self.n_agents), dtype=bool)
        
        rewards = [score for _ in range(self.n_agents)]
        infos = {"state": self.current_state}
        return next_obs, rewards, dones, infos

    def seed(self, seed):
        np.random.seed(seed)