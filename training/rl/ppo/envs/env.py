import random
import numpy as np
from training.rl.ppo.models.prm import ProcessRM
from utils.helper import load_dataset 
from utils.constants import Prompts, SPLIT_TOKEN


class RiskEnv:
    def __init__(
        self, 
        rank, 
        dataset_name, 
        dataset_path, 
        mode,
        ):
        
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
        
        self.instruction = Prompts.INSTRUCTION_STEP_BY_STEP.value

    def reset(self):
        self.set_seed(42)
        data_dict = random.choice(self.dataset)
            
        self.problem = self.instruction + data_dict["query_cot"]
        self.label = data_dict["choices"][int(data_dict["gold"])]  # "good" or "bad"
        
        self.current_state = self.problem + "\n"  ## FIXME this needs adaptation
        obs = np.array([self.current_state], dtype=np.object_)
        self.step_count = 0
        return obs
    
    def step(self, action):
        self.step_count += 1
        
        # Get next state
        action = action[0].replace(self.step_tag, "").strip()
        self.current_state = self.current_state + action + " " + self.step_tag + "\n"
        next_obs = np.array([self.current_state], dtype=np.object_)
        
        # FIXME need to account for the cases where the conclusion comes within the first few steps
        if SPLIT_TOKEN.lower() in action.lower():
            dones = np.ones((self.n_agents), dtype=bool)
        elif self.step_count >= self.max_step:
            dones = np.ones((self.n_agents), dtype=bool)
        else:
            dones = np.zeros((self.n_agents), dtype=bool)
        
        infos = {
            "state": self.current_state
            }
        return next_obs, dones, infos

    def set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)