import random
import numpy as np
from training.rl.ppo.models.prm import ProcessRM
from utils.helper import load_dataset 
from utils.constants import Prompts, SPLIT_TOKEN, STEP_TAG


class RiskEnv:
    def __init__(
        self, 
        rank, 
        dataset_name, 
        dataset_path, 
        mode,
        n_agents=1,
        max_step=30,
        ):
        
        self.rank = rank
        self.mode = mode
        self.dataset_name = dataset_name
        self.dataset = load_dataset(dataset_path)
        self.n_agents = n_agents
        self.max_step = max_step
        self.step_count = 0
        
        self.problem = None
        self.label = None
        self.good_assessment = SPLIT_TOKEN + ": " + 'good'
        self.bad_assessment = SPLIT_TOKEN + ": " + 'bad'
        self.current_state_len = None
        self.empirical_state_window = None  # A highly problem-specific attribute
        self.step_tag = STEP_TAG
        self.step_tag_real = "ки"
        self.current_state = None
        self.instruction = Prompts.INSTRUCTION_STEP_BY_STEP.value
        
        # FIXME this setting is problem specific and needs adaptation in the future
        # Windows are counts of characters in problem / resoning steps
        self.forward_window = 3000
        self.backward_window = 1000
        # This is to make sure that when checking backwards, we do not check the forwrd window.
        assert self.forward_window > self.backward_window, "forward_window should be larger than backward_window"
        
        self.last_few_steps = None

    def reset(self):
        data_dict = random.choice(self.dataset)
            
        self.problem = self.instruction + data_dict["query_cot"]
        self.label = data_dict["choices"][int(data_dict["gold"])]  # "good" or "bad"
        
        # NOTE "\n" here has an impact on the implementation in prm.py in the generate_input_texts method
        self.current_state = self.problem + "\n" 
        self.empirical_state_window = len(self.current_state) + self.forward_window # would be around 4000
        obs = np.array([self.current_state], dtype=np.object_)
        self.step_count = 0
        return obs
    
    def step(self, action):
        self.step_count += 1
        # Get next state
        action_str = action[0]
        self.current_state = self.current_state + action_str
        self.current_state_len = len(self.current_state)
        next_obs = np.array([self.current_state], dtype=np.object_)
        self.last_few_steps = self.current_state.replace(
            self.problem, ""
            )[-self.backward_window:].lower()
        
        if (
            self.current_state_len > self.empirical_state_window # reasoning has been going on for a few steps
            and (
                self.good_assessment.lower() in self.last_few_steps
                or
                self.bad_assessment.lower() in self.last_few_steps
            )
        ):
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