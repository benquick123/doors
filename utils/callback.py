import os
from collections import deque, defaultdict

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean


class CustomLoggingCallback(BaseCallback):
    
    def __init__(self, verbose: int = 0, reward_shaper_logging=False, reward_shaper_save_path=None):
        super().__init__(verbose)
        
        if reward_shaper_logging:
            self.reward_shaper_save_path = reward_shaper_save_path
            if self.reward_shaper_save_path is not None:
                os.makedirs(self.reward_shaper_save_path)
        else:
            self.reward_shaper_save_path = None
        
    def _on_training_start(self) -> None:
        list_generator_fn = lambda: [list() for _ in range(self.model.env.num_envs)]
        deque_generator_fn = lambda: deque(maxlen=5 * self.model.env.num_envs)
        self.reward_buffers = defaultdict(list_generator_fn)
        self.master_reward_buffer = defaultdict(deque_generator_fn)
        
    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
        for env_idx, (done, info) in enumerate(zip(dones, infos)):
            if done and self.locals["log_interval"] is not None and (self.model._episode_num + 1) % self.locals["log_interval"] == 0:
                self._dump_logs(env_idx)
                
                if self.reward_shaper_save_path is not None:
                    self.training_env.env_method("save_wrapper", 
                                                    path=os.path.join(self.reward_shaper_save_path, "reward_shaper_env=%01d_ep=%05d.pkl" % (env_idx, self.model._episode_num)), 
                                                    indices=[env_idx])
                    
        return True
    
    def _dump_logs(self, env_idx):
        master_log = self.model.env.env_method("get_master_log", indices=[env_idx])[0]
        
        suffix = "" if self.training_env.num_envs == 1 else f":{env_idx}"
        for key, log_values in master_log.items():
            self.logger.record(f"{key}{suffix}", safe_mean(log_values))
        