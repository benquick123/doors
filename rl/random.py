import stable_baselines3 as sb3
import torch as th


class FlatRandomPolicy(sb3.common.policies.BasePolicy):
    
    def __init__(self, *args, **kwargs):
        if "use_sde" in kwargs:
            del kwargs["use_sde"]
        
        super(FlatRandomPolicy, self).__init__(*args, **kwargs)
        
    def _predict(self, *args, **kwargs):
        return th.tensor(self.action_space.sample(), device=self.device)


class Random(sb3.common.off_policy_algorithm.OffPolicyAlgorithm):
    
    policy_aliases = {
        "MlpPolicy": FlatRandomPolicy,
        "MultiInputPolicy": FlatRandomPolicy,
        "CnnPolicy": FlatRandomPolicy,
    }
    
    def __init__(self, *args, **kwargs):
        if "policy_kwargs" in kwargs:
            del kwargs["policy_kwargs"]
            
        kwargs["learning_rate"] = 0.0
        kwargs["buffer_size"] = 1
        super(Random, self).__init__(*args, **kwargs)
        
        self._setup_model()
        
    def train(self, *args, **kwargs):
        pass
    
    
        