import math
import pickle
from collections import defaultdict, deque, OrderedDict
from functools import partial

import gymnasium as gym
from stable_baselines3.common.env_checker import _is_goal_env
import numpy as np
import torch as th

from networks.mdma import MDMA
from networks.mdn import MixtureDensityNetwork
from networks.latent_gce import LatentGCE
from networks.discriminator import DIAYNDiscriminator
from utils.replay_buffer import MDNDataset, RepeatingSampler, StateDataset, EmpowermentDataset, DIAYNDataset
from env_utils.helpers import default


#### CUSTOM RESET FN WRAPPER ####

class CustomResetFnWrapper(gym.Wrapper):
    
    def __init__(self, env, env_initialization_fn):
        super().__init__(env)
        
        self.env_initialization_fn = env_initialization_fn
        
    def reset(self, seed=None, options=None):
        return self.env_initialization_fn(self.env, reset_kwargs=dict(seed=seed, options=options))
    

#### DICT OBSERVATION WRAPPER ####
class DictObservationWrapper(gym.ObservationWrapper):
    
    def __init__(self, env):
        super().__init__(env)
        
        self.is_dict_obs_space = True
        if not isinstance(self.observation_space, gym.spaces.Dict):
            self.is_dict_obs_space = False
            self.observation_space = gym.spaces.Dict({"observation": self.observation_space})
        
    def observation(self, observation):
        if isinstance(observation, dict):
            return observation
        elif isinstance(observation, np.ndarray):
            return {"observation": observation}
        
        
class DummyZObservationWrapper(gym.ObservationWrapper):
    
    def __init__(self, env, n_skills=64, fix_z=None):
        super().__init__(env)
        self.n_skills = n_skills
        assert fix_z is None or isinstance(fix_z, int), "The `fix_z` must be an integer or None."
        self.fix_z = fix_z
        
        assert isinstance(self.observation_space, gym.spaces.Dict), "The observation space must be a Dict space. Use `DictObservationWrapper`."
        new_observation_space = self.observation_space.spaces.copy()
        new_observation_space["z"] = gym.spaces.Box(low=0, high=1, shape=(n_skills, ), dtype=new_observation_space["observation"].dtype)
        self.observation_space = gym.spaces.Dict(new_observation_space)
        
    def reset(self, *, seed = None, options = None):
        self.current_z = np.zeros((self.n_skills, ))
        if self.fix_z is None:
            self.current_z[self.np_random.integers(0, self.n_skills)] = 1.0
        else:
            self.current_z[self.fix_z] = 1.0
        
        return super().reset(seed=seed, options=options)
    
    def observation(self, observation):
        observation["z"] = self.current_z
        return observation


#### BASE REWARD WRAPPERS ####

class ZeroRewardWrapper(gym.RewardWrapper):
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        
    def step(self, action):
        obs, reward, terminal, truncated, info = self.env.step(action)
        info["dense_reward"] = reward
        return obs, 0.0, terminal, truncated, info


class SparseRewardWrapper(gym.RewardWrapper):
    
    def __init__(self, env, reward_threshold, direction="larger"):
        super().__init__(env)
        self.reward_threshold = reward_threshold
        self.direction = direction
        
        self.logging_buffer = defaultdict(list)
        self.master_log = defaultdict(partial(deque, maxlen=5))
        
    def get_master_log(self):
        super_master_log = dict()
        if hasattr(self.env, "get_master_log"):
            super_master_log = self.env.get_master_log()
            
        merged_log = self.master_log | super_master_log
        if len(merged_log) < len(self.master_log) + len(super_master_log):
            duplicates = set(super_master_log.keys()).intersection(self.master_log.keys())
            print(f"Duplicate keys found in the logs: {duplicates}")
            
        return merged_log
    
    def reset(self, seed=None, options=None):
        for key in self.logging_buffer:
            if len(self.logging_buffer[key]) > 0:
                self.master_log[key].append(np.sum(self.logging_buffer[key]))
                self.logging_buffer[key] = []
        
        return super().reset(seed=seed, options=options)
        
    def step(self, action):
        obs, reward, terminal, truncated, info = self.env.step(action)
        info["dense_reward"] = reward
        
        self.logging_buffer["rollout/dense_reward"].append(reward)
        
        return obs, self.reward(reward), terminal, truncated, info
        
    def reward(self, reward):
        if self.direction == "larger":
            return 1.0 if reward >= self.reward_threshold else 0.0
        elif self.direction == "smaller":
            return 1.0 if reward <= self.reward_threshold else 0.0
        else:
            raise ValueError("The direction must be either 'larger' or 'smaller'.")
        
    def compute_reward(self, obs, next_obs, info):
        if hasattr(self.env, "compute_reward"):
            # go deeper in the wrappers
            rewards = self.env.compute_reward(obs, next_obs, info)
        elif hasattr(self.env.unwrapped, "compute_reward"):
            # compute reward from the env directly
            rewards = self.env.unwrapped.compute_reward(next_obs["achieved_goal"], obs["desired_goal"], info)
        else:
            raise NotImplementedError("This should not be called, since the env is not GoalEnv.")

        if self.direction == "larger":
            rewards = np.where(rewards >= self.reward_threshold, 1.0, 0.0)
        elif self.direction == "smaller":
            rewards = np.where(rewards <= self.reward_threshold, 1.0, 0.0)
        else:
            raise ValueError("The direction must be either 'larger' or 'smaller'.")
        
        return rewards


#### REWARD SHAPING WRAPPERS ####

class ShapingRewardWrapper(gym.Wrapper):
    
    def __init__(self, 
                 env, 
                 shaping_reward_weight=0.0, 
                 rollout_shaping_reward=True,
                 reuse_rollout_shaping=False,
                 shaping_preprocess_obs_fn=default.preprocess_obs_fn, 
                 inference_preprocess_obs_fn=default.preprocess_obs_fn):
        super().__init__(env)
        
        assert not reuse_rollout_shaping or (reuse_rollout_shaping and rollout_shaping_reward), \
            "The `reuse_rollout_shaping` can only be set to True if the `rollout_shaping_reward` is also True."
        
        self.shaping_reward_weight = shaping_reward_weight
        self.rollout_shaping_reward = rollout_shaping_reward
        self.reuse_rollout_shaping = reuse_rollout_shaping
        
        self.shaping_preprocess_obs_fn = shaping_preprocess_obs_fn
        self.inference_preprocess_obs_fn = inference_preprocess_obs_fn
        
        obs_sample = self.observation_space.sample()
        train_shaping_shape = self.shaping_preprocess_obs_fn(obs_sample).shape
        inference_shaping_shape = self.inference_preprocess_obs_fn(obs_sample).shape
        assert train_shaping_shape == inference_shaping_shape, \
            f"The `shaping_preprocess_obs_fn` and `inference_preprocess_obs_fn` must return the same shape. Got {inference_shaping_shape} and {train_shaping_shape}."
        
        self.logging_buffer = defaultdict(list)
        self.master_log = defaultdict(partial(deque, maxlen=5))
        
    def reset(self, seed=None, options=None):
        for key in self.logging_buffer:
            if len(self.logging_buffer[key]) > 0:
                self.master_log[key].append(np.sum(self.logging_buffer[key]))
                self.logging_buffer[key] = []
        
        return super().reset(seed=seed, options=options)
        
    def step(self, action):
        obs, reward, terminal, truncated, info = super().step(action)  
        
        info["main_reward"] = reward
        self.logging_buffer["rollout/main_reward"].append(reward)
        
        if self.rollout_shaping_reward:
            info["shaping_reward"] = self.get_shaping_reward(obs)[0]
            self.logging_buffer["rollout/shaping_reward"].append(info["shaping_reward"])
            reward += self.shaping_reward_weight * info["shaping_reward"]
            
        return obs, reward, terminal, truncated, info
        
    def compute_reward(self, obs, next_obs, info):
        if _is_goal_env(self.env):
            if hasattr(self.env, "compute_reward"):
                # go deeper in the wrappers
                rewards = self.env.compute_reward(obs, next_obs, info)
            elif hasattr(self.env.unwrapped, "compute_reward"):
                # compute reward directly from env
                rewards = self.env.unwrapped.compute_reward(next_obs["achieved_goal"], obs["desired_goal"], info)
        else:
            # just read the original main reward, since the env is not GoalEnv
            assert "main_reward" in info[0], "The original reward must be stored in the info dictionary."
            rewards = np.array([_info["main_reward"] for _info in info], dtype=np.float32)
        
        if self.shaping_reward_weight > 0.0:
            if self.reuse_rollout_shaping:
                shaping_rewards = np.array([_info["shaping_reward"] for _info in info], dtype=np.float32)
                rewards += self.shaping_reward_weight * shaping_rewards
            else:
                rewards += self.shaping_reward_weight * self.get_shaping_reward(obs)
        
        # returns a reward shaped like (batch_size, ) = (obs.shape[0], )
        # `get_shaping_reward()` reshapes obs from shape=(obs_dim, ) to shape=(1, obs_dim)
        return rewards
    
    def get_shaping_reward(self, obs):
        # should return the shaping reward shape=(batch_size, )
        raise NotImplementedError("The `get_shaping_reward` method must be implemented.")
    
    def get_master_log(self):
        super_master_log = dict()
        if hasattr(self.env, "get_master_log"):
            super_master_log = self.env.get_master_log()
            
        merged_log = self.master_log | super_master_log
        if len(merged_log) < len(self.master_log) + len(super_master_log):
            duplicates = set(super_master_log.keys()).intersection(self.master_log.keys())
            print(f"Duplicate keys found in the logs: {duplicates}")
            
        return merged_log
    
    def save_wrapper(self, path, return_snapshot=False):
        raise NotImplementedError("The `save_wrapper` method must be implemented.")
    
    def load_wrapper(self, path, return_snapshot=False):
        snapshot = pickle.load(open(path, "rb"))
        if return_snapshot:
            return snapshot
     

class MDNRewardWrapper(ShapingRewardWrapper):
    
    def __init__(self, env,
                 *args,
                 
                 reuse_rollout_shaping=False,
                 shaping_preprocess_obs_fn=None,
                 inference_preprocess_obs_fn=None,
                 
                 mdn_update_freq=20,
                 mdn_kwargs={},
                 mdn_buffer_size=5000,
                 
                 shaping_path=None,
                 shaping_warmup_eps=1,
                 shaping_reward_weight=0.1,
                 
                 shaping_lr=5e-4,
                 # !!! `initial_n_epochs` does not get scaled by mdn_update_freq !!! #
                 initial_n_epochs=10000,
                 shaping_n_epochs=50,
                 batch_size=2 ** 9,
                 inference_batch_size=2 ** 9,
                 use_lr_scheduler=True,
                 
                 rollout_shaping_reward=True,
                 horizon=None,
                 query_horizon=20,
                 verbose=0,
                 device="cuda" if th.cuda.is_available() else "cpu",
                 double_precision=False,
                 **kwargs):
        assert shaping_preprocess_obs_fn is not None, "The `shaping_preprocess_obs_fn` must be provided."
        super().__init__(env, shaping_reward_weight, rollout_shaping_reward, reuse_rollout_shaping, shaping_preprocess_obs_fn, inference_preprocess_obs_fn)
        
        self.verbose = verbose
        
        # env initialization and processing related functions
        self.inference_batch_size = inference_batch_size
        
        # episode related stats
        self.episode_num = 0
        self.episode_obs_buffer = []
        
        # set the device
        self.device = th.device(device)
        self.double_precision = double_precision
        
        # horizon
        self.horizon = horizon
        if self.horizon is None:
            self.horizon = self.env.spec.max_episode_steps
        # precompute the horizon
        self._horizon = th.linspace(0 + 1/self.horizon, 1, self.horizon).reshape(-1, 1).to(self.device)
        
        if query_horizon is None:
            self.query_horizon = self.horizon
            self._query_horizon = self._horizon
        else:
            self.query_horizon = query_horizon
            base = self.horizon ** (1 / (query_horizon - 1))
            self._query_horizon = (base ** (th.arange(query_horizon))) / self.horizon
            self._query_horizon = self._query_horizon.reshape(-1, 1).to(self.device)
            
        # define helper functions for mean entropy querying
        # since we use logarithmic horizon spacing, we can integrate by weighting individual entropies
        self._query_diffs = th.diff(self._query_horizon, dim=0).reshape(-1)
        
        # initialize / load MDN
        self.flattened_observation_shape = self.shaping_preprocess_obs_fn(self.observation_space.sample())[0].shape[0]
        
        self.mdn_updated = False
        self.mdn_update_freq = float(mdn_update_freq)
        self.shaping_warmup_eps = float(shaping_warmup_eps)
        
        mdn_kwargs["dim_in"] = self.flattened_observation_shape + 1
        mdn_kwargs["dim_out"] = self.flattened_observation_shape
        mdn_kwargs["n_components"] = mdn_kwargs.get("n_components", 1)
        mdn_kwargs["hidden_dim"] = mdn_kwargs.get("hidden_dim", 512)
        mdn_kwargs["pi_layers"] = mdn_kwargs.get("pi_layers", 2)
        mdn_kwargs["normal_layers"] = mdn_kwargs.get("normal_layers", 2)
        
        sample_obs = [th.tensor(self.shaping_preprocess_obs_fn(self.observation_space.sample())) for _ in range(mdn_kwargs["n_components"])]
        mdn_kwargs["mu_bias_init"] = th.cat(sample_obs, dim=0).reshape(-1)

        self.mdn_kwargs = mdn_kwargs
        self.mdn = MixtureDensityNetwork(**mdn_kwargs)
        self.mdn = self.mdn.to(self.device)
        if self.double_precision:
            self.mdn = self.mdn.double()
            self.dtype = th.float64
        else:
            self.mdn = self.mdn.float()
            self.dtype = th.float32
        
        # optimization hyperparameters
        self.initial_n_epochs = initial_n_epochs
        self.shaping_lr = shaping_lr
        self.shaping_n_epochs = shaping_n_epochs
        self.batch_size = batch_size
        self.use_lr_scheduler = use_lr_scheduler
        
        self.shaping_optimizer = th.optim.Adam(self.mdn.parameters(), lr=self.shaping_lr)
        
        # initialize mdn buffer
        self.mdn_buffer_size = mdn_buffer_size
        self.mdn_buffer = MDNDataset(max_len=mdn_buffer_size, max_horizon=self.horizon, transform_fn=shaping_preprocess_obs_fn, use_double_precision=double_precision)
        if shaping_path is not None and self.shaping_reward_weight > 0.0:
            MDNRewardWrapper.load_wrapper(self, shaping_path)
            self.mdn_update_freq = float("inf")
            
        self.mdn_dataloader = th.utils.data.DataLoader(self.mdn_buffer, batch_sampler=RepeatingSampler(self.mdn_buffer, batch_size=self.batch_size))
        self.mdn_dataloader = iter(self.mdn_dataloader)
        
        if len(args) > 0:
            print("args nobody asked for:", args)
        if len(kwargs) > 0:
            print("kwargs nobody asked for:", kwargs)
        
    def reset(self, seed=None, options=None):
        self.episode_num += 1
        if len(self.episode_obs_buffer) > 0:
            self.mdn_buffer.append(self.episode_obs_buffer)
        
        if self.episode_num > self.shaping_warmup_eps and self.episode_num % self.mdn_update_freq == 0:
            self.update_mdn()
            
        obs, info = super().reset(seed=seed, options=options)
        
        self.episode_obs_buffer = []
        self.episode_obs_buffer.append(obs)
        
        return obs, info
    
    def step(self, action):
        obs, reward, terminal, truncated, info = super().step(action)  
        self.episode_obs_buffer.append(obs)
        
        return obs, reward, terminal, truncated, info
    
    def get_shaping_reward(self, obs):
        obs = self.inference_preprocess_obs_fn(obs)
        
        entropies = np.zeros(obs.shape[0])
        for obs_idx in range(0, obs.shape[0], self.inference_batch_size):
            _obs = obs[obs_idx:obs_idx+self.inference_batch_size]
            _obs = th.tensor(_obs, dtype=self.dtype).repeat([1, self.query_horizon])
            _obs = _obs.reshape(-1, obs.shape[1]).to(self.device)
            
            _repeat_query_horizon = self._query_horizon.repeat([_obs.shape[0] // self.query_horizon, 1])
            context = th.cat([_obs, _repeat_query_horizon], dim=1)
            
            with th.no_grad():
                entropy_mean = self.mdn.entropy(context).detach()
                
            entropy_mean = entropy_mean.reshape(-1, self.query_horizon)
            if self.query_horizon != 1:
                entropy_mean = 0.5 * (entropy_mean[:, :-1] + entropy_mean[:, 1:]) * self._query_diffs.reshape(1, -1)
            entropy_mean = entropy_mean.sum(dim=1).cpu().numpy()
            entropies[obs_idx:obs_idx+self.inference_batch_size] = entropy_mean
            
        # take into account the dimensionality of the observation space
        # to ensure approaximately equal scaling
        entropies = entropies / self.flattened_observation_shape
        # TODO: add scaling that takes into account the size of the covariance matrix 
        # that the determinant is computed for. an useful notion the determinant per row, 
        # where the result is scaled by the product of l2 norms of the covariance matrix rows.
        
        # transform entropies so they are > 0
        entropies = np.where(entropies > 0, entropies + 1, np.exp(entropies))
        
        return entropies
    
    def update_mdn(self):
        n_epochs = self.initial_n_epochs if not self.mdn_updated else int(self.shaping_n_epochs * self.mdn_update_freq)
            
        if self.use_lr_scheduler:
            scheduler = th.optim.lr_scheduler.CosineAnnealingLR(self.shaping_optimizer, n_epochs)
        
        self.mdn.train()
        losses = []
        for _ in range(n_epochs):
            s0, h, s_h = next(self.mdn_dataloader)
            context = th.cat([s0, h], dim=1).to(self.device)
            s_h = s_h.to(self.device)
            
            self.shaping_optimizer.zero_grad()
            loss = self.mdn.loss(context, s_h).mean()
            
            loss.backward()
            self.shaping_optimizer.step()
            if self.use_lr_scheduler:
                scheduler.step()
            
            losses.append(loss.item())
            
        if self.verbose == 1:
            print("Mean MDN (shaper) loss", np.mean(losses))
            
        self.mdn_updated = True
        self.mdn.eval()

        self.master_log["shaper/loss_mdn"].append(np.mean(losses))

    def save_wrapper(self, path, return_snapshot=False):
        snapshot = {"mdn": self.mdn.state_dict(),
                    "mdn_updated": self.mdn_updated,
                    "episode_num": self.episode_num,
                    "mdn_buffer": self.mdn_buffer.buffer}
        
        if return_snapshot:
            return snapshot
        else:
            pickle.dump(snapshot, open(path, "wb"))

    def load_wrapper(self, path, return_snapshot=False):
        snapshot = super().load_wrapper(path, return_snapshot=True)
        self.mdn.load_state_dict(snapshot["mdn"])
        self.mdn_updated = snapshot["mdn_updated"]
        self.episode_num = snapshot["episode_num"]
        self.mdn_buffer.buffer = snapshot["mdn_buffer"]
        
        if return_snapshot:
            return snapshot
        
    def __getstate__(self) -> object:
        state = super().__getstate__().copy()
        del state["mdn"]
        del state["mdn_buffer"]
        del state["mdn_dataloader"]
        return state
    
    def __setstate__(self, state: object) -> None:
        self.__dict__.update(state)
        
        self.mdn = MixtureDensityNetwork(**self.mdn_kwargs)
        self.mdn = self.mdn.to(self.device)
        if self.double_precision:
            self.mdn = self.mdn.double()
        
        self.mdn_buffer = MDNDataset(max_len=self.mdn_buffer_size, max_horizon=self.horizon, transform_fn=self.shaping_preprocess_obs_fn, use_double_precision=self.double_precision)
        self.mdn_dataloader = th.utils.data.DataLoader(self.mdn_buffer, batch_sampler=RepeatingSampler(self.mdn_buffer, batch_size=self.batch_size))
        self.mdn_dataloader = iter(self.mdn_dataloader)
        

class CountBasedRewardWrapper(ShapingRewardWrapper):
    
    def __init__(self, env,
                 *args,
                 
                 count_lr=1e-2,
                 count_n_epochs=50,
                 count_model_update_freq=1,
                 count_buffer_size=5000,
                 count_model_kwargs=dict(),
                 
                 use_stable_nll=True,
                 stable_nll_iters=5,
                 batch_size=2 ** 9,
                 inference_batch_size=2 ** 9,
                 
                 shaping_path=None,
                 shaping_reward_weight=0.1,
                 shaping_warmup_eps=20,
                 shaping_preprocess_obs_fn=None,
                 inference_preprocess_obs_fn=None,
                 reuse_rollout_shaping=False,
                 
                 rollout_shaping_reward=True,
                 verbose=0,
                 device="cuda" if th.cuda.is_available() else "cpu",
                 double_precision=False,
                 
                 **kwargs):
        
        assert shaping_preprocess_obs_fn is not None, "The `shaping_preprocess_obs_fn` must be provided."
        super().__init__(env, shaping_reward_weight, rollout_shaping_reward, reuse_rollout_shaping, shaping_preprocess_obs_fn, inference_preprocess_obs_fn)
        
        self.verbose = verbose
        
        # episode related stats
        self.episode_num = 0
        self.episode_obs_buffer = []
        
        # set the device & precision
        self.device = th.device(device)
        self.double_precision = double_precision
        
        self.flattened_observation_shape = self.shaping_preprocess_obs_fn(self.observation_space.sample())[0].shape[0]
        
        # get the weight_model_kwargs in order
        count_model_kwargs["d"] = self.flattened_observation_shape
        count_model_kwargs["l"] = count_model_kwargs.get("l", 3) # Depth of the univariate density networks
        count_model_kwargs["r"] = count_model_kwargs.get("r", 3) # Width of the univariate density network
        count_model_kwargs["m"] = count_model_kwargs.get("m", 1000) # Width of MDMA
        count_model_kwargs["adaptive_coupling"] = count_model_kwargs.get("adaptive_coupling", True)
        self.count_model_kwargs = count_model_kwargs
        
        # initialize MDMA
        self.count_model = MDMA(**count_model_kwargs)
        self.count_model = self.count_model.to(self.device)
        if self.double_precision:
            self.count_model = self.count_model.double()
            self.dtype = th.float64
        else:
            self.count_model = self.count_model.float()
            self.dtype = th.float32

        self.stable_nll_iters = stable_nll_iters
        self.use_stable_nll = use_stable_nll
        
        # optimization hyperparameters & optimizer
        self.batch_size = batch_size
        self.inference_batch_size = inference_batch_size
        self.shaping_warmup_eps = shaping_warmup_eps
        self.count_model_update_freq = float(count_model_update_freq) if isinstance(count_model_update_freq, str) else count_model_update_freq
        self.count_lr = count_lr
        self.count_n_epochs = count_n_epochs
        self.count_optimizer = th.optim.Adam(self.count_model.parameters(), lr=count_lr, weight_decay=0, amsgrad=True)
        
        if shaping_path is not None:
            self.load_wrapper(self, shaping_path)
            self.count_model_update_freq = float("inf")
        
        # buffer initialization
        self.count_buffer_size = count_buffer_size
        self.total_steps_buffer = deque(maxlen=count_buffer_size)
        self.total_steps = sum(self.total_steps_buffer)
            
        self.state_buffer = StateDataset(max_len=count_buffer_size, transform_fn=self.shaping_preprocess_obs_fn, use_double_precision=self.double_precision)
        self.state_dataloader = th.utils.data.DataLoader(self.state_buffer, batch_sampler=RepeatingSampler(self.state_buffer, batch_size=self.batch_size))
        self.state_dataloader = iter(self.state_dataloader)
        
        if len(args) > 0:
            print("args nobody asked for:", args)
        if len(kwargs) > 0:
            print("kwargs nobody asked for:", kwargs)
        
    def reset(self, seed=None, options=None):
        self.episode_num += 1
        if len(self.episode_obs_buffer) > 0:
            self.state_buffer.append(self.episode_obs_buffer)
            self.total_steps_buffer.append(len(self.episode_obs_buffer))
            self.total_steps = sum(self.total_steps_buffer)
        
        if self.episode_num > self.shaping_warmup_eps and self.episode_num % self.count_model_update_freq == 0:
            self.update_count()
 
        obs, info = super().reset(seed=seed, options=options)
        # self.env_initialization_fn(super().reset, reset_kwargs=dict(seed=seed, options=options))
        
        self.episode_obs_buffer = []
        self.episode_obs_buffer.append(obs)
        
        return obs, info
    
    def step(self, action):
        obs, reward, terminal, truncated, info = super().step(action)  
        self.episode_obs_buffer.append(obs)
            
        return obs, reward, terminal, truncated, info
        
    def update_count(self):
        use_stable_nll = self.use_stable_nll
        losses = []
        
        self.count_model.train()
        
        for epoch_idx in range(self.count_n_epochs * self.count_model_update_freq):
            s = next(self.state_dataloader).to(self.device)

            if epoch_idx == self.stable_nll_iters:
                use_stable_nll = False

            self.count_optimizer.zero_grad()
            log_density = self.count_model.log_density(s, stabilize=use_stable_nll)
            
            loss = -log_density.mean()            
            if th.isnan(loss).any():
                print("NaN loss detected. Skipping the update.")
                break
            
            loss.backward()
            self.count_optimizer.step()

            losses.append(loss.item())
            
        if self.verbose == 1:
            print("Mean MDMA (counter) loss", np.mean(losses))
        
        self.count_model.eval()
        
        self.master_log["shaper/loss_count"].append(np.mean(losses))
    
    def get_shaping_reward(self, obs):
        obs = self.inference_preprocess_obs_fn(obs)
        densities = np.zeros(obs.shape[0])
        
        for obs_idx in range(0, obs.shape[0], self.inference_batch_size):
            _obs = obs[obs_idx:obs_idx+self.inference_batch_size]
            _obs = th.tensor(_obs, device=self.device, dtype=self.dtype)
            
            with th.no_grad():
                _densities = self.count_model.log_density(_obs).exp().cpu().numpy()
                
            densities[obs_idx:obs_idx+self.inference_batch_size] = _densities
                
        weights = (self.total_steps * densities + 1e-2) ** (-1/2)
        return weights
    
    def save_wrapper(self, path, return_snapshot=False):
        snapshot = {
            "count_model": self.count_model.state_dict(),
            "state_buffer": self.state_buffer.buffer,
            "episode_num": self.episode_num,
            "total_steps_buffer": self.total_steps_buffer
        }
                
        if return_snapshot:
            return snapshot
        else:
            pickle.dump(snapshot, open(path, "wb"))
    
    def load_wrapper(self, path, return_snapshot=False):
        snapshot = super().load_wrapper(path, return_snapshot=True)
        
        self.episode_num = snapshot["episode_num"]
        
        self.count_model.load_state_dict(snapshot["count_model"])
        self.state_buffer.buffer = snapshot["state_buffer"]
        
        self.total_steps_buffer = snapshot["total_steps_buffer"]
        self.total_steps = sum(snapshot["total_steps_buffer"])
        
        if return_snapshot:
            return snapshot
        
    def __getstate__(self):
        state =  super().__getstate__()
        del state["count_model"]
        del state["state_buffer"]
        del state["state_dataloader"]
        
        del state["total_steps_buffer"]
        return state
    
    def __setstate__(self, state):
        super().__setstate__(state)
        
        self.count_model = MDMA(**self.count_model_kwargs)
        self.count_model = self.count_model.to(self.device)
        
        self.state_buffer = StateDataset(max_len=self.count_buffer_size, transform_fn=self.shaping_preprocess_obs_fn, use_double_precision=self.double_precision)
        self.state_dataloader = th.utils.data.DataLoader(self.state_buffer, batch_sampler=RepeatingSampler(self.state_buffer, batch_size=self.batch_size))
        self.state_dataloader = iter(self.state_dataloader)
        
        self.total_steps_buffer = deque(maxlen=self.count_buffer_size)
        self.total_steps = sum(self.total_steps_buffer)


class EmpowermentRewardWrapper(ShapingRewardWrapper):
    
    def __init__(self, env, 
                 *args,
                 
                 gce_update_freq=1,
                 gce_kwargs={},
                 exponential_empowerment=True,
                 
                 water_filling_power=1,
                 water_filling_iters=10000,
                 water_filling_tolerance=0.001,
                 
                 reuse_rollout_shaping=False,
                 shaping_preprocess_obs_fn=None,
                 inference_preprocess_obs_fn=None,
                 shaping_path=None,
                 shaping_warmup_eps=1,
                 shaping_reward_weight=0.1, 
                 rollout_shaping_reward=True,
                 
                 learning_rate=1e-3,
                 batch_size=2 ** 10,
                 inference_batch_size=2 ** 10,
                 n_epochs=1000,
                 horizon=50,
                 buffer_size=5000,
                 
                 verbose=0,
                 device="cuda" if th.cuda.is_available() else "cpu",
                 double_precision=False,
                 **kwargs):
        
        assert shaping_preprocess_obs_fn is not None, "The `shaping_preprocess_obs_fn` must be provided."
        super().__init__(env, shaping_reward_weight, rollout_shaping_reward, reuse_rollout_shaping, shaping_preprocess_obs_fn, inference_preprocess_obs_fn)
        
        self.verbose = verbose
        self.maybe_exponential_empowerment_fn = th.exp if exponential_empowerment else lambda x: x
        self.water_filling_power = water_filling_power
        self.water_filling_iters = water_filling_iters
        self.water_filling_tolerance = water_filling_tolerance
        
        # episode related stats
        self.episode_num = 0
        self.episode_obs_buffer = []
        self.episode_act_buffer = []
        
        # set the device & precision
        self.device = th.device(device)
        self.double_precision = double_precision
        
        # initialize / load LatentGCE
        self.horizon = horizon
        
        self.flattened_observation_shape = self.shaping_preprocess_obs_fn(self.observation_space.sample())[0].shape[0]
        
        gce_kwargs["obs_dim"] = self.flattened_observation_shape
        gce_kwargs["action_dim"] = self.action_space.shape[0] * self.horizon
        self.gce_kwargs = gce_kwargs
        
        self.latent_gce = LatentGCE(**gce_kwargs)
        self.latent_gce = self.latent_gce.to(self.device)
        if self.double_precision:
            self.latent_gce = self.latent_gce.double()
            self.dtype = th.float64
        else:
            self.latent_gce = self.latent_gce.float()
            self.dtype = th.float32
        
        self.optimizer = th.optim.Adam(self.latent_gce.parameters(), lr=learning_rate)
        
        # initialize dataset & dataloader
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.inference_batch_size = inference_batch_size
        self.n_epochs = n_epochs
        self.gce_update_freq = gce_update_freq
        self.shaping_warmup_eps = float(shaping_warmup_eps)
        
        self.buffer_size = buffer_size
        self.buffer = EmpowermentDataset(max_len=self.buffer_size, 
                                         transform_fn=self.shaping_preprocess_obs_fn, 
                                         horizon=self.horizon,
                                         use_double_precision=double_precision)
        
        self.dataloader = th.utils.data.DataLoader(self.buffer, batch_sampler=RepeatingSampler(self.buffer, batch_size=self.batch_size))
        self.dataloader = iter(self.dataloader)
        
        if shaping_path is not None:
            self.load_wrapper(shaping_path)
            self.shaping_warmup_eps = float("inf")
        
        if len(args) > 0:
            print("args nobody asked for:", args)
        if len(kwargs) > 0:
            print("kwargs nobody asked for:", kwargs)
        
    def reset(self, seed=None, options=None):
        self.episode_num += 1
        if len(self.episode_obs_buffer) > 0:
            self.buffer.append(self.episode_obs_buffer, self.episode_act_buffer)
        
        if self.episode_num > self.shaping_warmup_eps and self.episode_num % self.gce_update_freq == 0:
            self.update_empowerment()
            
        obs, info = super().reset(seed=seed, options=options) 
        
        self.episode_act_buffer = []
        self.episode_obs_buffer = []
        self.episode_obs_buffer.append(obs)
        
        return obs, info
    
    def step(self, action):
        obs, reward, terminal, truncated, info = super().step(action)  
        self.episode_obs_buffer.append(obs)
        self.episode_act_buffer.append(action)
            
        return obs, reward, terminal, truncated, info
        
    def update_empowerment(self):
        self.latent_gce.train()
        losses = []
        for _ in range(self.n_epochs * self.gce_update_freq):
            s0, a, s_h = next(self.dataloader)
            s0 = s0.to(self.device)
            a = a.to(self.device)
            s_h = s_h.to(self.device)
            
            self.optimizer.zero_grad()
            loss = self.latent_gce.loss(s0, a, s_h)
            
            loss.backward()
            self.optimizer.step()
            
            losses.append(loss.item())
            
        if self.verbose == 1:
            print("Mean Empowerment (shaper) loss", np.mean(losses))
            
        self.latent_gce.eval()
            
        self.master_log["shaper/loss_empowerment"].append(np.mean(losses))
    
    def get_shaping_reward(self, obs):
        obs = self.inference_preprocess_obs_fn(obs)
        
        singular_values = th.zeros((obs.shape[0], min(self.flattened_observation_shape, self.action_space.shape[0] * self.horizon)))
            
        for obs_idx in range(0, obs.shape[0], self.inference_batch_size):
            _obs = obs[obs_idx:obs_idx+self.inference_batch_size]
            _obs = th.tensor(_obs, device=self.device, dtype=self.dtype)
            
            with th.no_grad():
                _singular_values = self.latent_gce.forward_svd(_obs).detach().cpu()
            singular_values[obs_idx:obs_idx+self.inference_batch_size] = _singular_values
            
        # batch water filling
        noise = 1 / (singular_values + 1e-6)
        water = th.min(noise, dim=1, keepdim=True).values + self.water_filling_power / singular_values.shape[1]
        for _ in range(self.water_filling_iters):
            current_power = th.clip(water - noise, min=0)
            current_power = current_power.sum(dim=1, keepdim=True)
            
            if th.abs(current_power - self.water_filling_power).max() < self.water_filling_tolerance:
                break
            else:
                water += (self.water_filling_power - current_power) / singular_values.shape[1]
            
        empowerments = 0.5 * th.log2(th.clip(water - noise, min=0) / noise + 1).sum(dim=1)
        empowerments = self.maybe_exponential_empowerment_fn(empowerments).cpu().numpy()
        
        return empowerments
    
    def save_wrapper(self, path, return_snapshot=False):
        snapshot = {"latent_gce": self.latent_gce.state_dict(),
                    "episode_num": self.episode_num,
                    "buffer": self.buffer.buffer}
        
        if return_snapshot:
            return snapshot
        else:
            pickle.dump(snapshot, open(path, "wb"))
    
    def load_wrapper(self, path, return_snapshot=False):
        snapshot = super().load_wrapper(path, return_snapshot=True)
        self.latent_gce.load_state_dict(snapshot["latent_gce"])
        self.episode_num = snapshot["episode_num"]
        self.buffer.buffer = snapshot["buffer"]
        
        if return_snapshot:
            return snapshot
    
    def __getstate__(self):
        state = super().__getstate__()
        del state["latent_gce"]
        del state["buffer"]
        del state["dataloader"]
        return state
    
    def __setstate__(self, state):
        super().__setstate__(state)
        
        self.latent_gce = LatentGCE(**self.gce_kwargs)
        self.latent_gce = self.latent_gce.to(self.device)
        if self.double_precision:
            self.latent_gce = self.latent_gce.double()
        
        self.buffer = EmpowermentDataset(max_len=self.buffer_size, 
                                         transform_fn=self.shaping_preprocess_obs_fn, 
                                         horizon=self.horizon,
                                         use_double_precision=self.double_precision)
        self.dataloader = th.utils.data.DataLoader(self.buffer, batch_sampler=RepeatingSampler(self.buffer, batch_size=self.batch_size))
        self.dataloader = iter(self.dataloader)


class DIAYNRewardWrapper(ShapingRewardWrapper):
    
    def __init__(self,
                 env,
                 *args,
                 
                 shaping_reward_weight=0.1,
                 shaping_preprocess_obs_fn=None,
                 inference_preprocess_obs_fn=None,
                 shaping_path=None,
                 
                 discriminator_kwargs=dict(),
                 reward_type="original",
                 
                 shaping_warmup_eps=1,
                 discriminator_update_freq=1,
                 batch_size=2 ** 10,
                 inference_batch_size=2 ** 10,
                 learning_rate=1e-3,
                 n_epochs=100,
                 buffer_size=5000,
                 
                 rollout_shaping_reward=True,
                 reuse_rollout_shaping=False,
                 
                 verbose=0,
                 device="cuda" if th.cuda.is_available() else "cpu",
                 double_precision=False,
                 **kwargs):
        
        assert shaping_preprocess_obs_fn is not None, "The `shaping_preprocess_obs_fn` must be provided."
        super().__init__(env, shaping_reward_weight, rollout_shaping_reward, reuse_rollout_shaping, shaping_preprocess_obs_fn, inference_preprocess_obs_fn)
        
        self.episode_num = 0
        self.episode_obs_buffer = []
        
        self.verbose = verbose
        self.device = device
        self.double_precision = double_precision
        
        obs_sample = self.observation_space.sample()
        assert isinstance(obs_sample, dict), "The observation space must be a Dict. Use the `DictObservationWrapper`."
        assert "z" in obs_sample, "The observation space must be a Dict with 'z' key in it. Use the `DummyZObservationWrapper`."
        self.flattened_observation_shape = self.shaping_preprocess_obs_fn(obs_sample)[0].shape[0]
        
        self.shaping_warmup_eps = shaping_warmup_eps
        self.discriminator_update_freq = discriminator_update_freq
        self.batch_size = batch_size
        self.inference_batch_size = inference_batch_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        
        assert reward_type in {"original", "entropy"}, "The `reward_type` must be either 'original' or 'entropy'. Got {}".format(reward_type)
        self.reward_type = reward_type
        
        discriminator_kwargs["in_dim"] = self.flattened_observation_shape
        discriminator_kwargs["out_dim"] = obs_sample["z"].shape[0]
        self.discriminator_kwargs = discriminator_kwargs
        
        self.discriminator = DIAYNDiscriminator(**self.discriminator_kwargs)
        self.discriminator = self.discriminator.to(self.device)
        
        if self.double_precision:
            self.discriminator = self.discriminator.double()
            self.dtype = th.float64
        else:
            self.discriminator = self.discriminator.float()
            self.dtype = th.float32
        
        self.optimizer = th.optim.Adam(self.discriminator.parameters(), lr=learning_rate)
        
        self.buffer_size = buffer_size
        self.buffer = DIAYNDataset(max_len=self.buffer_size, transform_fn=self.shaping_preprocess_obs_fn, use_double_precision=double_precision)
        
        self.dataloader = th.utils.data.DataLoader(self.buffer, batch_sampler=RepeatingSampler(self.buffer, batch_size=self.batch_size))
        self.dataloader = iter(self.dataloader)
        
        if shaping_path is not None:
            self.load_wrapper(shaping_path)
            self.shaping_warmup_eps = float("inf")
        
        if len(args) > 0:
            print("args nobody asked for:", args)
        if len(kwargs) > 0:
            print("kwargs nobody asked for:", kwargs)
        
    def reset(self, seed=None, options=None):
        self.episode_num += 1
        if len(self.episode_obs_buffer) > 0:
            self.buffer.append(self.episode_obs_buffer)
        
        if self.episode_num > self.shaping_warmup_eps and self.episode_num % self.discriminator_update_freq == 0:
            self.update_diayn()
            
        obs, info = super().reset(seed=seed, options=options) 
        
        self.episode_obs_buffer = []
        self.episode_obs_buffer.append(obs)
            
        return obs, info
    
    def step(self, action):
        obs, reward, terminal, truncated, info = super().step(action)  
        self.episode_obs_buffer.append(obs)
        
        return obs, reward, terminal, truncated, info
    
    def update_diayn(self):
        self.discriminator.train()
        losses = []
        for _ in range(self.n_epochs * self.discriminator_update_freq):
            s, z = next(self.dataloader)
            s = s.to(self.device)
            z = z.to(self.device)
            
            self.optimizer.zero_grad()
            loss = self.discriminator.loss(s, z)
            
            loss.backward()
            self.optimizer.step()
            
            losses.append(loss.item())
            
        if self.verbose == 1:
            print("Mean DIAYN (shaper) loss", np.mean(losses))
            
        self.discriminator.eval()
            
        self.master_log["shaper/loss_diayn"].append(np.mean(losses))
    
    def get_shaping_reward(self, obs):
        # env is wrapped in DictObservationWrapper, but does not necessarily have z in the obs     
        if self.reward_type == "original":           
            zs = obs["z"]
        obs = self.inference_preprocess_obs_fn(obs)
        
        if self.reward_type == "original":
            zs = zs.reshape(obs.shape[0], -1)
        
        rewards = th.zeros((obs.shape[0]))
        
        for obs_idx in range(0, obs.shape[0], self.inference_batch_size):
            _obs = obs[obs_idx:obs_idx+self.inference_batch_size]
            _obs = th.tensor(_obs, device=self.device, dtype=self.dtype)
            
            with th.no_grad():
                if self.reward_type == "original":
                    _rewards = th.nn.functional.log_softmax(self.discriminator(_obs), dim=-1).detach().cpu()
                    _zs = th.tensor(zs[obs_idx:obs_idx+self.inference_batch_size], device="cpu", dtype=self.dtype)
                    _rewards = (_rewards * _zs).sum(dim=-1) - math.log(1 / _zs.shape[1])
                else:
                    _rewards = self.discriminator.get_entropy(_obs).detach().cpu()
                    
            rewards[obs_idx:obs_idx+self.inference_batch_size] = _rewards

        return rewards.cpu().numpy()
    
    def save_wrapper(self, path, return_snapshot=False):
        snapshot = {"discriminator": self.discriminator.state_dict(),
                    "episode_num": self.episode_num,
                    "buffer": self.buffer.buffer}
        
        if return_snapshot:
            return snapshot
        else:
            pickle.dump(snapshot, open(path, "wb"))
    
    def load_wrapper(self, path, return_snapshot=False):
        snapshot = super().load_wrapper(path, return_snapshot=True)
        self.discriminator.load_state_dict(snapshot["discriminator"])
        self.episode_num = snapshot["episode_num"]
        self.buffer.buffer = snapshot["buffer"]
        
        if return_snapshot:
            return snapshot
    
    def __getstate__(self):
        state = super().__getstate__()
        del state["discriminator"]
        del state["buffer"]
        del state["dataloader"]
        return state
    
    def __setstate__(self, state):
        super().__setstate__(state)
        
        self.discriminator = DIAYNDiscriminator(**self.discriminator_kwargs)
        self.discriminator = self.discriminator.to(self.device)
        if self.double_precision:
            self.discriminator = self.discriminator.double()
        
        self.buffer = DIAYNDataset(max_len=self.buffer_size, transform_fn=self.shaping_preprocess_obs_fn, use_double_precision=self.double_precision)
        self.dataloader = th.utils.data.DataLoader(self.buffer, batch_sampler=RepeatingSampler(self.buffer, batch_size=self.batch_size))
        self.dataloader = iter(self.dataloader)
    
