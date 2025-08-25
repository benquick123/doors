import copy
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
import torch as th
from gymnasium import Space
from numpy import ndarray
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples, ReplayBufferSamples
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from torch import device
from torch._C import device
from torch.utils.data import Dataset, Sampler


class RepeatingSampler(Sampler):
    
    def __init__(self, data_source, batch_size) -> None:
        self.data_source = data_source
        
        assert isinstance(batch_size, int) and batch_size > 0, "batch_size must be provided."
        self.batch_size = batch_size
        
    def __iter__(self):
        while True:
            yield np.random.randint(0, len(self.data_source), self.batch_size)
    
    def __len__(self):
        return float("inf")


class MDNDataset(Dataset):
    
    def __init__(self, buffer=None, max_len=None, max_horizon=1, transform_fn=lambda x : x, use_double_precision=False) -> None:
        if buffer is None:
            assert max_len is not None, "If buffer is None, `max_len` must be provided."
            buffer = deque(maxlen=max_len)
        self.buffer = buffer
        self.transform_fn = transform_fn
        self.max_horizon = max_horizon
        self.dtype = th.float64 if use_double_precision else th.float32
        
    def __len__(self) -> int:
        return len(self.buffer)
    
    def __getitem__(self, index) -> Any:
        ep_buffer = self.buffer[index]
        s0_index = np.random.randint(0, len(ep_buffer) - 1)
        
        s0 = ep_buffer[s0_index]
        h = np.random.randint(s0_index + 1, min(s0_index+self.max_horizon, len(ep_buffer))) - s0_index
        # h = np.random.randint(1, min(len(ep_buffer), self.max_horizon) - s0_index)
        
        s_h_index = np.random.randint(s0_index+1, s0_index+h+1)
        s_h = ep_buffer[s_h_index]
        
        s0 = self.transform_fn(s0)[0]
        s_h = self.transform_fn(s_h)[0]
        
        s0 = th.tensor(s0, dtype=self.dtype)
        h = th.ones(1, dtype=self.dtype) * (h / self.max_horizon)
        s_h = th.tensor(s_h, dtype=self.dtype)
        
        return s0, h, s_h
    
    def append(self, element) -> None:
        element = np.stack(element, axis=0)
        self.buffer.append(element)


class StateDataset(MDNDataset):
        
    def __getitem__(self, index) -> Any:
        ep_buffer = self.buffer[index]
        
        # the main difference from MDNDataset is that we sample from the WHOLE episode:
        s_index = np.random.randint(0, len(ep_buffer))
        
        s = ep_buffer[s_index]
        s = self.transform_fn(s)[0]
        s = th.tensor(s, dtype=self.dtype)
        return s


class EmpowermentDataset(Dataset):
    
    def __init__(self, buffer=None, max_len=None, horizon=10, transform_fn=lambda x : x, use_double_precision=False) -> None:
        if buffer is None:
            assert max_len is not None, "If buffer is None, `max_len` must be provided."
            buffer = deque(maxlen=max_len)
        self.buffer = buffer
        self.horizon = horizon
        self.transform_fn = transform_fn
        self.dtype = th.float64 if use_double_precision else th.float32
        
    def __len__(self) -> int:
        return len(self.buffer)
    
    def __getitem__(self, index) -> Any:
        observations, actions = self.buffer[index]
        
        s0_index = np.random.randint(0, len(observations) - self.horizon)
        
        s0 = observations[s0_index]
        s_h = observations[s0_index + self.horizon]
        a = actions[s0_index:s0_index+self.horizon].reshape(-1)
        
        s0 = self.transform_fn(s0)[0]
        s_h = self.transform_fn(s_h)[0]
        
        s0 = th.tensor(s0, dtype=self.dtype)
        a = th.tensor(a, dtype=self.dtype)
        s_h = th.tensor(s_h, dtype=self.dtype)
        
        return s0, a, s_h
    
    def append(self, observations, actions) -> None:
        assert len(observations) == len(actions) + 1 
        if len(actions) < self.horizon:
            return
        
        observations = np.stack(observations, axis=0)
        actions = np.stack(actions, axis=0)
        self.buffer.append((observations, actions))
    
    
class DIAYNDataset(Dataset):
    def __init__(self, buffer=None, max_len=None, transform_fn=lambda x : x, use_double_precision=False) -> None:
        if buffer is None:
            assert max_len is not None, "If buffer is None, `max_len` must be provided."
            buffer = deque(maxlen=max_len)
        self.buffer = buffer
        self.transform_fn = transform_fn
        self.dtype = th.float64 if use_double_precision else th.float32
        
    def __len__(self) -> int:
        return len(self.buffer)
    
    def __getitem__(self, index) -> Any:
        obs_ep_buffer = self.buffer[index]
        
        s_index = np.random.randint(0, len(obs_ep_buffer))
        
        s = obs_ep_buffer[s_index]
        z = th.tensor(s["z"], dtype=self.dtype)
        
        s = self.transform_fn(s)[0]
        s = th.tensor(s, dtype=self.dtype)
        
        return s, z
    
    def append(self, observations):
        observations = np.stack(observations, axis=0)
        self.buffer.append((observations))
        

class ChangingRewardReplayBuffer(ReplayBuffer):
    
    def __init__(self, 
                 buffer_size: int, 
                 observation_space: Space, 
                 action_space: Space, 
                 device: device | str = "auto", 
                 n_envs: int = 1, 
                 vec_env: VecEnv | None = None,
                 optimize_memory_usage: bool = False, 
                 handle_timeout_termination: bool = True):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage, handle_timeout_termination)
        
        self.vec_env = vec_env
        self.infos = np.array([[{} for _ in range(self.n_envs)] for _ in range(self.buffer_size)])
        
    def add(self, obs: ndarray, next_obs: ndarray, action: ndarray, reward: ndarray, done: ndarray, infos: List[Dict[str, Any]]) -> None:
        self.infos[self.pos] = infos
        return super().add(obs, next_obs, action, reward, done, infos)
    
    def _get_samples(self, batch_inds: ndarray, env: VecNormalize | None = None) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        obs = self.observations[batch_inds, env_indices, :]
        infos = self.infos[batch_inds, env_indices]
        
        rewards =  self.vec_env.env_method("compute_reward", obs, next_obs, infos, indices=[0])
        rewards = rewards[0].astype(np.float32)  # env_method returns a list containing one element

        data = (
            self._normalize_obs(obs, env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(rewards.reshape(-1, 1), env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))
    
    def __getstate__(self):
        state = self.__dict__.copy()
        del state["vec_env"]
        return state
    
    def __setstate__(self, state):
        assert "vec_env" not in state
        self.__dict__.update(state)
        self.vec_env = None
        
    def set_env(self, env):
        self.vec_env = env
        

class ChangingRewardHerReplayBuffer(HerReplayBuffer):
    
    def __init__(self, *args, **kwargs):
        kwargs["copy_info_dict"] = True
        super().__init__(*args, **kwargs)
        
    def _get_real_samples(self, batch_indices, env_indices, env = None):
        obs = {key: obs[batch_indices, env_indices, :] for key, obs in self.observations.items()}
        next_obs = {key: obs[batch_indices, env_indices, :] for key, obs in self.next_observations.items()}
        if self.copy_info_dict:
            # The copy may cause a slow down
            infos = copy.deepcopy(self.infos[batch_indices, env_indices])
        else:
            infos = [{} for _ in range(len(batch_indices))]
            
        assert (
            self.env is not None
        ), "You must initialize HerReplayBuffer with a VecEnv so it can compute rewards for virtual transitions"
        
        # recompute rewards since the aux. reward could be changing all the time
        rewards = self.env.env_method("compute_reward", obs, next_obs, infos, indices=[0])
        rewards = rewards[0].astype(np.float32)
        obs = self._normalize_obs(obs, env)  # type: ignore[assignment]
        next_obs = self._normalize_obs(next_obs, env)  # type: ignore[assignment]

        assert isinstance(obs, dict)
        assert isinstance(next_obs, dict)
        # Convert to torch tensor
        observations = {key: self.to_torch(obs) for key, obs in obs.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs.items()}

        return DictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_indices, env_indices]),
            next_observations=next_observations,
            dones=self.to_torch(self.dones[batch_indices, env_indices] * (1 - self.timeouts[batch_indices, env_indices])).reshape(-1, 1),
            rewards=self.to_torch(self._normalize_reward(rewards.reshape(-1, 1), env)),
        )
    
    def _get_virtual_samples(
        self,
        batch_indices: np.ndarray,
        env_indices: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> DictReplayBufferSamples:
        # Get infos and obs
        obs = {key: obs[batch_indices, env_indices, :] for key, obs in self.observations.items()}
        next_obs = {key: obs[batch_indices, env_indices, :] for key, obs in self.next_observations.items()}
        
        if len(batch_indices) == 0:
            return DictReplayBufferSamples(
                observations={key: self.to_torch(obs) for key, obs in obs.items()},
                actions=self.to_torch(self.actions[batch_indices, env_indices]),
                next_observations={key: self.to_torch(obs) for key, obs in next_obs.items()},
                dones=self.to_torch(self.dones[batch_indices, env_indices] * (1 - self.timeouts[batch_indices, env_indices])).reshape(-1, 1),
                rewards=self.to_torch(self._normalize_reward(np.zeros((0, 1), dtype=np.float32), env)),
            )
        
        if self.copy_info_dict:
            # The copy may cause a slow down
            infos = copy.deepcopy(self.infos[batch_indices, env_indices])
        else:
            infos = [{} for _ in range(len(batch_indices))]
            
        # Sample and set new goals
        new_goals = self._sample_goals(batch_indices, env_indices)
        obs["desired_goal"] = new_goals
        
        # The desired goal for the next observation must be the same as the previous one
        next_obs["desired_goal"] = new_goals

        assert (
            self.env is not None
        ), "You must initialize HerReplayBuffer with a VecEnv so it can compute rewards for virtual transitions"
        
        # Compute new reward
        if len(batch_indices) > 0:
            # in case we don't want virtual samples, HerReplayBuffer acts as a DictReplayBuffer
            rewards = self.env.env_method("compute_reward", obs, next_obs, infos, indices=[0])
            rewards = rewards[0].astype(np.float32)
        else:
            rewards = np.zeros((0, ), dtype=np.float32)
            
        obs = self._normalize_obs(obs, env)
        next_obs = self._normalize_obs(next_obs, env)

        # Convert to torch tensor
        observations = {key: self.to_torch(obs) for key, obs in obs.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs.items()}

        return DictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_indices, env_indices]),
            next_observations=next_observations,
            dones=self.to_torch(self.dones[batch_indices, env_indices] * (1 - self.timeouts[batch_indices, env_indices])).reshape(-1, 1),
            rewards=self.to_torch(self._normalize_reward(rewards.reshape(-1, 1), env)),
        )
        
    def set_env(self, env):
        # override this method so it doesn't raise an error upon reinitialization of the env.
        self.env = env