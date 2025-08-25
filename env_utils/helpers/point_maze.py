import numpy as np


def full_initialization_fn(env, reset_kwargs):
    if "options" not in reset_kwargs or reset_kwargs["options"] is None:
        reset_kwargs["options"] = {}
        
    obs_dict, info = env.reset(**reset_kwargs)
    env.unwrapped.point_env.init_qvel[:2] = np.random.uniform(-5.0, 5.0, size=(2,))
    
    obs, info = env.unwrapped.point_env.reset(seed=reset_kwargs["seed"])
    obs_dict["observation"] = obs
    
    return obs_dict, info


def preprocess_obs_fn(x):
    observation = x["observation"].copy()
    
    if len(observation.shape) == 1:
        observation = observation.reshape(1, -1)
        
    return observation


def preprocess_and_slice_obs_fn(x):
    observation = x["observation"].copy()
    
    if len(observation.shape) == 1:
        observation = observation.reshape(1, -1)
        
    return observation[:, :2]  # Only return the first two elements of the observation
