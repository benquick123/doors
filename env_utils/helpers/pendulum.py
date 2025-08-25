import numpy as np


INITIAL_STATE_THRESHOLD_PERC = 1 / 50
INITIAL_STATE = np.array([np.pi, 0])


def full_initialization_fn(env, reset_kwargs):
    if "options" not in reset_kwargs or reset_kwargs["options"] is None:
        reset_kwargs["options"] = {}
        
    # for initialization across whole state space
    reset_kwargs["options"]["x_init"] = np.pi
    reset_kwargs["options"]["y_init"] = 8
    return env.reset(**reset_kwargs)


def downward_initialization_fn(env, reset_kwargs):
    if "options" not in reset_kwargs or reset_kwargs["options"] is None:
            reset_kwargs["options"] = {}
        
    if not "seed" in reset_kwargs:
        reset_kwargs["seed"] = 0
    
    reset_kwargs["options"]["x_init"] = np.abs(INITIAL_STATE[0]) + 2 * np.pi * INITIAL_STATE_THRESHOLD_PERC
    reset_kwargs["options"]["y_init"] = np.abs(INITIAL_STATE[1]) + 16 * INITIAL_STATE_THRESHOLD_PERC
    
    initial_state = np.array([np.cos(INITIAL_STATE[0]), np.sin(INITIAL_STATE[0]), INITIAL_STATE[1]])
    
    in_bounds = False
    
    while not in_bounds:
        obs, info = env.reset(**reset_kwargs)
        if reset_kwargs["seed"] is not None:
            reset_kwargs["seed"] += 1
        
        angle_condition_cos = initial_state[0] - (2 * INITIAL_STATE_THRESHOLD_PERC) < obs[0] < initial_state[0] + (2 * INITIAL_STATE_THRESHOLD_PERC)
        angle_condition_sin = initial_state[1] - (2 * INITIAL_STATE_THRESHOLD_PERC) < obs[1] < initial_state[1] + (2 * INITIAL_STATE_THRESHOLD_PERC)
        velocity_condition = initial_state[2] - (16 * INITIAL_STATE_THRESHOLD_PERC) < obs[2] < initial_state[2] + (16 * INITIAL_STATE_THRESHOLD_PERC)
        
        if angle_condition_cos and angle_condition_sin and velocity_condition:
            in_bounds = True

    return obs, info


def preprocess_obs_fn(x):    
    if len(x.shape) == 1:
        x = x.copy().reshape(1, -1)
        
    # add random noise to vary the radius of the angle inferred from the sin and cos
    r = np.random.rand(x.shape[0], 1)
    x[:, :2] *= (2 * r) + 1e-8
    
    return x
