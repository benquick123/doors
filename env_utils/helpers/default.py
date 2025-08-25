

def preprocess_obs_fn(x):
    if isinstance(x, dict):
        x = x["observation"].copy()
        
    if len(x.shape) == 1:
        x = x.copy().reshape(1, -1)
    return x

def initialization_fn(env, reset_kwargs):
    return env.reset(**reset_kwargs)