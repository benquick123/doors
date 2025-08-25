import gymnasium as gym


def get_make_env_fn(env_id, wrappers, wrapper_kwargs):
    
    def make_env_fn(**env_kwargs):
        env = gym.make(env_id, **env_kwargs)
        for wrapper, _wrapper_kwargs in zip(wrappers, wrapper_kwargs):
            env = wrapper(env, **_wrapper_kwargs)
        return env
    
    return make_env_fn
