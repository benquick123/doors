import warnings


def warn(message, category='', stacklevel=1, source=''):
    conditions = [
        category == DeprecationWarning,
        message.startswith("rich is experimental/alpha"),
    ]
    if any(conditions):
        return None
    else:
        print(message, category, stacklevel, source, sep=" | ")

warnings.warn = warn

import os

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from env_utils.make_env import get_make_env_fn
from env_utils.wrappers import ShapingRewardWrapper
from rl.behavioral_cloning import behavioral_cloning
from utils.callback import CustomLoggingCallback
from utils.configuration import get_config, parse_args
from utils.utils import load_for_train
from utils.logging import save_self

if __name__ == "__main__":
    args, remaining_args = parse_args()
    
    config = get_config(args.config_path, args, remaining_args)
    save_self(args.config_path, args, remaining_args, config)
    
    make_env_fn = get_make_env_fn(config["env_name"], 
                                  config["wrappers"] + [config["shaper_class"]], 
                                  config["wrapper_kwargs"] + [config["shaper_kwargs"]])

    env = make_vec_env(make_env_fn, n_envs=1, seed=config["seed"], vec_env_cls=SubprocVecEnv, env_kwargs=config["env_kwargs"])
    
    callback = None
    if config["log"]:
        from stable_baselines3.common.callbacks import EvalCallback
        
        make_eval_env_fn = get_make_env_fn(config["env_name"], config["eval_wrappers"], config["eval_wrapper_kwargs"])
        eval_env = make_vec_env(make_eval_env_fn, n_envs=1, seed=config["seed"], vec_env_cls=SubprocVecEnv, env_kwargs=config["env_kwargs"])
        callback = [EvalCallback(eval_env=eval_env, warn=False, **config["eval_kwargs"]), CustomLoggingCallback(**config["logging_kwargs"])]
    
    learner = config["learner_class"](env=env, **config["learner_kwargs"])
    
    if args.continue_from is not None:
        learner = load_for_train(learner, env, config, args.continue_from, args.continue_mode, args.continue_what)
    learner.replay_buffer.set_env(env)
    
    if args.behavioral_cloning:
        behavioral_cloning(learner, env, args.behavioral_cloning, config["train_kwargs"].get("progress_bar", False))

    try:
        learner.learn(callback=callback, **config["train_kwargs"])
    except KeyboardInterrupt:
        print("Keyboard interrupt.")
    finally:        
        if config["log"]:
            learner.save(os.path.join(config["log_path"], config["log"], "final.zip"))
            learner.save_replay_buffer(os.path.join(config["log_path"], config["log"], "replay_buffer.pkl"))
            
            if env.env_is_wrapped(ShapingRewardWrapper, indices=[0])[0]:
                # assuming n_envs == 1
                env.env_method("save_wrapper", path=os.path.join(config["log_path"], config["log"], "reward_shaper.pkl"))
    