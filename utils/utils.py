import importlib
import json
import os

import torch as th
from torch import nn

from env_utils.wrappers import ShapingRewardWrapper


def get_module(path):
    path, target = path.rsplit(".", 1)
    master_module = importlib.import_module(path)
    return master_module.__dict__[target]


def load_for_test(path, env_kwargs={}, learner_path=None, reward_shaper_path=None, test_config=True, load_from=".", custom_cmd_args=[]):
    if load_from != ".":
        import sys
        sys.path.insert(0, os.path.join(load_from, "code"))
    
    from env_utils.make_env import get_make_env_fn
    from utils.configuration import get_config, parse_args
    
    args, remaining_args = parse_args()
    args.config_path = os.path.join(path, "config_original.json")
    remaining_args = json.load(open(os.path.join(path, "args.json"), "r"))["remaining_args"]
    for arg_idx in range(0, len(custom_cmd_args), 2):
        arg = custom_cmd_args[arg_idx]
        value = custom_cmd_args[arg_idx + 1]
        try:
            existing_idx = remaining_args.index(arg)
            remaining_args[existing_idx + 1] = value
        except ValueError:
            remaining_args += [arg, value]
    
    config = get_config(args.config_path, args, remaining_args)
    
    if test_config:
        wrappers = config["eval_wrappers"]
        wrapper_kwargs = config["eval_wrapper_kwargs"]
    else:
        config["shaper_kwargs"]["shaping_path"] = None
        wrappers = config["wrappers"] + [config["shaper_class"]]
        wrapper_kwargs = config["wrapper_kwargs"] + [config["shaper_kwargs"]]

    make_env_fn = get_make_env_fn(config["env_name"], wrappers, wrapper_kwargs)

    if "env_kwargs" not in config:
        config["env_kwargs"] = dict()
    _env_kwargs = dict(config["env_kwargs"])
    _env_kwargs.update(env_kwargs)
    env = make_env_fn(**_env_kwargs)
    
    if learner_path is None:
        learner = config["learner_class"](env=env, **config["learner_kwargs"])
    else:
        learner = config["learner_class"].load(learner_path, env=env)
    
    if not test_config and reward_shaper_path is not None:
        env.load_wrapper(reward_shaper_path)
        env.mdn_update_freq = float("inf")
        
    if load_from != ".":
        sys.path.pop(0)
    
    return learner, env


def load_for_train(learner, env, config, continue_from, continue_mode, continue_what):
    if continue_mode == "final":
        param_load_path = os.path.join(continue_from, "final.zip")
    elif continue_mode == "best":
        param_load_path = os.path.join(continue_from, "evaluations", "best_model.zip")
        
    if "all" in continue_what:
        learner = learner.load(param_load_path, env=env)
    else:
        # Load the entire model first
        prev_learner = config["learner_class"].load(param_load_path, env=env)
        
        # Selectively load only the policy parameters
        ignore_keys = {"replay_buffer", "reward_shaper"} # replay buffer and reward shaper are loaded later, separately
        for key in continue_what:
            if key in ignore_keys:
                continue
            
            module_keys = key.split(".")
            
            maybe_continue_module = prev_learner
            maybe_learner_module = learner
            for module_key in module_keys[:-1]:
                maybe_continue_module = getattr(maybe_continue_module, module_key)
                maybe_learner_module = getattr(maybe_learner_module, module_key)
            maybe_continue_module = getattr(maybe_continue_module, module_keys[-1])
            maybe_learner_module = getattr(maybe_learner_module, module_keys[-1])
            
            assert maybe_learner_module.__class__ == maybe_continue_module.__class__, f"Module {key} is of different type in the two models."
            if isinstance(maybe_learner_module, nn.Module):
                maybe_learner_module.load_state_dict(maybe_continue_module.state_dict())
            elif isinstance(maybe_learner_module, th.Tensor):
                maybe_learner_module.data.copy_(maybe_continue_module.data)
            else:
                raise NotImplementedError(f"Module {key} is of type {type(maybe_learner_module)}")
    
    if "all" in continue_what or "replay_buffer" in continue_what:
        learner.load_replay_buffer(os.path.join(continue_from, "replay_buffer.pkl"))
        # update n_sampled_goal.
        n_sampled_goal = config["learner_kwargs"]["replay_buffer_kwargs"].get("n_sampled_goal", learner.replay_buffer.n_sampled_goal)
        learner.replay_buffer.n_sampled_goal = n_sampled_goal
        learner.replay_buffer.her_ratio = 1 - (1.0 / (n_sampled_goal + 1))
    
    if ("all" in continue_what or "reward_shaper" in continue_what) and env.env_is_wrapped(ShapingRewardWrapper):
        env.env_method("load_wrapper", path=os.path.join(continue_from, "reward_shaper.pkl"))
            
    learner.set_env(env)
    return learner

