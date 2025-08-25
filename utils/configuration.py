import argparse
import re
import json
import os
from datetime import datetime
from utils.utils import get_module

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="", help="Path to experiment config.")
    parser.add_argument("--continue_from", type=str, default=None, help="Path to the log folder to continue training from.")
    parser.add_argument("--continue_mode", type=str, default="final", choices=["final", "best"], help="How to continue training; using last or best previous model.")
    parser.add_argument("--continue_what", default=["all"], nargs="+", help="What to continue from the previous model.")
    parser.add_argument("--behavioral_cloning", type=int, default=0, help="How many episodes to initialize the replay buffer with using the loaded policy. Value 0 disables behavioral cloning.")
    args, remaining_args = parser.parse_known_args()
    return args, remaining_args


def config_merge(config_main, config_other):
    # entries from config_main have a priority over values from config_other
    
    assert isinstance(config_main, dict)
    assert isinstance(config_other, dict)
    
    all_keys = set(config_main.keys()).union(config_other.keys())
    config = {}
    for key in all_keys:
        if key in config_other and key in config_main:
            if isinstance(config_main[key], dict):
                config[key] = config_merge(config_main[key], config_other[key])
            else:
                config[key] = config_main[key]
                print(f"Found two `{key}` keys in configs. Value after merging: {config[key]}")
        elif key in config_main:
            config[key] = config_main[key]
        else:
            config[key] = config_other[key]
            
    return config


def load_config(path):
    config_main = json.load(open(path, "r"))
    
    for key, value in list(config_main.items()):
        if isinstance(value, list):
            for _value in list(value):
                if isinstance(_value, str) and _value.endswith(".json"):
                    sub_config = load_config(_value)
                    config_main = config_merge(config_main, sub_config)
                    del config_main[key]
            
        elif isinstance(value, dict):
            for _value in list(value.values()):
                if isinstance(_value, str) and _value.endswith(".json"):
                    sub_config = load_config(_value)
                    config_main[key] = config_merge(config_main[key], sub_config)
                    del config_main[key]
                
        elif isinstance(value, str) and value.endswith(".json"):
            sub_config = load_config(value)
            config_main = config_merge(config_main, sub_config)
            del config_main[key]
            
    return config_main


def parse_cmd_args(remaining_args):
    _remaining_args = []
    arg_name_indices = [i for i, arg in enumerate(remaining_args) if arg.startswith("--")]
    for i_curr, i_next in zip(arg_name_indices, arg_name_indices[1:] + [len(remaining_args)]):
        arg = remaining_args[i_curr]
        arg_value = " ".join(remaining_args[i_curr+1:i_next])
        _remaining_args += [arg, arg_value]
    return _remaining_args


def update_config_from_cmd(config, remaining_args):
    # go through remaining args
    for i in range(0, len(remaining_args), 2): # zip(arg_name_indices, arg_name_indices[1:] + [len(remaining_args)]):
        arg = remaining_args[i].replace("--", "").split(".")
        arg_value = str(remaining_args[i+1])
    
        _current = config
        # from pprint import pprint
        # pprint(config)
        for _arg in arg[:-1]:
            if _arg.isdigit():
                _arg = int(_arg)
                assert len(_current) > _arg, "Provided index is out of range."
            elif _arg not in _current:
                _current[_arg] = dict()
                
            _current = _current[_arg]
        
        # take care of some basic type conversions
        try:
            arg_value = eval(arg_value)
        except:
            print(f"Couldn't parse remaining arg `{remaining_args[i]}` ({arg_value}). Will keep as str.")
        
        if isinstance(_current, list):
            arg[-1] = int(arg[-1])
        _current[arg[-1]] = arg_value
    
    return config


def parse_modules(config):
    for key in config.keys():
        if isinstance(config[key], dict):
            config[key] = parse_modules(config[key])
        elif isinstance(config[key], list):
            for i, value in enumerate(config[key]):
                if isinstance(value, dict):
                    config[key][i] = parse_modules(value)
                elif isinstance(value, str):
                    try:
                        config[key][i] = get_module(value)
                    except:
                        pass
        elif isinstance(config[key], str):
            try:
                config[key] = get_module(config[key])
            except:
                pass
    return config


def get_config(path, args, remaining_args):
    prev_remaining_args = None
    if args.continue_from is not None and "all" in args.continue_what:
        args.config_path = path = os.path.join(args.continue_from, "config_original.json")
        prev_args = json.load(open(os.path.join(args.continue_from, "args.json"), "r"))
        prev_remaining_args = prev_args["remaining_args"]
    
    config = load_config(path)
        
    if prev_remaining_args is not None:
        config = update_config_from_cmd(config, remaining_args=parse_cmd_args(prev_remaining_args))
    remaining_args = parse_cmd_args(remaining_args)
    config = update_config_from_cmd(config, remaining_args=remaining_args)

    ##### seeds #####
    config["seed"] = config.get("seed", np.random.randint(0, 99999999))
    config["learner_kwargs"]["seed"] = config["seed"] + 1
    
    ##### number of steps vs. buffer_size #####
    config["learner_kwargs"]["buffer_size"] = min(config["learner_kwargs"].get("buffer_size", float("inf")), config["train_kwargs"]["total_timesteps"])
    
    ##### logging #####
    if config["log"]:
        ##### construct log path #####
        if isinstance(config["log"], bool):
            config["log"] = str(datetime.now()).split(".")[0].replace(":", "-").replace(" ", "_")
        config["log_path"] = os.path.join(config.get("log_path", "."), config["env_name"].split("/")[-1].split(":")[-1], config["shaper_class"].split(".")[-1], config["learner_class"].split(".")[-1])
        
        if len(remaining_args) > 0:
            # make a custom folder name based on cmd arguments; ignore --log since it is already part of the folder name.
            add_to_filename = [arg_name[2:].split(".")[-1] + "=" + re.sub(r'["\'\[\]{}]', "", value.replace("/", "-")) for arg_name, value in zip(remaining_args[0::2], remaining_args[1::2]) if not arg_name[2:] == "log"]
            add_to_filename.sort(key=lambda x : len(x))
            config["log"] += "_" + "_".join(add_to_filename)
            config["log"] = config["log"].replace(" ", "")
            config["log"] = config["log"][:255] # max path length in Linux is 255 characters
        
        # config["tb_log_name"] = experiment_id
        config["learner_kwargs"]["tensorboard_log"] = os.path.join(config["log_path"], config["log"], "tb")
        
        ##### deal with evaluation #####
        config["eval_kwargs"] = config.get("eval_kwargs", dict())
        config["eval_kwargs"]["log_path"] = os.path.join(config["log_path"], config["log"], "evaluations")
        config["eval_kwargs"]["best_model_save_path"] = os.path.join(config["log_path"], config["log"], "evaluations")
    
        ##### callback args #####
        config["logging_kwargs"] = config.get("logging_kwargs", dict())
        config["logging_kwargs"]["reward_shaper_save_path"] = os.path.join(config["log_path"], config["log"], "reward_shaper")
                
    ##### learner cls #####
    config["learner_kwargs"]["replay_buffer_kwargs"] = config["learner_kwargs"].get("replay_buffer_kwargs", dict())
        
    if "train_freq" in config["learner_kwargs"] and isinstance(config["learner_kwargs"]["train_freq"], list):
        config["learner_kwargs"]["train_freq"] = tuple(config["learner_kwargs"]["train_freq"])
        
    ##### wrappers #####
    config["wrappers"] = config.get("wrappers", list())
    assert len(config["wrappers"]) == len(config["wrapper_kwargs"]), "Number of wrappers and wrapper_kwargs should be the same."
    
    ##### eval wrappers #####
    config["eval_wrappers"] = config.get("eval_wrappers", list())
    assert len(config["eval_wrappers"]) == len(config["eval_wrapper_kwargs"]), "Number of eval_wrappers and eval_wrapper_kwargs should be the same."
    
    ##### ensure env_kwargs #####
    config["env_kwargs"] = config.get("env_kwargs", dict())
    
    ##### parse modules #####
    config = parse_modules(config)
    
    return config
