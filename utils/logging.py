import contextlib
import copy
import json
import os

from utils.configuration import load_config


def save_self(config_path, args, remaining_args, config):
    if not config["log"]:
        return 
    
    path = os.path.join(config["log_path"], config["log"])
    os.makedirs(path)
    os.makedirs(os.path.join(path, "code"))
    ignore = set(open(".gitignore", "r").read().split("\n") + [".git", "configs", "ipynbs"])
    for root, dirs, files in os.walk("."):
        if any([ignore_string in root for ignore_string in ignore]):
            continue
        
        for file in files:
            if any([ignore_string in file for ignore_string in ignore]):
                continue
            os.makedirs(os.path.join(path, "code", root), exist_ok=True)
            os.system("cp '%s' '%s'" % (os.path.join(root, file), os.path.join(path, "code", root, file)))
        # print(root, dirs, files)

    def check_recursive(dictionary):
        for k, v in dictionary.items():
            if isinstance(v, dict):
                dictionary[k] = check_recursive(v)
                
            elif isinstance(v, list) or isinstance(v, tuple) or isinstance(v, set):
                dictionary[k] = list(v)
                for i, el in enumerate(v):
                    dictionary[k][i] = check_recursive({"k": el})["k"]
                
            elif isinstance(v, int) or isinstance(v, float) or isinstance(v, str) or isinstance(v, bool):
                continue
            
            else:
                dictionary[k] = str(v)
        return dictionary
    
    config_dict = check_recursive(copy.deepcopy(config))

    json.dump(config_dict, open(os.path.join(path, "config.json"), "w"), indent=4, sort_keys=True)
    
    # save the original config
    with contextlib.redirect_stdout(None):
        original_config = load_config(config_path)
    json.dump(original_config, open(os.path.join(path, "config_original.json"), "w"), indent=4, sort_keys=True)
    
    # save args and remaining args
    dict_args = {
        "args": vars(args),
        "remaining_args": list(remaining_args)
    }
    json.dump(dict_args, open(os.path.join(path, "args.json"), "w"), indent=4, sort_keys=True)    
    
    # save the environment yml
    os.system('conda env export | grep -v "^prefix: " > %s' % os.path.join(path, "environment.yml"))