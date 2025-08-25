import os
import multiprocessing as mp
import itertools
from time import sleep

def run_experiment(command):
    print("START EXPERIMENT:", command)
    os.system(command)
    print("END EXPERIMENT:", command)


if __name__ == "__main__":
    commands = [
        ########################################## STAGE 1 ##########################################
        ############################### Precompute heuristic rewards: ###############################
        ### Pendulum-v1 ###
        ##  Door(s)
        # "python train.py --config_path configs/pretrain_configs/mdn_random_pendulum.json --train_kwargs.total_timesteps 10000 --env_kwargs.max_episode_steps 210 --shaper_kwargs.mdn_update_freq 48 --shaper_kwargs.initial_n_epochs 10000 --shaper_kwargs.horizon 200 --shaper_kwargs.mdn_kwargs.n_components 1",
        ##  Empowerment
        # "python train.py --config_path configs/pretrain_configs/empowerment_random_pendulum.json --train_kwargs.total_timesteps 10000 --env_kwargs.max_episode_steps 210 --shaper_kwargs.gce_update_freq 48 --shaper_kwargs.n_epochs 210 --shaper_kwargs.horizon 8 --shaper_kwargs.gce_kwargs.action_encoder False",
        # "python train.py --config_path configs/pretrain_configs/empowerment_random_pendulum.json --train_kwargs.total_timesteps 10000 --env_kwargs.max_episode_steps 210 --shaper_kwargs.gce_update_freq 48 --shaper_kwargs.n_epochs 210 --shaper_kwargs.horizon 200 --shaper_kwargs.gce_kwargs.action_encoder True",
        # "python train.py --config_path configs/pretrain_configs/empowerment_random_pendulum.json --train_kwargs.total_timesteps 1000000 --env_kwargs.max_episode_steps 210 --shaper_kwargs.gce_update_freq 4761 --shaper_kwargs.n_epochs 21  --shaper_kwargs.horizon 200 --shaper_kwargs.gce_kwargs.action_encoder True",
        ##  DIAYN
        # "python train.py --config_path configs/pretrain_configs/diayn_random_pendulum.json --train_kwargs.total_timesteps 200000",
        
        ### PointMaze_LargeDense-v3
        ##  Door(s)
        # "python train.py --config_path configs/pretrain_configs/mdn_random_point_maze_large.json --train_kwargs.total_timesteps 1000000 --shaper_kwargs.mdn_update_freq 1251 --shaper_kwargs.initial_n_epochs 100000 --shaper_kwargs.horizon 500",
        ##  Empowerment (has action_encoder=True by default)
        # "python train.py --config_path configs/pretrain_configs/empowerment_random_point_maze_large.json --train_kwargs.total_timesteps 1000000 --shaper_kwargs.gce_update_freq 1251 --shaper_kwargs.n_epochs 80 --shaper_kwargs.horizon 500",
        # "python train.py --config_path configs/pretrain_configs/empowerment_random_point_maze_large.json --train_kwargs.total_timesteps 1000000 --shaper_kwargs.gce_update_freq 1251 --shaper_kwargs.n_epochs 80 --shaper_kwargs.horizon 100",
        # "python train.py --config_path configs/pretrain_configs/empowerment_random_point_maze_large.json --train_kwargs.total_timesteps 1000000 --shaper_kwargs.gce_update_freq 1251 --shaper_kwargs.n_epochs 80 --shaper_kwargs.horizon 50 --wrappers [\\'env_utils.wrappers.PointMazeWrapper\\', \\'env_utils.wrappers.CustomResetFnWrapper\\', \\'env_utils.wrappers.SparseRewardWrapper\\'] --wrapper_kwargs [{\\'frame_skip\\': 10}, {\\'env_initialization_fn\\': \\'env_utils.helpers.point_maze.full_initialization_fn\\'}, {\\'reward_threshold\\': 0.6376281516}]",
        
        ### Fetch{PickAndPlace|Push|Slide}Dense-v4
        ##  Door(s)
        # "python train.py --config_path configs/pretrain_configs/mdn_random_fetch_pick_and_place.json --shaper_kwargs.mdn_update_freq 20001 --shaper_kwargs.initial_n_epochs 100000 --shaper_kwargs.horizon 32",
        # "python train.py --config_path configs/pretrain_configs/mdn_random_fetch_push.json --shaper_kwargs.mdn_update_freq 20001 --shaper_kwargs.initial_n_epochs 100000 --shaper_kwargs.horizon 32",
        # "python train.py --config_path configs/pretrain_configs/mdn_random_fetch_slide.json --shaper_kwargs.mdn_update_freq 20001 --shaper_kwargs.initial_n_epochs 100000 --shaper_kwargs.horizon 32",
        ##  Empowerment (has horizon=32 by default)
        # "python train.py --config_path configs/pretrain_configs/empowerment_random_fetch_pick_and_place.json --shaper_kwargs.gce_update_freq 20001 --shaper_kwargs.n_epochs 5 --shaper_kwargs.gce_kwargs.action_encoder False",
        # "python train.py --config_path configs/pretrain_configs/empowerment_random_fetch_push.json --shaper_kwargs.gce_update_freq 20001 --shaper_kwargs.n_epochs 5 --shaper_kwargs.gce_kwargs.action_encoder False",
        # "python train.py --config_path configs/pretrain_configs/empowerment_random_fetch_slide.json --shaper_kwargs.gce_update_freq 20001 --shaper_kwargs.n_epochs 5 --shaper_kwargs.gce_kwargs.action_encoder False",
        # "python train.py --config_path configs/pretrain_configs/empowerment_random_fetch_pick_and_place.json --shaper_kwargs.gce_update_freq 20001 --shaper_kwargs.n_epochs 5 --shaper_kwargs.gce_kwargs.action_encoder False --shaper_kwargs.horizon 8 --train_kwargs.total_timesteps 1000000",
        # "python train.py --config_path configs/pretrain_configs/empowerment_random_fetch_push.json --shaper_kwargs.gce_update_freq 20001 --shaper_kwargs.n_epochs 5 --shaper_kwargs.gce_kwargs.action_encoder False --shaper_kwargs.horizon 8 --train_kwargs.total_timesteps 1000000",
        # "python train.py --config_path configs/pretrain_configs/empowerment_random_fetch_slide.json --shaper_kwargs.gce_update_freq 20001 --shaper_kwargs.n_epochs 5 --shaper_kwargs.gce_kwargs.action_encoder False --shaper_kwargs.horizon 8 --train_kwargs.total_timesteps 1000000",
        ##  DIAYN
        # "python train.py --config_path configs/pretrain_configs/diayn_random_fetch_pick_and_place.json --train_kwargs.total_timesteps 1000000 --shaper_kwargs.discriminator_update_freq 20001 --shaper_kwargs.n_epochs 5",
        # "python train.py --config_path configs/pretrain_configs/diayn_random_fetch_push.json --train_kwargs.total_timesteps 1000000 --shaper_kwargs.discriminator_update_freq 20001 --shaper_kwargs.n_epochs 5",
        # "python train.py --config_path configs/pretrain_configs/diayn_random_fetch_slide.json --train_kwargs.total_timesteps 1000000 --shaper_kwargs.discriminator_update_freq 20001 --shaper_kwargs.n_epochs 5",
        
        ########################################### STAGE 2 ##########################################
        ############################ Pretrain with heuristic rewards: ################################
        ### Pendulum-v1 ###
        # "python train.py --config_path configs/mdn_tqc_pendulum.json --wrapper_kwargs.1.reward_threshold 10.0 --log sparse --learner_kwargs.learning_starts 1000 --shaper_kwargs.shaping_reward_weight 0.0 --shaper_kwargs.mdn_kwargs.n_components 1 --shaper_kwargs.reuse_rollout_shaping True",
        # "python train.py --config_path configs/mdn_tqc_pendulum.json --wrapper_kwargs.1.reward_threshold 10.0 --learner_kwargs.learning_starts 1000 --shaper_kwargs.shaping_reward_weight 1.0 --shaper_kwargs.mdn_kwargs.n_components 1 --shaper_kwargs.reuse_rollout_shaping True",
        # "python train.py --config_path configs/empowerment_tqc_pendulum.json --wrapper_kwargs.1.reward_threshold 10.0 --learner_kwargs.learning_starts 1000 --shaper_kwargs.shaping_reward_weight 1.0 --shaper_kwargs.gce_kwargs.action_encoder False --shaper_kwargs.reuse_rollout_shaping True",
        
        ### PointMaze_LargeDense-v3 ###
        # "python train.py --config_path configs/mdn_tqc_point_maze_large.json --wrapper_kwargs.1.reward_threshold 10.0 --log sparse --shaper_kwargs.shaping_reward_weight 0.0 --shaper_kwargs.reuse_rollout_shaping True",
        # "python train.py --config_path configs/mdn_tqc_point_maze_large.json --wrapper_kwargs.1.reward_threshold 10.0 --shaper_kwargs.shaping_reward_weight 1.0 --shaper_kwargs.reuse_rollout_shaping True",
        # "python train.py --config_path configs/empowerment_tqc_point_maze_large_h=100.json --wrapper_kwargs.1.reward_threshold 10.0 --shaper_kwargs.shaping_reward_weight 1.0 --shaper_kwargs.reuse_rollout_shaping True",
        # "python train.py --config_path configs/empowerment_tqc_point_maze_large_h=500.json --wrapper_kwargs.1.reward_threshold 10.0 --shaper_kwargs.shaping_reward_weight 1.0 --shaper_kwargs.reuse_rollout_shaping True",
        # "python train.py --config_path configs/count_tqc_point_maze_large.json --wrapper_kwargs.1.reward_threshold 10.0 --shaper_kwargs.shaping_reward_weight 1.0",
        
        ### Fetch{PickAndPlace|Push|Slide}Dense-v4 ###
        ### Add flags for reward thresholds.
        # "python train.py --config_path configs/mdn_tqc_fetch_pick_and_place.json --shaper_kwargs.shaping_reward_weight 1.0 --wrapper_kwargs.1.reward_threshold 10.0 --shaper_kwargs.horizon 32 --train_kwargs.total_timesteps 1000000 --shaper_kwargs.reuse_rollout_shaping True",
        # "python train.py --config_path configs/empowerment_tqc_fetch_pick_and_place_h=32.json --shaper_kwargs.shaping_reward_weight 1.0 --wrapper_kwargs.1.reward_threshold 10.0 --shaper_kwargs.gce_kwargs.action_encoder False --train_kwargs.total_timesteps 1000000 --shaper_kwargs.reuse_rollout_shaping True",
        # "python train.py --config_path configs/empowerment_tqc_fetch_pick_and_place_h=8.json --shaper_kwargs.shaping_reward_weight 1.0 --wrapper_kwargs.1.reward_threshold 10.0 --shaper_kwargs.gce_kwargs.action_encoder False --train_kwargs.total_timesteps 1000000 --shaper_kwargs.reuse_rollout_shaping True",
        # "python train.py --config_path configs/count_tqc_fetch_pick_and_place.json --shaper_kwargs.shaping_reward_weight 1.0 --wrapper_kwargs.1.reward_threshold 10.0 --train_kwargs.total_timesteps 1000000",
        # "python train.py --config_path configs/diayn_tqc_fetch_pick_and_place.json --shaper_kwargs.shaping_reward_weight 1.0 --wrapper_kwargs.1.reward_threshold 10.0 --train_kwargs.total_timesteps 1000000",
        
        # "python train.py --config_path configs/mdn_tqc_fetch_push.json --shaper_kwargs.shaping_reward_weight 1.0 --wrapper_kwargs.1.reward_threshold 10.0 --shaper_kwargs.horizon 32 --shaper_kwargs.reuse_rollout_shaping True",
        # "python train.py --config_path configs/empowerment_tqc_fetch_push_h=32.json --shaper_kwargs.shaping_reward_weight 1.0 --wrapper_kwargs.1.reward_threshold 10.0 --shaper_kwargs.gce_kwargs.action_encoder False --shaper_kwargs.reuse_rollout_shaping True",
        # "python train.py --config_path configs/empowerment_tqc_fetch_push_h=8.json --shaper_kwargs.shaping_reward_weight 1.0 --wrapper_kwargs.1.reward_threshold 10.0 --shaper_kwargs.gce_kwargs.action_encoder False --train_kwargs.total_timesteps 1000000 --shaper_kwargs.reuse_rollout_shaping True",
        # "python train.py --config_path configs/count_tqc_fetch_push.json --shaper_kwargs.shaping_reward_weight 1.0 --wrapper_kwargs.1.reward_threshold 10.0",
        # "python train.py --config_path configs/diayn_tqc_fetch_push.json --shaper_kwargs.shaping_reward_weight 1.0 --wrapper_kwargs.1.reward_threshold 10.0 --train_kwargs.total_timesteps 1000000",
        
        # "python train.py --config_path configs/mdn_tqc_fetch_slide.json --shaper_kwargs.shaping_reward_weight 1.0 --wrapper_kwargs.1.reward_threshold 10.0 --shaper_kwargs.horizon 32 --shaper_kwargs.reuse_rollout_shaping True",
        # "python train.py --config_path configs/empowerment_tqc_fetch_slide_h=32.json --shaper_kwargs.shaping_reward_weight 1.0 --wrapper_kwargs.1.reward_threshold 10.0 --shaper_kwargs.gce_kwargs.action_encoder False --shaper_kwargs.reuse_rollout_shaping True",
        # "python train.py --config_path configs/empowerment_tqc_fetch_slide_h=8.json --shaper_kwargs.shaping_reward_weight 1.0 --wrapper_kwargs.1.reward_threshold 10.0 --shaper_kwargs.gce_kwargs.action_encoder False --train_kwargs.total_timesteps 1000000 --shaper_kwargs.reuse_rollout_shaping True",
        # "python train.py --config_path configs/count_tqc_fetch_slide.json --shaper_kwargs.shaping_reward_weight 1.0 --wrapper_kwargs.1.reward_threshold 10.0",
        # "python train.py --config_path configs/diayn_tqc_fetch_slide.json --shaper_kwargs.shaping_reward_weight 1.0 --wrapper_kwargs.1.reward_threshold 10.0 --train_kwargs.total_timesteps 1000000",
        
        ########################################## STAGE 3 ##########################################
        ### Pendulum-v1 ###
        ##  Naive baselines:
        # "python train.py --config_path configs/dense_tqc_pendulum.json",                                                                                                              # dense baseline
        # "python train.py --config_path configs/mdn_tqc_pendulum.json --log sparse --shaper_kwargs.shaping_reward_weight 0.0",                                                         # sparse baseline
        
        ### PointMaze_LargeDense-v3 ###
        ##  Naive baselines:
        # "python train.py --config_path configs/dense_tqc_point_maze_large.json",                                                                                                      # dense baseline
        # "python train.py --config_path configs/mdn_tqc_point_maze_large.json --log sparse --shaper_kwargs.shaping_reward_weight 0.0",                                                 # sparse baseline
        
        ### Fetch{PickAndPlace|Push|Slide}Dense-v4 ###
        ## Naive baselines:
        # "python train.py --config_path configs/dense_tqc_fetch_pick_and_place.json",                                                                                                  # dense baseline
        # "python train.py --config_path configs/mdn_tqc_fetch_pick_and_place.json --log sparse --shaper_kwargs.shaping_reward_weight 0.0",                                             # sparse baseline
        # "python train.py --config_path configs/dense_tqc_fetch_push.json",                                                                                                            # dense baseline
        # "python train.py --config_path configs/mdn_tqc_fetch_push.json --log sparse --shaper_kwargs.shaping_reward_weight 0.0",                                                       # sparse baseline
        # "python train.py --config_path configs/dense_tqc_fetch_slide.json --train_kwargs.total_timesteps 1000000",                                                                    # dense baseline
        # "python train.py --config_path configs/mdn_tqc_fetch_slide.json --log sparse --shaper_kwargs.shaping_reward_weight 0.0 --train_kwargs.total_timesteps 1000000",               # sparse baseline
        
        ################################## Train with heuristics: ###################################
        ### uncomment continue_what below
        
        ### Fetch{PickAndPlace|Push|Slide}Dense-v4 ###
        ##  Door(s)
        # "python train.py --config_path configs/mdn_tqc_fetch_pick_and_place.json --continue_from {path_to_pretrain_results} --learner_kwargs.learning_starts 1 --shaper_kwargs.shaping_reward_weight 0.0 --shaper_kwargs.mdn_update_freq inf --shaper_kwargs.rollout_shaping_reward False --behavioral_cloning 100",
        # "python train.py --config_path configs/mdn_tqc_fetch_push.json --continue_from {path_to_pretrain_results} --learner_kwargs.learning_starts 1 --shaper_kwargs.shaping_reward_weight 0.0 --shaper_kwargs.mdn_update_freq inf --shaper_kwargs.rollout_shaping_reward False --behavioral_cloning 100",
        # "python train.py --config_path configs/mdn_tqc_fetch_slide.json --continue_from {path_to_pretrain_results} --learner_kwargs.learning_starts 1 --shaper_kwargs.shaping_reward_weight 0.0 --shaper_kwargs.mdn_update_freq inf --shaper_kwargs.rollout_shaping_reward False --behavioral_cloning 100 --train_kwargs.total_timesteps 1000000",
        
        ##  Empowerment:
        # "python train.py --config_path configs/empowerment_tqc_fetch_pick_and_place_h=32.json --log horizon=32/ --continue_from {path_to_pretrain_results} --learner_kwargs.learning_starts 1 --shaper_kwargs.shaping_reward_weight 0.0 --shaper_kwargs.gce_update_freq inf --shaper_kwargs.rollout_shaping_reward False --shaper_kwargs.gce_kwargs.action_encoder False --behavioral_cloning 100",
        # "python train.py --config_path configs/empowerment_tqc_fetch_push_h=32.json --log horizon=32/ --continue_from {path_to_pretrain_results} --learner_kwargs.learning_starts 1 --shaper_kwargs.shaping_reward_weight 0.0 --shaper_kwargs.gce_update_freq inf --shaper_kwargs.rollout_shaping_reward False --shaper_kwargs.gce_kwargs.action_encoder False --behavioral_cloning 100",
        # "python train.py --config_path configs/empowerment_tqc_fetch_slide_h=32.json --log horizon=32/ --continue_from {path_to_pretrain_results} --learner_kwargs.learning_starts 1 --shaper_kwargs.shaping_reward_weight 0.0 --shaper_kwargs.gce_update_freq inf --shaper_kwargs.rollout_shaping_reward False --shaper_kwargs.gce_kwargs.action_encoder False --behavioral_cloning 100 --train_kwargs.total_timesteps 1000000",
        # "python train.py --config_path configs/empowerment_tqc_fetch_pick_and_place_h=8.json --log horizon=8/ --continue_from {path_to_pretrain_results} --learner_kwargs.learning_starts 1 --shaper_kwargs.shaping_reward_weight 0.0 --shaper_kwargs.gce_update_freq inf --shaper_kwargs.rollout_shaping_reward False --shaper_kwargs.gce_kwargs.action_encoder False --behavioral_cloning 100",
        # "python train.py --config_path configs/empowerment_tqc_fetch_push_h=8.json --log horizon=8/ --continue_from {path_to_pretrain_results} --learner_kwargs.learning_starts 1 --shaper_kwargs.shaping_reward_weight 0.0 --shaper_kwargs.gce_update_freq inf --shaper_kwargs.rollout_shaping_reward False --shaper_kwargs.gce_kwargs.action_encoder False --behavioral_cloning 100",
        # "python train.py --config_path configs/empowerment_tqc_fetch_slide_h=8.json --log horizon=8/ --continue_from {path_to_pretrain_results} --learner_kwargs.learning_starts 1 --shaper_kwargs.shaping_reward_weight 0.0 --shaper_kwargs.gce_update_freq inf --shaper_kwargs.rollout_shaping_reward False --shaper_kwargs.gce_kwargs.action_encoder False --behavioral_cloning 100 --train_kwargs.total_timesteps 1000000",
        
        ##  Count-based:
        # "python train.py --config_path configs/count_tqc_fetch_pick_and_place.json --continue_from {path_to_pretrain_results} --learner_kwargs.learning_starts 1 --shaper_kwargs.shaping_reward_weight 0.0 --shaper_kwargs.count_model_update_freq inf --shaper_kwargs.rollout_shaping_reward False --behavioral_cloning 100",
        # "python train.py --config_path configs/count_tqc_fetch_push.json --continue_from {path_to_pretrain_results} --learner_kwargs.learning_starts 1 --shaper_kwargs.shaping_reward_weight 0.0 --shaper_kwargs.count_model_update_freq inf --shaper_kwargs.rollout_shaping_reward False --behavioral_cloning 100",
        # "python train.py --config_path configs/count_tqc_fetch_slide.json --continue_from {path_to_pretrain_results} --learner_kwargs.learning_starts 1 --shaper_kwargs.shaping_reward_weight 0.0 --shaper_kwargs.count_model_update_freq inf --shaper_kwargs.rollout_shaping_reward False --behavioral_cloning 100 --train_kwargs.total_timesteps 1000000",
        
        ##  DIAYN:
        # "python train.py --config_path configs/diayn_tqc_fetch_pick_and_place.json --continue_from {path_to_pretrain_results} --learner_kwargs.learning_starts 1 --shaper_kwargs.shaping_reward_weight 0.0 --shaper_kwargs.rollout_shaping_reward False --behavioral_cloning 100",
        # "python train.py --config_path configs/diayn_tqc_fetch_push.json --continue_from {path_to_pretrain_results} --learner_kwargs.learning_starts 1 --shaper_kwargs.shaping_reward_weight 0.0 --shaper_kwargs.rollout_shaping_reward False --behavioral_cloning 100",
        # "python train.py --config_path configs/diayn_tqc_fetch_slide.json --continue_from {path_to_pretrain_results} --learner_kwargs.learning_starts 1 --shaper_kwargs.shaping_reward_weight 0.0 --shaper_kwargs.rollout_shaping_reward False --behavioral_cloning 100 --train_kwargs.total_timesteps 1000000",
    ]

    n_seeds = 10
    start_seed = 0
    max_processes = 16
    gpus = [0, 1, 2, 3]
    
    seeds = [("--seed", value) for value in range(start_seed, start_seed+n_seeds)]
    # continue_what = [("--continue_what", value) for value in ["actor"]]
    
    processes = []
    command_idx = 0
    for arguments in itertools.product(seeds):
        for command in commands:
            _command = command + " " + " ".join([f"{arg} {value}" for arg, value in arguments])
            _command = f"CUDA_VISIBLE_DEVICES={gpus[command_idx % len(gpus)]} " + _command
            p = mp.Process(target=run_experiment, args=(_command, ))
            p.start()
            
            processes.append(p)
            if len(processes) >= max_processes:
                for p in processes:
                    p.join()
                processes = []
                
            command_idx += 1
            sleep(0.05)
            
    if len(processes) > 0:
        for p in processes:
            p.join()
        processes = []
