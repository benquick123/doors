from collections import deque
from tqdm import tqdm

import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
            
            
def populate_replay_buffer(learner: OffPolicyAlgorithm, env: VecEnv, n_episodes: int, progress_bar: bool = False) -> None:
    """
    Collect experiences and store them into a ``ReplayBuffer``.

    :param learner: The algorithm used to collect the data
    :param env: The training environment
    :param n_episodes: Number of episodes to collect
    :return:
    """
    
    # Initialize buffers if they don't exist, or reinitialize if resetting counters
    learner.ep_info_buffer = deque(maxlen=learner._stats_window_size)
    learner.ep_success_buffer = deque(maxlen=learner._stats_window_size)
    
    # Initialize _last_obs before starting
    learner._last_obs = env.reset()  # type: ignore[assignment]
    learner._last_episode_starts = np.ones((env.num_envs,), dtype=bool)
    # Retrieve unnormalized observation for saving into the buffer
    if learner._vec_normalize_env is not None:
        learner._last_original_obs = learner._vec_normalize_env.get_original_obs()
    
    # Switch to eval mode (this affects batch norm / dropout)
    learner.policy.set_training_mode(False)

    num_collected_steps, num_collected_episodes = 0, 0

    assert isinstance(env, VecEnv), "You must pass a VecEnv"

    if learner.use_sde:
        learner.actor.reset_noise(env.num_envs)

    print("Collecting data for behavioral cloning...")
    if progress_bar:
        t = tqdm(total=n_episodes, unit="episode")

    while num_collected_episodes < n_episodes:
        if learner.use_sde and learner.sde_sample_freq > 0 and num_collected_steps % learner.sde_sample_freq == 0:
            # Sample a new noise matrix
            learner.actor.reset_noise(env.num_envs)

        # Select action randomly or according to policy
        actions, buffer_actions = learner._sample_action(0, learner.action_noise, env.num_envs)

        # Rescale and perform action
        new_obs, rewards, dones, infos = env.step(actions)
        num_collected_steps += 1

        # Retrieve reward and episode length if using Monitor wrapper
        learner._update_info_buffer(infos, dones)

        # Store data in replay buffer (normalized action and unnormalized observation)
        learner._store_transition(learner.replay_buffer, buffer_actions, new_obs, rewards, dones, infos)  # type: ignore[arg-type]

        # For DQN, check if the target network should be updated
        # and update the exploration schedule
        # For SAC/TD3, the update is dones as the same time as the gradient update
        # see https://github.com/hill-a/stable-baselines/issues/900
        learner._on_step()

        for idx, done in enumerate(dones):
            if done:
                # Update stats
                num_collected_episodes += 1

                if learner.action_noise is not None:
                    kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                    learner.action_noise.reset(**kwargs)
                    
                if progress_bar:
                    t.update(1)
        
    return num_collected_steps

def behavioral_cloning(learner, env, n_episodes, progress_bar=False):
    n_steps = populate_replay_buffer(learner, env, n_episodes, progress_bar)
    return