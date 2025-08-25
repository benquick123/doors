import numpy as np


MAX_STEPS = 100


def full_initialization_fn(env, reset_kwargs, min_object_distance=0.0):
    _, info = env.reset(**reset_kwargs)
    
    # Randomize start position of object.
    object_qpos = env.unwrapped._utils.get_joint_qpos(env.unwrapped.model, env.unwrapped.data, "object0:joint")
    object_qpos[:2] = env.unwrapped.initial_gripper_xpos[:2] + env.unwrapped.np_random.uniform(-env.unwrapped.obj_range, env.unwrapped.obj_range, size=2)
    env.unwrapped._utils.set_joint_qpos(env.unwrapped.model, env.unwrapped.data, "object0:joint", object_qpos)

    # env.unwrapped._mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
    obj_distance = 0.0
    while obj_distance <= min_object_distance:
        desired_gripper_xpos = env.unwrapped.initial_gripper_xpos + env.unwrapped.np_random.uniform(-env.unwrapped.target_range, env.unwrapped.target_range, size=3)
        obj_distance = np.linalg.norm(desired_gripper_xpos - object_qpos[:3])
    desired_gripper_xvel = np.zeros(3)
    desired_gripper_openess = np.random.rand(1) * 2 -1
    
    obs = env.unwrapped._get_obs()
    current_gripper_xpos = obs["observation"][0:3]
    current_gripper_xvel = obs["observation"][20:23] 

    xpos_diff = np.linalg.norm(desired_gripper_xpos - current_gripper_xpos)
    xvel_diff = np.linalg.norm(desired_gripper_xvel - current_gripper_xvel)
    
    n_steps = 0
    while xpos_diff > env.unwrapped.distance_threshold or xvel_diff > env.unwrapped.distance_threshold:
        action = np.concatenate([desired_gripper_xpos - current_gripper_xpos, desired_gripper_openess])
        action = np.clip(action, env.action_space.low, env.action_space.high)
        
        env.unwrapped._set_action(action)
        env.unwrapped._mujoco_step(action)
    
        obs = env.unwrapped._get_obs()
        current_gripper_xpos = obs["observation"][0:3]
        current_gripper_xvel = obs["observation"][20:23]
        
        xpos_diff = np.linalg.norm(desired_gripper_xpos - current_gripper_xpos)
        xvel_diff = np.linalg.norm(desired_gripper_xvel - current_gripper_xvel)
        
        n_steps += 1
        if n_steps > MAX_STEPS:
            break
        
    return obs, info


def constrained_initialization_fn(env, reset_kwargs):
    # max movement per step is 0.05 in x, y, z
    # we put in a larger value to make sure this becomes more difficult to compute for short horizon.
    min_object_distance = 3 * (3 * (0.05 ** 2)) ** 0.5
    return full_initialization_fn(env, reset_kwargs, min_object_distance=min_object_distance)


def preprocess_obs_fn(x):
    observation = x["observation"].copy()
    
    if len(observation.shape) == 1:
        observation = observation.reshape(1, -1)
        
    # first 6 elements are the robot EE's and object's positions.
    # these have absolute values, but are redundant to estimate the value of door(s).
    # observation[:, 6:9], on the other hand, are the differences between these two quantities.
    # however, the latter is invariant to EE and object position, and more useful in our scenario.
    observation = observation[:, 6:]
        
    return observation