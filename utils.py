import gymnasium as gym
import numpy as np
import torch
import random

from tianshou.env import ShmemVectorEnv, VectorEnvNormObs
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from gymnasium.wrappers import TimeLimit

def gen_env(env_name: str):
    env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name + "-v2-goal-observable"](seed=0)
    env.seeded_rand_vec = True
    env = TimeLimit(env, max_episode_steps=500)
    return env

def make_tianshou_metaworld_env(tasks, seed=42, training_num=1, test_num=1, obs_norm=True):
    training_envs = []
    testing_envs = []
    for task in tasks:
        env = gen_env(task)
        # train_envs = ShmemVectorEnv(
        #     [lambda: gen_env(task) for _ in range(training_num)]
        # )
        # test_envs = ShmemVectorEnv([lambda: gen_env(task) for _ in range(test_num)])
        env.unwrapped.seed(seed)
        # train_envs.seed(seed)
        # test_envs.seed(seed)
        # if obs_norm:
        #     # obs norm wrapper
        #     train_envs = VectorEnvNormObs(train_envs)
        #     test_envs = VectorEnvNormObs(test_envs, update_obs_rms=False)
        #     test_envs.set_obs_rms(train_envs.get_obs_rms())
        training_envs.append(env)
        testing_envs.append(env)
    return training_envs, testing_envs

def make_metaworld_env(env_ids, seed = 42):
    """ example usage: env_id = ['push-back-v2', 'sweep-into-v2', 'window-close-v2']
    """
    # meta-world env setup:
    import metaworld
    random.seed(seed)
    ml = metaworld.ML10(seed=seed)
    print(f"Setting up Meta-World ML45 env: {env_ids} with seed={seed}")

    training_envs = []
    testing_envs = []

    for phase in ['train', 'test']:
        classes = ml.train_classes if phase == 'train' else ml.test_classes
        tasks = ml.train_tasks if phase == 'train' else ml.test_tasks

        for name, env_cls in classes.items():
            if name in env_ids and phase == 'train':
                env = env_cls()
                task = random.choice([task for task in tasks if task.env_name == name])
                env.set_task(task)
                training_envs.append(env)
                print(f"ML45 Train env name: {name} initialized with action space {env.action_space}")
            elif phase == 'test':
                testing_envs.append(env)
                print(f"ML45 Test env name: {name} initialized with action space {env.action_space}")
    return training_envs, testing_envs
        
