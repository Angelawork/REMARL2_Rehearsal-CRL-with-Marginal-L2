import gymnasium as gym
import numpy as np
import torch
import random

def make_metaworld_env(env_ids, seed = 42):
    """ example usage: env_id = ['push-back-v2', 'sweep-into-v2', 'window-close-v2']
    """
    # meta-world env setup:
    import metaworld
    ml = metaworld.ML45(seed=seed)
    print(f"Setting up Meta-World ML45 env: {env_ids}")

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
        
