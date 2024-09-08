from dataclasses import dataclass
import numpy as np
import torch
import random
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  #deterministic results on GPUs
    
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPO_minatar_Agent(nn.Module):
    def __init__(self, envs, seed=None, hidden_size=256):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),#can be changed to 256
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, envs.single_action_space.n), std=0.01),
        )
        set_seed(seed)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

class RandomAgent(nn.Module):
    def __init__(self, envs):
        import gym
        self.envs = envs
        if isinstance(envs.single_action_space, gym.spaces.Discrete):
            self.action_dim = envs.single_action_space.n
        elif isinstance(envs.single_action_space, gym.spaces.Box):
            self.action_dim = envs.single_action_space.shape[0]
    def parameters(self):
        return []
    def get_action_and_value(self, x, action=None):
        #randomly sample 1 action from action space
        action = torch.tensor([self.envs.single_action_space.sample() for _ in range(len(x))])
        log_prob = torch.zeros(len(action)) 
        entropy = torch.zeros(len(action)) 
        value = torch.zeros(len(action)) 
        action = action.long()
        return action, log_prob, entropy, value

    def get_value(self, x):
        return torch.zeros(len(x))
    
class PPO_metaworld_Agent(nn.Module):
    def __init__(self, envs, seed=None,hidden_size=256):
        set_seed(seed)
        envs.seed(seed)
        envs.action_space.seed(seed)
        envs.observation_space.seed(seed)

        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, np.prod(envs.action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(np.prod(envs.action_space.shape))) 

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        log_prob = probs.log_prob(action)
        entropy = probs.entropy()

        if log_prob.dim() >=1:
            log_prob_sum = log_prob.sum().unsqueeze(0)
        else:
            log_prob_sum = log_prob.sum(1)
        if entropy.dim() >=1:
            entropy_sum = entropy.sum().unsqueeze(0)
        else:
            entropy_sum = entropy.sum(1)
        return action, log_prob_sum, entropy_sum, self.critic(x).unsqueeze(0)
