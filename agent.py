from dataclasses import dataclass
import numpy as np
import torch
import random
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import torch.nn.functional as F

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
        set_seed(seed)
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
        self.init_critic_params = self.get_flat_params(self.critic).detach()
        self.init_actor_params = self.get_flat_params(self.actor).detach()

    def get_value(self, x):
        return self.critic(x)

    def get_flat_params(self, module):
        return torch.cat([p.flatten() for p in module.parameters()])
    
    def compute_l2_loss(self, device= torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        curr_params = self.get_flat_params(self.critic)
        l2_loss = 0.5 * ((curr_params.to(device) - self.init_critic_params.to(device)) ** 2).sum()#.mean()
        return l2_loss
    
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
 
        self.init_critic_params = self.get_flat_params(self.critic).detach()
        self.init_actor_mean_params = self.get_flat_params(self.actor_mean).detach()
        self.init_actor_logstd = self.actor_logstd.detach()

    def get_value(self, x):
        return self.critic(x)

    def get_flat_params(self, module):
        return torch.cat([p.flatten() for p in module.parameters()])
    
    def compute_l2_loss(self, device= torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        curr_critic_params = self.get_flat_params(self.critic).to(device)
        curr_actor_mean_params = self.get_flat_params(self.actor_mean).to(device)

        l2_loss_critic = 0.5 * ((curr_critic_params - self.init_critic_params.to(device)) ** 2).sum()
        l2_loss_actor = 0.5 * ((curr_actor_mean_params - self.init_actor_mean_params.to(device)) ** 2).sum()
        l2_loss_actor_logstd = 0.5 * ((self.actor_logstd.to(device) - self.init_actor_logstd.to(device)) ** 2).sum()

        l2_loss = l2_loss_critic + l2_loss_actor + l2_loss_actor_logstd
        
        return l2_loss
    
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

class PPO_Conv_Agent(nn.Module):
    def __init__(self, envs, hidden_size=64, seed=None):
        super(PPO_Conv_Agent, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)

        input_channels = envs.single_observation_space.shape[0] 
        num_actions = envs.single_action_space.n

        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=2)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(512, hidden_size)

        self.actor_fc1 = nn.Linear(hidden_size, hidden_size)
        self.actor_fc2 = nn.Linear(hidden_size, hidden_size)
        self.actor_out = nn.Linear(hidden_size, num_actions)

        self.critic_fc1 = nn.Linear(hidden_size, hidden_size)
        self.critic_fc2 = nn.Linear(hidden_size, hidden_size)
        self.critic_out = nn.Linear(hidden_size, 1)

        self.init_critic_params = self.get_flat_params(self.critic_fc1, self.critic_fc2, self.critic_out).detach()
        self.init_actor_params = self.get_flat_params(self.actor_fc1, self.actor_fc2, self.actor_out).detach()

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))

        actor_mean = F.relu(self.actor_fc1(x))
        actor_mean = F.relu(self.actor_fc2(actor_mean))
        actor_mean = self.actor_out(actor_mean)

        critic_value = F.relu(self.critic_fc1(x))
        critic_value = F.relu(self.critic_fc2(critic_value))
        critic_value = self.critic_out(critic_value)

        return actor_mean, critic_value

    def get_value(self, x):
        _, value = self.forward(x)
        return value

    def get_action_and_value(self, x, action=None):
        logits, value = self.forward(x)
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action), probs.entropy(), value

    def get_flat_params(self, *modules):
        params = []
        for module in modules:
            params.append(torch.cat([p.flatten() for p in module.parameters()]))
        return torch.cat(params)

    def compute_l2_loss(self, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        curr_params = self.get_flat_params(self.critic_fc1, self.critic_fc2, self.critic_out)
        l2_loss = 0.5 * ((curr_params.to(device) - self.init_critic_params.to(device)) ** 2).sum()
        return l2_loss