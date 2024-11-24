# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import os
import random
import time
from dataclasses import dataclass, asdict, fields
import argparse
from typing import List
import gymnasium as gym
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from agent import PPO_minatar_Agent, PPO_metaworld_Agent, PPO_Conv_Agent, WeightClipping
from pathlib import Path
from collections import namedtuple

class RUNNINGSTATISTICS:
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def add(self, x):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def SampleMeanVar(self, x, mean, p):
        self.count += 1
        n=self.count
        delta = x - mean
        self.mean = mean+ (delta / n)
        delta2 = x - self.mean
        p = p + delta * delta2
        if n >= 2:
            var = p / (n-1)
        else:
            var = 1
        
        self.m2=p
        return p,self.mean,var

    def variance(self):
        return self.m2 / (self.count)

    def float_std(self):
        return self.variance() ** 0.5
    
    def std(self):
        return torch.tensor(self.variance() ** 0.5).to(device)

    def get_mean(self):
        return self.mean

def evaluate_single_env(agent, env, eval_episodes, device, max_steps=10000):
    agent.eval()
    total_rewards = []
    steps_per_episode = []

    for j in range(eval_episodes):
        obs, _ = env.reset(seed=args.seed + j)
        obs = torch.tensor(obs, dtype=torch.float32).to(device)
        
        done = False
        episode_reward = 0
        steps = 0

        while steps < max_steps and not done:
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs)
                action = action.cpu().numpy()
                
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                episode_reward += reward
                # reward_tensor = torch.tensor(reward).to(device).view(-1)
                # scaled_reward = reward_tensor / (reward_statistics.std() + 1e-8)

                # episode_reward += scaled_reward
                obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
                
                steps += 1

        total_rewards.append(episode_reward)
        steps_per_episode.append(steps)

    total_rewards = [r / eval_episodes for r in total_rewards]
    mean_reward = np.mean(total_rewards) if total_rewards else 0.0
    mean_steps = np.mean(steps_per_episode) if steps_per_episode else 0.0
    
    print(f"Total rewards over {eval_episodes} episodes: {total_rewards}")
    print(f"Steps per episode: {steps_per_episode}")
    
    return mean_reward, mean_steps

def evaluate_parallel_env(agent, envs, eval_episodes, device, max_steps=10000):
    agent.eval()
    total_rewards = []
    steps_per_episode = []

    for j in range(eval_episodes):
        obs, _ = envs.reset(seed=args.seed+j)
        obs = torch.tensor(obs, dtype=torch.float32).to(device)
        
        done = np.zeros(envs.num_envs, dtype=bool) 
        episode_rewards = np.zeros(envs.num_envs)   
        steps = 0

        while steps < max_steps and not np.all(done):
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs)
                action = action.cpu().numpy()
                
                next_obs, reward, terminations, truncations, _ = envs.step(action)
                done = np.logical_or(done, np.logical_or(terminations, truncations))

                reward = np.where(~done, reward, 0)
                episode_rewards += reward
                obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
                steps += 1

                if np.all(done):
                    break

        total_rewards.append(np.mean(episode_rewards))  # avg reward per episode (across envs)
        steps_per_episode.append(steps) 

    total_rewards = [r / eval_episodes for r in total_rewards]
    mean_reward = np.mean(total_rewards) if total_rewards else 0.0
    mean_steps = np.mean(steps_per_episode) if steps_per_episode else 0.0
    
    # print(f"total_rewards over {eval_episodes} episodes: {total_rewards}")
    # print(f"steps_per_episode per episode: {steps_per_episode}")
    
    return mean_reward, mean_steps

@dataclass
class Args:
    exp_type: str="ppo_minatar" # or "ppo_metaworld"
    exp_name: str = "PPO"#os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "PPO_minatar" # or "PPO_metaworld"
    """the wandb's project name"""
    wandb_entity: str = "angela-h"
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_ids: list = ("MinAtar/Breakout-v0", "MinAtar/Asterix-v0", "MinAtar/Freeway-v0") # or ('push-back-v2', 'sweep-into-v2', 'window-close-v2')
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""

    use_weight_clip: bool = False
    """weight clipping"""
    weight_clipping: float = 3.0
    """Weight Clipping"""
    clip_last_layer: int = 1
    """Clip the last layer of the network"""

    use_crelu: bool = False
    """Toggle for the usage of Concat ReLU"""

    use_ewc: bool = False
    """Toggle for the usage of EWC loss"""
    ewc_coef: float = 1.0

    use_l2_loss: bool = False
    """Toggle for the usage of L2 init loss"""
    use_l2_0_loss: bool = False
    """Toggle for the usage of L2-0 init loss"""
    l2_coef: float = 0.01
    """ l2 init loss's coefficient"""
    periodic_l2: bool = False
    """Toggle for the usage of L2 init loss on every parameter: theta_t-1 instead of paramater at theta_0"""
   
    wandb_log_off: bool = True
    """Toggle true for turning off wandb log files"""

    reward_rescale: bool = True
    """Toggle true for turning on reward rescale"""
    global_reward_rescale: bool = True
    """Toggle true for turning on global reward rescale"""

    value_norm: bool = True
    """Toggles value normalization"""
    global_value_norm: bool = True
    """Toggles global value normalization"""


    rolling_window: int = 100
    """ mean calculation's window size """
    eval_interval: int = 50000  
    """ Evaluate model performance every 50k steps"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

def parse_args():
    parser = argparse.ArgumentParser(description="PPO expr")
    def parse_bool(x):
        if isinstance(x, bool):
            return x
        if x.lower() in ('yes', 'true', 't', 'y'):
            return True
        elif x.lower() in ('no', 'false', 'f', 'n'):
            return False
        else:
            raise argparse.ArgumentTypeError(f'Bool value invalid:{x}')
    for field in fields(Args):
        arg_name = f"--{field.name.replace('_', '-')}"
        field_type = field.type
        default_value = field.default
        
        if field_type == bool:
            parser.add_argument(arg_name, type=parse_bool, help=field.metadata.get("help", ""), default=default_value)
        elif isinstance(default_value, list) or isinstance(default_value, tuple):
            field_type = str
            parser.add_argument(arg_name, nargs='+', help=field.metadata.get("help", ""), default=default_value)
        else:
            parser.add_argument(arg_name,nargs='?',type=field_type, help=field.metadata.get("help", ""), default=default_value)
    args = parser.parse_args()
    args_dict = vars(args)
    for field in fields(Args):
        if isinstance(field.default, list) or isinstance(field.default, tuple):
            args_dict[field.name] = tuple(args_dict[field.name])

    return Args(**args_dict)

if __name__ == "__main__":
    # args = tyro.cli(Args)
    args = parse_args()
    print(f"Args used for this expr: {args}")
    # example: args = Args(
    #     exp_name="PPO_minatar",
    #     seed=1,
    #     torch_deterministic=True,
    #     cuda=True,
    #     track=True,
    #     capture_video=False,
    #     env_ids=["MinAtar/Breakout-v0","MinAtar/Asterix-v0", "MinAtar/Freeway-v0"],
    #     # total_timesteps=1000,
    #     learning_rate=2.5e-4,
    #     # num_envs=2,
    #     # num_steps=50,
    # )

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.seed}_{args.exp_name}_{args.env_ids}"

    # Environment setup
    if args.exp_type == "ppo_metaworld":
        from utils import make_metaworld_env, make_tianshou_metaworld_env
        train_envs, test_envs = make_tianshou_metaworld_env(args.env_ids, seed = args.seed, training_num=args.num_envs)
        eval_envs, _ = make_tianshou_metaworld_env(args.env_ids, seed = args.seed+1, training_num=args.num_envs)
    elif args.exp_type == "ppo_minatar":
        from env import make_minatar_env
        train_envs=[]
        eval_envs=[]
        for env_id in args.env_ids:
            envs = gym.vector.SyncVectorEnv(
                [make_minatar_env(env_id, i, args.capture_video, run_name,seed=args.seed) for i in range(args.num_envs)]
            )
            assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
            train_envs.append(envs)
            eval_envs.append(gym.vector.SyncVectorEnv(
                [make_minatar_env(env_id, i, args.capture_video, run_name,seed=args.seed+1) for i in range(args.num_envs)]
            ))
        
    else:
        print(f"expr type not supported:{args.exp_type}")
        exit(1)

    if args.track:
        import wandb

        if args.wandb_log_off:
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                monitor_gym=True,
                save_code=False,
                settings=wandb.Settings(console="off",log_internal=str(Path(__file__).parent / 'wandb' / 'null')),
            )
        else:
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                monitor_gym=True,
                save_code=True
            )
    
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    global_step = 0
    start_time = time.time()

    print(f"Env tasks used in this run: {train_envs}")    
    #TODO: put MU/STD here?
    reward_statistics = RUNNINGSTATISTICS()
    R_t_1 = 0.0
    rew_P=0.0

    value_MU = 0.0  
    value_STD = 1.0 
    alpha = 0.001

    transition = namedtuple('Transition', ('obs', 'action'))
    class ewc_buffer:
        def __init__(self, buffer_size=100000):
            self.buffer_size = buffer_size  
            self.location = 0 
            self.buffer = []

        def add(self, obs, action):
            if len(self.buffer) < self.buffer_size:
                self.buffer.append(transition(obs, action))
            else:
                self.buffer[self.location] = transition(obs, action)
            self.location = (self.location + 1) % self.buffer_size

        def sample(self, batch_size):
            return random.sample(self.buffer, batch_size)

        def size(self):
            return len(self.buffer)

        def clear(self):
            self.buffer = []
            self.location = 0
    ewc_buffer = ewc_buffer(buffer_size=100000)
    
    for i,envs in enumerate(train_envs):
        print(f"Training on environment: {args.env_ids[i]}")
        if i==0:
            if args.exp_type == "ppo_minatar":
                agent=PPO_Conv_Agent(envs=envs,seed=args.seed,use_crelu=args.use_crelu).to(device)
            elif args.exp_type == "ppo_metaworld":
                agent = PPO_metaworld_Agent(envs=envs,seed=args.seed).to(device)

            if args.use_weight_clip:
                optimizer = WeightClipping(agent.parameters(), lr=args.learning_rate, eps=1e-5, beta=args.weight_clipping, clip_last_layer=args.clip_last_layer)
            else:
                optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

            # ALGO Logic: Storage setup
            obs_space_shape = envs.observation_space.shape if args.exp_type == "ppo_metaworld" else envs.single_observation_space.shape
            act_space_shape = envs.action_space.shape if args.exp_type == "ppo_metaworld" else envs.single_action_space.shape

            obs = torch.zeros((args.num_steps, args.num_envs) + obs_space_shape).to(device)
            actions = torch.zeros((args.num_steps, args.num_envs) + act_space_shape).to(device)
            logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
            rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
            dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
            values = torch.zeros((args.num_steps, args.num_envs)).to(device)
        else:
            if args.periodic_l2 and args.use_l2_loss:
                agent.set_flat_params()

        # TRY NOT TO MODIFY: start the game
        next_obs, _ = envs.reset(seed=args.seed)
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(args.num_envs).to(device)
        episode_success_rates=[]
        episode_rewards = []
        reward_window=[]
        
        # local values for rescale and normalization
        if args.value_norm and not args.global_value_norm:
            value_MU = 0.0  
            value_STD = 1.0 
            
        if args.reward_rescale and not args.global_reward_rescale:
            reward_statistics = RUNNINGSTATISTICS()
            R_t_1 = 0.0
            rew_P=0.0
        
        for iteration in range(1, args.num_iterations + 1):
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow
            
            for step in range(0, args.num_steps):
                global_step += args.num_envs

                obs[step] = next_obs
                if args.exp_type == "ppo_metaworld" and next_done.nelement() == 0:
                    # print("next_done invalid, reset as tensor([0.])")
                    next_done = torch.zeros(args.num_envs).to(device)
                dones[step] = next_done
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    if args.value_norm:
                        value = value * value_STD + value_MU

                    values[step] = value.flatten()

                actions[step] = action
                logprobs[step] = logprob
                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                
                if args.reward_rescale:
                    reward_tensor = torch.tensor(reward).to(device).view(-1)
                    

                    #old scaling
                    # R_t = args.gamma * R_t_1 + reward
                    # R_t_1 = R_t 
                    # reward_statistics.add(R_t)
                    # scaled_reward = reward_tensor / (reward_statistics.std() + 1e-8)

                    #new scaling 
                    R_t = args.gamma * R_t_1 + reward  
                    R_t_1 = R_t  
                    p,mean_,var=reward_statistics.SampleMeanVar(R_t, mean=0, p=rew_P)
                    rew_P=p
                    tmp_var=var
                    scaled_reward = reward_tensor / (torch.tensor(var, device=reward_tensor.device) ** 0.5 + 1e-8)

                    rewards[step] = scaled_reward
                else:
                    rewards[step] = torch.tensor(reward).to(device).view(-1)

                next_done = np.logical_or(terminations, truncations)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

                episode_rewards.append(torch.tensor(reward).to(device).view(-1))
                if args.exp_type == "ppo_metaworld":
                    episode_success_rates.append(infos["success"])
                    writer.add_scalar(f"charts/{args.env_ids[i]}_reward", reward, global_step)
                    writer.add_scalar(f"charts/{args.env_ids[i]}_success", infos["success"], global_step)
                    if global_step % 50000 == 0: #np.any(np.logical_or(terminations, truncations)):
                        if len(episode_success_rates) > 0:
                            success_rate_mean = np.mean(episode_success_rates)
                            writer.add_scalar(f"eval/success_rate_mean", success_rate_mean, global_step)
                            episode_success_rates = []
                        
                        if len(episode_rewards) > 0:
                            mean_reward = torch.stack(episode_rewards).mean().item()
                            writer.add_scalar(f"eval/mean_reward", mean_reward, global_step)
                            episode_rewards = []
                
                if args.reward_rescale:
                    #old rew scaling add .mean()
                    # writer.add_scalar(f"train/reward_mean", reward_statistics.get_mean().mean(), global_step)
                    # writer.add_scalar(f"train/reward_std", reward_statistics.float_std().mean(), global_step)
                    #new rew scaling dont need mean()
                    a=tmp_var ** 0.5 + 1e-8
                    writer.add_scalar(f"train/reward_mean", mean_.mean(), global_step)
                    try:
                        writer.add_scalar(f"train/reward_std", a, global_step)
                    except:
                        writer.add_scalar(f"train/reward_std", a.mean(), global_step)

                if "final_info" in infos:
                    for k, info in enumerate(infos["final_info"]):
                        if info and "episode" in info:
                            raw_reward = info['episode']['r'][0]
                            reward_window.append(raw_reward)
                            if args.reward_rescale:
                                #old rew scaling
                                # current_std = reward_statistics.std()[k] + 1e-8

                                #new rew scaling:
                                current_std = tmp_var[k] ** 0.5 + 1e-8
                                scaled_reward = raw_reward / current_std

                            if len(reward_window) > args.rolling_window:
                                reward_window.pop(0)

                            if global_step <=args.num_envs or (len(reward_window) == args.rolling_window and global_step % 10000 == 0) or ("freeway" in args.env_ids[i].lower()):
                                writer.add_scalar(f"train/rolling_episodic_return_raw", np.mean(reward_window), global_step)

                            writer.add_scalar(f"train/episodic_return_raw", raw_reward, global_step)
                            if args.reward_rescale:
                                writer.add_scalar(f"train/episodic_return_scaled", scaled_reward, global_step)
                            writer.add_scalar(f"train/episodic_length", info["episode"]["l"][0], global_step)
                if args.exp_type == "ppo_metaworld":#take it as a done flag
                    if truncations:
                        print("Truncated, env resetting!")
                        next_obs, _ = envs.reset(seed=args.seed)
                        next_obs = torch.Tensor(next_obs).to(device)
                        next_done = torch.zeros(args.num_envs).to(device)
                
                if global_step!=0 and global_step % args.eval_interval == 0:
                    # print("Evaluating model performance!")
                    for j, eval_env in enumerate(eval_envs):
                        mean_reward, mean_steps = evaluate_parallel_env(agent, eval_env, 1, device)
                        writer.add_scalar(f"test/{args.env_ids[j]}__eval_R", mean_reward, global_step)
                        writer.add_scalar(f"test/{args.env_ids[j]}__eval_steps", mean_steps, global_step)

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                #scale back
                if args.value_norm:
                    next_value = next_value * value_STD + value_MU

                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        if args.exp_type == "ppo_metaworld" and next_done.nelement() == 0:
                            next_done = torch.zeros(args.num_envs).to(device)

                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            obs_space_shape = envs.observation_space.shape if args.exp_type == "ppo_metaworld" else envs.single_observation_space.shape
            act_space_shape = envs.action_space.shape if args.exp_type == "ppo_metaworld" else envs.single_action_space.shape

            b_obs = obs.reshape((-1,) + obs_space_shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + act_space_shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    if args.use_ewc:
                        for mb_ind in mb_inds:
                            ewc_buffer.add(b_obs[mb_ind].detach().clone(), b_actions[mb_ind].detach().clone())
                    writer.add_scalar("losses/training_newvalue_raw", newvalue.mean().item(), global_step)
                    # if args.value_norm: TODO: scale newvalue or not
                    #     newvalue = newvalue * value_STD + value_MU

                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    writer.add_scalar("losses/b_returns[mb_inds]_raw", b_returns[mb_inds].mean().item(), global_step)
                    if args.value_norm:
                        mb_mean = b_returns[mb_inds].mean().item()
                        mb_std = b_returns[mb_inds].std().item()
                        value_MU = (1 - alpha) * value_MU + alpha * mb_mean
                        value_STD = (1 - alpha) * value_STD + alpha * mb_std
                        b_returns[mb_inds] = (b_returns[mb_inds] - value_MU) / (value_STD + 1e-8)

                        writer.add_scalar("losses/b_returns[mb_inds]_normalized", b_returns[mb_inds].mean().item(), global_step)
                    
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                    writer.add_scalar(f"train/original_ppo_loss", loss, global_step)
                    if args.use_l2_loss:
                        l2_loss_critic, l2_loss_actor = agent.compute_l2_loss(device=device)
                        l2_loss = l2_loss_critic + l2_loss_actor
                        loss += args.l2_coef * l2_loss
                        writer.add_scalar(f"train/actor_l2_loss", l2_loss_actor, global_step)
                        writer.add_scalar(f"train/critic_l2_loss", l2_loss_critic, global_step)
                    if args.use_l2_0_loss:
                        l2_0_loss = agent.compute_l2_0_loss()
                        loss += args.l2_coef * l2_0_loss
                        writer.add_scalar(f"train/l2_0_loss", l2_0_loss, global_step)
                    if args.use_ewc:
                        if i>0:
                            ewc_loss = agent.ewc_loss()
                            loss += args.ewc_coef * ewc_loss
                            writer.add_scalar(f"train/ewc_loss", ewc_loss.item(), global_step)
                        else:
                            writer.add_scalar(f"train/ewc_loss", 0, global_step)
                    writer.add_scalar(f"train/total_loss", loss, global_step)

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break
            #TODO: reset optimal weights here?
            # if args.use_ewc:
            #     #store optimal weights after the entire epoch
            #     agent.store_optimal_weights()

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            # print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        if args.use_ewc:
            agent.reset_fisher_information()
            sampled = ewc_buffer.sample(batch_size=10000)
            sampled_obs = torch.stack([t.obs for t in sampled])
            sampled_actions = torch.stack([t.action for t in sampled])
            #fisher-info matrix

            agent.compute_fisher_information(sampled_obs, sampled_actions)
            ewc_buffer.clear()
            agent.store_optimal_weights()
        envs.close()
    writer.close()