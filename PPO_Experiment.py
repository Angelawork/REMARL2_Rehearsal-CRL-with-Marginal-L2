# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import os
import random
import time
from dataclasses import dataclass, asdict, fields
from env import make_minatar_env
import argparse
from typing import List
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from agent import PPO_minatar_Agent, PPO_metaworld_Agent


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
    parser = argparse.ArgumentParser(description="Process experiment configuration.")
    for field in fields(Args):
        arg_name = f"--{field.name.replace('_', '-')}"
        field_type = field.type
        default_value = field.default
        
        if isinstance(default_value, list) or isinstance(default_value, tuple):
            field_type = str
            parser.add_argument(arg_name, nargs='+', help=field.metadata.get("help", ""), default=default_value)
        else:
            parser.add_argument(arg_name, type=field_type, help=field.metadata.get("help", ""), default=default_value)
    parser.add_argument("--env_ids", nargs='+', help=field.metadata.get("help", ""), default=["MinAtar/Breakout-v0", "MinAtar/Asterix-v0", "MinAtar/Freeway-v0"])
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
    exit(0)
    
    # Environment setup
    if args.exp_type == "ppo_metaworld":
        from utils import make_metaworld_env
        train_envs, test_envs = make_metaworld_env(args.env_ids, seed = args.seed)
    elif args.exp_type == "ppo_minatar":
        tasks=[]
        for env_id in args.env_ids:
            envs = gym.vector.SyncVectorEnv(
                [make_minatar_env(env_id, i, args.capture_video, run_name) for i in range(args.num_envs)]
            )
            assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
            tasks.append(envs)
        
    else:
        print(f"expr type not supported:{args.exp_type}")
        exit(1)

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
    run_name = f"{args.env_ids}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
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

    if args.exp_type == "ppo_metaworld":
        tasks = train_envs

    print(f"Env tasks used in this run: {tasks}")    
    for i,envs in enumerate(tasks):
        print(f"Training on environment: {args.env_ids[i]}")
        if i==0:
            if args.exp_type == "ppo_minatar":
                agent=PPO_minatar_Agent(envs).to(device)
            elif args.exp_type == "ppo_metaworld":
                agent = PPO_metaworld_Agent(envs).to(device)

            optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

            # ALGO Logic: Storage setup
            obs_space_shape = envs.observation_space.shape if args.exp_type == "ppo_metaworld" else envs.single_observation_space.shape

            obs = torch.zeros((args.num_steps, args.num_envs) + obs_space_shape).to(device)
            actions = torch.zeros((args.num_steps, args.num_envs) + obs_space_shape).to(device)
            logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
            rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
            dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
            values = torch.zeros((args.num_steps, args.num_envs)).to(device)

        # TRY NOT TO MODIFY: start the game
        next_obs, _ = envs.reset(seed=args.seed)
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(args.num_envs).to(device)

        for iteration in range(1, args.num_iterations + 1):
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, args.num_steps):
                global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                # print(next_obs)
                # print(next_obs.shape)
                # next_obs = next_obs.view(next_obs.size(0), -1, 84, 84)

                with torch.no_grad():
                    next_obs_flattened = next_obs.view(next_obs.size(0), -1)
                    action, logprob, _, value = agent.get_action_and_value(next_obs_flattened)

                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                if args.exp_type == "ppo_metaworld":
                    if truncations:
                        print("Truncated, env resetting!")
                        next_obs, _ = envs.reset(seed=args.seed)
                        next_obs = torch.Tensor(next_obs).to(device)
                        next_done = torch.zeros(args.num_envs).to(device)
                        continue
                    if step%100==0 and iteration%25==0:
                        print(f"\n*******************at step={step}, iteration={iteration} *******************\ninfos={infos}")
                        print(f"reward={reward}")
                        print(f"terminations={terminations}")
                        print(f"truncations={truncations}")

                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

                writer.add_scalar("charts/reward", reward, global_step)
                
                if args.exp_type == "ppo_metaworld":
                    writer.add_scalar("charts/success", infos["success"], global_step)
                    if "grasp_reward" in infos:
                        writer.add_scalar("charts/grasp_reward", infos["grasp_reward"], global_step)
                    if "grasp_success" in infos:
                        writer.add_scalar("charts/grasp_success", infos["grasp_success"], global_step)

                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

            # bootstrap value if not done
            with torch.no_grad():
                if args.exp_type == "ppo_metaworld":
                    next_value = agent.get_value(next_obs).reshape(1, -1)
                elif args.exp_type == "ppo_minatar":
                    next_obs_flattened = next_obs.reshape(-1, np.array(envs.single_observation_space.shape).prod())
                    next_value = agent.get_value(next_obs_flattened).reshape(1, -1)

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

                    if args.exp_type == "ppo_minatar":
                        # Reshape b_obs correctly for the actor network
                        mb_obs = b_obs[mb_inds].view(mb_inds.size, -1)  # Flatten the observation for each minibatch
                        _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, b_actions.long()[mb_inds])
                    elif args.exp_type == "ppo_minatar":
                        _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])

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
                    # print(f"pg_loss: {pg_loss}, v_loss: {v_loss}, entropy_loss: {entropy_loss}")
                    # print(f"old_approx_kl: {old_approx_kl}, approx_kl: {approx_kl}")
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

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
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        envs.close()
    writer.close()