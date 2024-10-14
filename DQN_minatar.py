import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import time
import random, numpy, argparse, logging, os
import wandb
import os
from pathlib import Path

from collections import namedtuple
from minatar import Environment

logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)
################################################################################################################
# Constants
#
################################################################################################################
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 100000
TARGET_NETWORK_UPDATE_FREQ = 1000
TRAINING_FREQ = 1
NUM_FRAMES = 5000000
FIRST_N_FRAMES = 100000
REPLAY_START_SIZE = 5000
END_EPSILON = 0.1
STEP_SIZE = 0.00025
GRAD_MOMENTUM = 0.95
SQUARED_GRAD_MOMENTUM = 0.95
MIN_SQUARED_GRAD = 0.01
GAMMA = 0.99
EPSILON = 1.0
SEED=10

def set_seed(seed):
    random.seed(seed)            
    numpy.random.seed(seed)       
    torch.manual_seed(seed)        
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)        
        torch.cuda.manual_seed_all(seed)     
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(QNetwork, self).__init__()
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)
        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16
        self.fc_hidden = nn.Linear(in_features=num_linear_units, out_features=128)
        self.output = nn.Linear(in_features=128, out_features=num_actions)


    def forward(self, x):
        x = f.relu(self.conv(x))
        x = f.relu(self.fc_hidden(x.view(x.size(0), -1)))
        return self.output(x)
transition = namedtuple('transition', 'state, next_state, action, reward, is_terminal')
class replay_buffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.location = 0
        self.buffer = []

    def add(self, *args):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition(*args))
        else:
            self.buffer[self.location] = transition(*args)
        self.location = (self.location + 1) % self.buffer_size

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

def get_state(s):
    return (torch.tensor(s, device=device).permute(2, 0, 1)).unsqueeze(0).float()

def world_dynamics(t, replay_start_size, num_actions, s, env, policy_net):
    if t < replay_start_size:
        action = torch.tensor([[random.randrange(num_actions)]], device=device)
    else:
        epsilon = END_EPSILON if t - replay_start_size >= FIRST_N_FRAMES else ((END_EPSILON - EPSILON) / FIRST_N_FRAMES) * (t - replay_start_size) + EPSILON
        if numpy.random.binomial(1, epsilon) == 1:
            action = torch.tensor([[random.randrange(num_actions)]], device=device)
        else:
            with torch.no_grad():
                action = policy_net(s).max(1)[1].view(1, 1)
    s_prime, reward, terminated, truncated, info = env.step(action)  # step() returns observation, reward, done, truncated, info
    s_prime = get_state(s_prime)

    info = {"episode": {"r": reward, "l": t}} if terminated else {}
    return s_prime, action, torch.tensor([[reward]], device=device).float(), torch.tensor([[terminated]], device=device), info

def train(sample, policy_net, target_net, optimizer):
    batch_samples = transition(*zip(*sample))
    states = torch.cat(batch_samples.state)
    next_states = torch.cat(batch_samples.next_state)
    actions = torch.cat(batch_samples.action)
    rewards = torch.cat(batch_samples.reward)
    is_terminal = torch.cat(batch_samples.is_terminal)

    Q_s_a = policy_net(states).gather(1, actions)
    none_terminal_next_state_index = torch.tensor([i for i, is_term in enumerate(is_terminal) if is_term == 0], dtype=torch.int64, device=device)
    none_terminal_next_states = next_states.index_select(0, none_terminal_next_state_index)

    Q_s_prime_a_prime = torch.zeros(len(sample), 1, device=device)
    if len(none_terminal_next_states) != 0:
        Q_s_prime_a_prime[none_terminal_next_state_index] = target_net(none_terminal_next_states).detach().max(1)[0].unsqueeze(1)

    target = rewards + GAMMA * Q_s_prime_a_prime
    loss = f.smooth_l1_loss(target, Q_s_a)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    wandb.log({"loss": loss.item()})

def dqn(env, replay_off, target_off, output_file_name, store_intermediate_result=False, load_path=None, step_size=STEP_SIZE):
    in_channels = 10 #env.state_shape()[2]
    num_actions = 6#env.num_actions()
    # wandb.log({f"training start-in_channels=": in_channels, "training start-num_actions":num_actions})

    policy_net = QNetwork(in_channels, num_actions).to(device)
    replay_start_size = 0
    if not target_off:
        target_net = QNetwork(in_channels, num_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())

    if not replay_off:
        r_buffer = replay_buffer(REPLAY_BUFFER_SIZE)
        replay_start_size = REPLAY_START_SIZE

    optimizer = optim.RMSprop(policy_net.parameters(), lr=step_size, alpha=SQUARED_GRAD_MOMENTUM, centered=True, eps=MIN_SQUARED_GRAD)

    e_init, t_init, policy_net_update_counter_init, avg_return_init = 0, 0, 0, 0.0
    data_return_init, frame_stamp_init = [], []

    if load_path is not None and isinstance(load_path, str):
        checkpoint = torch.load(load_path)
        policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        if not target_off:
            target_net.load_state_dict(checkpoint['target_net_state_dict'])
        if not replay_off:
            r_buffer = checkpoint['replay_buffer']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        e_init, t_init, policy_net_update_counter_init, avg_return_init = checkpoint['episode'], checkpoint['frame'], checkpoint['policy_net_update_counter'], checkpoint['avg_return']
        data_return_init, frame_stamp_init = checkpoint['return_per_run'], checkpoint['frame_stamp_per_run']
        policy_net.train()
        if not target_off:
            target_net.train()

    data_return = data_return_init
    frame_stamp = frame_stamp_init
    avg_return = avg_return_init

    t, e = t_init, e_init
    policy_net_update_counter = policy_net_update_counter_init
    t_start = time.time()

    while t < NUM_FRAMES:
        G = 0.0
        s, _ = env.reset(seed=SEED) 
        s = get_state(s)
        is_terminated = False

        steps_spent=0
        while(not is_terminated) and t < NUM_FRAMES:
            steps_spent+=1
            wandb.log({f"{args.env}_steps_spent": steps_spent})
            # Generate data
            s_prime, action, reward, is_terminated, info = world_dynamics(t, replay_start_size, num_actions, s, env, policy_net)

            sample = None
            if replay_off:
                sample = [transition(s, s_prime, action, reward, is_terminated)]
            else:
                # Write the current frame to replay buffer
                r_buffer.add(s, s_prime, action, reward, is_terminated)

                # Start learning when there's enough data and when we can sample a batch of size BATCH_SIZE
                if t > REPLAY_START_SIZE and len(r_buffer.buffer) >= BATCH_SIZE:
                    # Sample a batch
                    sample = r_buffer.sample(BATCH_SIZE)

            # Train every n number of frames defined by TRAINING_FREQ
            if t % TRAINING_FREQ == 0 and sample is not None:
                if target_off:
                    train(sample, policy_net, policy_net, optimizer)
                else:
                    policy_net_update_counter += 1
                    train(sample, policy_net, target_net, optimizer)

            if not target_off and policy_net_update_counter > 0 and policy_net_update_counter % TARGET_NETWORK_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

            G += reward.item()
            t += 1
            s = s_prime

        e += 1
        data_return.append(G)
        frame_stamp.append(t)

        avg_return = 0.99 * avg_return + 0.01 * G
        if e % 50 == 0:
            # logging.info(f"Episode {e} | Return: {G} | Avg return: {numpy.around(avg_return, 2)} | Frame: {t} | Time per frame: {(time.time()-t_start)/t}")
            print(f'"avg_return": {avg_return}, "episode": {e}, "return": {G}, "frame_step":{t}')
            wandb.log({f"{args.env}_avg_return": avg_return, "episode": e, f"{args.env}_return": G, "frame_step":t})

        # if "episode" in info:
        #     writer.add_scalar(f"charts/{avg_return}_episodic_return", info["episode"]["r"], t)
        #     writer.add_scalar(f"charts/{avg_return}_episodic_length", info["episode"]["l"], t)

        # if store_intermediate_result and e % 1000 == 0:
        #     torch.save({
        #         'episode': e,
        #         'frame': t,
        #         'policy_net_update_counter': policy_net_update_counter,
        #         'policy_net_state_dict': policy_net.state_dict(),
        #         'target_net_state_dict': target_net.state_dict() if not target_off else [],
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'avg_return': avg_return,
        #         'return_per_run': data_return,
        #         'frame_stamp_per_run': frame_stamp,
        #         'replay_buffer': r_buffer if not replay_off else [],
        #     }, output_file_name)

if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--env', type=str, default='breakout')
    parser.add_argument('--replay-off', action='store_true', default=False)
    parser.add_argument('--target-off', action='store_true', default=False)
    parser.add_argument('--load-path', type=str)
    args = parser.parse_args()

    SEED = args.seed
    set_seed(SEED)

    print(f"args used: {args}")
    import wandb
    config={
        "batch_size": BATCH_SIZE,
        "replay_buffer_size": REPLAY_BUFFER_SIZE,
        "target_network_update_freq": TARGET_NETWORK_UPDATE_FREQ,
        "training_freq": TRAINING_FREQ,
        "num_frames": NUM_FRAMES,
        "first_n_frames": FIRST_N_FRAMES,
        "replay_start_size": REPLAY_START_SIZE,
        "end_epsilon": END_EPSILON,
        "step_size": STEP_SIZE,
        "grad_momentum": GRAD_MOMENTUM,
        "squared_grad_momentum": SQUARED_GRAD_MOMENTUM,
        "min_squared_grad": MIN_SQUARED_GRAD,
        "gamma": GAMMA,
        "epsilon": EPSILON,
        "seed": SEED,
        "env": args.env
    }
    os.environ["WANDB_LOG_LEVEL"] = "error"
    
    wandb.init(project="DQN_minatar", 
               config=config,
               entity="angela-h",name=f"runs/{args.env}",
               settings=wandb.Settings(console="off",log_internal=str(Path(__file__).parent / 'wandb' / 'null')),
               )

    # env = Environment(args.env)
    from env import make_minatar_env
    env_ids=["MinAtar/Breakout-v0","MinAtar/Asterix-v0", "MinAtar/Freeway-v0", "MinAtar/Seaquest-v0", "MinAtar/SpaceInvaders-v0"]
    train_envs=[]
    for i in env_ids:
        train_envs.append(make_minatar_env(env_id=i, idx=0, capture_video=False, run_name="DQN_minatar",seed=SEED)())
    # writer = SummaryWriter(f"runs/{args.env}_replayon_targeton")
    for i,env in enumerate(train_envs):
        dqn(env, False, False, "dqn_out", load_path=args.load_path)
