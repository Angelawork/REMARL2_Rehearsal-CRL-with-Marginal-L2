from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
import gymnasium as gym
import numpy as np
import torch
import random
from gym import ObservationWrapper

class ResizeChannels(ObservationWrapper):
    def __init__(self, env, target_channel=7):
        super().__init__(env)
        self.target_channel = target_channel
        try:
            height, width = self.observation_space.shape[0], self.observation_space.shape[1]
        except:
            # if its minatar Environment obj
            height, width = self.env.state_shape()[0], self.env.state_shape()[1]
            
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(height, width, self.target_channel),
            dtype=np.uint8
        )

    def observation(self, observation):
      data = observation
      if isinstance(observation, tuple):
        obs, info = observation
        data = obs

      curr_channel = data.shape[-1]
      if curr_channel > self.target_channel:
        data = data[..., :self.target_channel]
      elif curr_channel < self.target_channel:
        #pad with zeros if fewer channels
        padding = np.zeros((*data.shape[:-1], self.target_channel - curr_channel), dtype=np.uint8)
        data = np.concatenate([data, padding], axis=-1)

      # print("*****************observation shape and type:*****************")
      # print(data.shape)
      # print(data.dtype)
      if isinstance(observation, tuple):
        return data, info
      else:
        return data

class BoolToUint8(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        try:
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=self.observation_space.shape, dtype=np.uint8
            )
        except:
            # if its minatar Environment obj
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=self.env.state_shape(), dtype=np.uint8
            )

    def observation(self, observation):
      data = observation
      if isinstance(observation, tuple):
        obs, info = observation
        data = obs

      if data.dtype != np.uint8:
        data = np.array(data, dtype=np.float32)
        data = (data / data.max()) * 255
        data = data.astype(np.uint8)

      if isinstance(observation, tuple):
        return data, info
      else:
        return data

def make_minatar_env(env_id, idx, capture_video, run_name, obs_size = (10,10)):
    """ available usage: 
    env_id from ["MinAtar/Asterix-v0", "MinAtar/Breakout-v0", "MinAtar/Freeway-v0", "MinAtar/Seaquest-v0", "MinAtar/SpaceInvaders-v0", "MinAtar/Asterix-v1", "MinAtar/Breakout-v1", "MinAtar/Freeway-v1", "MinAtar/Seaquest-v1", "MinAtar/SpaceInvaders-v1"]
    """
    print(f"Setting up MinAtar env: {env_id}")
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        
        #if use Environment
        # from minatar import Environment
        # ["asterix", "breakout", "freeway", "seaquest", "space_invaders", "asterix"]
        # env = Environment(env_id)

        # wrappers
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # env = MaxAndSkipEnv(env, skip=4)
        # env = EpisodicLifeEnv(env)
        # env = ClipRewardEnv(env)
        env = BoolToUint8(env)
        env = ResizeChannels(env)
        env = gym.wrappers.ResizeObservation(env, obs_size)
        env = gym.wrappers.FrameStack(env, 4)
        return env

    return thunk