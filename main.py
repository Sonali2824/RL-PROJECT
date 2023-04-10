import gymnasium as gym
import sys
import torch
import numpy as np
import highway_env
import warnings
import time
import pprint
from PPO import PPO

highway_env.register_highway_envs()

from rl_agents.agents.common.factory import  load_environment

warnings.filterwarnings('ignore')


env_config = {
    "id": "intersection-v0",
    "import_module": "highway_env",
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 15,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20]
        },
        "absolute": True,
        "order": "shuffled"
    },
    "destination":"o1"
}


env = load_environment(env_config)
env.reset()
pprint.pprint(env.config)

def train():
    print("train")
    #retrain
    PPO(env)

train()

