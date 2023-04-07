import argparse
from peaceful_pie.unity_comms import UnityComms
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np

from gymnasium import spaces
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import CnnLstmPolicy
from stable_baselines3.common.monitor import Monitor
import torch as th
import torch.nn as nn



from my_env import MyEnv

@dataclass
class MyVector3:
    x: float
    y: float
    z: float

@dataclass
class RlResult:
    reward: float
    done: bool
    obs: list


# Example of CustomNetwork structure to use the stable-baselines3 PPO implementation

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding="valid"),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=5),
    n_lstm=32,
)

def run(args: argparse.Namespace) -> None:
    unity_comms = UnityComms(port = 9000)
    my_env = MyEnv(unity_comms=unity_comms)
    my_env = Monitor(my_env)
    ppo = PPO(CnnLstmPolicy, env=my_env, policy_kwargs=policy_kwargs, verbose=1)

    # model_path = "saved_models/model_name"
    # ppo = PPO.load(model_path,env=my_env)

    for i in range(100):
        ppo.learn(10000)
        ppo.save("saved_models/model_lstm_"+str(i*10000))



    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=9000)
    args = parser.parse_args()
    run(args)