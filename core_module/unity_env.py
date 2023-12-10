import argparse
from peaceful_pie.unity_comms import UnityComms
from dataclasses import dataclass
from gymnasium import Env, spaces
import numpy as np
from typing import Tuple, List, Dict

from core_module.exceptions import UnityConnectionError

# testing testing 123

class MyVector3:
    """
    A data class representing a 3D vector.

    Attributes:
    - x (float): The x-coordinate.
    - y (float): The y-coordinate.
    - z (float): The z-coordinate.
    """
    x: float
    y: float
    z: float


@dataclass
class RlResult:
    """
    A data class representing the result of a reinforcement learning step.

    Attributes:
    - reward (float): The reward obtained in the step.
    - done (bool): A flag indicating whether the episode is done.
    - obs (list): A list representing the observation obtained in the step.
    """
    reward: float
    done: bool
    obs: list


@dataclass
class ImageObs:
    """
    A data class representing an image observation.

    Attributes:
    - obs (list): A list representing the image observation.
    """
    obs: list


@dataclass
class RlReset:
    """
    A data class representing the reset information in reinforcement learning.

    Attributes:
    - obs (list): A list representing the observation after the reset.
    - total_reward_available (float): The total available reward after the reset.
    """
    obs: list
    total_reward_available: float

class MyEnv(Env):
    """
    Custom environment class that inherits from OpenAI Gym's Env.

    Attributes:
    - unity_comms (UnityComms): An instance of UnityComms for communication with Unity environment.
    - config (dict): Configuration parameters for the environment.
    - action_space (Discrete): The action space for the environment.
    - observation_space (Box): The observation space for the environment.
    - total_reward (float): The total reward obtained in the environment.
    """
    def __init__(self, unity_comms: UnityComms) -> None:
        super().__init__()
        self.unity_comms = unity_comms
        self.config = {"width":32, "height" : 24}
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0,high=255,shape=(self.config["height"],self.config["width"],1), dtype=np.uint8)
        self.total_reward = 0


    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take a step in the environment based on the given action.

        Parameters:
        - action: The action to take in the environment.

        Returns:
        Tuple containing:
        - observation (np.ndarray): The observation after the step.
        - reward (float): The reward obtained in the step.
        - done (bool): A flag indicating whether the episode is done.
        - info (dict): Additional information about the step.
        """
        action = int(action)    #We need to use int and not np.int64 givent by spaces.Discrete
        rl_result : RlResult = self.unity_comms.Step(action=action, ResultClass=RlResult, retry=False)
        if rl_result is None:
            raise UnityConnectionError("The Unity connection was closed.")
        info = {"finished":rl_result.done}
        return np.array(rl_result.obs, dtype=np.uint8), rl_result.reward, rl_result.done, False, info


    def reset(self) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment.

        Returns:
        Tuple containing:
        - observation (np.ndarray): The observation after the reset.
        - info (dict): Additional information about the reset.
        """
        reset_res = self.unity_comms.Reset(ResultClass=RlReset, retry=False)
        if reset_res is None:
            raise UnityConnectionError("The Unity connection was closed.")
        obs = np.array(reset_res.obs, dtype=np.uint8)
        self.total_reward = reset_res.total_reward_available
        return obs, {}
        