import argparse
from peaceful_pie.unity_comms import UnityComms
from dataclasses import dataclass
from gymnasium import Env, spaces
import numpy as np
# from stable_baselines3.common.env_checker import check_env


# testing testing 123

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

@dataclass
class ImageObs:
    obs: list

@dataclass
class RlReset:
    obs: list
    total_reward_available: float

class MyEnv(Env):
    def __init__(self, unity_comms: UnityComms):
        super().__init__()
        self.unity_comms = unity_comms
        self.config = {"width":32, "height" : 24}
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0,high=255,shape=(self.config["height"],self.config["width"],1), dtype=np.uint8)


    def step(self, action):

        action = int(action)    #We need to use int and not np.int64 givent by spaces.Discrete

        rl_result : RlResult = self.unity_comms.Step(action=action, ResultClass=RlResult)
        info = {"finished":rl_result.done}
        return np.array(rl_result.obs, dtype=np.uint8), rl_result.reward, rl_result.done, False, info

    def reset(self):
        reset_res = self.unity_comms.Reset(ResultClass=RlReset)
        obs = np.array(reset_res.obs, dtype=np.uint8)
        self.total_reward = reset_res.total_reward_available
        return obs, {}
        

def run(args: argparse.Namespace) -> None:
    unity_comms = UnityComms(port = 9000)
    my_env = MyEnv(unity_comms=unity_comms)
    check_env(env=my_env)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=9000)
    args = parser.parse_args()
    run(args)
