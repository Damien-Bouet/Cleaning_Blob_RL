import argparse
from peaceful_pie.unity_comms import UnityComms
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np

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


def run(args: argparse.Namespace) -> None:
    unity_comms = UnityComms(port = 9000)
    reward = 0
    for _ in range(1):
        res = unity_comms.Step(action=args.action, ResultClass=RlResult)
        reward += res.reward
    print(" =================== REWARD ====================")
    print(reward)
    # plt.imshow(np.array(res.obs),cmap="gray")
    # plt.show()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=9000)
    parser.add_argument('--action', type=int, required=True)
    args = parser.parse_args()
    run(args)