import argparse
from peaceful_pie.unity_comms import UnityComms
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np

from core_module.utils import str2bool
from core_module.unity_env import RlResult

from core_module.exceptions import UnityConnectionError

def run(args: argparse.Namespace) -> None:
    """
    Apply number (default to 1) steps on the Agent :
        0 : do nothing
        1 : step forward
        2 : step backward
        3 : turn clockwise
        4 : turn counter-clockwise
    """
    
    unity_comms = UnityComms(port = args.port)
    
    if args.action not in list(range(5)):
        raise ValueError("action argument should be an integer between 0 and 4. \nActions : \n- 0 = do nothing,\n- 1 = step forward, \n- 2 = step backward,\n- 3 = turn clockwise,\n- 4 = turn counter-clockwise")

    reward = 0

    for _ in range(args.number):
        res = unity_comms.Step(action=args.action, ResultClass=RlResult, retry=False)
        if res is None:
            raise UnityConnectionError()
        reward += res.reward

    print(f"total reward for the {args.number} steps : {reward}")
    if args.number < 0.001:
        print("Don't hesitate to first reset the environnement to summon the algae and see the reward for the cleaned ones")
    
    if args.show_observation:
        plt.imshow(np.array(res.obs),cmap="gray")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=9000)
    parser.add_argument('--action', type=int, required=True, help="Action to send to the blob : 0=nothing, 1='step forward', 2='step backward', 3='turn clockwise', 4='turn counter-clockwise")
    parser.add_argument('--number', type=int, default=1, help="Number of identical action to send to the blob - default:1")
    parser.add_argument("--show_observation", type=str2bool, nargs='?', const=True, default=False, help="Show Blob Camera Observation - default:False")
    args = parser.parse_args()
    run(args)