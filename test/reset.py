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


def run(args: argparse.Namespace) -> None:
    unity_comms = UnityComms(port = 9000)
    res = unity_comms.Reset()

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=9000)
    args = parser.parse_args()
    run(args)