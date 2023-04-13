import argparse
from peaceful_pie.unity_comms import UnityComms
from dataclasses import dataclass

@dataclass
class MyVector3:
    x: float
    y: float
    z: float


def run(args: argparse.Namespace) -> None:
    unity_comms = UnityComms(port = 9000)
    res: MyVector3 = unity_comms.GetPos(ResultClass=MyVector3)
    print('res', res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=9000)
    args = parser.parse_args()
    run(args)