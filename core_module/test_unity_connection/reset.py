import argparse
from peaceful_pie.unity_comms import UnityComms

from core_module.exceptions import UnityConnectionError

def run(args: argparse.Namespace) -> None:
    """
    Reset the environnement by covering the boat with algae and setting the Agent position randomly
    """
    unity_comms = UnityComms(port = args.port)
    if unity_comms.Reset(retry=False) is not None:
        print("Environnement reseted, with the algae parameters set in Unity")
    else:
        raise UnityConnectionError()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=9000)
    args = parser.parse_args()
    run(args)