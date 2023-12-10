import argparse
from peaceful_pie.unity_comms import UnityComms

from core_module.exceptions import UnityConnectionError

def run(args: argparse.Namespace) -> None:
    """
    Apply a step of gravity in the Unity simulation

    Parameters:
    - args (argparse.Namespace): Command-line arguments.

    Returns:
    None
    """
    unity_comms = UnityComms(port = args.port)
    if unity_comms.Gravity(retry=False) is not None:
        print("Gravity applied")
    else:
        raise UnityConnectionError()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=9000)
    args = parser.parse_args()
    run(args)