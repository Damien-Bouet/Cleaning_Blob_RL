import argparse
from peaceful_pie.unity_comms import UnityComms

from core_module.unity_env import MyVector3
from core_module.exceptions import UnityConnectionError

def run(args: argparse.Namespace) -> None:
    """
    Run the main logic of the program.

    Parameters:
    - args (argparse.Namespace): Command-line arguments.

    Returns:
    None
    """
    
    unity_comms = UnityComms(port = args.port)
    res: MyVector3 = unity_comms.GetPos(ResultClass=MyVector3, retry=False)
    if res is not None:
        print(f'Agent position : {res}')
    else:
        raise UnityConnectionError()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=9000)
    args = parser.parse_args()
    run(args)