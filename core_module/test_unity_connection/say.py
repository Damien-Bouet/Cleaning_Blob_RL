import argparse
from peaceful_pie.unity_comms import UnityComms

from core_module.exceptions import UnityConnectionError

def run(args: argparse.Namespace) -> None:
    """
    Send the message (default to "Hello Wordl") to the Unity console
    """
    unity_comms = UnityComms(port = args.port)
    if unity_comms.Say(message=args.message, retry=False) is not None:
        print("Message sent to the unity console")
    else:
        raise UnityConnectionError()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--message', type=str, default="Hello World")
    parser.add_argument('--port', type=int, default=9000)
    args = parser.parse_args()
    run(args)