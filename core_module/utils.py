import argparse

def str2bool(v: str) -> bool:
    """
    Convert a string to a boolean value.

    Parameters:
    - v (str): The input string.

    Returns:
    bool: The boolean value corresponding to the input string.

    Raises:
    argparse.ArgumentTypeError: If the input string does not represent a boolean value.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

