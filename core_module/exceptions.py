
class UnityConnectionError(Exception):
    """Exception raised connection error between Unity and python.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message: str = "Connection error between Unity and the python script.\nPlease check that the scene is running and that the given localhost port is correct (default to 9000)."):
        self.message = message
        super().__init__(self.message)