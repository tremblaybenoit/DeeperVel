import os


def get_repo_root() -> str:
    """ Get the root directory of the repository.

        Returns:
        --------
        str: The root directory of the repository.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))


def get_config_path() -> str:
    """ Get the path to the configuration directory.

        Returns:
        --------
        str: The path to the configuration directory.
    """
    return os.path.join(get_repo_root(), "config")


def get_list(val, default, n: int) -> list:
    """ Convert a value to a list of length n.

        Parameters:
        -----------
        val: Any. The value to convert.
        default: Any. The default value to use if val is None.
        n: int. The desired length of the list.

        Returns:
        --------
        list: A list of length n containing the value or default.
    """

    # If val is None, return a list of default values
    if val is None:
        return [default] * n
    # If val is a single value, return a list of that value repeated n times
    if isinstance(val, list):
        # If val is a list, check if its length matches n
        if len(val) != n:
            raise ValueError(f"List argument must have same length as iters ({n})")
        return val
    return [val] * n
