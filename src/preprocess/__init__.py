"""
the preprocess module provides convenience functions for the ETL
"""
from box import Box
from pathlib import Path


def read_config(path: str = "config", *paths):
    """
    get config as dictionary with attribute access to values which can also be hashed
    :param path: paths to configuration files or directories wich configuration files
    :param paths: additional paths
    :return: box.Box: config as dictionary with attribute access to values

    Examples:
        .. code-block:: python

            from churn_model.preprocess import read_config

            # read the default config from the "config" directory and update it with the user config "my_config.yaml"
            config = read_config("config", "my+config.yaml")
    """
    # expand paths to yaml files
    file_paths = []
    for path in [path] + list(paths):
        if path.endswith(".yaml") or path.endswith(".yml"):
            file_paths.append(path)
        else:
            # Glob the given relative pattern in the directory
            # represented by this path, yielding all matching files (of any kind)
            file_paths.extend(list(Path(path).glob("*.yml")))
            file_paths.extend(list(Path(path).glob("*.yaml")))
    # read files
    config = Box()
    for path in file_paths:
        # Recursively merge dictionaries or Boxes together instead of replacing
        config.merge_update(Box.from_yaml(filename=path))

    return config