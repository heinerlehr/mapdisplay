import orjson
import pandas as pd
from pathlib import Path

def singleton(class_):
    """
    Decorator that implements the singleton pattern for a class.

    This decorator ensures that only one instance of the decorated class is created.
    Subsequent calls to instantiate the class will return the same instance.

    Args:
        class_: The class to be decorated as a singleton.

    Returns:
        function: A wrapper function that manages the singleton instance.

    Attributes:
        reset: A function attached to the wrapper that clears all singleton instances.

    Example:
        @singleton
        class MyClass:
            def __init__(self, value):
                self.value = value

        obj1 = MyClass(10)
        obj2 = MyClass(20)
        # obj1 and obj2 refer to the same instance
        # obj1.value == 10, obj2.value == 10

        # To reset and allow new instance creation:
        MyClass.reset()
    """

    instances = {}

    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]

    def reset():
        instances.clear()

    getinstance.reset = reset  # type: ignore
    return getinstance  # type: ignore


def read_json(path: str|Path) -> pd.DataFrame:
    """
    Reads a JSON file from the specified path and returns its contents as a pandas DataFrame.

    Args:
        path (str | Path): The path to the JSON file.
    Returns:
        pd.DataFrame: A DataFrame containing the data from the JSON file.
    """
    with open(path, 'rb') as f:
        data = orjson.loads(f.read())
    return pd.DataFrame(data)