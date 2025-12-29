from concurrent.futures import Executor
from typing import Any, Callable, Dict, Iterable


def submit_with_mapping(executor: Executor, items: Iterable[Any], fn: Callable[[Any], Any]):
    """
    Submit tasks to an executor and return a mapping of future -> item.
    """
    futures = [executor.submit(fn, item) for item in items]
    return {
        future: item
        for future, item in zip(futures, items)
    }
