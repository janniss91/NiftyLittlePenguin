import os
import sys

import torch


def create_dirs(path: str) -> None:
    """Create the directories for the given path."""
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)


def get_size(object, processed=None):
    """Recursively finds size of objects"""

    size = sys.getsizeof(object)
    if processed is None:
        processed = set()

    object_id = id(object)
    if object_id in processed:
        return 0

    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    processed.add(object_id)

    if isinstance(object, dict):
        # pylint: disable=consider-using-generator
        size += sum([get_size(v, processed) for v in object.values()])
        # pylint: disable=consider-using-generator
        size += sum([get_size(k, processed) for k in object.keys()])
    if isinstance(object, torch.Tensor):
        size += object.element_size() * object.nelement()
    if hasattr(object, "__dict__"):
        size += get_size(object.__dict__, processed)
    if hasattr(object, "__iter__") and not isinstance(
        object, (str, bytes, bytearray, torch.Tensor)
    ):
        # pylint: disable=consider-using-generator
        size += sum([get_size(i, processed) for i in object])

    return size


def convert_byte_unit(bytes: int) -> str:
    if bytes < 1024:
        return f"{bytes} B"
    if bytes < 1024**2:
        return f"{bytes / 1024:.2f} KB"
    if bytes < 1024**3:
        return f"{bytes / 1024 ** 2:.2f} MB"
    return f"{bytes / 1024 ** 3:.2f} GB"


def get_size_with_unit(object):
    return convert_byte_unit(get_size(object))
