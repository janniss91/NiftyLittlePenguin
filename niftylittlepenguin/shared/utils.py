import os


def create_dirs(path: str) -> None:
    """Create the directories for the given path."""
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
