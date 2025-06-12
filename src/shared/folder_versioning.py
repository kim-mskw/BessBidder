import os


def get_current_dir_version(dir_path):
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"The directory {dir_path} is empty.")
    exisiting_dirs = os.listdir(dir_path)
    exisiting_dirs = [int(x) for x in exisiting_dirs]
    max_dir = max(exisiting_dirs)
    folder_path = os.path.join(dir_path, str(max_dir))
    return folder_path


def create_new_dir_version(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    exisiting_dirs = os.listdir(dir_path)
    exisiting_dirs = [int(x) for x in exisiting_dirs]
    if len(exisiting_dirs) != 0:
        max_dir = max(exisiting_dirs) + 1
    else:
        max_dir = 0

    folder_path = os.path.join(dir_path, str(max_dir))
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    return folder_path


def make_versioned_dir(dir_path):
    exisiting_dirs = os.listdir(dir_path)
    exisiting_dirs = [int(x) for x in exisiting_dirs]
    max_dir = max(exisiting_dirs)

    folder_path = os.path.join(dir_path, str(max_dir))

    return folder_path
