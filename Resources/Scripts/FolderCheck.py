from os.path import isdir
from os import makedirs


def check_and_create_folder(folder_path: str):
    if isdir(folder_path):
        return

    makedirs(folder_path)
