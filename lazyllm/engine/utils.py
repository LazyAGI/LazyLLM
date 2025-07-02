import os
import shutil
from typing import List


def move_files_to_target_dir(file_list: List[str], target_dir: str) -> List[str]:
    if not file_list:
        return []

    target_dir = os.path.abspath(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    moved_files = []
    for file_path in file_list:
        if os.path.exists(file_path):
            filename = os.path.basename(file_path)
            target_path = os.path.join(target_dir, filename)
            shutil.move(file_path, target_path)
            moved_files.append(target_path)
        else:
            moved_files.append(file_path)

    return moved_files
