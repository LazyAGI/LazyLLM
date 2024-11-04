import os
import datetime

from lazyllm import LOG

def delete_old_files(directory):
    now = datetime.datetime.now()
    for root, dirs, files in os.walk(directory):
        for name in files:
            file_path = os.path.join(root, name)
            try:
                creation_time = datetime.datetime.fromtimestamp(os.path.getctime(file_path))
                if (now - creation_time).days > 1:
                    os.remove(file_path)
                    LOG.info(f"Deleted: {file_path}")
            except Exception as e:
                LOG.error(f"Error deleting file {file_path}: {e}")
        for name in dirs:
            dir_path = os.path.join(root, name)
            try:
                creation_time = datetime.datetime.fromtimestamp(os.path.getctime(dir_path))
                if (now - creation_time).days > 1:
                    os.rmdir(dir_path)
                    LOG.info(f"Deleted: {dir_path}")
            except Exception as e:
                LOG.error(f"Error deleting directory {dir_path}: {e}")
