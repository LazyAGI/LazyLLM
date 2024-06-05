
import os
import shutil
from typing import List

import lazyllm
from .base import DocImpl

DATA_DIR = "__data"
DEFAULT_DIR = "default"

class DocGroupImpl(lazyllm.ModuleBase):
    def __init__(self, dataset_path, embed) -> None:
        super().__init__()
        self._dataset_path = dataset_path
        self._embed = embed
        assert os.path.exists(self.dataset_path), f"{self.dataset_path} does not exist"

        if DEFAULT_DIR not in self.list_groups():
            self.new_group(DEFAULT_DIR)
        self._move_file_to_default()

        file_paths = self._list_all_files(self.dataset_path, lambda x: DATA_DIR in x)
        self._impl: DocImpl = DocImpl(doc_files=file_paths, embed=self._embed, doc_name="lazyllm_doc")

    @property
    def dataset_path(self):
        return self._dataset_path

    def _move_file_to_default(self):
        try:
            items = os.listdir(self.dataset_path)
            items = [os.path.join(self.dataset_path, item) for item in items]
            files = [item for item in items if os.path.isfile(item)]

            group_path = self.get_group_source_path(group_name=DEFAULT_DIR)
            for file in files:
                shutil.move(file, os.path.join(group_path, os.path.basename(file)))

        except Exception as e:
            return str(e)

    def new_group(self, group_name: str):
        if os.path.exists(self.get_group_path(group_name=group_name)):
            raise Exception(f"{group_name} already exists[{self.get_group_path(group_name=group_name)}]")

        for path in [
            self.get_group_path(group_name),
            self.get_gropu_data_path(group_name),
            self.get_group_source_path(group_name)
        ]:
            os.makedirs(path)

    def delete_group(self, group_name: str):
        try:
            shutil.rmtree(self.get_group_path(group_name))
            list_files = self.list_files(group_name)
            self._impl.delete_files(list_files)
        except Exception as e:
            raise Exception(f"{self.get_group_path(group_name)} delete error, exception:{e}")

        return f"delete {group_name} success"

    def list_groups(self):
        groups = self._list_all_subdirectories(self.dataset_path, lambda x: DATA_DIR not in x)
        return [dir[len(self.dataset_path) + 1:] for dir in groups]

    def add_files(self, group_name: str, files: List[str]):
        source_path = self.get_group_source_path(group_name)
        files = [os.path.join(source_path, file_path) for file_path in files]
        self._impl.add_files(files)

    def delete_files(self, group_name: str, files: List[str], is_del_source: bool = True):
        self._impl.delete_files(files)

        if not is_del_source: return

        source_path = self.get_group_source_path(group_name)
        for file_path in [os.path.join(source_path, file_path) for file_path in files]:
            os.remove(file_path)

    def list_files(self, group_name: str) -> List[str]:
        file_paths = self._list_all_files(self.get_group_source_path(group_name=group_name), lambda x: DATA_DIR in x)
        return [os.path.basename(file_path) for file_path in file_paths]

    def get_group_path(self, group_name: str):
        return os.path.join(self.dataset_path, group_name)

    def get_gropu_data_path(self, group_name: str):
        return os.path.join(self.get_group_path(group_name=group_name), DATA_DIR)

    def get_group_source_path(self, group_name: str):
        return os.path.join(self.get_gropu_data_path(group_name=group_name), "sources")

    def _list_all_subdirectories(self, directory, filter_dir=None):
        if not os.path.isabs(directory):
            raise ValueError("directory must be an absolute path")
        try:
            subdirectories = []

            for root, dirs, files in os.walk(directory):
                dirs = [os.path.join(root, dir) for dir in dirs]
                filtered_dirs = list(filter(filter_dir, dirs)) if filter else dirs
                subdirectories.extend(filtered_dirs)

            return subdirectories
        except Exception as e:
            lazyllm.LOG.error(f"Error while listing subdirectories in {directory}: {e}")
            return []

    def _list_all_files(self, directory, filter_func=None):
        if not os.path.isabs(directory):
            raise ValueError("directory must be an absolute path")

        try:
            files_list = []

            for root, dirs, files in os.walk(directory):
                files = [os.path.join(root, file_path) for file_path in files]
                filtered_files = list(filter(filter_func, files)) if filter_func else files
                files_list.extend(filtered_files)

            return files_list
        except Exception as e:
            lazyllm.LOG.error(f"Error while listing files in {directory}: {e}")
            return []

    def generate_signature(self, similarity, similarity_kw, parser):
        return self._impl.generate_signature(similarity, similarity_kw, parser)

    def query_with_sig(self, string, signature, parser):
        return self._impl._query_with_sig(string, signature, parser)

    def __repr__(self):
        return lazyllm.make_repr('Module', 'DocGroupImpl')
