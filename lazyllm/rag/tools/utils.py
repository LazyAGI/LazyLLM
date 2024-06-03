import os
import shutil
from typing import List, Callable, Generator, Dict, Any

import pydantic
from pydantic import BaseModel
from fastapi import UploadFile
from concurrent.futures import ThreadPoolExecutor, as_completed

import lazyllm

class BaseResponse(BaseModel):
    code: int = pydantic.Field(200, description="API status code")
    msg: str = pydantic.Field("success", description="API status message")
    data: Any = pydantic.Field(None, description="API data")

    class Config:
        json_schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
            }
        }

def run_in_thread_pool(
        func: Callable,
        params: List[Dict] = [],
) -> Generator:
    tasks = []
    with ThreadPoolExecutor() as pool:
        for kwargs in params:
            thread = pool.submit(func, **kwargs)
            tasks.append(thread)

        for obj in as_completed(tasks):
            yield obj.result()

Default_Suport_File_Types = [".docx", ".pdf", ".txt", ".json"]
def _save_file(_file: UploadFile, _file_path: str):
    file_content = _file.file.read()
    with open(_file_path, "wb") as f:
        f.write(file_content)

def _convert_path_to_underscores(file_path: str) -> str:
    return file_path.replace("/", "_").replace("\\", "_")

def _save_file_to_cache(file: UploadFile, cache_dir: str, suport_file_types: List[str]) -> list:
    to_file_path = os.path.join(cache_dir, file.filename)

    sub_result_list_real_name = []
    if file.filename.endswith(".tar"):
        def unpack_archive(tar_file_path: str, extract_folder_path: str):
            import tarfile
            out_file_names = []
            try:
                with tarfile.open(tar_file_path, 'r') as tar:
                    file_info_list = tar.getmembers()
                    for file_info in list(file_info_list):
                        file_extension = os.path.splitext(file_info.name)[-1]
                        if file_extension in suport_file_types:
                            tar.extract(file_info.name, path=extract_folder_path)
                            out_file_names.append(file_info.name)
            except tarfile.TarError as e:
                lazyllm.LOG.error(f'untar error: {e}')
                raise e

            return out_file_names

        _save_file(file, to_file_path)
        out_file_names = unpack_archive(to_file_path, cache_dir)
        sub_result_list_real_name.extend(out_file_names)
        os.remove(to_file_path)
    else:
        file_extension = os.path.splitext(file.filename)[-1]
        if file_extension in suport_file_types:
            if not os.path.exists(to_file_path):
                _save_file(file, to_file_path)
            sub_result_list_real_name.append(file.filename)
    return sub_result_list_real_name

def save_files_in_threads(
    files: List[UploadFile],
    override: bool,
    source_path,
    suport_file_types: List[str] = Default_Suport_File_Types
):
    real_dir = source_path
    cache_dir = os.path.join(source_path, "cache")

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    for dir in [real_dir, cache_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    param_list = [{"file": file, "cache_dir": cache_dir, "suport_file_types": suport_file_types} for file in files]

    result_list = []
    for result in run_in_thread_pool(_save_file_to_cache, params=param_list):
        result_list.extend(result)

    already_exist_files = []
    new_add_files = []
    overwritten_files = []

    for file_name in result_list:
        real_file_path = os.path.join(real_dir, _convert_path_to_underscores(file_name))
        cache_file_path = os.path.join(cache_dir, file_name)

        if os.path.exists(real_file_path):
            if not override:
                already_exist_files.append(file_name)
            else:
                os.rename(cache_file_path, real_file_path)
                overwritten_files.append(file_name)
        else:
            os.rename(cache_file_path, real_file_path)
            new_add_files.append(file_name)

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    return (already_exist_files, new_add_files, overwritten_files)

def rename_file(source, target):
    try:
        os.rename(source, target)
    except FileNotFoundError:
        lazyllm.LOG.error(f"file {source} not found")
    except Exception as e:
        lazyllm.LOG.error(f"error when batch rename file：{e}")
