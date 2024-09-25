import os
import re
import logging

from fastapi import UploadFile, File, HTTPException
from fastapi.responses import FileResponse

import lazyllm
from lazyllm import FastapiApp as app
from lazyllm.tools.rag.utils import BaseResponse
from lazyllm.tools.rag.db import FileRecord, FORDER_TYPE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure the directory exists
FILE_STORAGE_DIR = "/yhm/jisiyuan/LazyLLM/dataset/kb_server_test"
STORE_MODE = "minio"

def ensure_directory_exists(directory: str):
    """
    Ensure that a directory exists. If it doesn't, create it.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

ensure_directory_exists(FILE_STORAGE_DIR)

def validate_folder_name(folder_name: str) -> str:
    """
    Validate the folder name.
    """
    if not folder_name:
        raise HTTPException(status_code=400, detail="Folder name cannot be empty.")
    
    folder_name = folder_name.strip()
    if len(folder_name) < 1 or len(folder_name) > 50:
        raise HTTPException(status_code=400, detail="Folder name must be between 1 and 50 characters.")
    
    if not re.match(r'^[a-zA-Z0-9-_]+$', folder_name):
        raise HTTPException(status_code=400, detail="Folder name can only contain letters, numbers, hyphens, and underscores.")
    return folder_name

import s3fs
# MinIO 配置
minio_endpoint = "http://103.177.28.196:9000"
minio_access_key = "ROOTNAME"
minio_secret_key = "CHANGEME123"
# minio_bucket = "jsytest"

# 配置 s3fs
fs = s3fs.S3FileSystem(
    key=minio_access_key,
    secret=minio_secret_key,
    client_kwargs={'endpoint_url': minio_endpoint}
)

# file_url = f"{minio_endpoint}/{minio_bucket}/{relative_path}"


if STORE_MODE == "local":
    import fsspec
    from fsspec.implementations.local import LocalFileSystem
    fs = LocalFileSystem()
    # base_name = "kb_server_test"
elif STORE_MODE == "minio":
    import s3fs
    # MinIO 配置
    minio_endpoint = "http://103.177.28.196:9000"
    minio_access_key = "ROOTNAME"
    minio_secret_key = "CHANGEME123"
    # minio_bucket = "jsytest"

    # 配置 s3fs
    fs = s3fs.S3FileSystem(
        key=minio_access_key,
        secret=minio_secret_key,
        client_kwargs={'endpoint_url': minio_endpoint}
    )
    # base_name = "kb_server_test"



class FileServer:
    """
    File server for managing file operations.
    """

    @app.post("/create_folder")
    def create_folder(self, base_name: str = '', folder_path: str = None) -> BaseResponse:
    # def create_folder(self, target_path: str = '', folder_name: str = None) -> BaseResponse:
        """
        Create a new folder.
        """
        # folder_name = validate_folder_name(folder_name)
       
        try:
            real_folder_path = os.path.join(base_name, folder_path)
            if STORE_MODE == "local":
                real_folder_path = os.path.join(FILE_STORAGE_DIR, real_folder_path)
            if not fs.exists(real_folder_path):
                fs.mkdir(real_folder_path) 

            _location = os.path.join(base_name, folder_path)
            node = FileRecord.create(
                file_name=folder_path,
                file_path=_location,
                file_type=FORDER_TYPE,
                file_size=0
            )
            return BaseResponse(msg=f"Folder '{folder_path}' created successfully.", data_id=node.id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/upload_file")
    async def upload_file(self, file: UploadFile = File(...), base_name: str = '', target_path: str = '', is_overwrite: bool = True) -> BaseResponse:
    # async def upload_file(self, file: UploadFile = File(...), target_path: str = '', is_overwrite: bool = True) -> BaseResponse:
        """
        Upload a file to the server.
        """
        # file_location = os.path.join(FILE_STORAGE_DIR, target_path, file.filename) if target_path else os.path.join(FILE_STORAGE_DIR, file.filename)
        
        try:
            _location = os.path.join(base_name, target_path, file.filename)
            file_info = FileRecord.first(file_path=_location)
            if file_info and not is_overwrite:
                return BaseResponse(msg="File already exists and will not be overwritten.")
            
            if file_info and is_overwrite:
                FileRecord.del_node(file_path=_location)
            
            
            file_path = f"{base_name}/{target_path}/{file.filename}"
            if STORE_MODE == "local":
                file_path = os.path.join(FILE_STORAGE_DIR, file_path)
            if fs.exists(file_path):
                fs.rm(file_path)
                # print(f"remove {file.filename} from  {base_name}/{target_path}")
            # 保存文件
            cache_location = os.path.join(FILE_STORAGE_DIR, ".cache", file.filename)
            with open(cache_location, "wb+") as file_object:
                file_object.write(file.file.read())
            fs.put_file(cache_location, file_path)

            file_info = FileRecord(
                file_name=file.filename,
                file_path=_location,
                file_type=os.path.splitext(file.filename)[1][1:].upper(),
                file_size=file.size
            )
            FileRecord.add_node(file_info)
            return BaseResponse(msg=f"File '{file.filename}' uploaded successfully.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/list_files")
    def list_files(self, skip: int = 0, limit: int = 10) -> BaseResponse:
        """
        List files with pagination.
        """
        files = FileRecord.filter_by(skip=skip, limit=limit)
        files = [repr(file) for file in files]
        return BaseResponse(data=files)

    @app.get("/get_file")
    def get_file(self, file_id: int):
        """
        Download a file by its ID.
        """
        file_record = FileRecord.first(id=file_id)
        
        if not file_record:
            raise HTTPException(status_code=404, detail="File not found")

        if file_record.file_type != FORDER_TYPE:
            real_file_path = os.path.join(FILE_STORAGE_DIR, file_record.file_path) if STORE_MODE == "local" else file_record.file_path
            cache_file_path = os.path.join(FILE_STORAGE_DIR, ".cache", file_record.file_name)
            fs.get_file(real_file_path, cache_file_path)
            return FileResponse(path=cache_file_path, filename=file_record.file_name)
        else:
            files = FileRecord.filter_by_conditions(FileRecord.file_path.startswith(file_record.file_path))
            return self.build_file_tree(files)

    @app.get("/delete_file")
    def delete_file(self, file_id: int) -> BaseResponse:
        """
        Delete a file by its ID.
        """
        file_record = FileRecord.first(id=file_id)
        if not file_record:
            raise HTTPException(status_code=404, detail="File not found")
        
        try:
            os.remove(file_record.file_path)
        except Exception as e:
            logger.error(f"Failed to delete file {file_record.file_path}: {str(e)}")
        
        try:
            FileRecord.del_node(id=file_id)
        except Exception as e:
            logger.error(f"Failed to delete file record {file_id}: {str(e)}")

        return BaseResponse(msg=f"File ID {file_id} deleted successfully.")

    # @app.get("/get_file_tree")
    def build_file_tree(self, all_files) -> BaseResponse:
        """
        Get the file tree structure.
        """

        files_dict = {}
        for file in all_files:
            files_dict[file.file_path] = {
                "name": file.file_name,
                "type": "folder" if file.file_type == "folder" else "file",
                "children": []
            }

        file_tree = []
        for file in all_files:
            parent_path = os.path.dirname(file.file_path)
            if parent_path in files_dict:
                files_dict[parent_path]["children"].append(files_dict[file.file_path])
            else:
                file_tree.append(files_dict[file.file_path])
        return file_tree
    
    @app.get("/get_file_tree")
    def get_file_tree(self):
        all_files = FileRecord.all()  # Get all file information
        data=self.build_file_tree(all_files)
        return BaseResponse(data=data)
    

# m = lazyllm.ServerModule(FileServer(), launcher=lazyllm.launchers.empty(sync=False))
# m.start()