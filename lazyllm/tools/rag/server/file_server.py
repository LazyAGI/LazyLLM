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
FILE_STORAGE_DIR = ""

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


class FileServer:
    """
    File server for managing file operations.
    """

    @app.post("/create_folder", response_model=BaseResponse)
    async def create_folder(target_path: str = '', folder_name: str = None) -> BaseResponse:
        """
        Create a new folder.
        """
        folder_name = validate_folder_name(folder_name)
        folder_path = os.path.join(FILE_STORAGE_DIR, target_path, folder_name)
        try:
            os.mkdir(folder_path)
            _location = os.path.join(target_path, folder_name)
            node = FileRecord.create(
                file_name=folder_name,
                file_path=_location,
                file_type=FORDER_TYPE,
                file_size=0
            )
            return BaseResponse(msg=f"Folder '{folder_name}' created successfully.", data=node.id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/upload_file", response_model=BaseResponse)
    async def upload_file(file: UploadFile = File(...), target_path: str = '', is_overwrite: bool = True) -> BaseResponse:
        """
        Upload a file to the server.
        """
        file_location = os.path.join(FILE_STORAGE_DIR, target_path, file.filename) if target_path else os.path.join(FILE_STORAGE_DIR, file.filename)
        
        try:
            _location = os.path.join(target_path, file.filename)
            file_info = FileRecord.first(file_path=_location)
            if file_info and not is_overwrite:
                return BaseResponse(msg="File already exists and will not be overwritten.")
            
            if file_info and is_overwrite:
                FileRecord.del_node(file_path=_location)
                os.remove(file_location)

            ensure_directory_exists(os.path.dirname(file_location))
            with open(file_location, "wb+") as file_object:
                file_object.write(file.file.read())

            file_info = FileRecord(
                file_name=os.path.basename(file_location),
                file_path=_location,
                file_type=file.content_type,
                file_size=os.path.getsize(file_location)
            )
            FileRecord.add_node(file_info)
            return BaseResponse(msg=f"File '{file.filename}' uploaded successfully.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/list_files", response_model=BaseResponse)
    def list_files(skip: int = 0, limit: int = 10) -> BaseResponse:
        """
        List files with pagination.
        """
        files = FileRecord.filter_by(skip=skip, limit=limit)
        return BaseResponse(data=files)

    @app.get("/get_file/{file_id}", response_class=FileResponse)
    def get_file(file_id: int):
        """
        Download a file by its ID.
        """
        file_record = FileRecord.first(id=file_id)
        if not file_record:
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(path=file_record.file_path, filename=file_record.file_name)

    @app.delete("/delete_file/{file_id}", response_model=BaseResponse)
    def delete_file(file_id: int) -> BaseResponse:
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

    @app.get("/get_file_tree", response_model=BaseResponse)
    def get_file_tree() -> BaseResponse:
        """
        Get the file tree structure.
        """
        def _build_file_tree():
            all_files = FileRecord.all()  # Get all file information
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
        return BaseResponse(data=_build_file_tree())
    

m = lazyllm.ServerModule(FileServer(), launcher=lazyllm.launchers.empty(sync=False))
m.start()