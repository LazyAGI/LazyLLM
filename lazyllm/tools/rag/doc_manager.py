import os
import hashlib
import json
from typing import List, Optional, Dict
from pydantic import BaseModel

from starlette.responses import RedirectResponse
from fastapi import UploadFile

import lazyllm
from lazyllm import FastapiApp as app
from .utils import DocListManager, BaseResponse


class DocManager(lazyllm.ModuleBase):
    def __init__(self, dlm: DocListManager) -> None:
        super().__init__()
        self._manager = dlm

    @app.get("/", response_model=BaseResponse, summary="docs")
    def document(self):
        return RedirectResponse(url="/docs")

    @app.get("/list_kb_groups")
    def list_kb_groups(self):
        try:
            return BaseResponse(data=self._manager.list_all_kb_group())
        except Exception as e:
            return BaseResponse(code=500, msg=str(e), data=None)

    @app.post("/upload_files")
    def upload_files(self, files: List[UploadFile], override: bool = False,
                     metadatas: Optional[str] = None, user_path: Optional[str] = None):
        try:
            assert user_path is None or not user_path.startswith('/'), 'Cannot give absolute path'
            if metadatas:
                metadatas: Optional[List[Dict[str, str]]] = json.loads(metadatas)
                assert len(files) == len(metadatas), 'Length of files and metadatas should be the same'
            file_paths = [os.path.join(self._manager._path, user_path or '', file.filename) for file in files]
            lazyllm.LOG.warning(user_path)
            lazyllm.LOG.warning(file_paths)
            ids = self._manager.add_files(file_paths, metadatas=metadatas, status=DocListManager.Status.working)
            results = []
            for file, path in zip(files, file_paths):
                if os.path.exists(path):
                    if not override:
                        results.append('Duplicated')
                        continue

                content = file.file.read()
                directory = os.path.dirname(path)
                if directory:
                    os.makedirs(directory, exist_ok=True)

                with open(path, 'wb') as f:
                    f.write(content)
                file_id = hashlib.sha256(path.encode()).hexdigest()
                self._manager.update_file_status([file_id], status=DocListManager.Status.success)
                results.append('Success')

            return BaseResponse(data=[ids, results])
        except Exception as e:
            return BaseResponse(code=500, msg=str(e), data=None)

    @app.get("/list_files")
    def list_files(self, limit: Optional[int] = None, details: bool = True):
        try:
            return BaseResponse(data=self._manager.list_files(limit=limit, details=details))
        except Exception as e:
            return BaseResponse(code=500, msg=str(e), data=None)

    @app.get("/list_files_in_group")
    def list_files_in_group(self, group_name: str, limit: Optional[int] = None, alive: Optional[bool] = None):
        try:
            if group_name == '__None__': group_name = None
            Status = DocListManager.Status
            status = [Status.success, Status.waiting, Status.working] if alive else Status.all
            return BaseResponse(data=self._manager.list_kb_group_files(group_name, limit, details=True, status=status))
        except Exception as e:
            import traceback
            return BaseResponse(code=500, msg=str(e) + '\ntraceback:\n' + str(traceback.format_exc()), data=None)

    class FileGroupRequest(BaseModel):
        file_ids: List[str]
        group_name: str

    @app.post("/add_files_to_group_by_id")
    def add_files_to_group_by_id(self, request: FileGroupRequest):
        try:
            self._manager.add_files_to_kb_group(request.file_ids, request.group_name)
            return BaseResponse()
        except Exception as e:
            return BaseResponse(code=500, msg=str(e), data=None)

    @app.post("/add_files_to_group")
    def add_files_to_group(self, files: List[UploadFile], group_name: str,
                           override: bool = False, metadatas: Optional[str] = None):
        try:
            ids = self.upload_files(files, override=override, metadatas=metadatas)[0]
            self._manager.add_files_to_kb_group(ids, group_name)
            return BaseResponse(data=ids)
        except Exception as e:
            return BaseResponse(code=500, msg=str(e), data=None)

    @app.post("/delete_files")
    def delete_files(self, file_ids: List[str]):
        try:
            self._manager.update_kb_group_file_status(file_ids=file_ids, status=DocListManager.Status.deleting)
            self._manager.update_file_status(file_ids=file_ids, status=DocListManager.Status.deleting)
            return BaseResponse()
        except Exception as e:
            return BaseResponse(code=500, msg=str(e), data=None)

    @app.post("/delete_files_from_group")
    def delete_files_from_group(self, request: FileGroupRequest):
        try:
            self._manager.update_kb_group_file_status(
                file_ids=request.file_ids, status=DocListManager.Status.deleting, group=request.group_name)
            return BaseResponse()
        except Exception as e:
            return BaseResponse(code=500, msg=str(e), data=None)

    def __repr__(self):
        return lazyllm.make_repr("Module", "DocManager")
