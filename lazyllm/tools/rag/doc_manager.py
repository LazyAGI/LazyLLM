import os
import hashlib
from typing import List, Optional, Dict

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
    async def upload_files(self, files: List[UploadFile], override: bool,
                           metadatas: List[Dict], user_path: Optional[str] = None):
        try:
            assert user_path is None or not user_path.startswith('/'), 'Cannot give absolute path'
            assert not metadatas or len(files) == len(metadatas), 'Length of files and metadatas should be the same'
            file_paths = [os.path.join(self._manager._path, user_path or '', file.filename) for file in files]
            self._manager.add_files(file_paths, metadatas=metadatas, status=DocListManager.Status.working)
            results = []
            for file, path in zip(files, file_paths):
                content = await file.read()
                if os.path.exists(path):
                    if not override:
                        results.append('Duplicated')
                        continue
                with open(path, 'wb') as f:
                    f.write(content)
                file_id = hashlib.sha256(path.encode()).hexdigest()
                self._manager.update_file_status(file_id, status=DocListManager.Status.success)
                results.append('Success')

            return BaseResponse(data=results)
        except Exception as e:
            return BaseResponse(code=500, msg=str(e), data=None)

    @app.get("/list_files")
    def list_files(self, limit: Optional[int] = None, details: bool = True):
        try:
            return BaseResponse(data=self._manager.list_files(limit=limit, details=details))
        except Exception as e:
            return BaseResponse(code=500, msg=str(e), data=None)

    @app.get("/list_files_in_group")
    def list_files_in_group(self, group_name: str, limit: Optional[int] = None):
        try:
            return BaseResponse(data=self._manager.list_kb_group_files(group_name, limit, details=True))
        except Exception as e:
            return BaseResponse(code=500, msg=str(e), data=None)

    @app.post("/add_files_to_group")
    def add_files_to_group(self, files: List[UploadFile], group_name: str):
        try:
            self._manager.add_files_to_kb_group(files, group_name)
            return BaseResponse()
        except Exception as e:
            return BaseResponse(code=500, msg=str(e), data=None)

    @app.post("/delete_files")
    def delete_files(self, file_ids: str):
        try:
            self._manager.delete_files(file_ids)
            return BaseResponse()
        except Exception as e:
            return BaseResponse(code=500, msg=str(e), data=None)

    @app.post("/delete_files_from_group")
    def delete_files_from_group(self, group_name: str, file_ids: str):
        try:
            self._manager.delete_files_from_kb_group(file_ids, group_name)
            return BaseResponse()
        except Exception as e:
            return BaseResponse(code=500, msg=str(e), data=None)

    def __repr__(self):
        return lazyllm.make_repr("Module", "DocManager")
