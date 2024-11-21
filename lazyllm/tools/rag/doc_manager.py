import os
import hashlib
import json
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

from starlette.responses import RedirectResponse
from fastapi import UploadFile, Body

import lazyllm
from lazyllm import FastapiApp as app
from .utils import DocListManager, BaseResponse
from .global_metadata import RAG_DOC_ID, RAG_DOC_PATH


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

    # returns an error message if invalid
    @staticmethod
    def _validate_metadata(metadata: Dict) -> Optional[str]:
        if metadata.get(RAG_DOC_ID):
            return f"metadata MUST not contain key `{RAG_DOC_ID}`"
        if metadata.get(RAG_DOC_PATH):
            return f"metadata MUST not contain key `{RAG_DOC_PATH}`"
        return None

    @app.post("/upload_files")
    def upload_files(self, files: List[UploadFile], override: bool = False,  # noqa C901
                     metadatas: Optional[str] = None, user_path: Optional[str] = None):
        try:
            if user_path: user_path = user_path.lstrip('/')
            if metadatas:
                metadatas: Optional[List[Dict[str, str]]] = json.loads(metadatas)
                if len(files) != len(metadatas):
                    return BaseResponse(code=400, msg='Length of files and metadatas should be the same',
                                        data=None)
                for idx, mt in enumerate(metadatas):
                    err_msg = self._validate_metadata(mt)
                    if err_msg:
                        return BaseResponse(code=400, msg=f'file [{files[idx].filename}]: {err_msg}', data=None)

            file_paths = [os.path.join(self._manager._path, user_path or '', file.filename) for file in files]
            ids = self._manager.add_files(file_paths, metadatas=metadatas, status=DocListManager.Status.working)
            results = []
            for file, path in zip(files, file_paths):
                if os.path.exists(path):
                    if not override:
                        results.append('Duplicated')
                        continue

                try:
                    content = file.file.read()
                    directory = os.path.dirname(path)
                    if directory:
                        os.makedirs(directory, exist_ok=True)

                    with open(path, 'wb') as f:
                        f.write(content)
                except Exception as e:
                    lazyllm.LOG.error(f'writing file [{path}] to disk failed: [{e}]')
                    raise e

                file_id = hashlib.sha256(path.encode()).hexdigest()
                self._manager.update_file_status([file_id], status=DocListManager.Status.success)
                results.append('Success')

            return BaseResponse(data=[ids, results])
        except Exception as e:
            lazyllm.LOG.error(f'upload_files exception: {e}')
            return BaseResponse(code=500, msg=str(e), data=None)

    @app.post("/add_files")
    def add_files(self, files: List[str] = Body(...),
                  group_name: str = Body(None),
                  metadatas: Optional[str] = Body(None)):
        try:
            if metadatas:
                metadatas: Optional[List[Dict[str, str]]] = json.loads(metadatas)
                assert len(files) == len(metadatas), 'Length of files and metadatas should be the same'

            exists_files_info = self._manager.list_files(limit=None, details=True, status=DocListManager.Status.all)
            exists_files_info = {row[2]: row[0] for row in exists_files_info}

            exist_ids = []
            new_files = []
            new_metadatas = []
            id_mapping = {}

            for idx, file in enumerate(files):
                if os.path.exists(file):
                    exist_id = exists_files_info.get(file, None)
                    if exist_id:
                        update_kws = dict(fileid=exist_id, status=DocListManager.Status.success)
                        if metadatas: update_kws["metadata"] = metadatas[idx]
                        self._manager.update_file_message(**update_kws)
                        exist_ids.append(exist_id)
                        id_mapping[file] = exist_id
                    else:
                        new_files.append(file)
                        if metadatas:
                            new_metadatas.append(metadatas[idx])
                else:
                    id_mapping[file] = None

            new_ids = self._manager.add_files(new_files, metadatas=new_metadatas, status=DocListManager.Status.success)
            if group_name:
                self._manager.add_files_to_kb_group(new_ids + exist_ids, group=group_name)

            for file, new_id in zip(new_files, new_ids):
                id_mapping[file] = new_id
            return_ids = [id_mapping[file] for file in files]

            return BaseResponse(data=return_ids)
        except Exception as e:
            return BaseResponse(code=500, msg=str(e), data=None)

    @app.get("/list_files")
    def list_files(self, limit: Optional[int] = None, details: bool = True, alive: Optional[bool] = None):
        try:
            status = [DocListManager.Status.success, DocListManager.Status.waiting, DocListManager.Status.working,
                      DocListManager.Status.failed] if alive else DocListManager.Status.all
            return BaseResponse(data=self._manager.list_files(limit=limit, details=details, status=status))
        except Exception as e:
            return BaseResponse(code=500, msg=str(e), data=None)

    @app.get("/list_files_in_group")
    def list_files_in_group(self, group_name: Optional[str] = None,
                            limit: Optional[int] = None, alive: Optional[bool] = None):
        try:
            status = [DocListManager.Status.success, DocListManager.Status.waiting, DocListManager.Status.working,
                      DocListManager.Status.failed] if alive else DocListManager.Status.all
            return BaseResponse(data=self._manager.list_kb_group_files(group_name, limit, details=True, status=status))
        except Exception as e:
            import traceback
            return BaseResponse(code=500, msg=str(e) + '\ntraceback:\n' + str(traceback.format_exc()), data=None)

    class FileGroupRequest(BaseModel):
        file_ids: List[str]
        group_name: Optional[str] = Field(None)

    @app.post("/add_files_to_group_by_id")
    def add_files_to_group_by_id(self, request: FileGroupRequest):
        try:
            self._manager.add_files_to_kb_group(request.file_ids, request.group_name)
            return BaseResponse()
        except Exception as e:
            return BaseResponse(code=500, msg=str(e), data=None)

    @app.post("/add_files_to_group")
    def add_files_to_group(self, files: List[UploadFile], group_name: str, override: bool = False,
                           metadatas: Optional[str] = None, user_path: Optional[str] = None):
        try:
            response = self.upload_files(files, override=override, metadatas=metadatas, user_path=user_path)
            if response.code != 200: return response
            ids = response.data[0]
            self._manager.add_files_to_kb_group(ids, group_name)
            return BaseResponse(data=ids)
        except Exception as e:
            return BaseResponse(code=500, msg=str(e), data=None)

    @app.post("/delete_files")
    def delete_files(self, request: FileGroupRequest):
        try:
            if request.group_name:
                return self.delete_files_from_group(request)
            else:
                self._manager.update_kb_group_file_status(
                    file_ids=request.file_ids, status=DocListManager.Status.deleting)
                docs = self._manager.update_file_status(file_ids=request.file_ids, status=DocListManager.Status.deleting)

                for doc in docs:
                    if os.path.exists(path := doc[1]):
                        os.remove(path)

                self._manager.update_file_status(file_ids=request.file_ids, status=DocListManager.Status.deleted)
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
