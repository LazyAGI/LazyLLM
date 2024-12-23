import os
import json
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

from starlette.responses import RedirectResponse
from fastapi import UploadFile, Body

import lazyllm
from lazyllm import FastapiApp as app
from .utils import DocListManager, BaseResponse
from .global_metadata import RAG_DOC_ID, RAG_DOC_PATH


def gen_unique_filepaths(ori_filepath: str) -> str:
    if not os.path.exists(ori_filepath):
        return ori_filepath
    directory, filename = os.path.split(ori_filepath)
    name, ext = os.path.splitext(filename)
    ct = 1
    new_filepath = f"{os.path.join(directory, name)}_{ct}{ext}"
    while os.path.exists(new_filepath):
        ct += 1
        new_filepath = f"{os.path.join(directory, name)}_{ct}{ext}"
    return new_filepath


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
            for path in file_paths:
                directory = os.path.dirname(path)
                if directory:
                    os.makedirs(directory, exist_ok=True)
            ids, results = [], []
            for i in range(len(files)):
                file_path = file_paths[i]
                content = files[i].file.read()
                metadata = metadatas[i] if metadatas else None
                doc_path_parsing_result = self._manager.safe_parsing_path(file_path, override, content, metadata)
                # file_path may be changed
                file_path = doc_path_parsing_result.file_path
                msg = doc_path_parsing_result.msg
                if doc_path_parsing_result.is_new:
                    docs = self._manager.add_files([file_path], metadatas=[metadata], status=DocListManager.Status.success)
                    if not docs:
                        msg = "Failed: Call add_files failed"
                ids.append(doc_path_parsing_result.doc_id)
                results.append(msg)
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
                        if metadatas: update_kws["meta"] = json.dumps(metadatas[idx])
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
                documents = self._manager.delete_files(request.file_ids)
                deleted_ids = set([ele.doc_id for ele in documents])
                for doc in documents:
                    if os.path.exists(path := doc.path):
                        os.remove(path)
                results = ["Success" if ele.doc_id in deleted_ids else "Failed" for ele in documents]
                return BaseResponse(data=[request.file_ids, results])
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
