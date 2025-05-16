import os
import json
import traceback

from typing import List, Optional, Dict, Union
from pydantic import BaseModel, Field
from starlette.responses import RedirectResponse
from fastapi import UploadFile, Body

import lazyllm
from lazyllm import FastapiApp as app
from .utils import DocListManager, BaseResponse, gen_docid
from .global_metadata import RAG_DOC_ID, RAG_DOC_PATH
import uuid


class DocManager(lazyllm.ModuleBase):
    def __init__(self, dlm: DocListManager) -> None:
        super().__init__()
        # disable path monitoring in case of competition adding/deleting files
        self._manager = dlm
        self._manager.enable_path_monitoring = False

    def __reduce__(self):
        self._manager.enable_path_monitoring = False
        return (__class__, (self._manager,))

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

    def _gen_unique_filepath(self, file_path: str) -> str:
        suffix = os.path.splitext(file_path)[1]
        prefix = file_path[0: len(file_path) - len(suffix)]
        pattern = f"{prefix}%{suffix}"
        MAX_TRIES = 10000
        exist_paths = set(self._manager.get_existing_paths_by_pattern(pattern))
        if file_path not in exist_paths:
            return file_path
        for i in range(1, MAX_TRIES):
            new_path = f"{prefix}-{i}{suffix}"
            if new_path not in exist_paths:
                return new_path
        return f"{str(uuid.uuid4())}{suffix}"

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
            paths_is_new = [True] * len(file_paths)
            if override is True:
                is_success, msg, paths_is_new = self._manager.validate_paths(file_paths)
                if not is_success:
                    return BaseResponse(code=500, msg=msg, data=None)
            directorys = set(os.path.dirname(path) for path in file_paths)
            [os.makedirs(directory, exist_ok=True) for directory in directorys if directory]
            ids, results = [], []
            for i in range(len(files)):
                file_path = file_paths[i]
                content = files[i].file.read()
                metadata = metadatas[i] if metadatas else None
                if override is False:
                    file_path = self._gen_unique_filepath(file_path)
                with open(file_path, 'wb') as f: f.write(content)
                msg = "success"
                doc_id = gen_docid(file_path)
                if paths_is_new[i]:
                    docs = self._manager.add_files(
                        [file_path], metadatas=[metadata], status=DocListManager.Status.success)
                    if not docs:
                        msg = f"Failed: path {file_path} already exists in Database."
                else:
                    self._manager.update_kb_group(cond_file_ids=[doc_id], new_need_reparse=True)
                    msg = f"Success: path {file_path} will be reparsed."
                ids.append(doc_id)
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
            self._manager.update_kb_group(cond_file_ids=request.file_ids, cond_group=request.group_name,
                                          new_status=DocListManager.Status.deleting)
            return BaseResponse()
        except Exception as e:
            return BaseResponse(code=500, msg=str(e), data=None)

    class AddMetadataRequest(BaseModel):
        doc_ids: List[str]
        kv_pair: Dict[str, Union[bool, int, float, str, list]]

    @app.post("/add_metadata")
    def add_metadata(self, add_metadata_request: AddMetadataRequest):
        doc_ids = add_metadata_request.doc_ids
        kv_pair = add_metadata_request.kv_pair
        try:
            docs = self._manager.get_docs(doc_ids)
            if not docs:
                return BaseResponse(code=400, msg="Failed, no doc found")
            doc_meta = {}
            for doc in docs:
                meta_dict = json.loads(doc.meta) if doc.meta else {}
                for k, v in kv_pair.items():
                    if k not in meta_dict or not meta_dict[k]:
                        meta_dict[k] = v
                    elif isinstance(meta_dict[k], list):
                        meta_dict[k].extend(v) if isinstance(v, list) else meta_dict[k].append(v)
                    else:
                        meta_dict[k] = ([meta_dict[k]] + v) if isinstance(v, list) else [meta_dict[k], v]
                doc_meta[doc.doc_id] = meta_dict
            self._manager.set_docs_new_meta(doc_meta)
            return BaseResponse(data=None)
        except Exception as e:
            return BaseResponse(code=500, msg=str(e), data=None)

    class DeleteMetadataRequest(BaseModel):
        doc_ids: List[str]
        keys: Optional[List[str]] = Field(None)
        kv_pair: Optional[Dict[str, Union[bool, int, float, str, list]]] = Field(None)

    def _inplace_del_meta(self, meta_dict, kv_pair: Dict[str, Union[None, bool, int, float, str, list]]):
        # alert: meta_dict is not a deepcopy
        for k, v in kv_pair.items():
            if k not in meta_dict:
                continue
            if v is None:
                meta_dict.pop(k, None)
            elif isinstance(meta_dict[k], list):
                if isinstance(v, (bool, int, float, str)):
                    v = [v]
                # delete v exists in meta_dict[k]
                meta_dict[k] = list(set(meta_dict[k]) - set(v))
            else:
                # old meta[k] not a list, use v as condition to delete the key
                if meta_dict[k] == v:
                    meta_dict.pop(k, None)

    @app.post("/delete_metadata_item")
    def delete_metadata_item(self, del_metadata_request: DeleteMetadataRequest):
        doc_ids = del_metadata_request.doc_ids
        kv_pair = del_metadata_request.kv_pair
        keys = del_metadata_request.keys
        try:
            if keys is not None:
                # convert keys to kv_pair
                if kv_pair:
                    kv_pair.update({k: None for k in keys})
                else:
                    kv_pair = {k: None for k in keys}
            if not kv_pair:
                # clear metadata
                self._manager.set_docs_new_meta({doc_id: {} for doc_id in doc_ids})
            else:
                docs = self._manager.get_docs(doc_ids)
                if not docs:
                    return BaseResponse(code=400, msg="Failed, no doc found")
                doc_meta = {}
                for doc in docs:
                    meta_dict = json.loads(doc.meta) if doc.meta else {}
                    self._inplace_del_meta(meta_dict, kv_pair)
                    doc_meta[doc.doc_id] = meta_dict
                self._manager.set_docs_new_meta(doc_meta)
            return BaseResponse(data=None)
        except Exception as e:
            return BaseResponse(code=500, msg=str(e), data=None)

    class UpdateMetadataRequest(BaseModel):
        doc_ids: List[str]
        kv_pair: Dict[str, Union[bool, int, float, str, list]]

    @app.post("/update_or_create_metadata_keys")
    def update_or_create_metadata_keys(self, update_metadata_request: UpdateMetadataRequest):
        doc_ids = update_metadata_request.doc_ids
        kv_pair = update_metadata_request.kv_pair
        try:
            docs = self._manager.get_docs(doc_ids)
            if not docs:
                return BaseResponse(code=400, msg="Failed, no doc found")
            for doc in docs:
                doc_meta = {}
                meta_dict = json.loads(doc.meta) if doc.meta else {}
                for k, v in kv_pair.items():
                    meta_dict[k] = v
                doc_meta[doc.doc_id] = meta_dict
            self._manager.set_docs_new_meta(doc_meta)
            return BaseResponse(data=None)
        except Exception as e:
            return BaseResponse(code=500, msg=str(e), data=None)

    class ResetMetadataRequest(BaseModel):
        doc_ids: List[str]
        new_meta: Dict[str, Union[bool, int, float, str, list]]

    @app.post("/reset_metadata")
    def reset_metadata(self, reset_metadata_request: ResetMetadataRequest):
        doc_ids = reset_metadata_request.doc_ids
        new_meta = reset_metadata_request.new_meta
        try:
            docs = self._manager.get_docs(doc_ids)
            if not docs:
                return BaseResponse(code=400, msg="Failed, no doc found")
            self._manager.set_docs_new_meta({doc.doc_id: new_meta for doc in docs})
            return BaseResponse(data=None)
        except Exception as e:
            return BaseResponse(code=500, msg=str(e), data=None)

    class QueryMetadataRequest(BaseModel):
        doc_id: str
        key: Optional[str] = None

    @app.post("/query_metadata")
    def query_metadata(self, query_metadata_request: QueryMetadataRequest):
        doc_id = query_metadata_request.doc_id
        key = query_metadata_request.key
        try:
            docs = self._manager.get_docs(doc_id)
            if not docs:
                return BaseResponse(data=None)
            doc = docs[0]
            meta_dict = json.loads(doc.meta) if doc.meta else {}
            if not key:
                return BaseResponse(data=meta_dict)
            if key not in meta_dict:
                return BaseResponse(code=400, msg=f"Failed, key {key} does not exist")
            return BaseResponse(data=meta_dict[key])
        except Exception as e:
            return BaseResponse(code=500, msg=str(e), data=None)

    def __repr__(self):
        return lazyllm.make_repr("Module", "DocManager")
