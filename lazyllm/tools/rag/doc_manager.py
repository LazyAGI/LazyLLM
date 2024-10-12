from typing import List

from starlette.responses import RedirectResponse
from fastapi import UploadFile

import lazyllm
from lazyllm import FastapiApp as app
from .utils import DocListManager

from .utils import BaseResponse


class DocManager(lazyllm.ModuleBase):
    def __init__(self, dlm: DocListManager) -> None:
        super().__init__()
        self._manager = dlm

    @app.get("/", response_model=BaseResponse, summary="docs")
    def document(self):
        return RedirectResponse(url="/docs")

    @app.get("/list_kb_groups")
    def list_kb_groups(self):
        pass

    @app.post("/upload_files")
    def upload_files(self, files: List[UploadFile], override: bool):
        pass

    @app.get("/list_files")
    def list_files(self, group_name: str):
        pass

    @app.post("/add_files_to_group")
    def add_files_to_group(self, files: List[UploadFile], group_name: str):
        pass

    @app.post("/delete_files")
    def delete_file(self, file_names: str):
        pass

    @app.post("/delete_files_from_group")
    def delete_files_from_group(self, group_name: str, file_names: str):
        pass

    def __repr__(self):
        return lazyllm.make_repr("Module", "DocManager")
