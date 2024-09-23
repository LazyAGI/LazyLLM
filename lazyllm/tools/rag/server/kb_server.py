from typing import List

from fastapi import Body, UploadFile
from fastapi.responses import RedirectResponse

import lazyllm
from lazyllm import ServerModule
from lazyllm import FastapiApp as app
from lazyllm.tools.rag.utils import BaseResponse, save_files_in_threads
from lazyllm.tools.rag.db import KBInfoRecord, KBFileRecord, FileRecord, FileState

class KBServer(lazyllm.ModuleBase):
    """
    Document server for managing knowledge bases and file uploads.
    """
    def __init__(self) -> None:
        super().__init__()

    @app.get("/", response_model=BaseResponse, summary="docs")
    def document(self):
        """
        Redirects to the documentation page.
        """
        return RedirectResponse(url="/docs")

    @app.post("/create_knowledge_base")
    def create_knowledge_base(
        self, 
        kb_name: str = Body(..., examples=["samples"]),
        kb_info: str = Body(..., examples=["samples info"])
    ):
        """
        Creates a new knowledge base.
        """
        KBInfoRecord.create(kb_name=kb_name, kb_info=kb_info)
        return BaseResponse(msg=f"create {kb_name} success")

    @app.post("/delete_knowledge_base")
    def delete_knowledge_base(self, kb_name: str):
        """
        Deletes an existing knowledge base.
        """
        KBInfoRecord.del_node(kb_name=kb_name)
        return BaseResponse(msg=f"delete {kb_name} success")

    @app.get("/list_knowledge_bases")
    def list_knowledge_bases(self):
        """
        Lists all knowledge bases.
        """
        kbs = KBInfoRecord.all()
        return BaseResponse(data=kbs)

    @app.post("/upload_files")
    def upload_files(self, files: List[UploadFile], kb_name: str, override: bool):
        """
        Uploads files to a knowledge base.
        """
        already_exist_files, new_add_files, overwritten_files = save_files_in_threads(
            files=files,
            source_path="",
            override=override,
        )

        for file_name in overwritten_files:
            file_id = FileRecord.first(file_name=file_name).id
            KBFileRecord.update(
                func=lambda x: x.set(state=FileState.WAIT_DELETE), 
                file_id=file_id
            )

        for file_name in overwritten_files + new_add_files:
            file_id = 0
            KBFileRecord.create(kb_name=kb_name, file_id=file_id)

        return BaseResponse(
            data={
                "already_exist_files": already_exist_files,
                "new_add_files": new_add_files,
                "overwritten_files": overwritten_files,
            }
        )

    @app.get("/list_files")
    def list_files(self, kb_name: str):
        """
        Lists all files in a knowledge base.
        """
        file_list = KBFileRecord.all(kb_name=kb_name)
        return BaseResponse(data=file_list)

    @app.post("/delete_file")
    def delete_file(self, kb_name: str, file_name: str):
        """
        Deletes a file from a knowledge base.
        """
        KBFileRecord.update(
            func=lambda x: x.set(state=FileState.WAIT_DELETE), 
            kb_name=kb_name, 
            file_name=file_name
        )
        return BaseResponse(msg=f"delete {file_name} success")

    def forward(self, *args, **kwargs):
        """
        Forwards a function call to the underlying implementation.
        """
        pass

    def __repr__(self):
        """
        String representation of the DocumentServer instance.
        """
        return lazyllm.make_repr("Module", "DocManager")
    
    @staticmethod
    def start_server(launcher=None):
        launcher = launcher if launcher else lazyllm.launchers.remote(sync=False)
        doc_server = ServerModule(KBServer(), launcher=launcher)
        doc_server.start()
