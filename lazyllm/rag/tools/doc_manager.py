from typing import List

from starlette.responses import RedirectResponse
from fastapi import UploadFile

from lazyllm import FastapiApp as app

from .doc_impl import DocumentImpl
from .utils import BaseResponse, save_files_in_threads

class DocumentManager():
    def __init__(self, doc_impl: DocumentImpl) -> None:
        self.doc_impl = doc_impl

    @app.get('/', response_model=BaseResponse, summary='docs')
    def document(self):
        return RedirectResponse(url='/docs')

    @app.post('/new_group')
    def new_group(self, group_name: str):
        self.doc_impl.new_group(group_name)
        return BaseResponse(msg=f"create {group_name} success")

    @app.post('/delete_group')
    def delete_group(self, group_name: str):
        self.doc_impl.delete_group(group_name)
        return BaseResponse(msg=f"delete {group_name} success")

    @app.get('/list_groups')
    def list_groups(self):
        gourp_list = self.doc_impl.list_groups()
        return BaseResponse(data=gourp_list)

    @app.post('/upload_files')
    def upload_files(
        self, files: List[UploadFile], group_name: str, override: bool
    ):
        already_exist_files, new_add_files, overwritten_files = save_files_in_threads(
            files=files,
            source_path=self.doc_impl.get_group_source_path(group_name=group_name),
            override=override
        )

        self.doc_impl.sub_doc.delete_files(overwritten_files)
        self.doc_impl.add_files(group_name, new_add_files + overwritten_files)
        return BaseResponse(data={
            'already_exist_files': already_exist_files,
            'new_add_files': new_add_files,
            'overwritten_files': overwritten_files,
        })

    @app.get('/list_files')
    def list_files(self, group_name: str):
        file_list = self.doc_impl.list_files(group_name)
        return BaseResponse(data=file_list)

    @app.post('/delete_file')
    def delete_file(self, group_name: str, file_name: str):
        self.doc_impl.delete_files(group_name, files=[file_name])
        return BaseResponse(msg=f"delete {file_name} success")

    def __call__(self, input):
        if isinstance(input, dict):
            string_value = input['string']
            parser_value = input['parser']
            signature_value = input['signature']
            return self.doc_impl.sub_doc._query_with_sig(string_value, signature_value, parser_value)

        return ''
