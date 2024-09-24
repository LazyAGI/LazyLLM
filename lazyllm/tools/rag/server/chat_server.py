import threading
from typing import List, Optional, Union, Tuple, Callable

from fastapi import Body
from fastapi.responses import RedirectResponse

from lazyllm import FlowBase
from lazyllm import FastapiApp as app

from ..utils import BaseResponse
from ..document import Document
from .kb_server import KBServer, DocCreater

PipelineCreater = Callable[[str, List[Document]], Tuple[FlowBase, FlowBase]]

class ChatServer(KBServer):
    """
    Chat server for managing knowledge bases and handling file uploads.
    """
    def __init__(self, doc_creater: DocCreater, pipeline_creater: PipelineCreater) -> None:
        super().__init__(doc_creater=doc_creater)

        self._pipeline_creater = pipeline_creater
        self._pipeline_dict = {}
        self._condition = threading.Condition()

    @app.get("/", response_model=BaseResponse, summary="docs")
    def document(self):
        """
        Redirects to the documentation page.
        """
        return RedirectResponse(url="/docs")

    @app.post("/chat/knowledge_base_chat", summary="Chat with knowledge base")
    async def knowledge_base_chat(
        self,
        query: str = Body(..., description="User input", examples=["Hello"]),
        session_id: Optional[str] = Body(None, description="Session ID", examples=["session_id"]),
        kb_name: Union[str, List[str]] = Body(..., description="Knowledge base name", examples=["samples", ["sample1", "sample2"]]),
        stream: bool = Body(False, description="Stream output"),
        project_name: Optional[str] = Body(None, description="Project name", examples=["project_name"])
    ):  
        """
        Handles chat requests with the knowledge base.
        """
        kb_list = kb_name if isinstance(kb_name, list) else [kb_name]
        pipeline = self._create_query_pipeline(project_name, kb_list)
        return pipeline(query=query, session_id=session_id, stream=stream)

    @app.post("/chat/search_docs", summary="Search knowledge base")
    async def knowledge_base_search(
        self,
        query: str = Body(..., description="User input", examples=["Hello"]),
        kb_name: Union[str, List[str]] = Body(..., description="Knowledge base name", examples=["samples", ["sample1", "sample2"]]),
        top_k: int = Body(2, description="Number of matching vectors", ge=1, le=100)
    ) -> List[dict]:
        """
        Searches documents within the knowledge base.
        """
        kb_list = kb_name if isinstance(kb_name, list) else [kb_name]
        pipeline = self._create_search_pipeline(kb_list)
        return pipeline(query=query, top_k=top_k)

    def _create_doc(self, kb_name: str):
        with self._condition:
            if kb_name in self._doc_dict:
                return self._doc_dict[kb_name]
            
            doc = self._doc_creater(kb_name)
            self._doc_dict[kb_name] = doc
            return doc
    
    def _create_pipeline(self, project_name: str, kb_names: List[str], index: int):
        with self._condition:
            hash_key = hash("#".join(kb_names))
            if hash_key in self._pipeline_dict:
                return self._pipeline_dict[hash_key][index]

            docs = [self._create_doc(kb_name) for kb_name in kb_names]
            pipelines = self._pipeline_creater(project_name, docs)
            self._pipeline_dict[hash_key] = pipelines
            return pipelines[index]

    def _create_search_pipeline(self, project_name: str, kb_names: List[str]):
        return self._create_pipeline(project_name, kb_names, 1)
    
    def _create_query_pipeline(self, project_name: str, kb_names: List[str]):
        return self._create_pipeline(project_name, kb_names, 0)
    
    @staticmethod
    def start_server(doc_creater: DocCreater = None, launcher=None):
        if doc_creater is None:
            return KBServerBase.start_server(launcher=launcher)

        launcher = launcher if launcher else lazyllm.launchers.remote(sync=False)
        doc_server = ServerModule(KBServer(doc_creater=doc_creater), launcher=launcher)
        doc_server.start()