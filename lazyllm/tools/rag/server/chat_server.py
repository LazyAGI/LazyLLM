from typing import List, Optional, Union

from fastapi import Body
from fastapi.responses import RedirectResponse

import lazyllm
from lazyllm import FastapiApp as app

from ..utils import BaseResponse

class ChatServer(lazyllm.ModuleBase):
    """
    Chat server for managing knowledge bases and handling file uploads.
    """
    def __init__(self) -> None:
        super().__init__()

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
        kb_list: List[str] = Body(..., description="Selected knowledge base list", examples=[["sample1"]]),
        stream: bool = Body(False, description="Stream output")
    ):  
        """
        Handles chat requests with the knowledge base.
        """
        pass

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
        pass