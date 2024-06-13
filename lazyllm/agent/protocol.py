from pydantic import BaseModel
from typing import Optional, Union, Literal, List


DEFAULT_SYSTEM_MESSAGE = 'You are a helpful assistant.'

ROLE = 'role'
CONTENT = 'content'
NAME = 'name'

SYSTEM = 'system'
USER = 'user'
ASSISTANT = 'assistant'
TOOL = 'tool'


class BaseModelDict(BaseModel):
    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)
    
    def __str__(self):
        return f'{self.model_dump()}'
    
    def get(self, item, default=None):
        return getattr(self, item, default)
    
    def items(self):
        return self.__dict__.items()

class Function(BaseModelDict):
    arguments: str
    name: str

class ToolCall(BaseModelDict):
    id: str
    function: Function
    type: Literal["function"] = "function"

class Message(BaseModelDict):
    role: Literal["system", "user", "assistant"] = "assistant"
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None