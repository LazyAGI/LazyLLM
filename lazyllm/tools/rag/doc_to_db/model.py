from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Type
from dataclasses import dataclass
from pydantic import BaseModel, Field
from sqlalchemy.orm import DeclarativeBase

class _TableBase(DeclarativeBase):
    '''Lightweight base for dynamic ORM classes.'''

class Default(BaseModel):
    title: str = Field(default='', description='Title or main heading of the file')
    description: str = Field(description='Short description of the file')
    summary: str = Field(default='', description='Concise summary of the file content')
    keywords: List[str] = Field(default_factory=list, description='Keywords or tags extracted from the document')
    author: Optional[str] = Field(default=None, description='Owner or author of the file')

class ExtractionMode(Enum):
    TEXT = 'text'
    MULTIMODAL = 'multimodal'

class ExtractClue(BaseModel):
    '''Clue for extraction'''
    reason: str = Field(default='', description='The reason for extraction')
    citation: List[str] = Field(default_factory=list, description='Citation of the clue')

class ExtractMeta(BaseModel):
    '''Extra information for extraction'''
    schema_set_id: str
    mode: ExtractionMode = Field(default=ExtractionMode.TEXT, description='Extraction mode')
    algo_id: str = Field(default='', description='Algorithm ID')
    kb_id: str = Field(default='', description='KB ID')
    doc_id: str = Field(default='', description='Document ID')
    clues: Dict[str, ExtractClue] = Field(default_factory=dict)
    num_doc_tokens: int = Field(default=0, description='Number of tokens in the document')
    num_output_tokens: int = Field(default=0, description='Number of tokens in the output')

class ExtractResult(BaseModel):
    data: Dict[str, Any]
    metadata: ExtractMeta = Field(default_factory=dict)

@dataclass
class SchemaSetInfo:
    schema_set_id: str
    schema_model: Type[BaseModel]

TABLE_SCHEMA_SET_INFO = {
    'name': 'lazyllm_table_schema_set',
    'comment': 'Manage schema sets, which contains the infomation of fields for the extraction.',
    'columns': [
        {'name': 'id', 'data_type': 'integer', 'nullable': False, 'is_primary_key': True,
         'comment': 'Auto increment ID'},
        {'name': 'schema_set_id', 'data_type': 'string', 'nullable': False,
         'comment': 'External schema set identifier'},
        {'name': 'schema_set_json', 'data_type': 'string', 'nullable': False,
         'comment': 'Schema set JSON string'},
        {'name': 'desc', 'data_type': 'string', 'nullable': False,
         'comment': 'description of the schema set'},
        {'name': 'idem_key', 'data_type': 'string', 'nullable': False,
         'comment': 'unique key for the schema set'},
        {'name': 'created_at', 'data_type': 'datetime', 'nullable': False,
         'comment': 'Creation time (auto-generated)', 'default': datetime.now()},
        {'name': 'updated_at', 'data_type': 'datetime', 'nullable': False,
         'comment': 'Update time (auto-generated)', 'default': datetime.now()},
    ]
}

Table_ALGO_KB_SCHEMA = {
    'name': 'lazyllm_algo_kb_schema',
    'comment': 'Record the kb from a certain algorithm is using which schema set.',
    'columns': [
        {'name': 'id', 'data_type': 'integer', 'nullable': False, 'is_primary_key': True,
         'comment': 'record ID'},
        {'name': 'algo_id', 'data_type': 'string', 'nullable': False,
         'comment': 'Algorithm ID'},
        {'name': 'kb_id', 'data_type': 'string', 'nullable': False,
         'comment': 'KB ID'},
        {'name': 'schema_set_id', 'data_type': 'string', 'nullable': False,
         'comment': 'Schema set ID'},
    ]
}
