from dataclasses import dataclass, field
from typing import TypedDict

@dataclass
class GraphChunkNode:
    chunk_id: str
    content: str
    chunk_order_index: int
    full_doc_id: str = ""
    tokens: int = -1

@dataclass
class GraphEntityNode:
    entity_name: str
    entity_type: str = ""
    description: str = ""
    source_chunk_ids: list[int] = field(default_factory=list)

@dataclass
class GraphRelationNode:
    src_id: str
    tgt_id: str
    weight: float = 0.0
    keywords: str = ""
    description: str = ""
    source_chunk_ids: list[int] = field(default_factory=list)


class EntityDict(TypedDict):
    entity_name: str

class RelationDict(TypedDict):
    src_id: str
    tgt_id: str