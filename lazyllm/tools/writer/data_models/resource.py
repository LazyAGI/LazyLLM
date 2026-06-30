from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from .writing import WritingOutline


class ResourceProfile(BaseModel):
    resource_id: str = ""
    resource_role: Literal["spec", "background", "example"]
    template_usage: Optional[Literal["structure", "style", "both", "none"]] = None
    confidence: float = 1.0
    summary: Optional[str] = None
    extracted_constraints: Dict[str, Any] = Field(default_factory=dict)
    extracted_outline: Optional[WritingOutline] = None
    key_facts: List[str] = Field(default_factory=list)
    style_notes: List[str] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)
