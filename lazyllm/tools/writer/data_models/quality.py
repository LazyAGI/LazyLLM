from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from ..utils.artifact import ArtifactModel


class AuditIssue(BaseModel):
    severity: Literal['high', 'medium', 'low']
    category: Literal['format', 'coverage', 'relevance', 'evidence', 'style']
    location: Optional[str] = None
    description: str
    suggestion: str


class AuditResult(BaseModel):
    is_passed: bool
    score: int
    summary: str
    issues: List[AuditIssue] = Field(default_factory=list)


class ReviewReport(ArtifactModel):
    report_id: Optional[str] = None
    target: Optional[str] = None
    result: AuditResult
    meta: Dict[str, Any] = Field(default_factory=dict)
