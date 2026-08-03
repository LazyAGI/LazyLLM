from .task import InputResource, Selection, TargetDocument, WritingTask
from .document import ContentRef, DocumentFormat, DocumentPayload
from .resource import MaterialStyle, ResourceProfile
from .context import (
    BlockRelationGraph,
    BlockSummary,
    DocumentFact,
    DocumentSummary,
    StyleProfile,
    WritingContext,
)
from .planning import SectionInstruction, SectionInstructionList
from .revision import (
    LocateResult,
    ModifyInstruction,
    ModifyPlan,
    PatchHunk,
    PatchResult,
    PatchSet,
    StringReplace,
    StringReplaceResult,
    StringReplaceSet,
)
from .multimodal import MediaAsset, MediaAssetLibrary, VisualInstruction
from .quality import AuditIssue, AuditResult, ReviewReport
from .writer_ir import WriterBlock, WriterDocument, WriterSpan, WriterStage

__all__ = [
    'InputResource',
    'Selection',
    'TargetDocument',
    'WritingTask',
    'ContentRef',
    'DocumentFormat',
    'DocumentPayload',
    'MaterialStyle',
    'ResourceProfile',
    'BlockRelationGraph',
    'BlockSummary',
    'DocumentFact',
    'DocumentSummary',
    'StyleProfile',
    'WritingContext',
    'SectionInstruction',
    'SectionInstructionList',
    'ModifyInstruction',
    'ModifyPlan',
    'PatchHunk',
    'PatchResult',
    'PatchSet',
    'StringReplace',
    'StringReplaceResult',
    'StringReplaceSet',
    'LocateResult',
    'MediaAsset',
    'MediaAssetLibrary',
    'VisualInstruction',
    'AuditIssue',
    'AuditResult',
    'ReviewReport',
    'WriterBlock',
    'WriterDocument',
    'WriterSpan',
    'WriterStage',
]
