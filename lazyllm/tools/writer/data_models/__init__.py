from .task import InputResource, Selection, TargetDocument, WritingTask
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
    LocatedContent,
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
from .writer_ir import ContentRef, WriterBlock, WriterDocument, WriterSpan, WriterStage

__all__ = [
    'InputResource',
    'Selection',
    'TargetDocument',
    'WritingTask',
    'ContentRef',
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
    'LocatedContent',
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
