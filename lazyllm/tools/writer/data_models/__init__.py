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
from .writing import (
    DraftBlock,
    DraftDocument,
    DraftSection,
    OutlineNode,
    SectionInstruction,
    SectionInstructionList,
    WritingOutline,
    WritingOutput,
    WritingSubTask,
)
from .docir import Anchor, DocBlock, DocIR, DocSpan
from .revision import ModifyInstruction, ModifyPlan, PatchHunk, PatchResult, PatchSet
from .multimodal import MediaAsset, MediaAssetLibrary, VisualInstruction
from .quality import AuditIssue, AuditResult, ReviewReport

__all__ = [
    'InputResource',
    'Selection',
    'TargetDocument',
    'WritingOutput',
    'WritingTask',
    'MaterialStyle',
    'ResourceProfile',
    'BlockRelationGraph',
    'BlockSummary',
    'DocumentFact',
    'DocumentSummary',
    'StyleProfile',
    'WritingContext',
    'DraftBlock',
    'DraftDocument',
    'DraftSection',
    'OutlineNode',
    'SectionInstruction',
    'SectionInstructionList',
    'WritingOutline',
    'WritingSubTask',
    'Anchor',
    'DocBlock',
    'DocIR',
    'DocSpan',
    'ModifyInstruction',
    'ModifyPlan',
    'PatchHunk',
    'PatchResult',
    'PatchSet',
    'MediaAsset',
    'MediaAssetLibrary',
    'VisualInstruction',
    'AuditIssue',
    'AuditResult',
    'ReviewReport',
]
