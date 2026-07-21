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
from .revision import Anchor, LocateResult, ModifyInstruction, ModifyPlan, PatchBlock, PatchHunk, PatchResult, PatchSet
from .multimodal import MediaAsset, MediaAssetLibrary, VisualInstruction
from .quality import AuditIssue, AuditResult, ReviewReport
from .writer_ir import WriterAuthoring, WriterBlock, WriterConstraints, WriterDocument, WriterSpan, WriterStage

__all__ = [
    'InputResource',
    'Selection',
    'TargetDocument',
    'WritingTask',
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
    'Anchor',
    'ModifyInstruction',
    'ModifyPlan',
    'PatchBlock',
    'PatchHunk',
    'PatchResult',
    'PatchSet',
    'LocateResult',
    'MediaAsset',
    'MediaAssetLibrary',
    'VisualInstruction',
    'AuditIssue',
    'AuditResult',
    'ReviewReport',
    'WriterAuthoring',
    'WriterBlock',
    'WriterConstraints',
    'WriterDocument',
    'WriterSpan',
    'WriterStage',
]
