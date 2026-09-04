# flake8: noqa E501
import functools
import importlib
from .. import utils

add_writer_chinese_doc = functools.partial(utils.add_chinese_doc, module=importlib.import_module('lazyllm.tools.writer'))
add_writer_english_doc = functools.partial(utils.add_english_doc, module=importlib.import_module('lazyllm.tools.writer'))
add_writer_models_chinese_doc = functools.partial(utils.add_chinese_doc, module=importlib.import_module('lazyllm.tools.writer.data_models'))
add_writer_models_english_doc = functools.partial(utils.add_english_doc, module=importlib.import_module('lazyllm.tools.writer.data_models'))
add_writer_revision_models_chinese_doc = functools.partial(
    utils.add_chinese_doc, module=importlib.import_module('lazyllm.tools.writer.data_models.revision')
)
add_writer_revision_models_english_doc = functools.partial(
    utils.add_english_doc, module=importlib.import_module('lazyllm.tools.writer.data_models.revision')
)
add_writer_adapter_chinese_doc = functools.partial(utils.add_chinese_doc, module=importlib.import_module('lazyllm.tools.writer.adapter.feishu'))
add_writer_adapter_english_doc = functools.partial(utils.add_english_doc, module=importlib.import_module('lazyllm.tools.writer.adapter.feishu'))
add_writer_notion_adapter_chinese_doc = functools.partial(
    utils.add_chinese_doc, module=importlib.import_module('lazyllm.tools.writer.adapter.notion'))
add_writer_notion_adapter_english_doc = functools.partial(
    utils.add_english_doc, module=importlib.import_module('lazyllm.tools.writer.adapter.notion'))
add_writer_execution_chinese_doc = functools.partial(
    utils.add_chinese_doc, module=importlib.import_module('lazyllm.tools.writer.tools.execution_tools'))
add_writer_execution_english_doc = functools.partial(
    utils.add_english_doc, module=importlib.import_module('lazyllm.tools.writer.tools.execution_tools'))
add_writer_stream_chinese_doc = functools.partial(
    utils.add_chinese_doc, module=importlib.import_module('lazyllm.tools.writer.tools.stream_tools'))
add_writer_stream_english_doc = functools.partial(
    utils.add_english_doc, module=importlib.import_module('lazyllm.tools.writer.tools.stream_tools'))
add_writer_serialization_chinese_doc = functools.partial(
    utils.add_chinese_doc, module=importlib.import_module('lazyllm.tools.writer.utils.serialization'))
add_writer_serialization_english_doc = functools.partial(
    utils.add_english_doc, module=importlib.import_module('lazyllm.tools.writer.utils.serialization'))


def _add_bilingual_docs(chinese_adder, english_adder, entries):
    for target, chinese, english in entries:
        chinese_adder(target, chinese)
        english_adder(target, english)

add_writer_chinese_doc('WriterToolBase', '''
写作工具基类，封装共享的模型、适配器和产物存储。
''')

add_writer_english_doc('WriterToolBase', '''
Base class for writer tools with shared model, adapter, and artifact storage support.
''')

add_writer_revision_models_chinese_doc('RevisionBlockContent.validate_spans', '''
校验修订内容中的文本与富文本片段保持一致。
''')

add_writer_revision_models_english_doc('RevisionBlockContent.validate_spans', '''
Validate that revision text matches its rich-text spans.
''')

add_writer_chinese_doc('WriterPlanningTools.generate_outline', '''
根据写作任务和上下文生成 Writer IR 或 Markdown 大纲。
''')

add_writer_english_doc('WriterPlanningTools.generate_outline', '''
Generate a Writer IR or Markdown outline from a writing task and context.
''')

add_writer_chinese_doc('WriterPlanningTools.generate_section_instructions', '''
为大纲中的各章节生成写作指令。
''')

add_writer_english_doc('WriterPlanningTools.generate_section_instructions', '''
Generate drafting instructions for the sections in an outline.
''')

add_writer_execution_chinese_doc('WriterExecutionTools.execute_writing_subtasks', '''
并行执行大纲中的写作子任务，并记录重试、进度和结果。
''')

add_writer_execution_english_doc('WriterExecutionTools.execute_writing_subtasks', '''
Execute writing subtasks from an outline in parallel while recording retries, progress, and results.
''')

add_writer_models_chinese_doc('WriterDocument.iter_blocks', '''
按文档顺序遍历全部内容块。
''')

add_writer_models_english_doc('WriterDocument.iter_blocks', '''
Iterate over all blocks in document order.
''')

add_writer_models_chinese_doc('WriterDocument.block_by_id', '''
按节点标识查找内容块。
''')

add_writer_models_english_doc('WriterDocument.block_by_id', '''
Find a block by its node identifier.
''')

add_writer_models_chinese_doc('WriterBlock.iter_blocks', '''
遍历当前内容块及其后代。
''')

add_writer_models_english_doc('WriterBlock.iter_blocks', '''
Iterate over this block and its descendants.
''')

add_writer_chinese_doc('NaiveWriterWorkflow', '''
协调默认的规划、起草和修订流程。
''')

add_writer_english_doc('NaiveWriterWorkflow', '''
Coordinate the default planning, drafting, and revision workflow.
''')

add_writer_chinese_doc('NaiveWriterWorkflow.write', '''
执行写作任务的完整工作流。
''')

add_writer_english_doc('NaiveWriterWorkflow.write', '''
Run the complete workflow for a writing task.
''')

add_writer_chinese_doc('NaiveWriterWorkflow.revise', '''
执行已有文档的修订工作流。
''')

add_writer_english_doc('NaiveWriterWorkflow.revise', '''
Run the revision workflow for an existing document.
''')

add_writer_chinese_doc('ArtifactModel.save', '''
将模型保存为带版本信息的 JSON 产物。
''')

add_writer_english_doc('ArtifactModel.save', '''
Save the model as a versioned JSON artifact.
''')

add_writer_chinese_doc('ArtifactModel.load', '''
从 JSON 产物加载并校验模型。
''')

add_writer_english_doc('ArtifactModel.load', '''
Load and validate the model from a JSON artifact.
''')

add_writer_chinese_doc('WriterQualityTools.validate_section', '''
根据章节指令校验草稿章节。
''')

add_writer_english_doc('WriterQualityTools.validate_section', '''
Validate a drafted section against its section instruction.
''')

add_writer_chinese_doc('WriterQualityTools.validate_draft_document', '''
校验完整的草稿文档。
''')

add_writer_english_doc('WriterQualityTools.validate_draft_document', '''
Validate a complete draft document.
''')

add_writer_chinese_doc('WriterQualityTools.validate_patch_set', '''
根据修订任务校验补丁集。
''')

add_writer_english_doc('WriterQualityTools.validate_patch_set', '''
Validate a patch set against its revision task.
''')

add_writer_chinese_doc('WriterQualityTools.validate_string_replace_set', '''
根据修订任务校验 Markdown 字符串替换集。
''')

add_writer_english_doc('WriterQualityTools.validate_string_replace_set', '''
Validate a Markdown string replacement set against its revision task.
''')

add_writer_models_chinese_doc('PatchHunk.validate_operation', '''
校验各补丁操作所需的字段。
''')

add_writer_models_english_doc('PatchHunk.validate_operation', '''
Validate the fields required by each patch operation.
''')

add_writer_adapter_chinese_doc('FeishuWriterAdapter.merge_refreshed_document', '''
将刷新的飞书绑定信息合并到修订后的文档。
''')

add_writer_adapter_english_doc('FeishuWriterAdapter.merge_refreshed_document', '''
Merge refreshed Feishu bindings into a revised document.
''')

add_writer_notion_adapter_chinese_doc('NotionWriterAdapter.has_reusable_image_payload', '''
判断 Writer 图片块是否保留了可复用的 Notion 图片载荷。
''')

add_writer_notion_adapter_english_doc('NotionWriterAdapter.has_reusable_image_payload', '''
Return whether a Writer image block retains a reusable Notion image payload.
''')

add_writer_notion_adapter_chinese_doc('NotionWriterAdapter.merge_refreshed_document', '''
将刷新后的 Notion 块绑定合并回修订文档，同时保留 Writer 节点标识。
''')

add_writer_notion_adapter_english_doc('NotionWriterAdapter.merge_refreshed_document', '''
Merge refreshed Notion block bindings into a revised document while preserving Writer node identifiers.
''')

add_writer_chinese_doc('WriterToolKit', '''
将共享依赖下的写作工具组合成工具包。
''')

add_writer_english_doc('WriterToolKit', '''
Bundle writer tools that share the same dependencies.
''')

add_writer_chinese_doc('WriterToolKit.as_tool_groups', '''
按工作流阶段返回写作工具分组。
''')

add_writer_english_doc('WriterToolKit.as_tool_groups', '''
Return writer tools grouped by workflow stage.
''')

add_writer_chinese_doc('WriterRevisionTools.locate_revision_target', '''
定位修订任务涉及的文档块。
''')

add_writer_english_doc('WriterRevisionTools.locate_revision_target', '''
Locate document blocks targeted by a revision task.
''')

add_writer_chinese_doc('WriterRevisionTools.generate_modify_plan', '''
为定位到的修订目标生成修改计划。
''')

add_writer_english_doc('WriterRevisionTools.generate_modify_plan', '''
Generate a modification plan for located revision targets.
''')

add_writer_chinese_doc('WriterRevisionTools.generate_patch_set', '''
根据修改计划生成补丁集。
''')

add_writer_english_doc('WriterRevisionTools.generate_patch_set', '''
Generate a patch set from a modification plan.
''')

add_writer_chinese_doc('WriterRevisionTools.apply_patch', '''
将补丁集应用到写作文档。
''')

add_writer_english_doc('WriterRevisionTools.apply_patch', '''
Apply a patch set to a writer document.
''')

add_writer_chinese_doc('WriterDraftingTools.generate_draft_section', '''
为单个章节生成 Writer IR 或 Markdown 草稿。
''')

add_writer_english_doc('WriterDraftingTools.generate_draft_section', '''
Generate a Writer IR or Markdown draft for one section.
''')

add_writer_chinese_doc('WriterDraftingTools.generate_draft_document', '''
将 Writer IR 内容块或 Markdown 章节组装为草稿文档。
''')

add_writer_english_doc('WriterDraftingTools.generate_draft_document', '''
Assemble Writer IR blocks or Markdown sections into a draft document.
''')

add_writer_chinese_doc('WriterDraftingTools.generate_final_document', '''
将草稿渲染为最终文档。
''')

add_writer_english_doc('WriterDraftingTools.generate_final_document', '''
Render a draft as a final document.
''')

_add_bilingual_docs(
    add_writer_revision_models_chinese_doc,
    add_writer_revision_models_english_doc,
    [
        ('ModifyInstruction.validate_visual_instruction', '校验可视化指令仅用于创建操作。',
         'Validate that visual instructions are used only for create operations.'),
        ('GeneratedRevision.normalize_single_block_changes', '将单内容块修订规范化为内容块列表。',
         'Normalize single-block revision changes into block lists.'),
        ('StringReplace.validate_replacement', '校验 Markdown 字符串替换内容。',
         'Validate a Markdown string replacement.'),
    ],
)

_add_bilingual_docs(
    add_writer_chinese_doc,
    add_writer_english_doc,
    [
        ('WriterDraftingTools.generate_short_document', '根据整篇短文写作计划生成无章节标题的 Markdown 或 Writer IR 草稿。',
         'Generate a flat Markdown or Writer IR draft from a whole-document short writing plan.'),
        ('WriterDraftingTools.stream_short_document', '流式生成无章节标题的 Markdown 或 Writer IR 短文预览。',
         'Stream a flat Markdown or Writer IR short-document preview.'),
        ('WriterDraftingTools.stream_short_document_ir', '流式生成无章节标题的 Writer IR 短文及其 Markdown 预览。',
         'Stream a flat Writer IR short document and its Markdown preview.'),
        ('WriterDraftingTools.stream_draft_section', '流式生成 Markdown 章节草稿。',
         'Stream a Markdown section draft.'),
        ('WriterDraftingTools.stream_draft_section_ir', '流式生成 Writer IR 章节草稿的 Markdown 预览。',
         'Stream a Markdown preview of a Writer IR section draft.'),
        ('WriterPlanningTools.generate_short_writing_plan', '为整篇无章节标题的短文生成统一写作计划。',
         'Generate a whole-document writing plan for a flat short document.'),
        ('WriterPlanningTools.generate_short_visual_plan', '为整篇无章节标题的短文生成文档级可视化规划。',
         'Generate a document-level visual plan for a flat short document.'),
        ('WriterPlanningTools.generate_rewrite_outline', '为完整文档重写生成新大纲。',
         'Generate a new outline for a complete document rewrite.'),
        ('WriterPlanningTools.generate_rewrite_section_instructions', '为完整文档重写生成章节指令。',
         'Generate section instructions for a complete document rewrite.'),
        ('WriterPlanningTools.generate_visual_plan', '根据写作大纲生成可视化规划。',
         'Generate a visual plan from a writing outline.'),
        ('WriterQualityTools.validate_revision_set', '根据文档类型校验修订集。',
         'Validate a revision set for its document representation.'),
        ('WriterRevisionTools.generate_revision_set', '根据文档类型生成修订集。',
         'Generate a revision set for its document representation.'),
        ('WriterRevisionTools.generate_string_replace_set', '根据修改计划生成 Markdown 字符串替换集。',
         'Generate Markdown string replacements from a modification plan.'),
        ('WriterRevisionTools.apply_revision', '根据文档类型应用修订集。',
         'Apply a revision set for its document representation.'),
        ('WriterRevisionTools.apply_string_replace', '将字符串替换集应用到 Markdown 文档。',
         'Apply a string replacement set to a Markdown document.'),
    ],
)

_add_bilingual_docs(
    add_writer_stream_chinese_doc,
    add_writer_stream_english_doc,
    [
        ('IRPreviewOutput', '管理 Writer IR 流式预览的输出。', 'Manage streamed Writer IR preview output.'),
        ('IRPreviewOutput.mark_body', '标记预览流已经输出正文内容。',
         'Mark that the preview stream has emitted body content.'),
        ('IRPreviewOutput.start_item', '开始输出新的预览项。', 'Start a new preview item.'),
        ('IRPreviewOutput.append_complete', '追加完整的 Markdown 预览项。',
         'Append a complete Markdown preview item.'),
        ('DraftPreviewStream', '在后台生成草稿并流式输出预览。',
         'Generate a draft in the background and stream its preview.'),
        ('DraftPreviewStream.result', '返回已完整消费的流式结果。',
         'Return the result of a fully consumed stream.'),
        ('DraftPreviewStream.close', '关闭预览流并取消未完成任务。',
         'Close the preview stream and cancel unfinished work.'),
        ('MarkdownStreamNormalizer', '规范化模型的 Markdown 流式输出。',
         'Normalize streamed Markdown model output.'),
        ('MarkdownStreamNormalizer.feed', '接收并规范化一段流式文本。',
         'Consume and normalize a streamed text fragment.'),
        ('MarkdownStreamNormalizer.finish', '完成流式规范化并返回剩余内容。',
         'Finish normalization and return remaining content.'),
        ('IRBlockStreamState', '跟踪单个 Writer IR 内容块的流式解析状态。',
         'Track streaming parse state for one Writer IR block.'),
        ('IRBlockStreamState.set_type', '设置当前内容块类型。', 'Set the current block type.'),
        ('IRBlockStreamState.set_numbering', '设置当前内容块编号信息。',
         'Set numbering metadata for the current block.'),
        ('IRBlockStreamState.feed_content', '追加当前内容块的流式文本。',
         'Append streamed text for the current block.'),
        ('IRBlockStreamState.finish_content', '完成当前内容块的文本输出。',
         'Finish text output for the current block.'),
        ('IRBlockStreamState.prepare_children', '准备解析当前内容块的子节点。',
         'Prepare to parse child blocks.'),
        ('IRBlockStreamState.finish', '完成当前内容块的流式解析。',
         'Finish streaming parsing for the current block.'),
        ('IRJSONMarkdownParser.emit', '输出一段 Markdown 预览内容。',
         'Emit a Markdown preview fragment.'),
        ('IRJSONMarkdownParser.feed', '接收并解析一段 Writer IR JSON。',
         'Consume and parse a Writer IR JSON fragment.'),
        ('IRJSONMarkdownParser.finish', '完成解析并与最终 WriterBlock 校验一致性。',
         'Finish parsing and validate against the final WriterBlock.'),
        ('IRJSONMarkdownParser.finish_document', '完成文档级 Writer IR 解析并输出最终 Markdown 预览。',
         'Finish document-level Writer IR parsing and emit the final Markdown preview.'),
        ('IRPreviewStream', '将结构化 Writer IR 生成结果转换为 Markdown 预览流。',
         'Convert structured Writer IR generation results into a Markdown preview stream.'),
        ('OutlineIRStream', '将 Writer IR 大纲生成结果流式输出为 Markdown 预览。',
         'Stream a Writer IR outline generation result as a Markdown preview.'),
        ('DraftMarkdownStream', '流式输出 Markdown 草稿。', 'Stream a Markdown draft.'),
        ('DraftIRStream', '将 Writer IR 草稿流式输出为 Markdown 预览。',
         'Stream a Writer IR draft as a Markdown preview.'),
    ],
)

_add_bilingual_docs(
    add_writer_chinese_doc,
    add_writer_english_doc,
    [
        ('WriterPlanningTools.stream_outline', '流式生成 Writer IR 或 Markdown 大纲预览。',
         'Stream a Writer IR or Markdown outline preview.'),
    ],
)

add_writer_serialization_chinese_doc('MarkdownSelectionError', '''
表示 Markdown 选区无法被唯一、安全定位。
''')
add_writer_serialization_english_doc('MarkdownSelectionError', '''
Indicate that a Markdown selection cannot be located uniquely and safely.
''')
