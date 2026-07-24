# flake8: noqa E501
import functools
import importlib
from .. import utils

add_writer_chinese_doc = functools.partial(utils.add_chinese_doc, module=importlib.import_module('lazyllm.tools.writer'))
add_writer_english_doc = functools.partial(utils.add_english_doc, module=importlib.import_module('lazyllm.tools.writer'))
add_writer_models_chinese_doc = functools.partial(utils.add_chinese_doc, module=importlib.import_module('lazyllm.tools.writer.data_models'))
add_writer_models_english_doc = functools.partial(utils.add_english_doc, module=importlib.import_module('lazyllm.tools.writer.data_models'))
add_writer_adapter_chinese_doc = functools.partial(utils.add_chinese_doc, module=importlib.import_module('lazyllm.tools.writer.adapter.feishu'))
add_writer_adapter_english_doc = functools.partial(utils.add_english_doc, module=importlib.import_module('lazyllm.tools.writer.adapter.feishu'))

add_writer_chinese_doc('WriterToolBase', '''
写作工具基类，封装共享的模型、适配器和产物存储。
''')

add_writer_english_doc('WriterToolBase', '''
Base class for writer tools with shared model, adapter, and artifact storage support.
''')

add_writer_models_chinese_doc('ModifyInstruction.validate_move', '''
校验移动指令所需的锚点和位置。
''')

add_writer_models_english_doc('ModifyInstruction.validate_move', '''
Validate the anchor and position required by a move instruction.
''')

add_writer_chinese_doc('WriterPlanningTools.generate_outline', '''
根据写作任务和上下文生成结构化大纲。
''')

add_writer_english_doc('WriterPlanningTools.generate_outline', '''
Generate a structured outline from a writing task and context.
''')

add_writer_chinese_doc('WriterPlanningTools.generate_section_instructions', '''
为大纲中的各章节生成写作指令。
''')

add_writer_english_doc('WriterPlanningTools.generate_section_instructions', '''
Generate drafting instructions for the sections in an outline.
''')

add_writer_models_chinese_doc('WriterDocument.validate_document', '''
校验文档标识以及块标识的唯一性。
''')

add_writer_models_english_doc('WriterDocument.validate_document', '''
Validate the document identifier and block identifier uniqueness.
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

add_writer_models_chinese_doc('WriterBlock.validate_block', '''
校验内容块标识和可见内容。
''')

add_writer_models_english_doc('WriterBlock.validate_block', '''
Validate a block identifier and its visible content.
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

add_writer_models_chinese_doc('PatchHunk.validate_operation', '''
校验各补丁操作所需的字段。
''')

add_writer_models_english_doc('PatchHunk.validate_operation', '''
Validate the fields required by each patch operation.
''')

add_writer_models_chinese_doc('WriterConstraints.validate_word_range', '''
校验最小和最大字数限制。
''')

add_writer_models_english_doc('WriterConstraints.validate_word_range', '''
Validate the minimum and maximum word count constraints.
''')

add_writer_adapter_chinese_doc('FeishuWriterAdapter.merge_refreshed_document', '''
将刷新的飞书绑定信息合并到修订后的文档。
''')

add_writer_adapter_english_doc('FeishuWriterAdapter.merge_refreshed_document', '''
Merge refreshed Feishu bindings into a revised document.
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
为单个章节生成草稿内容块。
''')

add_writer_english_doc('WriterDraftingTools.generate_draft_section', '''
Generate a draft block for one section.
''')

add_writer_chinese_doc('WriterDraftingTools.generate_draft_document', '''
将草稿内容块组装为草稿文档。
''')

add_writer_english_doc('WriterDraftingTools.generate_draft_document', '''
Assemble draft blocks into a draft document.
''')

add_writer_chinese_doc('WriterDraftingTools.generate_final_document', '''
将草稿渲染为最终文档。
''')

add_writer_english_doc('WriterDraftingTools.generate_final_document', '''
Render a draft as a final document.
''')
