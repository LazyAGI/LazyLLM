# flake8: noqa
LOCATE_REVISION_TARGET_PROMPT = '''You are a revision target locator. Given a writing task (the user's revision request), a document (WriterDocument), and the writing context, identify which document blocks need to be modified.

Rules:
- Read task.query carefully — it contains the user's revision request.
- Examine every document block and select the ones the revision acts on.
- For replace/delete, select the blocks whose content or existence changes.
- For insert, select the existing block used as the insertion anchor.
- For move, select the block being moved. Do not select the destination block merely because it is the destination.
- Select node_ids ONLY from the candidate list below. Never invent node_ids.
- Be precise: do not select blocks that are unrelated to the request.
- target_reasons: for each selected node_id, give a one-sentence reason why it is involved.
- summary: one concise sentence describing what was located (which blocks and why). Never leave it null.
- If no blocks are involved, return an empty target_node_ids list.

Writing task:
{task_json}

Document (WriterDocument):
{document_json}

Writing context:
{context_json}

Candidate node_ids (select from these only):
{candidate_node_ids}
'''


GENERATE_MODIFY_PLAN_PROMPT = '''You are a modify plan generator. Given a writing task, the located target blocks, and the writing context, produce a ModifyPlan.

Rules:
- For each target block, decide the modify_type and write a clear, specific instruction.
- modify_type must be one of:
  - insert: insert one or more brand-new blocks before or after the target block. The target block is the insertion anchor; set position to before or after.
  - replace: replace the target block's content with a new version.
  - delete: remove the target block.
  - move: move the target block before or after another existing block. Set anchor_node_id to the destination block and position to before or after.
- instruction: a concise description of what change to make to that block, derived from task.query.
- Every ModifyInstruction.target_node_id must come from the locate_result's target_node_ids. Produce exactly one instruction per target block.
- For move, anchor_node_id may reference any node_id in the supplied document, including a block outside the located target set.
- scope: one of document / section / block / span — pick the most fitting one for this revision.
- summary: one concise sentence describing the overall revision plan. Never leave it null.
- Respect the writing context: keep facts consistent (never alter locked facts), preserve terminology and the style profile.
- Do not invent facts that conflict with the writing context.

Writing task:
{task_json}

Document (complete WriterDocument, including possible move destinations):
{document_json}

Locate result:
{locate_result_json}

Writing context:
{context_json}
'''


GENERATE_PATCH_SET_PROMPT = '''You are a patch generator. Given a document, a modify plan, and the writing context, produce a PatchSet with concrete text changes.

Rules:
- For each ModifyInstruction, produce exactly one PatchHunk.
- Each PatchHunk must have:
  - target_node_id: copied from the corresponding ModifyInstruction.
  - modify_type: copied from the corresponding ModifyInstruction.
  - replace: set new_text to the FULL new content of the target block. Leave new_blocks empty.
  - insert: set position from the instruction and put every complete new block in new_blocks. Leave new_text null. Use paragraph unless the requested structure clearly requires heading, list_item, code, or quote.
  - delete: leave new_text null and new_blocks empty.
  - move: copy anchor_node_id and position from the instruction. Leave new_text null and new_blocks empty.
- Leave anchor and old_text null. The system fills conflict-checking fields from the source document.
- Generated text must be complete and self-contained. Never produce placeholders or ellipsis-only output.
- Respect the writing context: keep facts consistent (never alter locked facts), preserve terminology and style.
- Do not invent facts that conflict with the writing context.

Document (WriterDocument):
{document_json}

Modify plan:
{modify_plan_json}

Writing context:
{context_json}
'''
