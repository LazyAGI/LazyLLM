# flake8: noqa
LOCATE_REVISION_TARGET_PROMPT = '''You are a revision target locator. Given a writing task (the user's revision request), a document (DocIR), and the writing context, identify which document blocks need to be modified.

Rules:
- Read task.query carefully — it contains the user's revision request.
- Examine every document block and select the ones the revision acts on.
- This covers blocks to change, and — when the request asks to add new content — the existing block right after which the new block should be inserted (the anchor).
- Select block_ids ONLY from the candidate list below. Never invent block_ids.
- Be precise: do not select blocks that are unrelated to the request.
- target_reasons: for each selected block_id, give a one-sentence reason why it is involved.
- summary: one concise sentence describing what was located (which blocks and why). Never leave it null.
- If no blocks are involved, return an empty target_block_ids list.

Writing task:
{task_json}

Document (DocIR):
{doc_ir_json}

Writing context:
{context_json}

Candidate block_ids (select from these only):
{candidate_block_ids}
'''


GENERATE_MODIFY_PLAN_PROMPT = '''You are a modify plan generator. Given a writing task, the located target blocks, and the writing context, produce a ModifyPlan.

Rules:
- For each target block, decide the modify_type and write a clear, specific instruction.
- modify_type must be one of:
  - rewrite: rewrite the whole target block into a new version.
  - insert: insert a brand-new block right after the target block.
  - replace: replace specific portions of the target block, leaving the rest intact.
  - delete: remove the target block.
- instruction: a concise description of what change to make to that block, derived from task.query.
- Every ModifyInstruction.target_block_id must come from the locate_result's target_block_ids. Produce exactly one instruction per target block.
- scope: one of document / section / block / span — pick the most fitting one for this revision.
- summary: one concise sentence describing the overall revision plan. Never leave it null.
- Respect the writing context: keep facts consistent (never alter locked facts), preserve terminology and the style profile.
- Do not invent facts that conflict with the writing context.

Writing task:
{task_json}

Document (DocIR, target blocks only):
{doc_ir_json}

Locate result:
{locate_result_json}

Writing context:
{context_json}
'''


GENERATE_PATCH_SET_PROMPT = '''You are a patch generator. Given a document, a modify plan, and the writing context, produce a PatchSet with concrete text changes.

Rules:
- For each ModifyInstruction, produce exactly one PatchHunk.
- Each PatchHunk must have:
  - target_block_id: copied from the corresponding ModifyInstruction.
  - modify_type: copied from the corresponding ModifyInstruction.
  - new_text depends on modify_type:
    - rewrite / replace: the FULL new text for that block after applying the instruction.
    - insert: the FULL content of the new block to insert right after the target block.
    - delete: leave new_text null.
- Leave anchor and old_text as null — the system fills them from the document automatically.
- new_text must be complete, self-contained prose. Never produce partial text, placeholders, or ellipsis-only output.
- Respect the writing context: keep facts consistent (never alter locked facts), preserve terminology and style.
- Do not invent facts that conflict with the writing context.

Document (DocIR):
{doc_ir_json}

Modify plan:
{modify_plan_json}

Writing context:
{context_json}
'''
