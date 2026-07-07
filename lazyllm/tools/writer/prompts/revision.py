# flake8: noqa
LOCATE_REVISION_TARGET_PROMPT = '''You are a revision target locator. Given a writing task (the user's revision request), a document (DocIR), and the writing context, identify which document blocks need to be modified.

Rules:
- Read task.query carefully — it contains the user's revision request.
- Examine every document block and select the ones that genuinely need modification to fulfill the request.
- Select block_ids ONLY from the candidate list below. Never invent block_ids.
- Be precise: only select blocks that actually need changes. Do not select blocks that are already correct.
- target_reasons: for each selected block_id, give a one-sentence reason why it needs modification.
- If no blocks need modification, return an empty target_block_ids list.

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
- modify_type must be one of: rewrite, polish, insert, delete, move, split, merge.
- instruction: a concise description of what change to make to that block, derived from task.query.
- Every ModifyInstruction.target_block_id must come from the locate_result's target_block_ids. Produce exactly one instruction per target block.
- scope: one of document / section / block / span — pick the most fitting one for this revision.
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
  - new_text: the FULL new text for that block after applying the instruction.
- Leave anchor and old_text as null — the system fills them from the document automatically.
- new_text must be complete, self-contained prose. Never produce partial text, placeholders, or ellipsis-only output.
- For modify_type=delete, set new_text to an empty string.
- Respect the writing context: keep facts consistent (never alter locked facts), preserve terminology and style.
- Do not invent facts that conflict with the writing context.

Document (DocIR):
{doc_ir_json}

Modify plan:
{modify_plan_json}

Writing context:
{context_json}
'''
