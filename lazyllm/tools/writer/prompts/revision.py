# flake8: noqa
LOCATE_REVISION_TARGET_PROMPT = '''You are a revision target locator. Given a writing task (the user's revision request), a document (WriterDocument), and the writing context, identify which document blocks need to be modified.

Rules:
- Read task.query carefully — it contains the user's revision request.
- Examine every document block and select the ones the revision acts on.
- If the request changes the document title, set target_title to true. The document
  title is WriterDocument.title, not a content block; do not select a document/root
  block merely to change the title.
- If the request does not change the document title, set target_title to false.
- For update/delete, select the blocks whose content, formatting, type, or existence changes.
- For create, select the existing block used as the insertion anchor.
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
- If locate_result.target_title is true, set title_instruction to a clear instruction
  describing the requested document-title change. Otherwise leave title_instruction null.
- A document-title change is separate from block instructions and does not need a
  synthetic document/root block instruction.
- For each target block, decide the modify_type and write a clear, specific instruction.
- modify_type must be one of:
  - create: insert one or more brand-new blocks before or after the target block. The target block is the insertion anchor; set position to before or after.
  - update: update any user-visible field of the target block, including content, type, numbering, and spans.
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


GENERATE_PATCH_SET_PROMPT = '''You are a document revision executor. Given a complete WriterDocument, a modify plan, and the writing context, return the complete revised WriterDocument.

Rules:
- Return a complete WriterDocument, not a PatchSet.
- Preserve document_id, stage, revision, metadata, provider_binding, and ui_editable.
- Preserve node_id, provider_binding, provider_payload, and editable for every existing block.
- update: change the requested user-visible block fields. Heading levels belong in
  type="heading" and numbering.level. Inline formatting belongs in spans.
- create: add complete WriterBlock objects at the requested positions. Assign each new
  block a unique node_id beginning with "writer-new-" and leave provider_binding and
  provider_payload empty.
- delete: remove the requested block subtree.
- move: move the existing WriterBlock without changing its node_id or provider fields.
- If title_instruction is absent, preserve the title exactly.
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
