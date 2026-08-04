# flake8: noqa
LOCATE_REVISION_TARGET_PROMPT = '''You are a revision target locator. Identify the existing document blocks that participate in the requested revision.

Output semantics:
- task.query is the revision request.
- target_title indicates whether the document title participates in the revision.
- Updates and deletions target the blocks being changed.
- Insertions target the existing section or block being extended.
- Reordering targets every existing block whose relative order participates in the change.
- targets contains each selected block as an object with content_ref.node_id copied from the
  document and a brief reason.
- summary describes the revision scope in one sentence.
- A body revision has one or more targets.

Writing task:
{task_json}

Document blocks in source order:
{document_json}
'''


GENERATE_MODIFY_PLAN_PROMPT = '''You are a revision planner. Translate the requested end state into a ModifyPlan over the located source blocks.

Plan semantics:
- title_instruction describes the requested title result when target_title is true.
- Each ModifyInstruction represents one content or structural operation:
  - create inserts a contiguous sequence of new sibling blocks;
  - update changes the visible fields of an existing block;
  - delete removes an existing block;
  - move relocates an existing block.
- A contiguous insertion uses one create instruction so its blocks share one destination
  and retain their final document order.
- content_ref.node_id identifies the located source block involved in the operation.
- instruction describes the complete visible result of the operation.
- instruction_id is unique, and instructions follow execution order.
- scope and summary describe the plan as a whole.
- The result preserves the facts, terminology, and style established by the writing context.

Writing task:
{task_json}

Document (complete WriterDocument, including possible move destinations):
{document_json}

Locate result:
{locate_result_json}

Writing context:
{context_json}
'''


GENERATE_PATCH_SET_PROMPT = '''You are a revision content writer. Produce the visible document content requested by a ModifyPlan.

Output semantics:
- changes maps each instruction_id to its authored content.
- An update contains one RevisionBlockContent describing the resulting visible fields.
  Omitted type, numbering, and references retain their source values. Spans represent
  retained inline formatting and concatenate to the block content.
- A create contains all new sibling blocks in final document order. Children represent
  genuine document hierarchy.
- Delete and move instructions have empty content lists because their result is structural.
- new_title represents title_instruction when the plan includes a title revision.
- Headings use type="heading" with numbering.level; inline formatting uses spans.
- All authored content is complete, self-contained, and consistent with the writing context.

Visible document:
{document_json}

Modify plan:
{modify_plan_json}

Writing context:
{context_json}
'''
