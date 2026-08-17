# flake8: noqa
LOCATE_REVISION_TARGET_PROMPT = '''You are a revision target locator. Identify the existing document content that participates in the requested revision. The document may be represented as Writer IR or Markdown.

Output semantics:
- task.query is the revision request.
- target_title indicates whether the document title participates in the revision.
- Updates and deletions target the blocks being changed.
- Insertions target the existing section or block being extended.
- Reordering targets every existing block whose relative order participates in the change.
- targets contains the relevant content_ref values copied exactly from the document candidates.
- Each target contains content_ref and a brief reason.
- For Writer IR, each content_ref must be exactly {{"node_id": "<string copied from a candidate>"}}.
- node_id must be a string. Never put an object inside node_id, and do not add
  heading_path, document_root, occurrence, or other fields when node_id is present.
- Plain text without headings uses content_ref.document_root=true.
- summary describes the revision scope in one sentence.
- A body revision has one or more targets.

Writing task:
{task_json}

Document candidates in source order:
{document_content}
'''


GENERATE_MODIFY_PLAN_PROMPT = '''You are a revision planner. Translate the requested end state into a ModifyPlan over the located source blocks.

Plan semantics:
- title_instruction describes the requested title result when target_title is true.
- Each ModifyInstruction represents one content or structural operation:
  - create inserts a contiguous sequence of new sibling blocks;
  - update changes the visible fields of an existing block;
  - delete removes an existing block;
  - move relocates an existing block.
- Choose operations that match the requested structural outcome, not merely an
  approximately similar textual result:
  - If task.query explicitly asks to delete or remove N paragraphs, sections, or
    blocks, produce exactly N delete instructions targeting those complete existing
    blocks. Do not simulate structural deletion by shortening them with update.
  - If task.query asks to merge or consolidate blocks, update the retained block with
    the essential combined content and delete each absorbed block.
  - If task.query specifies only a substantial length reduction, choose update,
    delete, or a coherent combination based on semantic redundancy. Prefer removing
    or consolidating whole redundant blocks when that preserves the narrative or
    argument; do not mechanically shrink every located block by the same ratio.
  - Do not delete a semantically distinct block solely to satisfy a length target when
    its essential content cannot be preserved in a retained block.
- Image-specific revision semantics:
  - A create instruction that adds an image must include visual_instruction.
  - visual_instruction.need_id must equal instruction_id and its content_ref must
    equal the create instruction's content_ref.
  - visual_instruction.visual_type must be "image" for this revision workflow.
  - visual_instruction.purpose is the semantic image requirement used to match an
    uploaded asset or acquire a new image. required must be true.
  - visual_instruction.preferred_strategy must be null or "image_generation"
    for a revision image create.
  - A delete instruction targeting an existing image must not include visual_instruction.
  - Existing image blocks must not be updated or moved. Text blocks continue to support
    create, update, delete, and move.
- A contiguous insertion uses one create instruction so its blocks share one destination
  and retain their final document order.
- content_ref identifies the located content involved in the operation.
- Create must provide position; content_ref and position identify the insertion location.
- For move, content_ref identifies the content being moved, while destination_ref and
  position identify its destination. Move must provide both destination_ref and position.
- instruction describes the complete visible result of the operation.
- Preserve existing cross-reference links. Do not plan to remove or rewrite an
  internal reference unless the user explicitly asks to change that reference.
- Preserve every explicit structural constraint from task.query in instruction itself,
  including paragraph count, list-item count, heading level, and ordering. Render distinct
  Markdown paragraphs with blank lines and render lists/headings with their Markdown syntax;
  do not record required structure only in meta.
- When task.query requires inserted or updated content to reference an existing section or
  image, keep that requirement in instruction; the content writer emits an internal_ref span
  for the existing target.
- For an image create, describe one image block and its final caption in instruction;
  do not invent media_asset IDs, file paths, URLs, or provider identifiers.
- instruction_id is unique, and instructions follow execution order.
- scope and summary describe the plan as a whole.
- The result preserves the facts, terminology, and style established by the writing context.

Writing task:
{task_json}

Document, including possible move destinations:
{document_content}

Locate result:
{locate_result_json}

Writing context:
{context_json}
'''


GENERATE_STRING_REPLACE_SET_PROMPT = '''You are a Markdown revision writer. Convert the ModifyPlan into a StringReplaceSet that applies the requested revision directly to the supplied Markdown.

Output semantics:
- Each replacement contains an exact old_string copied from the Markdown and its complete new_string.
- content_ref uses heading_path and occurrence to identify the affected Markdown section.
- Plain text without headings uses content_ref.document_root=true.
- Update replaces the selected content, delete replaces it with an empty string, and create inserts content before or after its content_ref.
- Move is represented by replacements that remove the source content and insert it at destination_ref.
- Every ModifyPlan instruction must be implemented by one or more replacements; do not omit an instruction.
- Every replacement must make a real change: new_string must differ from old_string and implement the instruction.
- Never return an unchanged section as a replacement.
- new_string must visibly preserve the instruction's required Markdown structure and exact
  counts/order. In particular, distinct paragraphs are separated by a blank line; do not
  collapse them into sentences in one paragraph even if replacement meta describes them.
- Replacements are returned in application order and preserve unaffected Markdown exactly.
- Preserve existing <a id="block-..."></a> anchors and [](#block-...) links exactly.
  Do not rename or drop them unless the instruction explicitly targets that reference.
- Image handling:
  - When an instruction creates an image, put exactly `![<caption>](media-placeholder://<need_id>)`
    in new_string at the insertion position. Use the need_id from that create instruction's
    visual_instruction; do not reuse a need_id from a different instruction.
    Never use Obsidian/wiki syntax such as `![[...]]`, a local filename/path, a raw URL,
    or any other image syntax.
  - When an instruction deletes an image, old_string must be the complete image line
    (a line beginning with `![` and ending with `)`, including its complete image target/path). Identify
    the intended image line by caption or document order when the request references
    "first"/"second"/a caption.
  - Never invent need_id values, asset IDs, paths, or URLs.

Markdown document:
{document_content}

Modify plan:
{modify_plan_json}

Writing context:
{context_json}
'''


REWRITE_MARKDOWN_BLOCK_PROMPT = '''You are revising exactly one Markdown block.

Return one StringReplace. Copy the selected block exactly into old_string and
return the complete replacement paragraph in new_string. Set content_ref to
document_root=true.
Preserve unaffected inline formatting.
Preserve existing [](#block-...) links and any inline formatting inside the selected block.
Do not return surrounding document content or explanations.

Instruction:
{instruction}

Selected complete Markdown block:
{markdown_block}

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
- For an image create, return exactly one new block with type="image". Its content is
  the final caption. Do not invent references or asset IDs; the system adds the single
  resolved media_asset reference after generation.
- Preserve existing internal_ref spans in updated text blocks. Do not invent new
  target_node_id values; to reference an existing heading or image block, copy its exact
  node_id from the visible document into an internal_ref span with empty text. The system
  renders the reference and assigns created targets.
- A paragraph that references an existing block uses one empty internal_ref span between
  its text spans. content equals the concatenation of the text spans; an internal_ref
  span keeps text empty, so do not include the target title in content. The system inserts
  the rendered reference. Example: spans=[{{"text":"详见"}},{{"text":"","style":{{"link":{{"type":"internal_ref","target_node_id":"sec-related"}}}}}},{{"text":"中的定义。"}}] with content="详见中的定义。" renders as "详见第2章中的定义。".

Visible document:
{document_json}

Modify plan:
{modify_plan_json}

Writing context:
{context_json}
'''
