# flake8: noqa
RETRY_MARKDOWN_IMAGE_REFERENCES_PROMPT = '''
The previous replacements failed image validation: {error}
Regenerate the complete StringReplaceSet against the original document and plan.
Preserve existing images. Every new image must use ![caption](media-placeholder://<need_id>)
with the exact visual_instruction.need_id from the plan. Include every required image.
Do not copy invented paths from the plan's prose. Image acquisition is already handled separately.
'''

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
- For Markdown, copy heading_path and occurrence exactly from one candidate. Never add
  node_id or placeholder_id to a Markdown heading reference.
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
  - visual_instruction.visual_type may be "image", "diagram", "chart", or "table".
  - visual_instruction.purpose is the semantic image requirement used to match an
    uploaded asset or acquire a new image. required must be true.
  - visual_instruction.preferred_strategy may be null or "image_generation". For
    "image" and "diagram", it may also be "web_search".
  - A delete instruction targeting an existing image must not include visual_instruction.
  - Existing image blocks must not be updated or moved. Text blocks continue to support
    create, update, delete, and move.
- A contiguous insertion uses one create instruction so its blocks share one destination
  and retain their final document order.
- content_ref identifies the located content involved in the operation.
- Every content_ref and destination_ref must copy exactly one reference from locate_result
  or the visible document. Locator kinds are mutually exclusive; never combine node_id,
  heading_path, placeholder_id, or document_root in one reference. Describe a fragment
  inside a Markdown section in instruction while retaining the section heading_path.
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


LOCATE_REVISION_TARGET_MARKDOWN_PROMPT = '''You are a Markdown revision target locator. Identify the existing document content that participates in the requested revision.

Output semantics:
- task.query is the revision request.
- target_title indicates whether the document title participates in the revision.
- Updates and deletions target the blocks being changed.
- Insertions target the existing section or block being extended.
- Reordering targets every existing block whose relative order participates in the change.
- targets contains the relevant content_ref values copied exactly from the document candidates.
- For deletion requests, select only candidates containing content to delete. Content mentioned
  solely to be preserved is a constraint, not an additional target. Return each selected section
  reference once, even when several paragraphs inside it are mentioned in the request.
- Each target contains content_ref and a brief reason.
- For Markdown, copy heading_path and occurrence exactly from one candidate. Never add
  node_id or placeholder_id to a Markdown heading reference.
- The supplied candidate references are the authority for the document's heading structure.
  Locate a paragraph by its meaning within a candidate's content, then copy that candidate's
  content_ref unchanged. Do not append body text to heading_path to identify the paragraph.
  Stage names, bold text, and standalone short lines inside candidate content do not create
  additional heading levels, even when they look like headings or the request calls them sections.
  Example: a candidate has heading_path=["Guide", "Workflow"] and its body contains the
  standalone line "Final phase" followed by a paragraph. To revise that paragraph, return
  ["Guide", "Workflow"], not ["Guide", "Workflow", "Final phase"]. The containing section
  reference locates the paragraph; it does not expand the requested edit to the whole section.
- occurrence distinguishes repeated sections with the same complete heading_path. It is not
  a paragraph number, sentence number, or the candidate's position in the document. Never derive
  it from "first/second paragraph". If the candidate omits occurrence, leave it omitted;
  its default is 1. Use a different occurrence only when supplied by the selected candidate.
  Example: one section contains two paragraphs, and the request deletes the first while retaining
  the second. Return that section's candidate reference once; do not invent occurrence=2 for
  the retained paragraph. Distinct supplied section occurrences remain distinct references.
- Markdown HTML anchors, including image anchors, are source text rather than Writer IR
  node IDs. Locate an image or paragraph through its containing section candidate.
- Plain text without headings uses content_ref.document_root=true.
- summary describes the revision scope in one sentence.
- A body revision has one or more targets.

Writing task:
{task_json}

Document candidates in source order:
{document_content}
'''


GENERATE_MODIFY_PLAN_MARKDOWN_PROMPT = '''You are a Markdown revision planner. Translate the requested end state into a MarkdownModifyPlan over the located document content.

Plan semantics:
- title_instruction describes the requested title result when target_title is true.
- Each MarkdownModifyInstruction represents one content or structural operation:
  - create inserts a contiguous sequence of new sibling blocks;
  - update changes the visible fields of an existing block;
  - delete removes existing content at the extent selected by target_scope;
  - move relocates an existing block.
- Choose modify_type and target_scope from the requested edit, independently of the
  containing section used as content_ref. If the request only removes specified existing
  text and preserves the rest verbatim, use delete. Determine target_scope from the complete
  unit requested below. Do not express a pure deletion as update or expand its requested extent.
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
  - visual_instruction.visual_type may be "image", "diagram", "chart", or "table".
  - visual_instruction.purpose is the semantic image requirement used to match an
    uploaded asset or acquire a new image. required must be true.
  - visual_instruction.preferred_strategy may be null or "image_generation". For
    "image" and "diagram", it may also be "web_search".
  - A delete instruction targeting an existing image must not include visual_instruction.
  - Existing image blocks must not be updated or moved. Text blocks continue to support
    create, update, delete, and move.
- A contiguous insertion uses one create instruction so its blocks share one destination
  and retain their final document order.
- content_ref identifies the located content involved in the operation.
- For every non-create instruction, content_ref must copy a reference from locate_result.targets.
  A create content_ref and a move destination_ref may also copy a reference from the visible document.
  Locator kinds are mutually exclusive; never combine node_id, heading_path, placeholder_id,
  or document_root in one reference.
- For Markdown, references use the containing section's heading_path and occurrence, or
  document_root=true for text without headings. Never use node_id or placeholder_id;
  HTML anchors such as <a id="block-IMAGE-2"></a> are not Writer IR node IDs.
- For each Markdown instruction, identify the requested unit and choose target_scope in this order:
  - section: the complete section, including its heading, owned anchors, body, images,
    and descendant subsections. Use only when the requested operation concerns that whole section.
  - paragraph: exactly one complete prose paragraph inside the referenced section.
    A request to delete a complete paragraph requires paragraph even if that paragraph contains
    only one sentence, is identified by meaning rather than a quotation, or is the section's
    only body paragraph and its heading will remain. A prose paragraph can contain multiple
    sentences or source lines; sentence count and physical line count do not determine its scope.
    Set locator_text to a short verbatim excerpt that identifies that paragraph. Prefer
    task.selection.text when it identifies the requested paragraph. Do not copy the whole
    long paragraph or invent a paragraph ID. Each paragraph operation has its own instruction.
    If the user describes the paragraph by meaning without quoting it, first identify the
    paragraph in the supplied document, then copy a short identifying excerpt into locator_text.
    The semantic description itself is not a verbatim locator_text.
  - fragment: part of a prose paragraph, such as a sentence or phrase, or an image or other
    non-prose fragment. Describe the exact target in instruction, retain the containing section's
    content_ref. For move, locator_text must copy the complete fragment verbatim from the raw
    Markdown, including punctuation and any enclosing inline markup. For other fragment operations,
    set locator_text=null. A complete paragraph is not a fragment merely because
    it occupies only part of a section or the user did not quote its opening words.
  Set target_scope explicitly. Before returning, check that it agrees with the unit described
  in instruction: deleting a complete paragraph must not be labeled fragment.
  Examples: "删除这个只有一句话的完整段落" requires delete + paragraph;
  "删除多句段落中的第二句" requires delete + fragment;
  "删除正文段落，保留所属标题" requires delete + paragraph;
  "改写这一段，使其更简洁" requires update + paragraph.
  The presence of delete and heading_path alone does not imply section scope.
- A section delete already removes its contained paragraphs, images, and subsections.
  Do not add separate delete instructions for content already covered by that section delete.
- Create must provide position; content_ref and position identify the insertion location.
- For move, content_ref identifies the content being moved, while destination_ref and
  position identify its destination. Move must provide both destination_ref and position.
  Set destination_scope to section, paragraph, or fragment independently of target_scope.
  For a paragraph destination, destination_locator_text is a short verbatim identifying excerpt;
  for a fragment destination, it is the complete raw Markdown fragment; for section, it is null.
  before/after refers to the start/end of the entire destination unit, including descendants for
  a section. Source and destination may share a section reference when moving distinct paragraphs
  or fragments within it. Copy complete inline markup; never cut inside links, emphasis, or code.
- instruction describes the complete visible result of the operation.
- For a pure deletion, instruction specifies the exact target and boundaries and requires
  all other content to remain unchanged. Describe preservation constraints briefly instead of
  copying retained sentences, paragraphs, or sections into instruction. For a complete paragraph,
  use its short locator_text and request deletion of the whole paragraph rather than quoting it
  in full. For a sentence or phrase, quote only the deletion target when exact text is needed;
  describe retained neighbors by position or role. The containing section reference only locates
  the operation; it does not determine the deletion range.
- Preserve existing cross-reference links. Do not plan to remove or rewrite an
  internal reference unless the user asks to change it or it belongs to content being deleted.
  An image deletion may also remove its owned anchor and the corresponding prose link.
- Preserve every explicit structural constraint from task.query in instruction itself,
  including paragraph count, list-item count, heading level, and ordering. Render distinct
  Markdown paragraphs with blank lines and render lists/headings with their Markdown syntax;
  do not record required structure only in meta.
- When task.query requires inserted or updated content to reference an existing section or
  image, keep that requirement in instruction; the content writer emits a Markdown internal
  link using the existing target's exact anchor.
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


RETRY_LOCATE_REVISION_TARGET_MARKDOWN_PROMPT = '''

Your previous response could not be parsed. Return valid JSON only.
Copy heading_path and occurrence exactly from a Markdown candidate, or use document_root=true
for text without headings. Never use node_id or placeholder_id, including HTML image anchor IDs.
'''


RETRY_MODIFY_PLAN_MARKDOWN_PROMPT = '''

Your previous plan has invalid content references or missing required fields: {invalid_shapes}
For non-create instructions, copy content_ref exactly from locate_result.targets.
Use heading_path/occurrence or document_root only; HTML image anchors are not node_id values.
For target_scope=paragraph, locator_text must be a non-empty short verbatim excerpt from the
requested paragraph. Correct the reported errors and preserve the other operations and their order.
A whole-section deletion already includes all images within it. Return valid JSON only.

Previous plan:
{previous_plan_json}
'''


COMPLETE_MARKDOWN_PLAN_FIELDS_PROMPT = '''Complete only the missing fields in the existing Markdown revision plan.
Return a MarkdownPlanCompletion, not a replacement plan or document edits.

- instruction_index is the 1-based position in the existing instructions list. Never reorder operations.
- Each entry contains instruction_index and only missing fields listed in allowed_fields for that index.
  Do not repeat or change existing fields, even when their values would remain the same.
- Infer values from the existing operation and the supplied document. Preserve its requested outcome.
- A move requires destination_ref, destination_scope and position. Do not infer destination_scope
  from target_scope: the source and destination may have different scopes.
- When supplying destination_scope=paragraph or fragment, also supply destination_locator_text if
  it is missing. For paragraph, copy a short unique verbatim excerpt; for fragment, copy the complete
  fragment verbatim, including inline markup. For section, no locator text is required.
- Missing locator_text follows the same paragraph/fragment rules for the source target_scope.
- Use Markdown heading_path/occurrence or document_root references, never node_id or placeholder_id.
- Return only the small JSON completion. Do not output old_string/new_string or copy whole sections.

Previous plan:
{previous_plan_json}

Validation errors:
{errors_json}

Allowed fields by instruction_index (including conditional destination_locator_text):
{allowed_fields_json}

Markdown document (reference data):
{document_content}
'''


GENERATE_STRING_REPLACE_SET_PROMPT = '''You are a Markdown revision writer. Convert the ModifyPlan into a StringReplaceSet that applies the requested revision directly to the supplied Markdown.

Output semantics:
- Each replacement contains an exact old_string copied from the Markdown and its complete new_string.
- content_ref uses heading_path and occurrence to identify the affected Markdown section.
- Plain text without headings uses content_ref.document_root=true.
- Update replaces the selected content, and create inserts content before or after its content_ref.
- Delete removes exactly the requested content. For each pure sentence or phrase deletion:
  1. Extract the exact deletion target from the supplied Markdown, excluding retained neighbors.
  2. Check whether that target occurs once within content_ref. If it does, return exactly one
     replacement for this target with old_string equal to the target and new_string="".
  3. Only when the target actually repeats and needs disambiguation, include the smallest
     necessary surrounding context in old_string and preserve that context verbatim in new_string.
  4. Before returning, check whether old_string/new_string repeat any retained prefix or suffix.
     Remove unnecessary shared context while keeping the intended occurrence identifiable.
     A unique deletion target must end with new_string=""; do not rewrite retained neighbors.
  Any quoted preservation context or locator_text in the plan is a location hint, not an instruction
  to include that text in the replacement.
  Examples for unique targets: from "第一句。第二句。第三句。", deleting only the first sentence
  uses old_string="第一句。", new_string=""; deleting only the middle sentence uses
  old_string="第二句。", new_string=""; deleting only the last sentence uses
  old_string="第三句。", new_string="". From "术语（补充说明）", deleting only the parenthesis
  and its contents uses old_string="（补充说明）", new_string=""; leave "术语" outside the patch.
- Copy old_string character for character from the supplied Markdown, including punctuation,
  whitespace, line breaks, Markdown syntax, and backslashes.
  Preserve the original form of symbols and punctuation in copied text: Chinese forms must remain Chinese
  and English forms must remain English; never convert between them.
  Never invent blank lines between sentences or normalize the source text.
  JSON escaping must decode to the exact original text.
- Move is represented by replacements that remove the source content and insert it at destination_ref.
  Respect destination_scope and destination_locator_text as well as position. Generate local
  replacements for the affected content; do not rewrite the entire document to implement a move.
- Every ModifyPlan instruction must be implemented by one or more replacements; do not omit an instruction.
- Respect target_scope even when deterministic location was unavailable: section replaces the
  complete section, paragraph replaces the complete paragraph, and fragment changes only the
  requested fragment. locator_text identifies a paragraph; it is not itself the deletion range.
- A section starts with its owned anchor(s), if any, followed by its heading, and ends before
  the next heading of the same or higher level and that heading's owned anchors. Include all
  descendant subsections. Heading-like text inside fenced code is not a section boundary.
- Markdown HTML anchors are not node_id values. Use heading_path/occurrence or document_root only.
- Every replacement must make a real change: new_string must differ from old_string and implement the instruction.
- Never return an unchanged section as a replacement.
- new_string must visibly preserve the instruction's required Markdown structure and exact
  counts/order. In particular, distinct paragraphs are separated by a blank line; do not
  collapse them into sentences in one paragraph even if replacement meta describes them.
- Replacements are returned in application order and preserve unaffected Markdown exactly.
- Preserve existing <a id="block-..."></a> anchors and internal links exactly.
  A whole-section deletion includes its owned anchors and contained images. An image deletion
  may include its owned anchor and corresponding prose link; preserve unrelated references.
- Unless the user explicitly asks to modify the relevant structure, preserve non-standard Markdown
  extensions and structural markers exactly: double-bracket links and embeds including their targets;
  Callout prefixes such as > [!note] and their + or - fold markers; inline comments; block-ID markers;
  and complete query fenced blocks including their contents. Callout titles and bodies may be rewritten.
  Do not convert double-bracket links into ordinary Markdown links or URLs.
- Image handling:
  - When an instruction creates a new image, put exactly `![<caption>](media-placeholder://<need_id>)`
    in new_string at the insertion position. Use the need_id from that create instruction's
    visual_instruction; do not reuse a need_id from a different instruction.
    Never use non-standard embed syntax, a local filename/path, a raw URL, or any other image syntax.
  - When an instruction deletes an image, old_string must include the complete image line
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


GENERATE_MARKDOWN_REVISION_BATCH_PROMPT = '''You are a Markdown revision writer. Complete all operations together against the original document and return one MarkdownRevisionBatch.

Output semantics:
- Operation keys are supplied by the program and are separate from instruction_id. Copy them exactly.
- For an update with located_old_string, return contents[key] containing only new_string.
  Rewrite that complete range, including required headings and structure, without copying old_string.
- A delete with located_old_string is performed by the program. Omit it from both output mappings.
- For every operation without located_old_string, return a non-empty replacements[key] list
  of StringReplace objects implementing that operation. Each old_string must be copied exactly
  from the original document. Do not include changes assigned to another operation.
- If title_instruction is non-null, return its replacements under the special key "title".
- Return exactly the required mapping keys. Do not omit operations or invent keys.
- The program assembles the title replacements first, then the operations in supplied order.
  Keep each replacement list in application order. Do not pre-apply an earlier operation when
  copying old_string: all operations are generated together against the same original document.
- Update replaces the selected content, create inserts before or after content_ref, and move
  removes source content and inserts it at destination_ref. Delete removes only the requested content.
- For each pure sentence or phrase deletion requiring model replacements, first extract only
  the exact deletion target, then check whether it occurs once within content_ref. If unique,
  return exactly one replacement for this target with old_string equal to it and new_string="".
  Only actual repeated targets needing disambiguation may include the smallest necessary context,
  preserved verbatim in new_string. Before returning, remove shared retained prefixes or suffixes
  that are unnecessary to identify the intended occurrence. Quoted preservation context or
  locator_text in the plan is a location hint, not part of the deletion target.
  Examples for unique targets in "第一句。第二句。第三句。": deleting the middle sentence uses
  old_string="第二句。", new_string=""; deleting the last sentence uses
  old_string="第三句。", new_string="". In "术语（补充说明）", deleting only the parenthesis
  and its contents uses old_string="（补充说明）", new_string=""; do not reproduce "术语".
  Never invent whitespace or line breaks; JSON escaping must decode to the exact source text.
- References use heading_path and occurrence, or document_root=true. Never use node_id or
  placeholder_id for Markdown HTML anchors.
- Respect target_scope: section includes its owned anchors, heading, body, images and descendants,
  ending before the next peer or ancestor heading and its anchors. Fenced code is not a heading.
  paragraph means the complete paragraph identified by locator_text, not only that excerpt.
  fragment changes only the requested sentence, phrase, image or other partial content.
- Preserve unaffected Markdown, facts, anchors, links, image paths, whitespace, and non-standard
  extensions including double-bracket links/embeds, Callouts, comments, block IDs and query fences.
  Preserve required paragraph counts, heading levels and order using actual Markdown syntax.
- For newly created content, emit no block anchors or IDs; the program assigns them.
  Preserve the original anchors and links of retained or moved existing content.
- A whole-section deletion includes its contained images and anchors. An image deletion must
  include the complete image line and may also remove its owned anchor and corresponding prose link.
- Create an image only as ![<caption>](media-placeholder://<need_id>), using the exact need_id
  from that create instruction's visual_instruction. Never invent IDs, paths or URLs.
- Do not return unchanged replacements, surrounding unrelated content, or explanatory text.

Original Markdown document:
{document_content}

Operations in execution order, with program locations when available:
{operations_json}

Title instruction:
{title_instruction_json}

Writing context:
{context_json}
'''


GENERATE_MARKDOWN_REVISION_CONTENT_PROMPT = '''Rewrite the complete, program-located Markdown range below.

Return only new_string in the structured response. The program already has the exact old_string;
do not copy the old Markdown into a separate response field. Implement the instruction over this
entire range and retain unaffected content, formatting, anchors, image paths, and internal links.
Preserve non-standard Markdown extensions, including double-bracket links/embeds, Callouts,
inline comments, block-ID markers, and query fences. Do not invent IDs, paths, or URLs.
Respect target_scope: section includes its heading and descendants; paragraph is the complete
located paragraph. Follow requested paragraph counts and structure using Markdown syntax.
Do not include content outside the located range or add explanatory text or an outer code fence.
Preserve the range's leading and trailing line breaks so adjacent Markdown stays separate.

Instruction:
{instruction_json}

Complete original Markdown range:
{markdown_block}

Writing context:
{context_json}
'''


REWRITE_MARKDOWN_BLOCK_PROMPT = '''You are revising exactly one Markdown block.

Return one StringReplace. Copy the selected block exactly into old_string and
return the complete replacement paragraph in new_string. Set content_ref to
document_root=true.
Preserve unaffected inline formatting.
Preserve existing internal links and any inline formatting inside the selected block.
Unless the user explicitly asks to modify the relevant structure, preserve non-standard Markdown
extensions and structural markers exactly: double-bracket links and embeds including their targets;
Callout prefixes such as > [!note] and their + or - fold markers; inline comments; block-ID markers;
and complete query fenced blocks including their contents. Callout titles and bodies may be rewritten.
Do not convert double-bracket links into ordinary Markdown links or URLs.
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
- Omit delete and move instructions from changes; the program executes these structural operations.
- new_title represents title_instruction when the plan includes a title revision.
- Headings use type="heading" with numbering.level; inline formatting uses spans.
- Tables use table children of type table_row, whose children are table_cell blocks. Update one cell by targeting its
  table_cell; add, delete, or move rows and cells structurally. Never put a Markdown table in table.content.
- All authored content is complete, self-contained, and consistent with the writing context.
- For an image create, return exactly one new block with type="image". Its content is
  the final caption. Do not invent references or asset IDs; the system adds the single
  resolved media_asset reference after generation.
- Preserve existing internal_ref spans in updated text blocks. Do not invent new
  target_node_id values; to reference an existing heading or image block, copy its exact
  node_id from the visible document into a non-empty internal_ref span containing the
  natural words that carry the link.
- content equals the concatenation of all span text. Example:
  spans=[{{"text":"详见"}},{{"text":"架构设计","style":{{"link":{{"type":"internal_ref","target_node_id":"sec-related"}}}}}},{{"text":"中的定义。"}}]
  with content="详见架构设计中的定义。".

Visible document:
{document_json}

Modify plan:
{modify_plan_json}

Writing context:
{context_json}
'''
