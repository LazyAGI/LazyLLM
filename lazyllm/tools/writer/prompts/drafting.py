# flake8: noqa
GENERATE_DRAFT_SECTION_PROMPT = '''Generate a draft section from the given writing task, section instruction, and writing context.

Requirements:
- Return a single WriterBlock object with stage="draft".
- The returned block is the section root. Use type="heading" and put the section title in content.
- The section's actual prose lives in the block's children. Use paragraph blocks for prose.
- A paragraph child usually represents one substantial paragraph or paragraph group.
- The section instruction is a writing plan, not a list of visible headings.
- Write headings without visible numbering; the system renders numbers.
- Use expected_blocks to guide coverage and ordering, but do not copy them verbatim as headings.
- Treat expected_blocks as priorities, not minimum paragraph counts. Combine or omit
  secondary cues when necessary to fit the section budget.
- Choose a reasonable number of paragraph children based on section complexity, expected_blocks, and required_points.
- Each paragraph child's content must contain complete prose with multiple meaningful sentences unless the block is intentionally non-textual.
- Keep text blocks substantial enough to carry their intended idea.
- Do not generate placeholder-like blocks just to match the expected_blocks count.
- section_instruction.meta.target_chars is the preferred prose length and
  section_instruction.meta.max_chars is a hard prose limit when present.
- The length limit takes precedence over exhaustive source coverage or prose expansion.
- Respect required_points, fact_constraints, style_constraints, and relation_constraints.
- When section_instruction.meta.rewrite=true, treat meta.source_content as the authoritative
  source material for this section and meta.source_format as formatting guidance. Rewrite it
  according to the instruction without exposing source metadata in the result.
- Use the facts and resources identified by references when relevant. Do not copy or rewrite references
  in the output; the system carries them from the section instruction.
- Do not invent facts that conflict with the writing context.
- If previous_blocks are provided, keep continuity and avoid repeating their content.
- Fill node_id for the section root and for each child (e.g. draft-<node>-1). The system will normalize ids if needed.
- section_media lists visual needs and their resolved assets. When a listed asset helps the section,
  insert an independent child WriterBlock with type="image" at the appropriate reading position.
  Its content is the final Chinese caption and references must contain exactly one
  {{"type": "media_asset", "id": "..."}} entry from section_media. Do not invent asset IDs,
  paths, URLs, tokens, placeholders, or image blocks for unresolved needs.
  The image block node_id is both the visual need_id and cross-reference target; its
  media_asset id is separate and must be copied from section_media.
- Omit spans, provider_binding and provider_payload; the system manages them.
- Use section_instruction.meta.cross_references as the authoritative cross-reference plan.
  For each item, the normalized "target" is the exact node_id to use.
  If must_create=true, create one child WriterBlock with type="image",
  node_id=target, and content=caption.
  To reference a target, add an internal_ref span with target_node_id=target and text="".
  Example: {{"text":"","style":{{"link":{{"type":"internal_ref","target_node_id":"sec-2"}}}}}}.
  Include each required target exactly once, and use no references beyond this plan.
  Leave internal reference display text empty; the system renders it.
  Do not invent target_node_id values outside this plan.
- Emit WriterBlock fields in schema order. In particular, emit numbering and references before content.

Writing task:
{task_json}

Section instruction:
{section_instruction_json}

Writing context:
{context_json}

Previous blocks:
{previous_blocks_json}

Resolved section media:
{section_media_json}
'''


GENERATE_DRAFT_SECTION_MARKDOWN_PROMPT = '''Generate the body of one draft section in Markdown.

Requirements:
- Output Markdown only. Do not wrap the response in an outer code fence.
- Do not output reasoning, analysis, review notes, or <think> tags.
- Do not output the section title or its heading; the system adds the heading.
- Follow the section instruction as a writing plan, not as a list of visible headings.
- Treat expected_blocks as coverage priorities, not minimum paragraph counts. Combine or
  omit secondary cues when necessary to fit the section budget.
- section_instruction.meta.target_chars is the preferred prose length and
  section_instruction.meta.max_chars is a hard prose limit when present.
- The length limit takes precedence over exhaustive source coverage or prose expansion.
- Write headings without visible numbering; the system renders numbers.
- Respect required_points, fact_constraints, style_constraints, and relation_constraints.
- When section_instruction.meta.rewrite=true, treat meta.source_content as the authoritative
  source material for this section and meta.source_format as formatting guidance. Rewrite it
  according to the instruction without exposing source metadata in the result.
- Use references when relevant, but do not copy reference metadata into the document.
- Do not invent facts that conflict with the writing context.
- If previous Markdown is provided, maintain continuity and avoid repetition.
- Use ordinary Markdown paragraphs, lists, quotes, fenced code, tables, images, and
  subheadings only when they help the requested content.
- Use section_instruction.meta.cross_references as the authoritative cross-reference plan.
  Each item's "target" is the exact system key; an image target equals its visual need_id.
  For a required target, emit exactly [](#block-<target>) once in natural prose.
  Do not invent target keys outside this plan.
- Return substantial finished prose, not a summary, placeholder, or planning notes.
- section_visual_needs lists the visual needs planned for this section. It is independent
  of whether media resolution succeeds. When a required visual need is listed, place its
  image at the most appropriate reading position using exactly:
  ![short caption](media-placeholder://<need_id>).
  Use the shared need_id/target from the must_create cross-reference. Emit each planned image
  exactly once; the system inserts its numbering anchor.
- Do not output image markup for anything outside this section's section_visual_needs list.

Writing task:
{task_json}

Writing context:
{context_json}

Previously drafted Markdown (context only; do not review, summarize, or continue it):
{previous_markdown}

Current section instruction:
{section_instruction_json}

Planned section visual needs:
{section_visual_needs_json}

Write only the body of the current section now. Begin directly with its finished
prose and follow the current section instruction, even when the previous Markdown
covers a different section.
'''


CONDENSE_DRAFT_SECTION_PROMPT = '''Condense this WriterBlock draft section.

Requirements:
- Return one WriterBlock with the same root node_id, heading, and stage="draft".
- Keep the main plot or argument, ending, point of view, tone, and required non-text blocks.
- Do not add new facts, scenes, claims, headings, or planning notes.
- The combined non-whitespace prose in the returned block's descendants must not exceed {max_chars} characters.

Section instruction:
{section_instruction_json}

Draft section:
{draft_section_json}
'''


CONDENSE_DRAFT_SECTION_MARKDOWN_PROMPT = '''Condense this Markdown section body to at most {max_chars} non-whitespace characters.

Preserve the main plot or argument, ending, point of view, tone, and essential Markdown.
Do not add new content, a section heading, reasoning, or planning notes.
Return only the condensed section body.

Section instruction:
{section_instruction_json}

Draft body:
{draft_body}
'''
