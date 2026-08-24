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
- Sections may be drafted independently and in parallel. Treat document-global point of view,
  tense, narrative voice, character identity, and naming constraints as strict. Do not invent a
  proper name for an unnamed protagonist. If multiple POV options remain without an explicit
  selection, use third-person limited consistently.
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
- Omit provider_binding and provider_payload; the system manages them.
- Use section_instruction.meta.cross_references as the authoritative cross-reference plan.
  For each item, the normalized "target" is the exact node_id to use.
  If must_create=true, create one child WriterBlock with type="image",
  node_id=target, and content=caption.
  To reference a target, use a non-empty internal_ref span for the natural words that
  carry the link, with target_node_id=target. All spans together must contain the complete sentence.
  Example: {{"text":"架构设计","style":{{"link":{{"type":"internal_ref","target_node_id":"sec-2"}}}}}}.
  A required image item produces both its image child block and one internal_ref span in prose.
  Include each required target exactly once, and use no references beyond this plan.
  Do not use target_node_id values outside section_instruction.meta.cross_reference_targets.
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
- Sections may be drafted independently and in parallel. Treat document-global point of view,
  tense, narrative voice, character identity, and naming constraints as strict. Do not invent a
  proper name for an unnamed protagonist. If multiple POV options remain without an explicit
  selection, use third-person limited consistently.
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
  For a required target, use its guidance to link natural, non-empty wording as
  [reference wording](#block-<target>) exactly once inside a complete sentence.
  Do not use target keys outside section_instruction.meta.cross_reference_targets.
- Return substantial finished prose, not a summary, placeholder, or planning notes.
- The system places planned images after the prose link. Do not output image markup.

Writing task:
{task_json}

Writing context:
{context_json}

Previously drafted Markdown (context only; do not review, summarize, or continue it):
{previous_markdown}

Current section instruction:
{section_instruction_json}

Write only the body of the current section now. Begin directly with its finished
prose and follow the current section instruction even when the previous Markdown
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


GENERATE_SHORT_DOCUMENT_MARKDOWN_PROMPT = '''Write one complete short article from its whole-document writing plan.

Requirements:
- Output Markdown body text only. Do not wrap the response in an outer code fence.
- Do not output the article title; the system adds the single H1 title.
- Do not output headings or subheadings of any level.
- Write the complete article in one pass as continuous prose. Paragraph breaks are allowed.
- Use prose paragraphs rather than lists, tables, block quotes, or planning labels.
- short_visuals contains resolved visuals for the complete article. For every item, output exactly one
  standalone Markdown image at a natural paragraph boundary using its exact need_id:
  ![concise caption](media-placeholder://<need_id>)
- Use purpose and placement_hint to choose the caption and location. Do not output unplanned images,
  asset paths, URLs, HTML anchors, or prose links to the image. When short_visuals is empty, output no images.
- Treat expected_blocks as an internal order and coverage guide. Do not copy its entries as headings,
  labels, a checklist, or separately generated fragments.
- Express core_viewpoint clearly while covering required_points within the available length.
- Respect fact_constraints and style_constraints.
- Use relevant references as source guidance, but do not copy reference metadata into the article.
- Do not invent facts that conflict with the writing context.
- task.constraints.target_chars is the preferred body length and task.constraints.max_chars is a hard
  non-whitespace body limit when present. The hard limit takes precedence over exhaustive coverage.
- Return substantial finished prose, not an outline, summary, review, or planning notes.

Writing task:
{task_json}

Whole-document writing plan:
{short_writing_plan_json}

Writing context:
{context_json}

Resolved short-document visuals:
{short_visuals_json}

Write only the finished article body now.
'''


GENERATE_SHORT_DOCUMENT_IR_PROMPT = '''Generate one complete short document as a WriterDocument.

Requirements:
- Return one WriterDocument object with stage="draft".
- Put the exact article title in the document title field. Do not create a heading block
  for the title or any other subsection heading.
- Put the article body in the document blocks. Use paragraph blocks for prose and choose
  other block types only when they materially help the requested content.
- The document must be flat: do not create blocks with type="heading" anywhere in the body.
- Each block and child block must have a stable non-empty node_id. The system may normalize
  the document id, stage, title, and editability metadata after generation.
- Respect the writing plan's required_points, references, fact_constraints, style_constraints,
  expected_blocks, and length limits. Do not copy planning metadata into visible prose.
- Do not invent facts that conflict with the writing context. Do not return reasoning, review
  notes, or planning commentary.
- For each resolved visual need in short_visuals, you may insert an image block at the most
  appropriate position. Its node_id must equal the need_id and its references must contain
  exactly one {{"type":"media_asset","id":"..."}} entry using an asset id listed for that
  need. Do not invent image blocks, asset ids, paths, or URLs for unresolved needs.
- Keep prose within short_writing_plan.meta.max_chars when that limit is present.

Writing task:
{task_json}

Short writing plan:
{short_writing_plan_json}

Writing context:
{context_json}

Resolved short-document visuals:
{short_visuals_json}

Return only the WriterDocument object now.
'''


CONDENSE_SHORT_DOCUMENT_IR_PROMPT = '''Condense this short WriterDocument without changing its flat structure.

Requirements:
- Return one WriterDocument with the same title, stage, and document meaning.
- Keep the main argument, conclusion, tone, required points, and existing image blocks.
- Do not add headings, new facts, planning notes, or unresolved media references.
- The combined non-whitespace prose in body blocks must not exceed {max_chars} characters.

Short writing plan:
{short_writing_plan_json}

Draft WriterDocument:
{draft_document_json}
'''
