# flake8: noqa
GENERATE_DRAFT_SECTION_PROMPT = '''Generate a draft section from the given writing task, section instruction, and writing context.

Requirements:
- Return a single WriterBlock object with stage="draft".
- The returned block is the section root. Use type="heading" and put the section title in content.
- The section's actual prose lives in the block's children. Use paragraph blocks for prose.
- A paragraph child usually represents one substantial paragraph or paragraph group.
- The section instruction is a writing plan, not a list of visible headings.
- Use expected_blocks to guide coverage and ordering, but do not copy them verbatim as headings.
- expected_blocks are minimum coverage cues, not a maximum block count. Expand them when needed.
- Choose a reasonable number of paragraph children based on section complexity, expected_blocks, and required_points.
- Each paragraph child's content must contain complete prose with multiple meaningful sentences unless the block is intentionally non-textual.
- Keep text blocks substantial enough to carry their intended idea.
- Do not generate short summary-like or placeholder-like blocks just to match the expected_blocks count.
- If expected_blocks is too coarse, add additional content blocks for setup, transition, evidence/detail, consequence, or closing as appropriate.
- Respect required_points, fact_constraints, style_constraints, and relation_constraints.
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
- Omit spans, provider_binding and provider_payload; the system manages them.
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
- Respect required_points, fact_constraints, style_constraints, and relation_constraints.
- Use references when relevant, but do not copy reference metadata into the document.
- Do not invent facts that conflict with the writing context.
- If previous Markdown is provided, maintain continuity and avoid repetition.
- Use ordinary Markdown paragraphs, lists, quotes, fenced code, tables, images, and
  subheadings only when they help the requested content.
- Return substantial finished prose, not a summary, placeholder, or planning notes.

Writing task:
{task_json}

Writing context:
{context_json}

Previously drafted Markdown (context only; do not review, summarize, or continue it):
{previous_markdown}

Current section instruction:
{section_instruction_json}

Write only the body of the current section now. Begin directly with its finished
prose and follow the current section instruction, even when the previous Markdown
covers a different section.
'''
