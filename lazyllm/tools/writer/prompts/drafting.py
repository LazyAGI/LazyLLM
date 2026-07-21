# flake8: noqa
GENERATE_DRAFT_SECTION_PROMPT = '''Generate a draft section from the given writing task, section instruction, and writing context.

Requirements:
- Return a single WriterBlock object with stage="draft".
- The returned block is the section root. Use type="heading" and put the section title in content.
- The section's actual prose lives in the block's children, each a WriterBlock with type="paragraph".
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
- Leave spans, status, authoring, references, provider_binding and provider_payload empty.

Writing task:
{task_json}

Section instruction:
{section_instruction_json}

Writing context:
{context_json}

Previous blocks:
{previous_blocks_json}
'''
