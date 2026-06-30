GENERATE_DRAFT_SECTION_PROMPT = """Generate a draft section from the given writing task, section instruction, and writing context.

Requirements:
- Return a DraftSection object.
- Generate content only for the provided section_instruction.
- Preserve section_instruction.outline_node_id as DraftSection.outline_node_id.
- Preserve section_instruction.instruction_id as DraftSection.instruction_id.
- DraftSection.title should match section_instruction.section_title.
- DraftBlock is a basic document content block. It usually represents one substantial paragraph or paragraph group.
- DraftBlock.heading is optional. Do not fill heading unless the block itself truly needs a visible subheading.
- section_instruction.expected_blocks is an internal writing plan, not a list of visible headings.
- Use expected_blocks to guide coverage and ordering, but do not copy them into DraftBlock.heading by default.
- expected_blocks are minimum coverage cues, not a maximum block count. Expand them when needed.
- Choose a reasonable number of DraftBlock entries based on section complexity, expected_blocks, and required_points.
- Each text DraftBlock.content must contain complete prose with multiple meaningful sentences unless the block is intentionally non-textual.
- Keep text blocks substantial enough to carry their intended idea.
- Do not generate short summary-like or placeholder-like blocks just to match the expected_blocks count.
- If expected_blocks is too coarse, add additional content blocks for setup, transition, evidence/detail, consequence, or closing as appropriate.
- Respect required_points, fact_constraints, style_constraints, relation_constraints, and source_refs.
- Do not invent facts that conflict with the writing context.
- If previous_sections are provided, keep continuity and avoid repeating their content.

Writing task:
{task_json}

Section instruction:
{section_instruction_json}

Writing context:
{context_json}

Previous sections:
{previous_sections_json}
"""
