# flake8: noqa
GENERATE_OUTLINE_PROMPT = '''Generate a writing outline from the given writing task and context.

Requirements:
- Return a WriterDocument object with stage="outline".
- Set document_id to the value given in document_id_hint below.
- Generate at least 3 top-level blocks unless the task explicitly asks for fewer.
- Each top-level block is a section. Use type="heading" for section blocks.
- Put the section title in block.content, and place a concrete writing instruction in
  block.authoring.instruction.
- Fill node_id for every block. Use stable ids such as section-1, section-2, section-1-1.
- Use block.numbering.level for the heading level: 1 for top-level sections, incrementing for children.
  Put child sections under block.children.
- Constraints live in block.authoring.constraints and may only use the fields defined by WriterConstraints:
  section_goal, required_points, fact_constraints, style_constraints, relation_constraints,
  min_words, max_words, pov, tone, must_include, must_avoid.
- Do not invent other constraints fields. Fill only the fields useful for the section.
- block.references holds identifiers for facts or resources the section depends on.
- Each element of block.references is an object with at least an "id" field. The id must match a
  DocumentFact.fact_id or ResourceProfile.resource_id present in the input. Additional fields are allowed.
- references is a WriterBlock field only. Never put references inside block.authoring or
  block.authoring.constraints.
- fact_constraints contains the literal factual statements that the section must preserve. Do not put
  fact IDs or resource IDs in fact_constraints; identifiers belong only in references.
- Every fact_constraints statement must be grounded in the writing context. If no factual statement
  applies to a section, leave fact_constraints empty.
- Prefer the target document title or task intent as document.title.
- Use the writing context and resource profiles as constraints, not as content to copy blindly.
- Leave spans, status, provider_binding and provider_payload empty; the system manages them.

Writing task:
{task_json}

document_id_hint: {document_id_hint}

Writing context:
{context_json}

Resource profiles:
{resource_profiles_json}

Execution results:
{execution_results_json}
'''


GENERATE_SECTION_INSTRUCTIONS_PROMPT = '''Generate section-level writing instructions from the outline and writing context.

Requirements:
- Return a SectionInstructionList object.
- Generate exactly one SectionInstruction for every top-level block listed in target_outline_blocks.
- Each instruction's outline_node_id MUST equal the corresponding outline block's node_id.
- section_title MUST equal the corresponding outline block's content.
- instruction_id should be stable, such as instruction-section-1 or instruction-ch01.
- section_goal should be concrete and actionable.
- required_points should contain the key content that must appear in the section.
- fact_constraints should preserve the literal text of locked facts and important context facts
  relevant to this section. It must not contain fact IDs or resource IDs.
- fact_constraints MUST only contain factual statements actually present in the writing context.
- references are owned by the authoritative outline. Leave references empty; the system copies them from
  the matching outline block.
- style_constraints should include tone, pov, audience, and style requirements when applicable.
- relation_constraints should describe dependencies on previous or later sections when useful.
- expected_blocks should be a concise block-level content plan for the draft tool.
- For a normal section, expected_blocks should usually contain 3 to 6 planned content blocks unless the section is explicitly very short.
- expected_blocks are planning labels for coverage and ordering, not visible headings that must appear in final text.
- Do not invent facts that conflict with writing context.

Outline (authoritative structure):
{outline_json}

Target outline blocks to author:
{target_outline_blocks_json}

Writing context:
{context_json}

Execution results:
{execution_results_json}
'''
