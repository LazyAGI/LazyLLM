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
- block.source_refs holds resource/fact identifiers the section depends on.
- Each element of block.source_refs is an object with at least an "id" field (the resource_id
  or fact key) and a "kind" field ("resource" or "fact"). Example: {{"id": "res-3", "kind": "resource"}}.
- fact_constraints and source_refs MUST only reference facts and resource_ids actually present
  in the writing context or resource profiles. If none applies to a section, leave the list empty.
- Prefer the target document title or task intent as document.title.
- Use the writing context and resource profiles as constraints, not as content to copy blindly.
- Leave spans, status, references, provider_binding and provider_payload empty; the system manages them.

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
- Return a WriterDocument object that mirrors the input outline structure.
- Echo document_id from the input outline verbatim. Keep stage="outline".
- For every top-level block, echo its node_id and type verbatim, leave content/numbering/children empty,
  and fill only block.authoring. The system merges your authoring back onto the authoritative outline;
  never invent or reorder blocks, and never change node_id.
- Each WriterAuthoring.origin_node_id MUST equal the outline block's node_id.
- instruction_id should be stable, such as instruction-section-1 or instruction-ch01.
- instruction should restate or sharpen the section goal as a concrete, actionable writing directive.
- constraints.section_goal should be concrete and actionable.
- constraints.required_points should contain the key content that must appear in the section.
- constraints.fact_constraints should preserve locked facts and important context facts relevant to this section.
- fact_constraints and source_refs MUST only reference facts and resource_ids actually present
  in the writing context or resource profiles. If no specific fact or resource applies to a section,
  leave the list empty.
- constraints.style_constraints should include tone, pov, audience, and style requirements when applicable.
- constraints.relation_constraints should describe dependencies on previous or later sections when useful.
- expected_blocks should be a concise block-level content plan for the draft tool.
- For a normal section, expected_blocks should usually contain 3 to 6 planned content blocks unless the section is explicitly very short.
- expected_blocks are planning labels for coverage and ordering, not visible headings that must appear in final text.
- Do not invent facts that conflict with writing context.

Outline (authoritative structure):
{outline_json}

Target outline blocks to author (one block in your output per node_id here):
{target_outline_blocks_json}

Writing context:
{context_json}

Execution results:
{execution_results_json}
'''
