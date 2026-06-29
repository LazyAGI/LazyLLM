GENERATE_OUTLINE_PROMPT = """Generate a writing outline from the given writing task and context.

Requirements:
- Return a WritingOutline object.
- Generate at least 3 top-level sections unless the task explicitly asks for fewer.
- Each top-level section must have a clear title, a concrete writing instruction, and useful constraints.
- Fill node_id for every node. Use stable ids such as section-1, section-2, section-1-1.
- Set level=1 for top-level sections and increment levels for children.
- The constraints field must only use fields defined by OutlineNodeConstraints:
  section_goal, required_points, fact_constraints, style_constraints, relation_constraints, source_refs,
  min_words, max_words, pov, tone, must_include, must_avoid.
- Do not invent other constraints fields.
- Constraints fields are optional. Fill only the fields that are useful for the section.
- Prefer the target document title or task intent as the outline title.
- Use the writing context and resource profiles as constraints, not as content to copy blindly.

Writing task:
{task_json}

Writing context:
{context_json}

Resource profiles:
{resource_profiles_json}

Execution results:
{execution_results_json}
"""


GENERATE_SECTION_INSTRUCTIONS_PROMPT = """Generate section-level writing instructions from the outline and writing context.

Requirements:
- Return a SectionInstructionList object.
- Generate one SectionInstruction for every top-level outline node listed in target_outline_nodes.
- Each instruction must use the outline node's node_id as outline_node_id.
- instruction_id should be stable, such as instruction-section-1 or instruction-ch01.
- section_title must match the outline node title.
- section_goal should be concrete and actionable.
- required_points should contain the key content that must appear in the section.
- fact_constraints should preserve locked facts and important context facts relevant to this section.
- style_constraints should include tone, pov, audience, and style requirements when applicable.
- relation_constraints should describe dependencies on previous or later sections when useful.
- expected_blocks should be a concise block-level plan for the draft tool.
- Do not invent facts that conflict with writing context.

Outline:
{outline_json}

Target outline nodes:
{target_outline_nodes_json}

Writing context:
{context_json}

Execution results:
{execution_results_json}
"""
