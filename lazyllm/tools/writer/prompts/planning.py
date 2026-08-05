# flake8: noqa
GENERATE_OUTLINE_MARKDOWN_PROMPT = '''Generate a writing outline in Markdown from the given task and context.

Requirements:
- Output Markdown only. Do not wrap the response in an outer code fence.
- Do not output reasoning, analysis, review notes, or <think> tags.
- Start with exactly one H1 document title.
- Add at least one H2 section directly under the H1 title.
- Every H2 section title must be unique.
- Use H3-H6 only for optional subsection planning under an H2 section.
- Keep the outline concise but concrete enough to guide drafting.
- Use resource profiles and execution results as constraints, not as text to copy blindly.
- Do not invent facts that conflict with the writing context.

Writing task:
{task_json}

Writing context:
{context_json}

Resource profiles:
{resource_profiles_json}

Execution results:
{execution_results_json}
'''


GENERATE_OUTLINE_PROMPT = '''Generate a writing outline from the given writing task and context.

Requirements:
- Return a WriterDocument object with stage="outline".
- Set document_id to the exact document_id below.
- Generate at least 3 top-level blocks unless the task explicitly asks for fewer.
- Each top-level block is a section. Use type="heading" for section blocks.
- All user-visible outline text MUST use the same document tree contract as draft and final content:
  put section titles in heading block.content, and put section descriptions and key points in
  paragraph or list_item blocks under block.children.
- Fill node_id for every block. Use stable ids such as section-1, section-2, section-1-1.
- Use block.numbering.level for the heading level: 1 for top-level sections, incrementing for children.
  Put child sections under block.children as heading blocks alongside any visible description blocks.
- block.references holds identifiers for facts or resources the section depends on.
- Each element of block.references is an object with at least an "id" field. The id must match a
  DocumentFact.fact_id or ResourceProfile.resource_id present in the input.
- Prefer the target document title or task intent as document.title.
- Use the writing context and resource profiles as constraints, not as content to copy blindly.
- Omit spans, provider_binding and provider_payload; the system manages them.

Writing task:
{task_json}

document_id: {document_id}

Writing context:
{context_json}

Resource profiles:
{resource_profiles_json}

Execution results:
{execution_results_json}
'''


GENERATE_VISUAL_PLAN_PROMPT = '''Generate a visual plan for this Writer IR outline.

Requirements:
- Return a VisualPlan object.
- Create a visual only when the user explicitly requires it or it materially improves the section.
- Each content_ref must contain only node_id for one top-level heading in the outline.
- Use visual_type image or diagram and preferred_strategy image_generation.
- purpose must state what the visual communicates for its section.
- Set required=true only when the user explicitly requires the visual.
- Do not change the outline. Do not generate asset IDs, paths, URLs, captions, placeholders, or upload details.

Writing task:
{task_json}

Writing context:
{context_json}

Outline:
{outline_json}
'''


GENERATE_SECTION_INSTRUCTIONS_PROMPT = '''Generate section-level writing instructions from the outline and writing context.

Requirements:
- Return a SectionInstructionList object.
- Generate exactly one SectionInstruction for every item listed in target_outline_blocks.
- Copy each target's content_ref exactly. For Writer IR this contains node_id; for Markdown it
  contains heading_path and occurrence. Do not mix locator types.
- section_title MUST equal the corresponding target's section_title.
- instruction_id should be stable, such as instruction-section-1 or instruction-ch01.
- section_goal should be concrete and actionable.
- required_points should contain the key content that must appear in the section.
- fact_constraints should preserve the literal text of locked facts and important context facts
  relevant to this section. It must not contain fact IDs or resource IDs.
- fact_constraints MUST only contain factual statements actually present in the writing context.
- references are owned by the authoritative outline. Omit references; the system normalizes them.
- style_constraints should include tone, pov, audience, and style requirements when applicable.
- relation_constraints should describe dependencies on previous or later sections when useful.
- Use the visual plan to shape section goals, ordering, and transitions when its content_ref targets
  the same section. Do not copy visual needs into SectionInstruction or generate asset IDs, paths,
  placeholders, captions, or acquisition instructions.
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

Visual plan:
{visual_plan_json}
'''
