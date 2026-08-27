# flake8: noqa
GENERATE_OUTLINE_MARKDOWN_PROMPT = '''Generate a writing outline in Markdown from the given task and context.

Requirements:
- Output Markdown only. Do not wrap the response in an outer code fence.
- Do not output reasoning, analysis, review notes, or <think> tags.
- Start with exactly one H1 document title.
- Add at least one H2 section directly under the H1 title.
- Every H2 section title must be unique.
- Use H3-H6 only for optional subsection planning under an H2 section.
- Write titles and section titles without visible numbering; the system renders numbers.
- Treat the outline as the exact structural skeleton of the final deliverable: every H2
  will become a visible section in the drafted document.
- Do not create meta-planning H2 sections such as background and setting, character
  profiles, themes and symbols, writing style, chapter-plan commentary, or writing plan,
  unless the user explicitly requested those sections in the final deliverable.
- For fiction and other narrative writing, use H2 only for actual chapter titles and use
  H3-H6 for scenes, plot beats, or events within that chapter. Distribute character,
  setting, theme, and style requirements into the relevant chapters instead of exposing
  them as standalone planning sections.
- For non-fiction, reports, and articles, use H2 only for sections that should appear in
  the final document.
- Do not create image-annotation headings such as "图片：..." or "Image: ..." at any level.
  Image needs are planned by the visual plan step; the outline must stay pure text structure.
- Do not emit Markdown image syntax, HTML image tags, image paths, or image placeholders.
  The visual plan and media resolver exclusively own image selection and placement.
- Keep the outline concise but concrete enough to guide drafting.
- Treat task.constraints.target_chars and task.constraints.max_chars as limits for the
  entire final document, not for each section.
- When max_chars is at most 1200 and the user did not explicitly request multiple
  chapters or sections, prefer one H2 section and merge the essential material into it.
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
- Generate at least 3 top-level blocks unless the task asks for a short document or
  explicitly asks for fewer.
- Treat task.constraints.target_chars and task.constraints.max_chars as limits for the
  entire final document, not for each section.
- When max_chars is at most 1200 and the user did not explicitly request multiple
  chapters or sections, generate one top-level block and merge the essential material into it.
- Each top-level block is a section. Use type="heading" for section blocks.
- Treat the outline as the exact structural skeleton of the final deliverable: every
  top-level heading block will become a visible section in the drafted document.
- Do not create meta-planning top-level sections such as background and setting,
  character profiles, themes and symbols, writing style, chapter-plan commentary, or
  writing plan, unless the user explicitly requested those sections in the final
  deliverable.
- For fiction and other narrative writing, use top-level heading blocks only for actual
  chapter titles. Put scenes, plot beats, and events in child blocks, and distribute
  character, setting, theme, and style requirements into the relevant chapters.
- For non-fiction, reports, and articles, use top-level heading blocks only for sections
  that should appear in the final document.
- Do not create heading blocks named "图片：..." or "Image: ..." for image planning; visual
  needs are handled by the visual plan step, not by outline headings.
- All user-visible outline text MUST use the same document tree contract as draft and final content:
  put section titles in heading block.content, and put section descriptions and key points in
  paragraph or list_item blocks under block.children.
- Fill node_id for every block. Use stable ids such as section-1, section-2, section-1-1.
- Use block.numbering.level for the heading level: 1 for top-level sections, incrementing for children.
  Put child sections under block.children as heading blocks alongside any visible description blocks.
- Write titles and section titles without visible numbering; the system renders numbers.
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


GENERATE_REWRITE_OUTLINE_PROMPT = '''Generate a new writing outline for a complete rewrite of the source document.

Requirements:
- Return a WriterDocument object with stage="outline".
- Set document_id to the exact document_id below.
- Treat the user's requested end state as authoritative, including chapter count, section order,
  compression, expansion, merging, splitting, and renamed sections.
- Use top-level heading blocks only for sections that will appear in the rewritten document.
- Preserve source facts, plot points, terminology, characters, and style unless the task asks to change them.
- Include concise source-grounded descriptions and key points under each heading so drafting can recreate
  the requested document without reading the source again.
- Do not copy the source structure when it conflicts with the requested end state.
- Do not add planning commentary or other sections that should not appear in the final document.
- Put section titles in heading block.content and descriptions or key points in child paragraph or list_item blocks.
- Fill node_id for every block with stable ids such as rewrite-section-1 and rewrite-section-1-1.
- Use block.numbering.level for heading levels.
- Omit spans, references, provider_binding and provider_payload; the system manages them.
- Preserve the source title unless the task explicitly requests a new title.

Writing task:
{task_json}

document_id: {document_id}

Writing context:
{context_json}

Complete source document:
{source_document_json}
'''


GENERATE_REWRITE_SECTION_INSTRUCTIONS_PROMPT = '''Generate the section-level writing instructions for a complete rewrite.

Requirements:
- Return a SectionInstructionList object. Do not return an outline or draft prose.
- The user's requested end state is authoritative. You may preserve, rename, merge, split, reorder,
  expand, or remove source sections as required by the task.
- meta.document_title must be the title of the rewritten document. Preserve the source title unless
  the task explicitly requests a new title.
- Produce one SectionInstruction for every top-level section in the rewritten document, in final order.
- Treat task.constraints.target_chars and task.constraints.max_chars as limits for the
  entire final document, not for each section.
- When max_chars is at most 1200 and the user did not explicitly request multiple
  chapters or sections, normally produce one SectionInstruction and merge source sections.
- When multiple sections are necessary, set each SectionInstruction.meta.target_chars to
  a positive relative length budget based on its narrative or explanatory importance.
  Do not divide the budget evenly unless the sections are genuinely equally important.
- instruction_id must be stable and unique. section_title and section_goal must be non-empty.
- required_points must retain the source facts, plot points, terminology, and details needed by drafting.
- expected_blocks must be a concise content plan, not visible headings.
- Sections may be drafted independently and in parallel. Preserve one document-global point
  of view, tense, narrative voice, character identity, and naming policy by copying the same
  continuity constraints into every relevant section. Do not invent a proper name for an
  unnamed protagonist, and do not leave mutually exclusive POV choices unresolved.
- Keep required_points and expected_blocks selective enough to fit the final length budget.
- references must contain only exact source_ref objects copied from source_sections. Use them to identify
  which source sections inform each rewritten section; do not invent reference fields or values.
- For representation="ir", content_ref must contain only a unique node_id such as rewrite-section-1.
- For representation="markdown", content_ref must contain only heading_path=[document title, section title]
  and occurrence=1.
- Respect the writing context and do not invent facts that conflict with it.
- Do not add source excerpts, paths, URLs, provider bindings, media assets, or planning commentary.

Representation:
{representation}

Writing task:
{task_json}

Source title:
{source_title}

Source sections:
{source_sections_json}

Writing context:
{context_json}
'''


GENERATE_VISUAL_PLAN_PROMPT = '''Generate a visual plan for this Writer IR outline.

Requirements:
- Return a VisualPlan object.
- Create a visual only when the user explicitly requires it or it materially improves the section.
- Each content_ref must contain only node_id for one top-level heading in the outline.
- Use the most appropriate visual_type. preferred_strategy is optional; if omitted, the system
  derives it from visual_type. Do not use image_generation for chart or table.
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


GENERATE_VISUAL_PLAN_MARKDOWN_PROMPT = '''Generate a visual plan for this Markdown outline.

Requirements:
- Return a VisualPlan object.
- Create a visual only when the user explicitly requires it or it materially improves the section.
- Each content_ref must target exactly one H2 section from target_sections: use its exact
  heading_path and occurrence. Do not use node_id or document_root.
- Use the most appropriate visual_type. preferred_strategy is optional; if omitted, the system
  derives it from visual_type. Do not use image_generation for chart or table.
- purpose must state what the visual communicates for its section.
- Set required=true only when the user explicitly requires the visual.
- Do not change the outline. Do not generate asset IDs, paths, URLs, captions, placeholders, or upload details.

Writing task:
{task_json}

Writing context:
{context_json}

Outline:
{outline_json}

Target H2 sections:
{target_sections_json}
'''


GENERATE_SHORT_VISUAL_PLAN_PROMPT = '''Generate a visual plan for one flat short article.

Requirements:
- Return a VisualPlan object.
- An explicit user requirement that the article contain any image, illustration, chart, table, diagram,
  or other visual has highest priority. In that case, instructions MUST NOT be empty; never return an empty
  object or an empty instructions list. Create enough instructions to satisfy the requested visual content
  and count, and mark every explicitly required instruction with required=true.
- Treat restrictions on the visual source separately from whether a visual is required. For example,
  "do not use uploaded images" still requires a non-empty plan when the user asks for a generated image.
  When the user explicitly asks the system to generate an image, set preferred_strategy=image_generation.
- Return an empty instructions list only when the user forbids visuals, or when the user has not explicitly
  required any visual and no visual materially improves the article.
- Use the short writing plan to decide what each visual communicates and where it naturally belongs.
- Each content_ref must contain only document_root=true. Do not use node_id, heading_path, or placeholder_id.
- visual_type must be image, chart, table, or diagram.
- preferred_strategy must be null or exactly one of web_search, kb_search, image_generation, or code_render.
  Leave it null when an uploaded or otherwise available input image should be reused.
- Put natural-language placement guidance only in meta.placement_hint. Never put placement guidance in
  preferred_strategy.
- purpose must state what the visual communicates for the complete article.
- Set required=true only when the user explicitly requires that visual.
- Do not impose a visual count limit; decide from the requested content, length, and explicit requirements.
- Do not generate asset IDs, paths, URLs, captions, placeholders, or upload details.
- Keep the plan selective enough to satisfy writing_task.constraints.target_chars and max_chars.

Writing task:
{task_json}

Short writing plan:
{short_writing_plan_json}

Writing context:
{context_json}
'''


GENERATE_SECTION_INSTRUCTIONS_PROMPT = '''Generate section-level writing instructions from the outline and writing context.

Requirements:
- Return a SectionInstructionList object.
- Generate exactly one SectionInstruction for every item listed in target_outline_blocks.
- Treat writing_task.constraints.target_chars and writing_task.constraints.max_chars as
  limits for the entire final document, not for each section.
- Set each SectionInstruction.meta.target_chars to a positive relative length budget based
  on the section's importance. Do not divide evenly unless the sections are genuinely equal.
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
- Sections may be drafted independently and in parallel. Resolve document-global narrative
  choices during planning: choose one point of view, tense, narrative voice, and protagonist
  naming policy, then copy those exact continuity constraints into every relevant section.
  If the task, context, and outline do not name a protagonist, do not invent a proper name.
  If they allow multiple points of view without choosing one, select one and use it throughout.
- relation_constraints should describe ordinary continuity with neighboring sections;
  the drafting model expresses that continuity in prose.
- Use the visual plan to shape section goals, ordering, and transitions when its content_ref targets
  the same section. Keep acquisition details out of SectionInstruction; the system binds every
  planned visual need to its created-image cross-reference target.
- expected_blocks should be a concise block-level content plan for the draft tool.
- For a normal-length section, expected_blocks should usually contain 3 to 6 planned content
  blocks. Use fewer and merge coverage cues when the total document budget is short.
- expected_blocks are planning labels for coverage and ordering, not visible headings that must appear in final text.
- Plan a section cross-reference when this section's text will point readers to
   another section for a specific definition, result, or method - for example a
   conclusion citing the experiments it summarizes. This is expected when the writing
   task asks for cross-references; in that case include at least one section reference.
   Represent every planned reference in SectionInstruction.meta.cross_references as
   an object with target_ref, kind, required, and guidance.
   Background continuity readers are assumed to know
   (such as narrative chapters building on earlier events) belongs in relation_constraints.
   For Writer IR, target_ref is {{"node_id": "..."}} copied from the outline.
   For Markdown, target_ref is the target section's {{"heading_path": [...], "occurrence": 1}}.
   Its guidance names the information the reader needs from that target.
- Visual plan needs own created images. Do not add must_create image objects; the system adds them.
   To reference a planned image in this or another section, use
   target_ref: {{"node_id": "<that visual need_id>"}} with must_create=false and kind="image".
- Use guidance to describe what natural wording should carry each reference link.
- Do not invent facts that conflict with writing context.

Writing task:
{task_json}

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


GENERATE_SHORT_WRITING_PLAN_PROMPT = '''Generate one whole-document writing plan for a short article.

Requirements:
- Return one ShortWritingPlan object. Do not return an outline, chapter plan, or draft prose.
- The plan covers the complete article in one writing pass.
- content_ref must contain only document_root=true.
- section_title is the article title. Prefer writing_task.target_document.title when provided.
- section_goal states the writing objective for the complete article.
- core_viewpoint states the central claim or message that the complete article must communicate.
- required_points contains the essential content that must appear in the article.
- fact_constraints contains only factual statements actually present in the writing context.
- references identifies relevant context facts or resources and must not invent identifiers.
- style_constraints includes the requested genre, audience, tone, point of view, and style when applicable.
- Keep visual_needs empty. Visuals are planned separately with a strongly typed VisualPlan.
- expected_blocks is a concise content-order plan for continuous prose. Its entries are internal coverage
  cues, not visible headings, separate generations, or minimum paragraph counts.
- Do not create relation_constraints, cross-references, section links, chapters, or subheadings.
- Keep the plan selective enough to satisfy task.constraints.target_chars and max_chars.
- Set meta.representation to the requested writing_task.output.representation.
- Do not invent facts that conflict with the writing context.

Writing task:
{task_json}

Writing context:
{context_json}

Execution results:
{execution_results_json}
'''
