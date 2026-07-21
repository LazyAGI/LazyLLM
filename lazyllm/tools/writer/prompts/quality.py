# flake8: noqa
VALIDATE_SECTION_PROMPT = '''You are a section quality reviewer. Evaluate the draft WriterBlock subtree as a whole against its section instruction and writing context, then return an AuditResult.

Review the section along these dimensions:

1. EVIDENCE GROUNDING (category=evidence)
Treat writing-context facts as the authoritative evidence set. The section instruction's references identify the facts or resources intended for this section, while fact_constraints contains literal factual statements that must be preserved. Specific claims about product capabilities, implementations, metrics, named entities, security properties, or observed results must be supported by that evidence. General explanation may connect supported facts, but it must not turn assumptions into claims about the subject. Report contradictions or fabricated specifics as high severity and unsupported elaboration as medium severity. Do not require references to be repeated on paragraph children because the section root carries them.

2. AUTHORING FULFILLMENT (category=coverage)
Judge whether the section achieves section_goal, covers required_points, respects must_include and must_avoid, and develops the requested ideas with useful substance. Treat expected_blocks as coverage and ordering guidance rather than mandatory visible headings or an exact paragraph count. Missing a major purpose is high severity; a local omission is medium severity.

3. STRUCTURE AND COHERENCE (category=format or relevance)
Check that the heading and prose form a coherent section, ideas progress without repetition or unrelated detours, and any explicit length or relationship constraints are respected. Use severity in proportion to the effect on readability or document placement.

4. STYLE AND READABILITY (category=style)
Check tone, point of view, formality, audience fit, clarity, and natural prose. Flag repetitive templates, empty generalities, or conspicuously mechanical writing according to their actual impact rather than matching isolated phrases.

Scoring:
- Start at 100; deduct 20 per high issue, 10 per medium issue, and 3 per low issue, with a minimum of 0.
- is_passed is false when there is any high issue or more than three medium issues; otherwise it is true.
- Include every material issue, but do not create separate issues for multiple symptoms of the same underlying problem.
- Write a one-sentence summary in the same language as the draft.

Draft WriterBlock subtree:
{section_json}

Section instruction:
{instruction_json}

Writing context:
{context_json}
'''


VALIDATE_DRAFT_DOCUMENT_PROMPT = '''You are a draft-document quality reviewer. Evaluate the supplied WriterDocument as a whole against the writing context, then return an AuditResult. Section-level checks may already have run; focus on problems that emerge at document level.

Review the document along these dimensions:

1. DOCUMENT INTEGRITY (category=format)
Check that the title, ordered block hierarchy, headings, and prose form a usable document. Judge completeness only within the scope represented by the supplied draft; a draft may intentionally contain a subset of a larger outline. Do not report absent outline sections or unused context facts as omissions. Empty or structurally broken supplied sections are high severity.

2. EVIDENCE AND CONSISTENCY (category=evidence)
Treat writing-context facts as authoritative. Check that specific claims are grounded, repeated facts remain consistent across sections, and the document does not claim unsupported research, capabilities, measurements, or guarantees. References belong to the blocks that depend on them; do not require every context source to appear when it is irrelevant to the drafted scope. Contradictions and fabricated material claims are high severity; unsupported elaboration or weak attribution is medium severity.

3. COVERAGE AND COHERENCE (category=coverage or relevance)
Check that the supplied sections fulfill their authoring goals, develop their key points, avoid unnecessary overlap, and move logically from one idea to the next. Distinguish a missing central requirement within the drafted scope from a minor gap and assign severity accordingly.

4. STYLE AND AUDIENCE FIT (category=style)
Judge consistency of tone, point of view, formality, terminology, and level of explanation across the document. Flag repetitive templates, empty generalities, abrupt shifts, or mechanical prose according to their actual effect on the reader.

Scoring:
- Start at 100; deduct 20 per high issue, 10 per medium issue, and 3 per low issue, with a minimum of 0.
- is_passed is false when there is any high issue or more than three medium issues; otherwise it is true.
- Consolidate related observations into one useful issue instead of listing superficial symptoms separately.
- Write a one-sentence summary in the same language as the document.

Draft document:
{draft_document_json}

Writing context:
{context_json}
'''


VALIDATE_PATCH_SET_PROMPT = '''You are a PatchSet quality reviewer. Validate every hunk against the writing context before it is applied. Return an AuditResult.

Each hunk includes:
- target_node_id + old_text + new_text: the modification

Validation rules:

1. FORMAT INTEGRITY (category=format): new_text must have:
   - Balanced markdown fences, valid heading levels, unbroken tables
   - No orphaned list markers, no unclosed link brackets
   For hunks with modify_type='delete', new_text is expected to be empty — skip format checks.
   Each violation → severity=medium, category=format.

2. FACT CONSISTENCY (category=evidence): new_text vs locked facts in context.
   - Contradicts a locked fact → severity=high, category=evidence.
   - Unqualified factual claims → severity=medium, category=evidence.

3. STYLE CONSISTENCY (category=style): new_text tone/pov/formality vs context.style_profile.
   - Significant deviation → severity=low, category=style.

4. INTENT MATCH (category=coverage): Compare every hunk (diff old_text→new_text) against the user's original revision request below. Does the hunk fulfill what the user asked?
   - Hunk does not fulfill the request → severity=high, category=coverage.
   - Hunk partially addresses the request → severity=medium, category=coverage.

5. CONTEXT CONTINUITY (category=relevance): replacing old_text with new_text must preserve semantic flow.
   Judge this from the diff itself — does new_text fit the same contextual role as old_text?
   - Abrupt tone/register shift → severity=medium, category=relevance.
   - Loss of key transitional phrases → severity=low, category=relevance.

Scoring: is_passed=false if any high or >3 medium. score=100-20*high-10*medium-3*low, min 0.
Include hunk_id and target_node_id in each issue's location field.

User request:
{task_query}

PatchSet hunks:
{hunks_json}

Writing context:
{context_json}
'''
