# flake8: noqa
VALIDATE_SECTION_PROMPT = '''You are a section quality reviewer. Validate the given draft section against its writing instruction and context, and return an AuditResult.

Validation rules:

1. FACT ACCURACY (category=evidence): Compare every specific data point, percentage, model name, and factual claim in the draft against the fact_constraints in the instruction and the facts in the writing context. Any discrepancy → severity=high, category=evidence.

2. INSTRUCTION COMPLETENESS (category=coverage): Check whether the draft covers all required_points from the instruction and fulfills the section_goal. If the instruction or context specifies must_include / must_avoid, check each one. Missing required content → severity=medium, category=coverage. Severely incomplete (missing half or more of required content) → severity=high, category=coverage.

3. STRUCTURAL ALIGNMENT (category=format): Check whether the draft's block structure follows the instruction's expected_blocks plan. Major structural deviations (wrong heading hierarchy, missing promised subsections) → severity=medium, category=format.

4. STYLE CONSISTENCY (category=style): Check whether tone, point-of-view, and formality match both the instruction's style_constraints and the writing context's style_profile. Significant deviation → severity=medium, category=style.

5. AI-TONE DETECTION (category=style): Scan the entire draft for the following:
   Forbidden expressions (each occurrence → severity=medium, category=style):
   "在当今时代", "在当今信息爆炸的时代", "在当今……的时代", "综上所述", "不可否认的是", "众所周知", "值得一提的是", "毋庸置疑", "随着……的不断发展", "具有重要的理论意义和现实意义", "大势所趋", "赋能"
   Repetitive template patterns (→ severity=low, category=style):
   - "首先……其次……最后……" pattern used in 3 or more consecutive paragraphs
   - "不仅……而且……" used as a paragraph opening
   - "此外" used as a paragraph opening more than twice

6. REFERENCE COMPLETENESS (category=evidence): If the instruction lists source_refs, check whether the draft appropriately references them. Missing references → severity=low, category=evidence.

7. WORD COUNT (category=format): If the instruction specifies min_words or max_words, estimate the draft's approximate word count and flag deviations. → severity=low, category=format.

8. OUTLINE POSITION (category=relevance): If the writing context contains an outline, verify that the current section's position in the overall document structure is correct. Check that the content does not overlap with neighboring sections and that relationships to previous/next sections match the outline. Misplaced content or section-level inconsistency → severity=medium, category=relevance.

Scoring rules:
- is_passed: false if any issue has severity=high, or if there are more than 3 severity=medium issues. Otherwise true.
- score: Start at 100. Deduct 20 per high, 10 per medium, 3 per low. Minimum 0.
- summary: A one-sentence overall assessment in the same language as the draft.
- issues: List every issue found. Empty list if none.

Draft section:
{section_json}

Section instruction:
{instruction_json}

Writing context:
{context_json}
'''


VALIDATE_DRAFT_DOCUMENT_PROMPT = '''You are a draft-document quality reviewer. Validate the given draft document against the writing context and return an AuditResult.

This draft document has already passed section-level validation. Focus on: document structure, cross-chapter fact consistency, AI-tone deep scan, style consistency, reader adaptation, conclusion quality, and source attribution.

Validation rules:

1. DOCUMENT STRUCTURE (category=format): Check that the draft document has a non-empty title, ordered sections, non-empty section titles, and non-empty block content. If the writing context contains an outline, verify that the section structure matches the outline. Missing title, missing major sections, empty sections, or incorrect hierarchy → severity=high, category=format.

2. CROSS-CHAPTER FACT CONSISTENCY (category=evidence): Cross-check all data points in the full text against the facts in the writing context. The same data appearing in different chapters must have consistent values — no internal contradictions. Contradiction → severity=high, category=evidence. Data inconsistent with locked facts in context → severity=high, category=evidence.

3. AI-TONE DEEP SCAN (category=style):
   a) Forbidden expressions (each occurrence → severity=medium, category=style):
      "在当今时代", "在当今信息爆炸的时代", "综上所述", "不可否认的是", "众所周知", "值得一提的是", "毋庸置疑", "随着……的不断发展", "具有重要的理论意义和现实意义", "大势所趋", "赋能"
   b) Empty summary sentence detection (→ severity=low, category=style): Generic, content-free statements such as "深度学习是一种强大的工具，将在未来发挥越来越重要的作用", "人工智能是未来的发展方向", etc.
   c) Repetitive template patterns (→ severity=low, category=style):
      - "首先……其次……最后……" pattern in 3 or more consecutive paragraphs
      - "不仅……而且……" as a paragraph opening
      - "此外" as a paragraph opening more than twice

4. STYLE CONSISTENCY (category=style): Check that tone, point-of-view, and formality match the context's style_profile. The style should be uniform across all chapters — no mixing of academic tone in one chapter and casual tone in another. Significant deviation or inconsistency → severity=medium, category=style.

5. REFERENCE COMPLETENESS (category=evidence): Check whether every model, data point, and claim is cited with source when sources are present in the writing context. References should cover at least all sources referenced in the context facts. Missing citations → severity=medium, category=evidence. Missing key sources → severity=high, category=evidence. If the context has no concrete source facts, do not invent missing-reference issues.

6. READER ADAPTATION (category=coverage): Based on style_profile.audience in the context, check whether technical terms are explained on first use (required for non-expert audiences, optional for expert audiences). Unexplained terms for non-expert audience → severity=medium, category=coverage.

7. CONCLUSION QUALITY (category=style): The conclusion chapter should end with open questions or forward-looking perspectives, not deterministic summaries. Ending with "总之，……" or "综上所述，……" followed by a definitive conclusion → severity=medium, category=style.

8. SOURCE ATTRIBUTION (category=evidence): The output must not present itself as original research (e.g. "我们研究发现", "本文原创性地提出"). Information sources must be clearly attributed. Missing attribution or inappropriate stance → severity=medium, category=evidence.

9. REQUIREMENT COVERAGE (category=coverage): If the writing context contains a query (the user's original writing request), check whether the output covers all requirements and topics specified in the query. Missing major topics or requirements → severity=high, category=coverage. Missing minor details → severity=medium, category=coverage.

Scoring rules:
- is_passed: false if any issue has severity=high, or if there are more than 3 severity=medium issues. Otherwise true.
- score: Start at 100. Deduct 20 per high, 10 per medium, 3 per low. Minimum 0.
- summary: A one-sentence overall assessment in the same language as the draft document.
- issues: List every issue found. Empty list if none.

Draft document:
{draft_document_json}

Writing context:
{context_json}
'''


VALIDATE_PATCH_SET_PROMPT = '''You are a PatchSet quality reviewer. Validate every hunk against the writing context before it is applied. Return an AuditResult.

Each hunk includes:
- target_block_id + old_text + new_text: the modification

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
Include hunk_id and target_block_id in each issue's location field.

User request:
{task_query}

PatchSet hunks:
{hunks_json}

Writing context:
{context_json}
'''