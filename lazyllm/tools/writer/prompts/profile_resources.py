# flake8: noqa
RESOURCE_PROFILE_PROMPT = '''\
Analyze the following material for a writing task.

Task: {query}
Task type: {task_type}
Constraints: {constraints}

Input context (do NOT copy these verbatim into output fields):
- Resource title: {title}
- Provided summary (if any): {summary}

Material content to analyze:
---
{content}
---

Analyze the content above and produce a JSON object:
- resource_role: "spec" (specification/requirement), "background" (background material), or "example" (reference/template)
- template_usage: "structure" (provides structure), "style" (provides style), "both", or "none"
- summary: YOUR one-sentence analysis of this resource's key contribution to the task (do not copy the provided summary)
- key_facts: 3-5 hard facts the writer must respect (empty list if none)
- style: {{"tone": "formal" / "informal" / "neutral" or null, "formality": "technical" / "business" / "casual" / "academic" or null, "audience": "professional" / "general" / "academic" or null, "notes": [YOUR style observations (tone, formality, key phrases)]}} or null
- confidence: 0.0-1.0, how confident you are in this analysis
- extracted_constraints: key-value pairs of constraints from the material (empty object if none), e.g. {{"word_limit": "10000", "format": "markdown"}}
- extracted_outline: a WriterDocument with stage="outline" that represents an outline explicitly found in the material, or null if the material does not provide one. Give every block a stable node_id, type="heading", stage="outline", section title in content, heading level in numbering.level, and nested sections in children. Do not invent an outline merely because the material contains several facts.
'''
