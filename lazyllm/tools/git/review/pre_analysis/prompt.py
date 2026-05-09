# Copyright (c) 2026 LazyAGI. All rights reserved.
from ..utils import JSON_OUTPUT_INSTRUCTION

_FRAMEWORK_GOTCHAS_NOTICE = '''\
### Framework-Specific Gotchas (auto-generated)
When reviewing this codebase, be aware of the following common pitfalls:
- Names that shadow Python builtins (e.g. `locals`, `globals`, `type`, `id`) may be \
framework-defined objects with dict-like or special behavior — do NOT assume they are \
Python built-in functions. Verify by checking imports or surrounding code before reporting.
- Module-level code guarded by environment variables (e.g. `if os.environ.get(...)`) or \
conditional imports is intentional — do NOT report it as "will always fail" without \
confirming the guard condition.
- Abstract base class method signature changes are only "breaking" if at least one concrete \
subclass has NOT been updated. Check the diff for subclass updates before reporting.
- `max(n, 1)` and similar floor-clamp patterns are defensive programming, not bugs.
- When code is MOVED between files (deleted in one, added in another with similar content), \
this is a refactoring — do NOT report it as "duplicate code" or "missing reuse".
- If a function accepts parameters that are only used in some call paths, this may be \
intentional interface design for extensibility — verify actual callers before reporting \
"unused parameter".
- A class with many parameters in `__init__` may be a configuration/builder pattern, not a \
god class — check if the parameters map to distinct concerns before reporting.
- Do NOT suggest introducing a new abstraction layer (base class, protocol, registry) unless \
at least 3 concrete implementations already exist or are planned in this PR. Premature \
abstraction is worse than duplication.
'''

_ARCH_OUTLINE_PROMPT = '''\
You are a senior software architect. Based on the project snapshot below, generate an outline \
for an architecture document with {max_sections} sections.

For each section output a JSON object with:
- "title": section name (e.g. "Module Responsibilities")
- "focus": one sentence describing what to cover
- "search_hints": list of 2-3 regex patterns for search_in_files to find relevant code \
(infer patterns from the actual project code visible in the snapshot — do NOT hardcode project-specific names)

The FIRST section MUST be:
{{"title": "Module Hierarchy",
 "focus": "Module layering: core/base modules, mid-layer, top-level business modules, \
and forbidden dependency directions",
 "search_hints": ["<generate 2-3 regex patterns based on the project import style visible in the snapshot>"]}}

The SECOND section MUST be:
{{"title": "Environment & Dependencies",
 "focus": "Python/compiler version requirements, key dependency packages and version constraints; \
which dependencies are optional/extras vs hard requirements, and the naming convention for optional groups",
 "search_hints": ["python_requires", "install_requires", "optional.dependencies", "extras_require"]}}

The THIRD section MUST be:
{{"title": "Module Ownership Rules",
 "focus": "Per-module/sub-package responsibility boundaries: which code must live in which module \
(e.g. config definitions inside feature modules, not top-level), and allowed cross-module reference directions",
 "search_hints": ["<generate 2-3 regex patterns for config/ownership patterns visible in the snapshot>"]}}

{{gotchas_instruction}}

The THIRD-TO-LAST section MUST be:
{{"title": "Typical Usage Patterns",
 "focus": "Typical calling conventions for public APIs, common multi-step composition patterns, \
known usage constraints and ordering requirements inferred from tests, examples, and documentation",
 "search_hints": ["<generate 2-3 regex patterns for test files, example scripts, or README usage blocks>"]}}

The SECOND-TO-LAST section MUST be:
{{"title": "Concurrency & Multi-User Conventions",
 "focus": "Thread-safety guarantees, per-request vs shared objects, lock usage conventions, \
tenant/user isolation mechanisms, async safety, ContextVar usage",
 "search_hints": ["threading\\.Lock|asyncio\\.Lock|threading\\.local", "session|user_id|tenant_id", \
"ContextVar|contextvars"]}}

The LAST section MUST be:
{{"title": "Key Utilities & Usage Notes",
 "focus": "Key helper functions, data structures, their typical usage and caveats. \
Format each entry as: FunctionName(args): one-line description",
 "search_hints": ["<generate 2-3 regex patterns for utility classes/functions visible in the snapshot>"]}}

''' + JSON_OUTPUT_INSTRUCTION + '''

<snapshot>
{snapshot}
</snapshot>
'''

_ARCH_GOTCHAS_INSTRUCTION = '''\
The SECOND-TO-LAST section MUST be:
{{"title": "Non-Obvious Behaviors & Gotchas", \
"focus": "Initial values, global state, thread-safety conventions, registry/factory behavior, \
lazy-loading / deferred-init patterns, private-attribute cross-module access conventions, \
easily misunderstood design decisions", \
"search_hints": ["<generate 2-3 regex patterns for non-obvious state management, lazy-init, \
or deferred-import patterns visible in the snapshot>"]}}'''

_ARCH_HAS_AGENT_INSTRUCTION = '''\
NOTE: This project already has an AGENTS.md (or equivalent) covering conventions and gotchas. \
Still include a "Non-Obvious Behaviors & Gotchas" section as the SECOND-TO-LAST section, but \
focus ONLY on framework mechanisms NOT already covered by AGENTS.md (e.g. lazy-loading wrappers, \
registry metaclasses, implicit state initialization). Do NOT duplicate content from AGENTS.md.'''

_ARCH_SECTION_PROMPT = '''\
You are analyzing a Python project. Fill in ONE section of the architecture document.

## Current Section
Title: {section_title}
Focus: {section_focus}

## Directory Overview
{dir_tree}

## Already Documented (brief)
{prev_summaries}

## Relevant Code Snippets
{code_snippets}

Based on the code snippets above, write the section content.
- Max 500 words for this section
- Plain text, no markdown headers
- Focus only on: {section_focus}
- Include key class names, function signatures, and usage patterns where relevant

Output ONLY the section content text.
'''

_ARCH_BATCH_SECTIONS_PROMPT = '''\
You are analyzing a Python project. Fill in EVERY architecture section listed below in one response.

## Directory Overview
{dir_tree}

## Already Documented (brief)
{prev_summaries}

{sections_block}

For each section: plain text (max 500 words), no markdown headers, focused on that section focus.

''' + JSON_OUTPUT_INSTRUCTION + '''
Each item: "title" (exact section title) and "content" (section body).
'''

_ARCH_STATIC_PROMPT_TMPL = '''\
You are a senior software architect. Analyze the following project snapshot and produce a concise \
architecture document covering: module responsibilities, class hierarchies, design patterns, \
public API conventions, key utilities usage, and notable constraints.
Be concise (max 800 words). Output plain text, no markdown headers.

<snapshot>
{snapshot}
</snapshot>
'''

_SYMBOL_SUMMARY_PROMPT_TMPL = '''\
You are a code analyst. Given the following class or function definition, write a ONE-sentence summary \
(max 100 words) covering: purpose, key constraints, and any important usage notes.

File: {file_path}
Symbol: {symbol_name}

```
{code_snippet}
```

Output ONLY the one-sentence summary.
'''

_PUBLIC_API_FILES_PROMPT_TMPL = '''\
You are analyzing a software project. Based on the directory tree below, identify files that \
serve as shared/public utility or base-class libraries — files whose functions and classes are \
intended to be reused by other modules.

For each such file output a JSON object with:
- "file": path relative to repo root (e.g. "common/utils.py")
- "scope": the path prefix under which this file is relevant.
  Use "global" if it is a top-level shared library usable by the entire project.
  Otherwise use the directory path of the module it belongs to (e.g. "tools/agent").
- "reason": one short phrase explaining why it is public (e.g. "top-level utils", "agent helpers")

Rules:
- Only include files that are clearly utility/helper/base libraries, NOT application logic files.
- Do NOT include test files, example files, migration scripts, or generated files.
- Limit to at most 30 files.
- ''' + JSON_OUTPUT_INSTRUCTION + '''

<directory_tree>
{dir_tree}
</directory_tree>
'''

_EXTRACT_RULES_PROMPT = '''\
You are a code review expert. The following are review comments from a single pull request.
Comments are prefixed with [MAINTAINER] (repo owner/member/collaborator) or [CONTRIBUTOR].
Extract REPOSITORY-LEVEL norms and review standards — not PR-specific bug fixes.

For each rule, ask yourself:
- Would a maintainer give this same feedback on a DIFFERENT PR touching different code?
- Is this about HOW things should be done in this repo, not WHAT was wrong in this specific PR?
- Does this reflect a technology choice, design pattern preference, or coding convention?

Prioritize [MAINTAINER] comments — they represent the repo's official review standards.

For each rule found, output a JSON object with these fields:
- "rule_id": string like "PR{pr_num}_ERR001", "PR{pr_num}_STY002" \
(ERR=error/exception, STY=style, DSN=design, PERF=performance, SEC=security, \
XFILE=cross-file consistency, PREF=development preference, CONV=repo convention)
- "title": short title (max 8 words)
- "severity": "P0" | "P1" | "P2"  (P0=must fix, P1=should fix, P2=nice to have)
- "scope": "repo_wide" | "pr_specific"
- "rationale": one sentence explaining WHY this is a repo-level norm (or "" if pr_specific)
- "detect": list of strings describing how to detect this issue (patterns, keywords, conditions)
- "bad_example": short code snippet showing the bad pattern (or "" if not applicable)
- "good_example": short code snippet showing the correct pattern (or "" if not applicable)
- "fix": one-sentence fix suggestion

Examples of repo-level norms (scope=repo_wide):
- "Prefer X over Y for this type of task" (PREF)
- "In scenario A, must use module B" (CONV)
- "No shortcut implementations allowed" (DSN)
- "Must use explicit dependency injection" (DSN)
- "Must register config via config.add()" (CONV)
- "Prefer readability over performance" (PREF)

Pay special attention to cross-file consistency issues (use rule_id prefix "PR{pr_num}_XFILE"):
- Interface changed but callers not updated
- Symmetric methods (encode/decode, open/close) only one side updated
- Registry/factory pattern: new entry added but docs/tests not updated
- Abstract method added to base class but not implemented in subclasses

Discard and do NOT extract:
- One-off bug fixes that won't recur in other PRs
- Typo corrections
- PR-specific logic errors tied to a single function

''' + JSON_OUTPUT_INSTRUCTION + '''
If no clear rules can be extracted: use <<<JSON_START>>>\n[]\n<<<JSON_END>>>

<review_comments>
{{comments_text}}
</review_comments>
'''

_MERGE_RULES_PROMPT = '''\
You are a code review expert. Below are rule cards extracted from multiple pull requests.
Your task:
1. Merge duplicate or highly similar rules into one (keep the most informative example and detect patterns).
2. Remove rules that are too vague, trivial, or project-unspecific.
3. Re-assign clean rule_ids using standard prefixes: ERR, STY, DSN, PERF, SEC, XFILE, PREF, CONV \
(e.g. ERR001, STY002, PREF001, CONV001).
4. Assign a "category" to each rule from: error_handling, style, design, performance, security, \
cross_file, concurrency, dependency, maintainability, type_safety, preference, convention.
5. Sort by severity (P0 first), then by frequency/importance.
6. Keep at most {max_rules} rules total.
7. GENERALIZE: Lift PR-specific patterns to repo-level norms. If multiple PRs show the same \
preference (e.g. "use module X"), merge into one repo-level rule with scope "repo_wide".
8. FILTER: Remove rules that are only applicable to one specific PR or one specific function. \
Keep only rules that a maintainer would enforce on ANY future PR.
9. PRIORITIZE: Rules from [MAINTAINER] comments outweigh [CONTRIBUTOR] comments. \
Rules with higher source_count (appeared in more PRs) get higher severity.
10. For rules with scope "repo_wide", preserve the "rationale" field explaining WHY.

For each final rule, output a JSON object with:
- "rule_id": clean id like "ERR001" (no PR prefix)
- "category": one of the categories above
- "scope": "repo_wide" | "pr_specific"
- "rationale": why this is a repo-level norm (for repo_wide rules)
- "source_count": how many PRs this rule appeared in (integer, from input or estimated)
- "title", "severity", "detect" (list), "bad_example", "good_example", "fix"

''' + JSON_OUTPUT_INSTRUCTION + '''

<rule_cards>
{rules_json}
</rule_cards>
'''

_RULE_CARD_TEMPLATE = '''\
[Rule ID] {rule_id}
[Category] {category}
[Title] {title}
[Severity] {severity}
[Scope] {scope}

[Detect]
{detect_bullets}

[Bad Example]
{bad_example}

[Good Example]
{good_example}

[Auto Fix Suggestion]
- {fix}
{rationale_line}'''

_VALIDATE_CONVENTION_PROMPT = '''\
You are analyzing a code review conversation to determine if it reveals a framework convention \
or a recurring AI reviewer false-positive pattern.

## Conversation Pattern: {pattern}

### Bot's Original Comment
{bot_comment}

### Response
{reply_text}

{ack_section}

## Task
Determine what this conversation reveals. Output a JSON object:
- "verdict": one of:
  - "framework_convention": the reviewer misunderstood a reusable framework mechanism
  - "ai_false_positive": the reviewer flagged something that is correct/intentional in this repo \
due to uncommon project conventions, domain-specific patterns, or codebase-specific idioms \
(e.g. unusual metaclass usage, custom descriptor protocols, non-standard naming that is intentional, \
deliberate defensive patterns, project-specific error handling strategies)
  - "design_tradeoff": a legitimate design discussion, not a clear mistake
  - "not_generalizable": too specific to this PR, not a recurring pattern
  - "author_wrong": the bot was correct, the author/maintainer was wrong
- "reasoning": one sentence
- If verdict is "framework_convention":
  - "trigger_pattern": the code pattern that triggered the false alarm
  - "actual_behavior": what actually happens in the framework
  - "do_not_flag": one sentence guideline for the reviewer
- If verdict is "ai_false_positive":
  - "trigger_pattern": the code pattern the AI incorrectly flagged
  - "why_correct": why this pattern is correct/intentional in this repo
  - "category": one of "metaclass_magic", "defensive_pattern", "naming_convention", \
"error_handling", "dynamic_dispatch", "test_pattern", "config_pattern", "other"
  - "do_not_flag": one sentence guideline for the reviewer

Output "framework_convention" for reusable framework mechanisms. \
Output "ai_false_positive" for project-specific patterns the AI should learn to skip.
''' + JSON_OUTPUT_INSTRUCTION

_PRE_ROUND_PROMPT_TMPL = '''\
You are a senior code reviewer. Summarize the following pull request diff concisely.
{lang_instruction}

## PR Title
{pr_title}

## PR Description
{pr_body}

## Diff (may be truncated)
{diff_text}

Produce a structured summary covering:
1. What is the purpose of this PR? (1-2 sentences)
2. Which files/modules are changed and why?
3. Key design decisions or trade-offs visible in the diff
4. Potential risk areas that deserve extra scrutiny

Be concise (max 400 words). Output plain text, no markdown headers.
'''
