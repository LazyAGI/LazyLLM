# LazyLLM Git PR Review — Complete Pipeline Reference

This document describes the end-to-end PR review flow of the `lazyllm.tools.git.review` module.
The entry point is the `review()` function in `runner.py`, which runs through pre-analysis
(architecture parsing, historical spec extraction), static lint analysis, six LLM stages
(R1 hunk review → R2a PR design doc → R2 architect review → R3 Agent verification →
RMod modification-necessity analysis → R4 merge & dedup),
usage-scenario inference (RScene) and call-chain bug analysis (RChain) running in parallel
with the main chain, a test coverage check (RCov), and finally publishes
`final_comments + rchain_issues + rcov_issues` to GitHub / GitLab / Gitee / GitCode.

Source directory: `lazyllm/tools/git/review/`
(`runner.py`, `pre_analysis.py`, `rounds.py`, `coverage_checker.py`, `constants.py`, `checkpoint.py`, `utils.py`, `poster.py`, `lint_runner.py`).

---

## 1. Architecture Overview

### 1.1 Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `runner.py` | **Main entry point**: orchestrates all sub-modules; diff fetching & truncation; strategy decision (`_ReviewStrategy`); meta warning generation; clone cleanup; comment publishing |
| `pre_analysis.py` | Repository clone; architecture doc generation (`analyze_repo_architecture`); local-repo arch analysis (local mode); historical review spec extraction (`analyze_historical_reviews`); PR summary; R3 Agent tool set construction |
| `rounds.py` | **Review round core**: R1 hunk-level analysis, R2a PR design doc, R2 architect review, R3 ReactAgent verification, RMod modification-necessity analysis, R4 merge & dedup |
| `coverage_checker.py` | **RCov**: test coverage check — identify testable symbols, group by dependency, grep test files, evaluate coverage gaps (LLM, parallel) |
| `lint_runner.py` | Static lint analysis (no LLM calls), results injected directly into R4 |
| `checkpoint.py` | Resume support: PR-level checkpoint; stage enum `ReviewStage`; soft invalidation (`resume_from`) |
| `constants.py` | Context budget constants; `BudgetManager`; issue density control; diff heuristic compression |
| `utils.py` | LLM call wrappers (retry/QPS); diff parsing; JSON parsing & repair; progress reporting |
| `poster.py` | Fetch existing PR comments; submit platform review (batch `submit_review` + per-comment fallback) |

### 1.2 End-to-End Flow

```
review(pr_number, ...)
  └─ Checkpoint initialization
  └─ Fetch PR info & diff
  └─ _compute_diff_stats + _decide_review_strategy
       ├─ diff exceeds max_diff_chars → truncate at file boundary → generate meta warning issue
       └─ _run_pre_analysis → arch_doc + review_spec + clone_dir + agent_instructions
  └─ _pre_round_pr_summary → pr_summary
  └─ _fetch_existing_pr_comments
  └─ _run_lint_analysis → lint_issues
  ├─ _run_four_rounds (R1 → R2a → R2 → R3 → RMod → R4) → final_comments   ┐ parallel
  └─ _run_rscene_rchain                                                      │
       ├─ infer_usage_scenarios (RScene) → usage_scenarios                   │
       └─ _rscenario_call_chain (RChain) → rchain_issues                    ┘
  └─ _run_coverage_check (RCov, independent thread, dynamic timeout) → rcov_issues
  └─ Merge: final_comments + rchain_issues + rcov_issues → all_comments
  └─ Clean up clone subdirectory
  └─ _post_review_comments → platform
```

### 1.3 Stage Order Summary

| # | Stage | Main Output |
|---|-------|-------------|
| 0 | Checkpoint initialization | `pr_dir`, `checkpoint.json`, `resume_from` soft invalidation |
| 1 | Diff fetch & truncation | `diff_text` (truncated to `max_diff_chars` at file boundary), `hunks` |
| 2 | Strategy decision | `_DiffStats`, `_ReviewStrategy` (adaptive R3 parameters) |
| 3 | Meta Warning | `source='meta'` issue inserted when truncation occurs |
| 4 | Pre-analysis | `arch_doc`, `review_spec`, `clone_dir`, `agent_instructions` |
| 5 | PR summary | `pr_summary` |
| 6 | Existing comments | `existing_comments` (for R4 dedup) |
| 7 | Lint analysis | `lint_issues` (no LLM, injected into R4) |
| 8 | Round 1 | Hunk-level static review issue list |
| 9 | Round 2a | PR design document (structured, 9 sections) |
| 10 | Round 2 | Architect-perspective global design review issue list |
| 11 | Round 3 | ReactAgent verification: validate R1+R2 issues + discover new cross-file issues |
| 12 | RMod | ReactAgent modification-necessity analysis: flag unnecessary changes per file |
| 13 | Round 4 | Deterministic dedup + LLM merge + lint fusion → `final_comments` |
| 14 | RScene | Scenario inference (parallel with R1–RMod): infer 2–4 typical usage scenarios → `usage_scenarios` |
| 15 | RChain | Call-chain bug analysis (after RScene): scenario-driven bug + usability issues → `rchain_issues` |
| 16 | RCov | Test coverage check: identify untested symbols, evaluate gaps → `rcov_issues` (bypasses R4 dedup, appended directly) |
| 17 | Publish | Merge `final_comments + rchain_issues + rcov_issues`, submit platform review; update checkpoint `UPLOAD` stage |
| 18 | Cleanup | Delete `{pr_dir}/clone/`; retain `checkpoint.json` |

---

## 2. Entry Point & Strategy Decision (`runner.py`)

### 2.1 `review()` Function Signature

```python
def review(
    pr_number: int,
    repo: str,
    token: str,
    llm: Optional[Any] = None,
    language: str = 'cn',
    post_to_github: bool = True,
    clone_target_dir: Optional[str] = None,
    arch_cache_path: Optional[str] = None,
    resume_from: Optional[str] = None,
    clear_checkpoint: bool = False,
    backend: str = 'github',
    max_diff_chars: int = 120000,
) -> Dict[str, Any]
```

- `language`: output language for review comments; `'cn'` = Simplified Chinese, `'en'` = English.
- `post_to_github`: when `False`, only returns the result dict without publishing comments — useful for debugging.
- `arch_cache_path`: architecture doc cache directory, shared across PRs to avoid redundant clone & analysis.
- `resume_from`: soft-invalidate from a specific `ReviewStage` without deleting other cached stages.
- `clear_checkpoint`: force-clear the checkpoint and restart from scratch.

### 2.2 Strategy Decision (`_ReviewStrategy`)

Automatically adjusts R3 parameters based on diff size to avoid excessive LLM calls on large PRs:

| PR Size | Condition | `large_file_threshold` | `max_files_for_r3` | `max_chunks_per_file` |
|---------|-----------|----------------------|-------------------|----------------------|
| Large | >3000 lines or >50 files | 100 chars | 10 | 2 |
| Medium | >1000 lines or >20 files | 150 chars | 15 | 2 |
| Small | otherwise | 200 chars (default) | 20 (default) | 3 (default) |

### 2.3 Diff Truncation Strategy

When the diff exceeds `max_diff_chars` (default 120000), it is truncated at file boundaries
(never mid-hunk), and a `source='meta'` warning issue is inserted to inform the user which
files were skipped.

---

## 3. Pre-Analysis (`pre_analysis.py`)

Pre-analysis runs before all review rounds, providing high-quality repository context to
subsequent LLM calls. It consists of four sub-stages, all of which support checkpoint caching.

### 3.1 Repository Clone (`_fetch_repo_code`)

- Uses `git clone --single-branch --depth 1` (without `pin_sha`) or a full clone (with `pin_sha`)
  to pull the repository locally.
- If the target directory already contains a complete clone, it is reused and a `git pull` is
  attempted to update it, avoiding redundant downloads.
- When `pin_sha` (the PR head commit) is specified, `_pin_clone_to_sha` switches the clone to
  the exact commit, ensuring the reviewed code matches the PR.
- The clone directory is stored under `arch_cache_path/{owner_repo}/clone/` and reused across PRs.
- **Local mode** (`review-local`): cloning is skipped. If `arch_doc` is empty, `analyze_repo_architecture`
  is called directly on the local repo path via `_run_local_arch_analysis`, so architecture context
  is still available without a network clone.

### 3.2 Architecture Document Generation (`analyze_repo_architecture`)

The architecture document is the "map" for the entire review flow, allowing the LLM to understand
the repository structure without reading all the code.

**Generation steps (five steps):**

1. **Structured snapshot** (`_collect_structured_snapshot`): scan the clone directory and collect
   the directory tree (2 levels), top-level `__init__.py`, sub-package `__init__.py`, core module
   files (`module.py`, `flow.py`, etc.), dependency config files (`pyproject.toml`,
   `requirements.txt`, etc.), `AGENTS.md`, etc. Concatenate into a structured snapshot text,
   constrained by the `_ARCH_SNAPSHOT_BUDGET` character budget.

2. **DeepWiki integration** (optional): if the `mcp` package is installed, fetch the repository's
   pre-indexed architecture summary from the DeepWiki MCP service (`_fetch_deepwiki_summary`) and
   append it to the snapshot as background reference. Always uses `base_repo` (the upstream
   repository) rather than the fork URL to avoid missing data on forks. DeepWiki data may be
   1–3 months stale; a staleness notice is included when injected.

3. **Outline generation** (`_arch_generate_outline`): feed the snapshot to the LLM to generate an
   N-section outline (at most `_ARCH_OUTLINE_MAX_SECTIONS_WITH_AGENT` sections when `AGENTS.md`
   is present, otherwise `_ARCH_OUTLINE_MAX_SECTIONS`). The outline always ends with two fixed
   sections:
   - Second-to-last: **Concurrency & Multi-User Conventions** (thread safety, ContextVar, lock
     conventions)
   - Last: **Testing & Examples** (test file locations, example scripts)
   The outline is cached as `arch_outline` to avoid regeneration.

4. **Batch section fill** (`_arch_fill_all_sections`): for each section, first use
   `_arch_collect_snippets_for_section` to grep relevant code snippets from the repository, then
   pack multiple sections into a single LLM call (`_arch_fill_batch_llm`) to reduce LLM call
   count. Each section is capped at 3500 characters and cached under an independent key
   (`arch_section_{title}`).

5. **Public API Catalog** (`_build_public_api_catalog`): the LLM identifies the repository's
   public API files, then regex-extracts public symbols (functions, classes, constants) from each
   file, generating a JSON-format API catalog appended to the architecture document. Supports
   Python, Go, TypeScript, Java, Rust, C++, and more.

**Final output**: architecture document + Public API Catalog; also generates `arch_index`
(section summary index) and `arch_symbol_index` (symbol → section mapping) for on-demand
retrieval by R1/R2/R3. All outputs are cached at `arch_cache_path/{owner_repo}/`; subsequent
PRs on the same repository reuse them directly.

**AGENTS.md support**: if `AGENTS.md` exists in the repository root, its content is injected as
`agent_instructions` into all LLM prompts, allowing project maintainers to customize review rules
(e.g. "ignore naming style in the tests/ directory").

### 3.3 Historical Review Spec Extraction (`analyze_historical_reviews`)

Distills project-specific coding conventions from historical PR review comments into a `review_spec`.

**How it works:**

1. Fetch review comments from the most recent N merged PRs (default 30).
2. Filter out bot comments, very short comments (<20 chars), and emoji-only comments.
3. Group comments by file type and feed them to the LLM to extract conventions:
   - Project-specific naming conventions
   - Forbidden patterns (e.g. "no direct `print`, must use `lazyllm.LOG`")
   - Mandatory architectural constraints
   - Testing requirements
4. The spec is cached at `arch_cache_path/{owner_repo}/review_spec.json` and reused across PRs.

This step enables the review to "remember" recurring issues from the project's history and avoid
re-raising already-known conventions.

### 3.4 PR Summary Generation (`_pre_round_pr_summary`)

Before the four review rounds begin, a single LLM call generates a PR summary (`pr_summary`)
containing:
- The core purpose of the PR (new feature / bugfix / refactoring)
- Main changed modules
- Potential impact scope

The PR summary is injected as context into R2 and R3 prompts, helping the LLM evaluate from the
perspective of "PR intent" rather than looking at each hunk in isolation.

### 3.5 R3 Agent Tool Set (`_build_scoped_agent_tools_with_cache`)

Builds a file-system tool set for R3's ReactAgent, with all tools scoped to the clone directory:

| Tool | Function |
|------|----------|
| `read_file_lines` | Read a specified line range from a file |
| `read_file_skeleton_scoped` | Read file skeleton (class names, function signatures, no implementation) |
| `grep_symbol` | Search for symbol definitions or references across the repository |
| `list_directory` | List directory contents |

Tool calls are capped at a maximum number of steps (default 15) to prevent the Agent from
exploring indefinitely and exhausting the budget.

---

## 4. Six-Round Analysis (`rounds.py`)

The six-round analysis is the core of the review. Each round has a clear responsibility,
forming a progressive structure: broad scan → design doc → architect review → deep verification
→ modification-necessity analysis → merge & dedup. In parallel, **RScene** (scenario inference)
and **RChain** (call-chain bug analysis) run alongside the main chain, with their results merged
after R4.

---

### 4.1 Round 1: Hunk-Level Static Review

**Goal**: fine-grained code quality review of each diff hunk to find concrete bugs, logic errors,
and security issues.

**How it works:**

1. Split the diff by file and hunk; each hunk is accompanied by context (N lines before and after).
2. Call the LLM independently for each hunk; the prompt includes:
   - The full diff hunk (with line number annotations)
   - File-level context (function signatures, class definitions)
   - Architecture document excerpt (relevant to this file)
   - PR summary
   - Historical review spec
3. Multiple hunks are processed concurrently (`ThreadPoolExecutor`), constrained by
   `TOTAL_CALL_BUDGET=60`.

**Checklist (`_R1_STRICT_RULES` + `_R1_REVIEW_CHECKLIST`):**

- **Bug**: null/out-of-bounds access, incorrect conditionals, loop boundary errors,
  resource leaks (unclosed files/connections)
- **Logic**: function behavior inconsistent with its name, unchecked return values,
  silently swallowed exceptions
- **Security**: SQL injection, path traversal, hardcoded secrets, unsafe deserialization
- **Concurrency**: race conditions, improper lock usage, unprotected shared state
- **Configuration**: config placed in the wrong location (should be in `configs.py` not `tracing/`),
  optional dependency added to required dependencies
- **Maintainability**: orphaned helpers/constants left after refactoring (symbols only used by
  deleted code)

**Strict exclusions** (to avoid noise):
- Issues in unchanged lines that pre-existed the diff
- Issues already covered by lint tools (unused imports, line length, complexity metrics)
- Pure style issues (unless they violate project conventions)

**Issue density control**: at most 5 issues per 100 effective diff lines, preventing large PRs
from generating excessive noise.

**Output format** (per issue):

```json
{
  "path": "lazyllm/tools/git/review/rounds.py",
  "line": 42,
  "severity": "critical|high|medium|normal",
  "bug_category": "bug|security|performance|maintainability|design|style",
  "title": "Short title",
  "description": "Problem description",
  "suggestion": "Fix suggestion (with code example)",
  "source": "r1"
}
```

---

### 4.2 Round 2a: PR Design Document Generation

**Goal**: before the architect review (R2), generate a structured PR design document as input
context for R2.

**How it works:**

A single LLM call on the full diff + PR title/description + architecture document produces a
9-section design document:

| Section | Content |
|---------|---------|
| 1. Background & Problem Definition | Problem the PR solves; its position in the existing architecture; new feature/bugfix/refactoring |
| 2. Design Goals | Expected outcome; design constraints (performance/extensibility/consistency) |
| 3. Design Approach | Core idea; rationale; alternative approaches; conformance to architectural layering |
| 4. Module Impact Analysis | Modified/added modules; responsibility changes; new dependencies introduced |
| 5. API Design | Added/modified interfaces; inputs and outputs; consistency with existing API style |
| 6. Usage Example | Typical calling example; impact on user interaction |
| 7. Compatibility & Impact Scope | Effect on existing functionality; whether it is a breaking change |
| 8. Risks & Edge Cases | Potential issues; uncovered scenarios; implicit assumptions |
| 9. Extensibility Analysis | Ease of extending for similar future requirements; design evolution space |

The design document is saved in the checkpoint (`pr_design_doc`) and also returned as part of
the final review result.

---

### 4.3 Round 2: Architect Design Review

**Goal**: review the entire PR from a global architectural perspective, finding design-level issues
that R1 cannot detect.

**How it works:**

Takes the full diff + PR design document (R2a output) + architecture document + PR summary as
input, and performs a global review in a single LLM call.

**11 evaluation dimensions (`_ROUND2_ARCHITECT_PROMPT_TMPL`):**

1. **Module Responsibility**: is the new code placed in the right module? Is any logic scattered
   where it does not belong? Should it live in a different module?

2. **Layering & Dependencies**: are existing layer boundaries respected? Are circular dependencies
   introduced? Is there cross-layer direct access (e.g. UI layer directly touching the database)?

3. **API Design**: are new interfaces concise, stable, and easy to use? Are there too many
   parameters? Are unnecessary internal details exposed?

4. **Consistency**:
   - Do classes/functions in the same module follow the same interface pattern?
   - If the diff adds a new class of the same type as an existing one (e.g. a new storage backend,
     a new model provider), do they share a common base class or Protocol?
   - Do similar classes follow the same construction pattern (`__init__` parameter order,
     factory/client entry point)?
   - Do similar classes implement the same set of key methods with consistent signatures
     (parameter names, order, return types)?
   - If the project uses `__call__` + `forward` or similar dispatch patterns, does the new class
     follow the same pattern?

5. **Abstraction & Reuse**:
   - Does logic already exist elsewhere in the codebase?
   - **Critical check**: if the system previously had only ONE implementation of a concept
     (e.g. one storage backend, one model provider) and this PR adds a SECOND one, does a common
     base class / Protocol / ABC exist? If not, this is a design issue — the two implementations
     should be unified under a shared abstraction.
   - Is the new abstraction at the right level (not too generic, not too specific)?

6. **Complexity**: is unnecessary complexity introduced? Are there multi-step pipelines that could
   be expressed as a single expression?

7. **Extensibility**: will similar future requirements be easy to extend? Are there hardcoded
   assumptions that would block extension?

8. **Coupling**: is inter-module coupling increased? Can it be reduced through interfaces, events,
   or dependency injection?

9. **Testability**: is the new code easy to test? Is there global state or hard dependencies that
   make testing difficult?

10. **Overall Design Verdict**: is there a simpler or more consistent alternative design? What is
    the single most important architectural improvement?

11. **Naming & Semantic Clarity**:
    - Are method/function names self-explanatory and concise?
    - Do method names avoid redundantly including the class name (e.g. prefer
      `ClassA.get_instance()` over `ClassA.get_classA_instance()`)?
    - Are parameter names consistent across similar methods in sibling classes?
    - If the project provides syntactic sugar (operator overloads like `__or__`, `__ror__`,
      `__getitem__`, etc.), does the sugar's semantics align with mainstream conventions
      (bash pipe `|`, Python slice `[]`, etc.)? Flag if it could confuse users.

---

### 4.4 Round 3: ReactAgent Deep Verification

**Goal**: R1 and R2 can only see the diff itself. R3 uses a ReactAgent to actively explore the
repository, validate the accuracy of existing issues, and discover new problems that require
cross-file context.

**How it works (two phases):**

#### Phase 1: Context Collect

For each diff file that needs verification, the ReactAgent uses the tool set to actively explore
the repository:

1. **Read file skeleton**: use `read_file_skeleton_scoped` to understand file structure before
   reading specific lines.
2. **Symbol tracing**: use `grep_symbol` to find all callers of modified symbols and assess
   impact scope.
3. **Base class / interface check**: read base class definitions to confirm whether subclasses
   satisfy interface contracts.
4. **Framework mechanism identification**: check for metaclass, `__init_subclass__`, registry,
   and other auto-registration mechanisms (to avoid falsely flagging auto-registered subclasses
   as "dead code").
5. **Sibling class exploration** (new): when the diff adds a new class, search for sibling classes
   with the same base class or in the same directory, and compare their construction patterns,
   key method signatures, and `__call__`/`forward` dispatch patterns.
6. **Operator semantics verification** (new): if the diff modifies `__or__`, `__getitem__`, or
   other dunder methods, grep their usage patterns to verify that the semantics align with
   mainstream conventions.

Context Collect output is structured JSON:

```json
{
  "explored_symbols": ["sym1", "sym2"],
  "related_files": [
    {"path": "relative/path.py", "reason": "one-line reason", "lines": [start, end]}
  ],
  "base_classes": [
    {"symbol": "BaseClassName", "file": "relative/path.py"}
  ],
  "framework_notes": ["one-line finding about framework mechanism"],
  "sibling_classes": [
    {"symbol": "ClassName", "file": "relative/path.py", "key_methods": ["method1(args)", "method2(args)"]}
  ]
}
```

#### Phase 2: Issue Extract

Using the context collected in Phase 1 + existing R1/R2 issues as input, two tasks are performed:

**Task 1: Validate existing issues**
- Confirm whether issues are real (eliminate false positives)
- Add cross-file evidence (e.g. "caller X will be affected")
- If the same pattern exists elsewhere in the repository, mark as "consistent with project
  convention, not an issue"

**Task 2: Discover new issues** (requiring cross-file context):
- **Caller breakage**: function signature changed but callers not updated
- **Orphaned symbols**: helpers/constants left after refactoring that were only used by deleted code
- **Base class contract violation**: subclass does not implement abstract methods, or overrides
  methods it should not
- **Base class abstraction gap** (new): if the diff adds a second implementation of a concept
  with no shared base class, report as a design issue
- **Sibling consistency** (new): if sibling classes (same base or same role) have inconsistent
  construction signatures, key method signatures, or dispatch patterns (`__call__`/`forward`),
  report the inconsistency
- **Naming clarity** (new): if new method/function names redundantly include the parent
  class/module name, or are not self-explanatory, report the naming issue
- **Syntactic sugar semantics** (new): if the diff adds or modifies operator overloads
  (`__or__`, `__getitem__`, `__lshift__`, etc.), verify the semantics align with mainstream
  language/shell conventions; report if it could mislead users

**R3 scale controls:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `R3_MAX_FILES` | 20 | Maximum files processed by R3 |
| `R3_MAX_CHUNKS_PER_FILE` | 3 | Maximum chunks per file |
| `R3_MAX_CHUNKS_HARD` | 8 | Hard upper bound on chunks per file |
| Agent step limit | 15 steps | Maximum tool calls per Context Collect invocation |

Large files (diff exceeding `large_file_threshold`) are skipped in R3 to avoid consuming
excessive budget.

---

### 4.5 RMod: Modification-Necessity Analysis (`_run_rmod_agent_round`)

**Goal**: for each modified file, use a ReactAgent to judge whether the changes are architecturally
justified — flagging unnecessary refactors, over-engineering, or changes that violate framework
conventions.

**How it works:**

1. `_rmod_collect_file_diffs` extracts per-file diffs; `_rmod_new_file_paths` identifies newly
   created files (correctly handles `--- /dev/null` → `+++ b/` ordering in unified diff format).
2. For each file, a `ReactAgent` is launched with the same file-exploration tools as R3
   (`read_file_scoped`, `search_scoped`, etc.).
   The agent is given the PR design doc, arch doc,
   and framework conventions as context.
3. The agent runs with a per-file timeout (`_RMOD_AGENT_TIMEOUT_SECS`). Files are processed in
   parallel (up to `max_workers` threads).
4. Issues are tagged `source='rmod'` and `bug_category='design'`.
5. Results are saved to checkpoint (`ReviewStage.RMOD`) for resume support.

**Key implementation notes:**
- `execute_in_sandbox` is disabled on tools before agent creation (not via private `_tools_manager`).
- `FuturesTimeoutError` (from `concurrent.futures`) is used instead of the built-in `TimeoutError`.

---

### 4.6 RScene: Usage Scenario Inference (`infer_usage_scenarios`)

**Goal**: understand the public APIs modified by the PR, infer 2–4 typical end-to-end usage
scenarios, and provide input for RChain. RScene runs **in parallel** with the main chain
(R1→R2→R3→RMod) and does not block it.

**How it works (two steps):**

**Step 1 — Understand the functional module**: the ReactAgent actively explores the repository
using the tool set:
- Use `read_file_scoped` to read the full skeleton (class definitions, method signatures, key
  constants) of each modified file
- Use `analyze_symbol` to understand the state machine and data flow of core classes
- Use `search_in_files` to search existing tests and examples for established usage patterns

**Step 2 — Infer typical usage scenarios**: based on the understanding, output 2–4 scenarios,
each containing:
- `title`: scenario name
- `description`: scenario description
- `api_sequence`: API call order (e.g. `init → configure → run → cleanup`)
- `call_chain`: expected call chain (list of function/method names)
- `edge_cases`: edge cases to check

**Scale controls**:
- Process at most `max_files_for_r3` files (shared strategy parameter with R3)
- Only process modified files (not newly created files); new files have no historical behavior
  to infer from
- At most 3 parallel workers (`_RSCENE_MAX_PARALLEL`)
- Per-agent timeout of `_RSCENE_AGENT_TIMEOUT_SECS` (180s), with up to `_RSCENE_AGENT_RETRIES`
  (10) retries

**Checkpoint**: scenario results are saved to `ReviewStage.RSCENE` (key: `rscene_all`); on
resume, the cached results are loaded directly.

---

### 4.7 RChain: Call-Chain Bug Analysis (`_rscenario_call_chain`)

**Goal**: driven by the usage scenarios inferred by RScene, perform deep bug analysis and API
usability review on the call chain of each scenario. RChain starts immediately after RScene
completes, running in parallel with the main chain.

**How it works (two tasks):**

**Task A — Call-chain bug analysis**:
1. Use `read_file_scoped` / `analyze_symbol` to trace each step in the call chain
2. Verify that input/output contracts between callers and callees are consistent
3. Check exception propagation paths: are exceptions correctly caught or propagated upward?
4. Check resource lifecycle: are files/connections/locks properly released on all paths?
5. Check concurrency safety: are there race conditions or shared-state issues in the call chain?

**Task B — API usability review**:
1. Verify that the API sequence is intuitive (parameter order, naming, default values)
2. Check whether error messages are clear enough for users to debug
3. Check for easily misused APIs (e.g. parameter order that is easy to confuse)

**Output**: issues are tagged `source='rchain'`; Task A uses `bug_category` values of
`bug`/`security`/`performance`/`concurrency`/`safety`/`type`; Task B uses `design`/`exception`.

**Scale controls**:
- Each scenario runs an independent ReactAgent; at most `_RCHAIN_MAX_PARALLEL_SCENARIOS` (3)
  run in parallel
- Per-agent timeout of `_RCHAIN_AGENT_TIMEOUT_SECS` (240s), with up to `_RCHAIN_AGENT_RETRIES`
  (6) retries
- Each scenario's diff budget is constrained by `_RCHAIN_FIXED_OVERHEAD`

**Checkpoint**: each scenario's result is cached independently (key:
`rchain_scene_{idx}_{title}`); the overall result is cached as `rchain_all`
(`ReviewStage.RCHAIN`).

**Relationship with the main chain**: RChain issues **bypass R4 dedup** and are appended
directly to `all_comments` after `final_comments` (similar to RCov). This is because RChain's
scenario-driven perspective is complementary to R1/R2/R3's diff-centric perspective; forcing
dedup would discard valuable scenario-level bugs.

---

### 4.8 Round 4: Merge & Dedup

**Goal**: merge the issue lists from R1, R2, R3, RMod, and `lint_issues` into a final,
deduplicated comment list. RCov and RChain issues bypass this stage and are appended directly
after R4.

**How it works (three steps):**

1. **Deterministic dedup**:
   - For the same `(path, line, bug_category)`, keep only the highest-severity issue
   - Issues already present in `existing_comments` are discarded (avoid duplicate comments)

2. **LLM semantic merge**:
   - For issues in the same file that are semantically similar but at different line numbers,
     the LLM judges whether they describe the same problem
   - If so, they are merged into one, retaining the more detailed description and suggestion

3. **Lint fusion**:
   - `lint_issues` (from `lint_runner.py`, no LLM) are appended directly to the final list
   - Lint issues have `source='lint'` and are specially marked in platform comments

**Output**: `final_comments` list; each item contains the full `path`, `line`, `severity`,
`bug_category`, `title`, `description`, `suggestion`, and `source` fields.

---

## 5. RCov: Test Coverage Check (`coverage_checker.py`)

RCov runs **after** R4 (independently, not merged through dedup) and checks whether new or
modified public symbols have adequate test coverage.

### 5.1 Three-Step Flow

**Step 1 — Identify testable symbols** (`_rcov_identify_symbols`):
- Clips the diff to `SINGLE_CALL_CONTEXT_BUDGET - 20000` chars.
- Calls the LLM to extract a list of public functions/classes that need test coverage.
- Filters out internal helpers (`is_internal=True`).

**Step 1.5 — Group by dependency** (`_rcov_group_symbols`):
- If more than one symbol is found, calls the LLM to group related symbols (e.g. a class and its
  key methods) so they are evaluated together.
- Falls back to one group per symbol if grouping fails.

**Step 2 — Grep + evaluate** (`_rcov_evaluate_groups`):
- `_find_test_files` scans `clone_dir` for test files (`test_*.py`, `*_test.py`, etc.).
- For each group, `_build_grep_results` greps test files in parallel for each symbol name.
  Symbol names are validated with `re.fullmatch` before being passed to `grep -F` to prevent
  regex injection.
- The LLM evaluates whether existing tests are sufficient; issues are tagged `source='rcov'`.
- Groups are evaluated in parallel (up to 4 workers); each group has a per-call timeout of
  `_RCOV_GROUP_TIMEOUT_SECS` (90s).

### 5.2 Timeout Strategy

The outer timeout in `runner.py` is derived dynamically from diff size:

```
timeout = clamp((3 * 90) + (len(diff_text) // 10000) * 30, min=300, max=900)  # seconds
# Note: base value 270 < min=300, so the actual minimum is 300 when diff is very small.
```

The larger the diff, the longer the timeout, avoiding false timeouts on large PRs while
preventing unnecessary waiting on small ones. The actual timeout value is logged at `INFO`
level for observability.

### 5.3 Checkpoint

RCov results are stored under the independent `ReviewStage.RCOV` checkpoint key (`rcov_issues`).
On resume, the cached results are loaded directly, skipping all LLM calls.

Output metrics:
- `rcov_issues_count`: number of issues found, or `None` if RCov was skipped (timeout / no clone).
- `rcov_ran`: boolean indicating whether RCov actually executed.

---

## 6. Static Lint Analysis (`lint_runner.py`)

Lint analysis runs independently before the LLM review rounds and consumes no LLM call budget.

**How it works:**

1. Extract changed files and line number ranges from the diff.
2. Run `ruff` / `flake8` on Python files, and `eslint` on JS/TS files (if installed).
3. Filter to only lint errors that appear on diff-changed lines (pre-existing issues in unchanged
   lines are not reported).
4. Convert results to the unified issue format (`source='lint'`) and inject into R4.

Lint issues bypass LLM validation and go directly into the final result, giving them high
precision with no hallucination risk.

---

## 7. Checkpoint System (`checkpoint.py`)

### 7.1 Design Goals

- **Resume support**: if the review fails mid-way (network timeout, LLM rate limit), it resumes
  from the last completed stage without repeating finished LLM calls.
- **Soft invalidation**: the `resume_from` parameter restarts from a specific stage without
  deleting other cached stages.
- **Version control**: when `_REVIEW_ROUND_VERSION` is incremented, the R4 (merge & dedup) cache
  is automatically invalidated, ensuring prompt changes trigger a recomputation of the final result.

### 7.2 Stage Enum (`ReviewStage`)

```
CLONE → ARCH → SPEC → PR_SUMMARY → R1 → R2A → R2 → R3 → RMOD →
RSCENE → RCHAIN → FINAL → RCOV → UPLOAD
```

After each stage completes, `ckpt.mark_stage_done(stage)` is called. On restart,
`ckpt.should_use_cache(stage)` determines whether to skip the stage.

`_KEY_TO_STAGE` is a class-level dict (initialized at class definition time, not lazily) mapping
checkpoint keys to their owning stage. This avoids TOCTOU races in concurrent scenarios.

### 7.3 Head SHA Rotation

If the PR is force-pushed during review (head SHA changes), the checkpoint is automatically
backed up and all review-round data is purged (clone, arch, and spec are retained), restarting
from R1.

---

## 8. Comment Publishing (`poster.py`)

### 8.1 Publishing Strategy

1. **Batch submit**: prefer the platform's `submit_review` API to submit all comments at once
   (GitHub Pull Request Review).
2. **Per-comment fallback**: if batch submission fails (e.g. a comment line is not in the diff
   range), retry each comment individually with `create_review_comment`.
3. **Rate limit retry**: on 429 / 403 rate-limit responses, retry with exponential backoff
   (up to 3 times).
4. **Merge order**: `all_comments = final_comments + rchain_issues + rcov_issues`; all three
   categories of issues are filtered through `_filter_commentable` before publishing.

### 8.2 Commentable Line Filtering (`_filter_commentable`)

Platforms only allow line-level comments on lines that are actually changed in the diff.
`_build_commentable_lines` parses the diff to build a `{path: set(lines)}` map.
`_filter_commentable` returns a **three-tuple** `(inline, general, dropped)`:
- `inline`: issues with a valid, in-diff line number → posted as line-level review comments.
- `general`: issues with `line=None` (e.g., RCov coverage issues) → posted as PR-level comments
  in the review body under a "General Review Comments" section.
- `dropped`: issues whose line number is outside the changed range and cannot be mapped → discarded.

### 8.3 Comment Body Policy

Each comment body includes the following enforcement statement:

> *Newly introduced architecture issues must be fixed before merging; pre-existing ones must be
> tracked via an issue (new or linked). Style issues must also be fixed; missing test coverage
> must be added.*

---

## 9. Budget & Rate Limiting (`constants.py`)

| Constant | Default | Description |
|----------|---------|-------------|
| `SINGLE_CALL_CONTEXT_BUDGET` | 120000 chars | Context limit per LLM call |
| `R1_DIFF_BUDGET` | 95000 chars | Diff content limit in R1 (25000 reserved for system prompt + arch doc) |
| `TOTAL_CALL_BUDGET` | 60 calls | Total LLM call limit for the entire review session |
| `ISSUE_DENSITY_LINE_BLOCK` | 100 lines | Block size for issue density control |
| `ISSUE_DENSITY_MAX_PER_BLOCK` | 5 issues | Maximum issues per 100 lines |
| `R3_MAX_FILES` | 20 | Maximum files processed by R3 |
| `R3_MAX_CHUNKS_PER_FILE` | 3 | Maximum chunks per file in R3 |
| `R3_MAX_CHUNKS_HARD` | 8 | Hard upper bound on chunks per file in R3 |

`BudgetManager` tracks the number of LLM calls consumed. Each round checks the remaining budget
before calling, and skips non-critical steps when the budget is exhausted.

---

## 10. Issue Field Specification

All rounds output issues in a unified format:

| Field | Type | Description |
|-------|------|-------------|
| `path` | str | File path relative to the repository root |
| `line` | int | Line number of the issue (new line number in the diff) |
| `severity` | str | `critical` / `high` / `medium` / `normal` |
| `bug_category` | str | `bug` / `security` / `performance` / `maintainability` / `design` / `style` |
| `title` | str | Short title (≤80 chars) |
| `description` | str | Detailed problem description |
| `suggestion` | str | Fix suggestion (with code example using markdown code blocks) |
| `source` | str | `r1` / `r2` / `r3` / `rmod` / `rcov` / `lint` / `meta` |

---

## 11. Known Limitations

| Limitation | Description |
|------------|-------------|
| Diff truncation | Very large PRs (>120K chars) only review the first N files; truncated files are reported via a meta warning |
| R3 file cap | On large PRs, R3 processes at most 10 files; the rest only go through R1/R2 |
| RCov clone dependency | RCov requires `clone_dir` to grep test files; in local mode without a clone, test file discovery falls back to the local repo path |
| Dynamic references | `grep_symbol` cannot trace runtime-generated symbol names (e.g. `getattr(obj, name)`) |
| Cross-repo dependencies | Only the current repository is analyzed; interface changes in external dependencies cannot be detected |
| Language support | Lint analysis currently focuses on Python (ruff/flake8); JS/TS requires eslint to be installed locally |

---

*If budget constants, `ReviewStage.ordered()`, round prompts, checklists, or new rounds are
changed in the future, please update this document accordingly.*
