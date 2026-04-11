# LazyLLM Git PR Review 完整流程说明

本文档描述 `lazyllm.tools.git.review` 中一次 PR review 的端到端流程：预分析、五轮分析、合并去重、发帖，以及缓存与断点续跑机制。涉及源码路径：`lazyllm/tools/git/review/`（`runner.py`、`pre_analysis.py`、`rounds.py`、`checkpoint.py`、`utils.py`、`poster.py`）。

---

## 1. 总览：从入口到结束

### 1.1 入口 `review()`（`runner.py`）

1. **Checkpoint 初始化**
   - `pr_dir = ~/.lazyllm/review/cache/{safe_repo}/{pr_number}/`
   - `checkpoint_path` 默认：`{pr_dir}/checkpoint.json`
   - `clear_checkpoint=True`：清空 checkpoint 文件并删除整个 `pr_dir`（含 clone），从零开始。
   - 否则：`resume_from=ReviewStage.X` 时写入 `_invalidated_from` 标记（**不物理删除**历史字段），见第 6 节。

2. **拉取 PR 与 diff**
   - `diff_text`：优先从 checkpoint 读；否则 API 拉取，可按 `max_diff_chars` 截断后写入 checkpoint。
   - `hunks = _parse_unified_diff(diff_text)`，可按 `max_hunks` 截断。

3. **预分析** `_run_pre_analysis(...)`：得到 `arch_doc`、`review_spec`、`clone_dir`、`agent_instructions`。

4. **PR 摘要** `_pre_round_pr_summary`：checkpoint 命中则跳过，否则 1 次文本 LLM。

5. **五轮** `_run_five_rounds`：R1→R2→R3→R4（合并设计文档 + 架构师评审）→FINAL（合并去重）。

6. **清理 clone**
   - 仅删除 `{pr_dir}/clone/`，**保留** `checkpoint.json`（避免误删断点）。

7. **可选发帖** `_post_review_comments`：需 `head_sha`；仅对带 `path`+`line` 的 issue 发行评。

8. **返回值**：含 `comments`、`pr_summary`、`pr_design_doc`（R4 合并阶段写入 checkpoint 的 `pr_design_doc`）等。

---

## 2. Token / 调用次数说明（估算方法）

代码中多为 **字符预算（chars）** 而非 token。下表将 **输入 ≈ 各段预算之和 + 固定 prompt 模板**，**输出** 按任务类型给出量级；**1 次调用** 指一次 `_safe_llm_call`（JSON）或 `_safe_llm_call_text`（纯文本）。实际 token 与模型分词有关，需按比例换算。

| 阶段 | 典型输入规模（字符级） | 典型输出 | 调用次数（量级） |
|------|------------------------|----------|------------------|
| 架构 outline | snapshot≤4k + 模板 | JSON 数组，小 | 1 |
| 每 arch section | 目录+摘要≤1.5k + snippets≤6k + 模板 | ≤3.5k 文本 | 2/section（填充+摘要） |
| Public API 文件列表 | 目录树≤4k + 模板 | JSON≤30 项 | 1 |
| 历史规则 / PR | 视 PR 数与评论长度 | 规则卡 JSON | 每 PR 1–2 + 合并 1 |
| R1 每 batch | arch≤3k + hunk + 上下文 + 模板 | JSON 数组 | 1/batch |
| R2 每文件 | Agent 多轮 + 1 次 issue 提取 | 结构化文本 + JSON | Agent 多轮 + 1 |
| R3 每文件 | 全量 arch + 规则 + diff≤4k + 模板 | JSON 数组 | 1 |
| R4a | arch≤2k + summary≤600 + diff≤8k | 长文本（9 节） | 1 |
| R4 | 全 arch + design≤3k + diff≤12k + 前序摘要≤2k | JSON 数组 | 1 |
| R5 | 压缩后的 new+existing JSON | JSON 数组 | 1 + 可选批量压缩 |

---

## 3. 预分析（`pre_analysis.py`）

### 3.1 编排 `_run_pre_analysis`

- **arch**
  - 若 `fetch_repo_code` 且 checkpoint 无 `arch_doc`：`clone` 到 `{pr_dir}/clone/`，再 `analyze_repo_architecture`。
  - 若已有 `arch_doc` 但需 agent：可能复用或重建 `clone_dir`，并读 `agent_instructions`（可来自 arch 侧缓存）。
- **review_spec**：`_run_spec_analysis` → `analyze_historical_reviews`（可跳过或从 checkpoint 加载）。
- 返回：`(arch_doc, review_spec, clone_dir, agent_instructions)`。

### 3.2 架构文档 `analyze_repo_architecture`

1. **`_collect_structured_snapshot`**（无 LLM）  
   总预算 `_ARCH_SNAPSHOT_BUDGET=6000`：2 级目录树、顶层/子包 `__init__.py`、关键文件头、依赖文件、`AGENTS.md` 等。

2. **`_arch_generate_outline`**（**1×JSON LLM**）  
   - 输入：`snapshot[:4000]`。  
   - 有 agent 指令时 section 数约 9，否则约 12；强制首尾章节（模块分层、环境依赖、Key Utilities；无 agent 时还有 Gotchas）。  
   - 输出：每节 `title` / `focus` / `search_hints`。

3. **`_arch_fill_all_sections`**（每节 **2×文本 LLM**）  
   - `_arch_collect_snippets`：按 `search_hints` 在 `*.py` 中搜索，每 pattern 最多约 8 条命中，拼接至多约 6k 字符。  
   - `_arch_fill_section`：目录树 400 + `prev_summaries` 总预算 `_ARCH_PREV_SUMMARY_BUDGET=1500` + snippets；单节输出截断 **3500**。  
   - `_summarize_section`：再生成 ≤200 字符摘要供后续节使用。  
   - 每节可缓存到 repo 级 `arch.json`（`arch_section_*`）。

4. **`_build_public_api_catalog`**（**1×JSON LLM + 无 LLM 扫描**）  
   - LLM：输入 3 级目录树≤4000，输出最多 30 个 `{file, scope, reason}`。  
   - 代码：按扩展名用正则提取公开符号（跳过 `_` 前缀等），每文件最多 40 条，按 `scope` 聚合成 JSON。  
   - 结果追加到 `arch_doc`：`[Public API Catalog]\n{json}`，并缓存 `public_api_catalog`。

5. **落盘** `_save_cache_multi(arch.json)`：`arch_doc`、`arch_index`、`arch_symbol_index`。

**本节 LLM 次数粗算**：`1 + 2×N_sections + 1`（N_sections≈9 或 12）。

### 3.3 历史 review 规则 `analyze_historical_reviews`

- 拉取已合并 PR 的评论；过滤 bot；长评论先压缩再抽规则；多 PR 规则再 **1 次合并** LLM。  
- 产出两级结构：`summaries`（轻量）+ `details`（完整规则卡），写入 `spec.json` / checkpoint。

### 3.4 PR 摘要 `_pre_round_pr_summary`

- **1×文本 LLM**：`pr_title` + `pr_body[:800]` + `diff[:5000]` + 语言指令；输出结构化短摘要（模板要求约 400 词内）。

### 3.5 上下文裁剪与规则查找

- **`_extract_arch_for_file(arch_doc, file_path, max_chars=3000)`**  
  - 解析 `[Section]` 块；对标题匹配 `_ARCH_ALWAYS_INJECT`（Module Hierarchy、Gotchas、Key Utilities 等）给高分。  
  - **Public API Catalog**：解析 JSON，用 **`_candidate_scopes(file_path)`**（`global` + 各级父路径）过滤，只注入与当前文件相关的 scope，避免全量灌入。  
  - 按分数降序拼块直到 `max_chars`。

- **`_lookup_relevant_rules`**：从 diff 前 200 行提关键词，匹配规则 `title`，最多加载 `max_detail` 条完整卡片，其余仅标题列表。

### 3.6 Round 2 用 Agent 工具（`clone_dir` 限定在仓库内）

`_build_scoped_agent_tools`：

| 工具 | 作用 |
|------|------|
| `read_file_scoped` | 按行范围读文件 |
| `read_files_batch` | 批量读，最多 6 文件×1600 字符 |
| `grep_callers` | 按符号搜调用点 |
| `search_scoped` | 正则搜索，默认约 40 结果 |
| `list_dir_scoped` | 列目录 |
| `shell_scoped` | 只读 shell（`allow_unsafe=False`） |

`_build_scoped_agent_tools_with_cache` 额外 **`analyze_symbol`**：定位定义、抽签名/docstring、**内部可再调 LLM** 生成一句摘要，结果进共享 `symbol_cache` 跨文件复用。

---

## 4. 五轮分析（`rounds.py`）

### 4.1 Round 1：按 hunk / 批次的静态审查

- **并发**：`ThreadPoolExecutor(max_workers=4)`，按文件分组。  
- **批处理**：同文件多 hunk 合并，单批 diff 总长约 **`_R1_BATCH_DIFF_LIMIT=60000`**；单 hunk 可走单条分析。  
- **每条输入**：  
  - 截断 hunk（如 `_truncate_hunk_content` 约 80 行）；  
  - **`_read_file_context`**：小文件全文带行号，大文件 hunk 前后 `_CONTEXT_LINES=50`，附封闭 scope（class/def）及兄弟签名；  
  - `arch_doc` → **`_extract_arch_for_file(..., 3000)`**；  
  - `review_spec`、`pr_summary`、`agent_instructions` 等截断；  
  - 可选从 arch 的 Key Utilities 建 **symbol_index**，匹配 diff 中符号注入说明。  
- **输出**：JSON 数组，`path/line/severity/bug_category/problem/suggestion`，**仅针对 diff 引入的问题**。  
- **LLM**：每批 **1 次** JSON 调用。

### 4.2 Round 2：按文件的 ReactAgent + 再判别

- **依赖**：`clone_dir` 必须存在。  
- **共享静态上下文** `_r2_build_shared_context`（无 LLM，预算 **`_R2_SHARED_CTX_BUDGET=1500`**）：跨文件重复 import、PR 内依赖、变更接口签名等。  
- **每文件 Step A**：`ReactAgent`（`max_retries=5`，`force_summarize=True`，`keep_full_turns=2`，timeout 300s），工具为上述 6+1；按 prompt 顺序读 AGENTS、对改动符号 `analyze_symbol`、`grep_callers`、必要时分析基类；输出短结构化 **`explored/callers/base_changes/risk`**（约 600 字符预算）。  
- **每文件 Step B**：**1×JSON LLM** `_r2_extract_issues`，输入：`arch`（每文件 `_R2_ARCH_BUDGET=3000`）、`shared_context`、`symbol_context[:3000]`、`round1`（**`_R2_R1_BUDGET=1200`**）、`diff` 分块 **`_R2_DIFF_CHUNK=4000`**；对 R1 条目 KEEP/MODIFY/DISCARD，并产出新 issue。  
- **LLM 次数**：每文件 **1 + Agent 内多轮**（常见约数轮至十余轮量级，依工具调用而定）。

### 4.3 Round 3：按文件的全局/规则视角

- **传入 `prev_issues` 实际为 `r1+r2`**，再按文件过滤。  
- **每文件 1×JSON LLM**：`arch_doc` **完整**（不截断）、`review_spec` 经 `_lookup_relevant_rules(fdiff, max_detail=8)`、`pr_summary[:400]`、`prev_json` 最多 30 条且总长约 1000、`diff` 单文件 **4000**；偏重 **design / maintainability**。

### 4.4 Round 4：设计文档 + 架构师评审（合并）

- **优先 1×JSON LLM**：单次输出 `{"pr_design_doc": "...", "issues": [...]}`（9 章设计文档 + 10 维架构师 issue 列表 + Reuse Check）；`arch`/`diff`/`prev_issues` 按 **约 80k 字符** 总预算裁剪。  
- 解析失败时 **fallback**：先 `_round4_generate_pr_doc`（文本），再 `_round4_architect_review`（JSON issues）。  
- **`pr_design_doc`** 与 **`r4`**（issues）分别写入 checkpoint；`runner.review()` 返回的 `pr_design_doc` 来自 **`pr_design_doc`** 键。

### 4.5 Round 5（FINAL）：合并去重

1. **`_compress_existing_comments`**：已有 PR 评论 `body>200` 则批量 **1×LLM** 压成一句（≤20 词）；否则截断 200。  
2. **`_compress_new_issues`**：`problem+suggestion>300` 则批量压缩；否则 `problem[:120]`。  
3. **1×JSON LLM** 去重合并：新 issue（带 `source`: r1/r2/r3/r4）与已有评论对比，同 path+line **r2 优先于 r1**，去掉已被人工评论覆盖的项，合并同根因，并按 severity 排序。  
4. 若 LLM 返回空，fallback 按 **critical > medium > normal** 排序。  
5. 每条最终 issue 带 **`_review_version: 2`**；旧 checkpoint 无此标记会触发 **final 整表重算**。

### 4.6 `_run_five_rounds` 与 R1 透传

- 合并进入 R5：`r1_passthrough`（规则见代码：R2 覆盖文件中丢弃被 R2 处理过的同 path:line）+ `r2` + `r3` + `r4`，并 `_tag(..., source=...)`。

---

## 5. 发帖（`poster.py`）

- **`_fetch_existing_pr_comments`**：`list_review_comments`，规范化 `body/path/line`。  
- **`_post_review_comments`**：先 **`submit_review`**（`commit_id=head_sha`，`event=COMMENT`，附 `review_body` + 批量行评）；失败则逐条 **`create_review_comment`**。  
- 无 `path`/`line` 的 issue **不会**作为行评发出。

---

## 6. 缓存与断点（`checkpoint.py` + `runner.py`）

### 6.1 两套存储

| 类型 | 路径 | 内容 |
|------|------|------|
| **PR checkpoint** | `{pr_dir}/checkpoint.json` | `diff_text`、`pr_summary`、`r1_hunk_*`、`r2_file_*`、`r2_disc_*`、`r3`、`pr_design_doc`、`r4`、`final`、`clone_dir`、`_stage_done_*`、`_invalidated_from` 等 |
| **Repo 级 arch/spec** | `~/.lazyllm/review/cache/{safe_repo}/arch.json` / `spec.json` | `arch_doc`、`arch_outline`、`arch_section_*`、`public_api_catalog`、`agent_instructions`、`review_spec` 等 |

`_load_cache` / `_save_cache` / `_save_cache_multi` 用于 **arch.json / spec.json**；`_ReviewCheckpoint` 用于 **checkpoint.json**。

### 6.2 `ReviewStage` 顺序

`CLONE → ARCH → SPEC → PR_SUMMARY → R1 → R2 → R3 → R4 → FINAL`

### 6.3 `resume_from` 与 `_invalidated_from`（只标记、不删数据）

- 构造 checkpoint 时若传入 `resume_from`：写入 **`_invalidated_from`** 为该 stage 的 value，**不** `pop` 旧字段。  
- **`get(key)`**：若 key 映射到的 stage 的 **index ≥ invalidation 起点**，则返回 `None`（逻辑上视为未缓存）。  
- **`should_use_cache(stage)`**：无 `resume_from` 时若存在 `_invalidated_from`，则仅 **`stage < invalidated`** 为 True；仍有 `resume_from` 参数时则用 **`stage < resume_from`**。  
- **`mark_stage_done(stage)`**：若 `stage.index() >= invalidated.index()`，会 **清除** `_invalidated_from` 并清空内存中的 `resume_from`，表示从该阶段起已重新跑通。

### 6.4 `clear_checkpoint=True`

删除 checkpoint 文件并 **删除整个 `pr_dir`**（与仅删 `clone/` 的成功收尾不同）。

### 6.5 成功结束后的目录

- 删除 **`{pr_dir}/clone/`** 释放磁盘；**保留** `checkpoint.json` 供下次续跑。

---

## 7. LLM 调用与工具小结

- **JSON 输出**：R1/R2 issue 提取、R3、R4、R5 去重、架构 outline、规则提取、Public API 文件列表等 → `_safe_llm_call`。  
- **文本输出**：各 arch section、section 摘要、PR 摘要、R4a 设计文档等 → `_safe_llm_call_text`。  
- **Agent**：仅 **Round 2** 使用 `ReactAgent`，工具为 scoped 读写/搜索/shell + **`analyze_symbol`**（内部可再调 LLM）。  
- **重试**：`utils` 中间层对 JSON 解析失败、限流等有重试与 `json_repair` 兜底。

---

## 8. 文件索引

| 模块 | 职责 |
|------|------|
| `runner.py` | 编排、diff、预分析、五轮、清理 clone、发帖、返回 `pr_design_doc` |
| `pre_analysis.py` | clone、架构、规则、PR 摘要、Public API Catalog、arch 按文件裁剪、agent 工具 |
| `rounds.py` | R1–R5、批处理与并发、Round 2 Agent、R4a/R4、合并去重 |
| `checkpoint.py` | PR 断点、stage、invalidation 标记、`get` 屏蔽逻辑 |
| `utils.py` | LLM 封装、diff 解析、评论规范化、review body |
| `poster.py` | 拉已有评论、提交 review / 行评 |

---

## 9. 设计评审：不合理之处（信息截断与一致性）

以下对照当前实现（`lazyllm/tools/git/review/*.py`）归纳：**多为硬截断（truncation），而非面向语义的压缩**，容易在「参考信息不足」或「前后轮不一致」时误判。

### 9.1 架构分析链路

| 现象 | 代码位置 | 风险 |
|------|----------|------|
| 快照预算 6000，但 outline 只用 `snapshot[:4000]` | `pre_analysis.py` `_arch_generate_outline` | 约 1/3 结构化快照未参与 outline，目录/依赖信息可能偏废。 |
| 每节 `_summarize_section` 只摘要 `content[:800]` | `pre_analysis.py` | 长 section 的摘要丢失后半段重点，后续 section 的「已文档化摘要」链式误差累积。 |
| Public API：目录树 `[:4000]` | `_build_public_api_catalog` | 深仓库前几屏占满预算，深层 `utils` 可能进不了 LLM 视野。 |

### 9.2 PR 预摘要与全链路 diff

| 现象 | 代码位置 | 风险 |
|------|----------|------|
| `pr_body[:800]`、`diff[:5000]` | `_pre_round_pr_summary` | 大 PR 的描述与 diff 头部占满窗口，**尾部变更**在预摘要阶段即不可见。 |
| `runner` 层 `max_diff_chars` 截断整份 diff | `runner.py` | 超大 PR 后续 hunk 根本不会进入 R1–R5（文档需与默认参数一并说明）。 |

### 9.3 Round 1

| 现象 | 代码位置 | 风险 |
|------|----------|------|
| `review_spec[:600]`、`pr_summary[:600]` | `rounds.py` `_round1_hunk_analysis` | 规则全文与 PR 摘要在首轮即被压成极短片段，**与 R3「全量 arch + 规则查找」不对称**，R1 可能漏掉依赖规则卡的结论。 |
| 每文件共用一份 `_extract_arch_for_file(..., 3000)` | `_r1_task_batch` | 合理，但若该文件与高分 section 无关，仍可能挤掉 Public API 等块（取决于打分）。 |

### 9.4 Round 2（问题最集中）

| 现象 | 代码位置 | 风险 |
|------|----------|------|
| Agent 上下文只用 **`fdiff[:4000]`** | `_r2_process_file_chunk` | 单文件 diff 超过 4k 时，**后半文件变更根本不会进入「探索上下文」的 Agent**，但后续仍按 chunk 做 issue 提取。 |
| Agent prompt 内再截断 **`diff_chunk[:2000]`**，`force_summarize_context` 仅 **`[:300]`** | `_r2_build_file_context` | 同一文件 diff 被三重截断；300 字符摘要几乎只能承载路径级提示，**语义信息量极低**。 |
| 每文件 **只跑 1 次** Agent，再对多 chunk 复用同一 `symbol_context` | `_r2_process_file_chunk` | 若关键符号/调用关系落在 **4000 字符之后**的 chunk，Agent 未探索过，R2 仍用旧上下文做提取，**跨 chunk 漏检风险高**。 |
| `r1_text` 上限 **1200** 字符 | `_r2_extract_issues` | R1 条目多时直接被截断，LLM 无法对完整 R1 列表做 KEEP/MODIFY/DISCARD。 |

### 9.5 Round 3

| 现象 | 代码位置 | 风险 |
|------|----------|------|
| `prev_json` 30 条且总长 **1000** | `rounds.py` `_round3_analyze_file` | 平均每条约 33 字符，**几乎只能看见路径:行号+极短 problem**，R1/R2 的 suggestion 基本丢失，与「全局分析」目标不匹配。 |
| 单文件 diff **`fdiff[:4000]`** | 同上 | 大文件后半段不参与 R3。 |

### 9.6 Round 4a / Round 4

| 现象 | 代码位置 | 风险 |
|------|----------|------|
| R4a：`arch_doc[:2000]` | `_round4_generate_pr_doc` | 完整 `arch_doc`（含多节 + Public API JSON）被压到 2k，**设计文档推断严重缺参考**。 |
| R4：`pr_design_doc[:3000]`、`diff[:12000]` | `_round4_architect_review` | 相对合理，但若与 R4a 的「缺 arch」叠加，架构评审仍可能建立在残缺设计上。 |

### 9.7 Round 5 与历史规则

| 现象 | 代码位置 | 风险 |
|------|----------|------|
| 合并规则时 `rules_json[:8000]` | `pre_analysis.py` 合并路径 | 规则过多时合并输入被砍，**合并结果可能不完整**。 |
| `_lookup_relevant_rules` 只用 diff **前 200 行**提关键词 | `pre_analysis.py` | 变更集中在 diff 尾部时，**规则标题匹配可能失准**。 |

### 9.8 与「检索增强」的关系

框架内可用 **`lazyllm.tools.rag.retriever.ContextRetriever`**（对内存长文本做 BM25/向量检索，`TempPathGenerator` + LRU 缓存），当前 review **管线未接入**。在以下场景用「按 query 从长 arch / 长 diff / review_spec 中取 top-k 片段」替代单纯头部截断，通常比继续缩小 `[:N]` 更合理：

- 按 **当前文件路径 + hunk 文本** 从 `arch_doc` 检索相关段落，替代仅靠 `_extract_arch_for_file` 静态打分。  
- 对 **超长 diff** 按文件或符号检索相关 hunk，再喂给 R2 Agent / R4。  
- 对 **review_spec** 按 diff 关键词检索规则全文，而非仅 title 关键字 + 前 200 行。

---

## 10. 设计评审：可优化方向

### 10.1 减少 LLM 调用次数

| 思路 | 说明 |
|------|------|
| 架构 section **批量填充** | 当前每节 1 次填充 + 1 次摘要；可改为「一次输出多节草稿」或「仅对超长节摘要」，减少 2N 中的常数因子。 |
| **合并 R4a 与 R4** | 若延迟允许，可用一次结构化输出同时产出「设计文档段落 + 问题列表」（需严格 JSON schema），减少 1 次往返（需评估输出长度与稳定性）。 |
| R5 **压缩批大小** | 已有批量压缩；若 existing/new  issue 数量稳定，可合并为单次「压缩+去重」提示词（当前已是先压再 dedup，可再减少一轮空转）。 |

### 10.2 用缓存减少重复模型调用（在已有 checkpoint 之外）

| 思路 | 说明 |
|------|------|
| **`r2_shared_context` 已缓存** | 同 PR 重跑时跳过静态分析；确保 `resume_from` 不误清该 key（当前逻辑依赖 checkpoint）。 |
| **按 `(repo, arch_doc hash)` 缓存 Public API 正则扫描结果** | LLM 文件列表不变时，clone 未变可跳过 `_extract_public_symbols` 的重扫。 |
| **Symbol 摘要缓存** | `analyze_symbol` 已对 `symbol_cache` 做进程内缓存；跨 PR 若 clone 路径与 commit 相同可考虑落盘 key（需谨慎失效条件）。 |

### 10.3 优先修复的高收益项（实施顺序建议）

1. **R2**：对 Agent 输入改为「每 chunk 各跑一次轻量探索」或「全量 fdiff 分块检索 + 合并上下文」，至少去掉 **`fdiff[:4000]` 与 `[:2000]/[:300]` 叠加截断**中的最严一层。  
2. **R3 `prev_json`**：改为按条数限制 + 每条约 **200–400 字符** 的摘要（或先 LLM 批量压缩 R1+R2 再喂 R3），避免 1000 字符硬切。  
3. **R4a `arch_doc[:2000]`**：改为与 R1 一致的 `_extract_arch_for_file` 或 ContextRetriever 取 top-k，与全文长度解耦。  
4. **预摘要 diff**：对大 PR 至少记录「已截断」并在后续轮次用完整 diff（或分块）补全关键阶段。

---

*文档版本与代码同步说明：若后续调整 budget 或 stage 枚举，请以源码常量及 `ReviewStage.ordered()` 为准。本节（9–10）为基于源码的静态评审，实施改动前请再跑回归与成本评估。*
