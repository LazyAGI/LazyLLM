# LazyLLM Git PR Review 完整流程说明

本文档描述 `lazyllm.tools.git.review` 模块的完整 PR review 端到端流程。入口为 `runner.py` 中的 `review()` 函数，依次经过预分析（架构解析、历史规范提取）、静态 Lint 分析、六轮 LLM 分析（R1 hunk 审查 → R2a PR 设计文档 → R2 架构师评审 → R3 Agent 验证 → RMod 修改必要性分析 → R4 合并去重）、与主链并行的场景推断（RScene）和调用链 bug 分析（RChain）、测试覆盖检查（RCov），最终将 `final_comments + rchain_issues + rcov_issues` 合并后发布到 GitHub / GitLab / Gitee / GitCode。

源码目录：`lazyllm/tools/git/review/`（`runner.py`、`pre_analysis.py`、`rounds.py`、`coverage_checker.py`、`constants.py`、`checkpoint.py`、`utils.py`、`poster.py`、`lint_runner.py`）。

---

## 1. 整体架构一览

### 1.1 模块职责

| 模块 | 职责 |
|------|------|
| `runner.py` | **主入口**：编排所有子模块；diff 拉取与截断；策略决策（`_ReviewStrategy`）；meta warning 生成；清理 clone 目录；发布评论 |
| `pre_analysis.py` | 仓库 clone；架构文档生成（`analyze_repo_architecture`）；本地模式 arch 分析；历史 review 规范提取（`analyze_historical_reviews`）；PR 摘要；R3 Agent 工具集构建 |
| `rounds.py` | **Review 轮次核心**：R1 hunk 级分析、R2a PR 设计文档、R2 架构师评审、R3 ReactAgent 验证、RMod 修改必要性分析、R4 合并去重 |
| `coverage_checker.py` | **RCov**：测试覆盖检查——识别可测符号、按依赖分组、grep 测试文件、评估覆盖缺口（LLM，并行） |
| `lint_runner.py` | 静态 Lint 分析（不调用 LLM），结果直接注入 R4 |
| `checkpoint.py` | 断点续传：PR 级 checkpoint；阶段枚举 `ReviewStage`；失效控制（`resume_from`） |
| `constants.py` | 上下文预算常量；`BudgetManager`；issue 密度控制；diff 启发式压缩 |
| `utils.py` | LLM 调用封装（重试/QPS）；diff 解析；JSON 解析与修复；进度报告 |
| `poster.py` | 拉取已有 PR 评论；提交平台 review（批量 `submit_review` + 逐条 fallback） |

### 1.2 端到端流程图

```
review(pr_number, ...)
  └─ Checkpoint 初始化
  └─ 获取 PR 信息 & diff
  └─ _compute_diff_stats + _decide_review_strategy
       ├─ diff 超出 max_diff_chars → 按文件边界截断 → 生成 meta warning issue
       └─ _run_pre_analysis → arch_doc + review_spec + clone_dir + agent_instructions
  └─ _pre_round_pr_summary → pr_summary
  └─ _fetch_existing_pr_comments
  └─ _run_lint_analysis → lint_issues
  ├─ _run_four_rounds (R1 → R2a → R2 → R3 → RMod → R4) → final_comments   ┐ 并行
  └─ _run_rscene_rchain                                                      │
       ├─ infer_usage_scenarios (RScene) → usage_scenarios                   │
       └─ _rscenario_call_chain (RChain) → rchain_issues                    ┘
  └─ _run_coverage_check (RCov，独立线程，动态超时) → rcov_issues
  └─ 合并：final_comments + rchain_issues + rcov_issues → all_comments
  └─ 清理 clone 子目录
  └─ _post_review_comments → 平台
```

### 1.3 阶段顺序总览

| # | 阶段 | 主要产物 |
|---|------|----------|
| 0 | Checkpoint 初始化 | `pr_dir`、`checkpoint.json`、`resume_from` 软失效控制 |
| 1 | Diff 拉取与截断 | `diff_text`（按文件边界截断至 `max_diff_chars`）、`hunks` |
| 2 | 策略决策 | `_DiffStats`、`_ReviewStrategy`（R3 参数自适应） |
| 3 | Meta Warning | 截断时插入 `source='meta'` 的 issue |
| 4 | 预分析 | `arch_doc`、`review_spec`、`clone_dir`、`agent_instructions` |
| 5 | PR 摘要 | `pr_summary` |
| 6 | 已有评论 | `existing_comments`（供 R4 去重） |
| 7 | Lint 分析 | `lint_issues`（直接注入 R4，不过 LLM） |
| 8 | Round 1 | hunk 级静态审查 issue 列表 |
| 9 | Round 2a | PR 设计文档（结构化，9 节） |
| 10 | Round 2 | 架构师视角全局设计评审 issue 列表 |
| 11 | Round 3 | ReactAgent 验证：验证 R1+R2 issue + 发现新跨文件问题 |
| 12 | RMod | ReactAgent 修改必要性分析：逐文件标记不必要的改动 |
| 13 | Round 4 | 确定性去重 + LLM 合并 + Lint 融合 → `final_comments` |
| 14 | RScene | 场景推断（与 R1-RMod 并行）：推断 2-4 个典型使用场景 → `usage_scenarios` |
| 15 | RChain | 调用链 bug 分析（RScene 完成后）：场景驱动的 bug + 可用性 issue → `rchain_issues` |
| 16 | RCov | 测试覆盖检查：识别未覆盖符号，评估缺口 → `rcov_issues`（绕过 R4 去重直接追加） |
| 17 | 发布 | 合并 `final_comments + rchain_issues + rcov_issues`，提交平台 review；更新 checkpoint `UPLOAD` 阶段 |
| 18 | 清理 | 删除 `{pr_dir}/clone/`；保留 `checkpoint.json` |

---

## 2. 入口与策略决策（`runner.py`）

### 2.1 `review()` 函数签名

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

- `language`：review 评论输出语言，`'cn'` 为简体中文，`'en'` 为英文。
- `post_to_github`：`False` 时只返回结果字典，不发布评论，适合调试。
- `arch_cache_path`：架构文档缓存目录，跨 PR 复用，避免重复 clone 分析。
- `resume_from`：指定从某个 `ReviewStage` 重新开始（软失效，不删除 checkpoint）。
- `clear_checkpoint`：强制清除 checkpoint，从头开始。

### 2.2 策略决策（`_ReviewStrategy`）

根据 diff 规模自动调整 R3 的参数，避免大 PR 消耗过多 LLM 调用：

| PR 规模 | 判断条件 | `large_file_threshold` | `max_files_for_r3` | `max_chunks_per_file` |
|---------|---------|----------------------|-------------------|----------------------|
| 大 PR | >3000 行 或 >50 文件 | 100 字符 | 10 | 2 |
| 中 PR | >1000 行 或 >20 文件 | 150 字符 | 15 | 2 |
| 小 PR | 其余 | 200 字符（默认） | 20（默认） | 3（默认） |

### 2.3 Diff 截断策略

当 diff 超过 `max_diff_chars`（默认 120000）时，按文件边界截断（不在 hunk 中间切断），并在结果中插入一条 `source='meta'` 的 warning issue，告知用户哪些文件被跳过。

---

## 3. 预分析（`pre_analysis.py`）

预分析在所有 review 轮次之前运行，为后续 LLM 提供高质量的仓库上下文。它分为四个子阶段，均支持 checkpoint 缓存。

### 3.1 仓库 Clone（`_fetch_repo_code`）

- 使用 `git clone --single-branch --depth 1`（无 pin_sha 时）或完整 clone（有 pin_sha 时）将仓库拉取到本地。
- 若目标目录已存在完整 clone，优先复用并尝试 `git pull` 更新，避免重复下载。
- 若指定了 `pin_sha`（PR head commit），通过 `_pin_clone_to_sha` 将 clone 切换到精确 commit，确保 review 的代码与 PR 一致。
- clone 目录保存在 `arch_cache_path/{owner_repo}/clone/` 下，跨 PR 复用。
- **本地模式**（`review-local`）：跳过 clone。若 `arch_doc` 为空，通过 `_run_local_arch_analysis` 直接对本地 repo 路径调用 `analyze_repo_architecture`，无需网络 clone 也能获得架构上下文。

### 3.2 架构文档生成（`analyze_repo_architecture`）

架构文档是整个 review 流程的"地图"，让 LLM 在不读取全量代码的情况下理解仓库结构。

**生成流程（五步）：**

1. **结构化快照**（`_collect_structured_snapshot`）：扫描 clone 目录，收集目录树（2 层）、顶层 `__init__.py`、子包 `__init__.py`、核心模块文件（`module.py`、`flow.py` 等）、依赖配置文件（`pyproject.toml`、`requirements.txt` 等）、`AGENTS.md` 等，拼接为结构化快照文本，受 `_ARCH_SNAPSHOT_BUDGET` 字符预算约束。

2. **DeepWiki 集成**（可选）：若安装了 `mcp` 包，通过 DeepWiki MCP 服务（`_fetch_deepwiki_summary`）拉取仓库的预索引架构摘要，注入快照末尾作为背景参考。始终使用 `base_repo`（上游仓库）而非 fork 地址，避免 fork 无数据。DeepWiki 数据可能有 1-3 个月延迟，注入时附带 stale 提示。

3. **大纲生成**（`_arch_generate_outline`）：将快照喂给 LLM，生成 N 个章节的大纲（有 `AGENTS.md` 时最多 `_ARCH_OUTLINE_MAX_SECTIONS_WITH_AGENT` 节，否则 `_ARCH_OUTLINE_MAX_SECTIONS` 节）。大纲固定包含两个末尾章节：
   - 倒数第二节：**Concurrency & Multi-User Conventions**（线程安全、ContextVar、锁约定）
   - 最后一节：**Testing & Examples**（测试文件位置、示例脚本）
   大纲缓存到 `arch_outline`，避免重复生成。

4. **批量填充章节**（`_arch_fill_all_sections`）：对每个章节，先用 `_arch_collect_snippets_for_section` 从仓库中 grep 相关代码片段，再将多个章节打包为一个 LLM 调用（`_arch_fill_batch_llm`），减少 LLM 调用次数。每章节内容上限 3500 字符，缓存到独立 key（`arch_section_{title}`）。

5. **Public API Catalog**（`_build_public_api_catalog`）：LLM 识别仓库的公开 API 文件列表，再通过正则提取各文件的公开符号（函数、类、常量），生成 JSON 格式的 API 目录，追加到架构文档末尾。支持 Python、Go、TypeScript、Java、Rust、C++ 等多语言。

**最终产物**：架构文档 + Public API Catalog，同时生成 `arch_index`（章节摘要索引）和 `arch_symbol_index`（符号→章节映射），供 R1/R2/R3 按需检索。所有产物缓存到 `arch_cache_path/{owner_repo}/`，同一仓库的后续 PR 直接复用。

**AGENTS.md 支持**：若仓库根目录存在 `AGENTS.md`，其内容作为 `agent_instructions` 注入所有 LLM prompt，让项目维护者可以自定义 review 规则（如"忽略 tests/ 目录的命名风格"）。

### 3.3 历史 Review 规范提取（`analyze_historical_reviews`）

从仓库历史 PR 的 review 评论中提炼出项目特有的代码规范，形成 `review_spec`。

**工作原理：**

1. 拉取最近 N 条已合并 PR（默认 30 条）的 review 评论。
2. 过滤掉机器人评论、过短评论（<20 字符）、纯表情评论。
3. 将评论按文件类型分组，喂给 LLM 提炼规范：
   - 项目特有的命名约定
   - 禁止的模式（如"不允许直接 print，必须用 lazyllm.LOG"）
   - 必须遵守的架构约束
   - 测试要求
4. 规范缓存到 `arch_cache_path/{owner_repo}/review_spec.json`，跨 PR 复用。

这一步使 review 能够"记住"项目历史上反复出现的问题，避免重复提出已知规范。

### 3.4 PR 摘要生成（`_pre_round_pr_summary`）

在四轮 review 开始前，先用一次 LLM 调用生成 PR 摘要（`pr_summary`），包含：
- PR 的核心目的（新功能 / bugfix / 重构）
- 主要改动模块
- 潜在影响范围

PR 摘要作为上下文注入 R2、R3 的 prompt，帮助 LLM 从"PR 意图"角度评审，而不是孤立地看每个 hunk。

### 3.5 R3 Agent 工具集（`_build_scoped_agent_tools_with_cache`）

为 R3 的 ReactAgent 构建一套文件系统工具，所有工具都限定在 clone 目录内：

| 工具 | 功能 |
|------|------|
| `read_file_lines` | 读取指定文件的指定行范围 |
| `read_file_skeleton_scoped` | 读取文件骨架（类名、函数签名，不含实现） |
| `grep_symbol` | 在仓库中搜索符号定义或引用 |
| `list_directory` | 列出目录内容 |

工具调用有步数上限（默认 15 步），防止 Agent 无限探索消耗预算。

---

## 4. 六轮分析（`rounds.py`）

六轮分析是 review 的核心，每轮有明确的职责分工，形成"广度扫描 → 设计文档 → 架构评审 → 深度验证 → 修改必要性 → 合并去重"的递进结构。与此同时，**RScene**（场景推断）和 **RChain**（调用链 bug 分析）与主链并行运行，最终结果在 R4 之后合并。

---

### 4.1 Round 1：Hunk 级静态审查

**目标**：对每个 diff hunk 进行细粒度的代码质量审查，发现具体的 bug、逻辑错误和安全问题。

**工作原理：**

1. 将 diff 按文件和 hunk 分割，每个 hunk 附带上下文（前后各 N 行）。
2. 对每个 hunk 独立调用 LLM，prompt 包含：
   - 完整 diff hunk（带行号标注）
   - 文件级上下文（函数签名、类定义）
   - 架构文档摘要（该文件相关部分）
   - PR 摘要
   - 历史 review 规范
3. 并发处理多个 hunk（`ThreadPoolExecutor`），受 `TOTAL_CALL_BUDGET=60` 约束。

**检查项（`_R1_STRICT_RULES` + `_R1_REVIEW_CHECKLIST`）：**

- **Bug 类**：空指针/越界访问、条件判断错误、循环边界错误、资源泄漏（文件/连接未关闭）
- **逻辑类**：函数行为与名称不符、返回值未检查、异常被静默吞掉
- **安全类**：SQL 注入、路径遍历、硬编码密钥、不安全的反序列化
- **并发类**：竞态条件、锁使用不当、共享状态未保护
- **配置类**：配置项放错位置（应在 `configs.py` 而非 `tracing/`）、可选依赖被加入必选
- **维护性**：重构后遗留的孤儿 helper/常量（原来只被删除代码使用的符号）

**严格排除项**（避免噪音）：
- 未改动行的已有代码问题
- lint 工具已覆盖的问题（未使用 import、行长度、复杂度指标）
- 纯风格问题（除非违反项目规范）

**issue 密度控制**：每 100 行有效 diff 最多输出 5 个 issue，防止大 PR 产生过多噪音。

**输出格式**（每个 issue）：

```json
{
  "path": "lazyllm/tools/git/review/rounds.py",
  "line": 42,
  "severity": "critical|high|medium|normal",
  "bug_category": "bug|security|performance|maintainability|design|style",
  "title": "简短标题",
  "description": "问题描述",
  "suggestion": "修复建议（含代码示例）",
  "source": "r1"
}
```

---

### 4.2 Round 2a：PR 设计文档生成

**目标**：在架构师评审（R2）之前，先生成一份结构化的 PR 设计文档，作为 R2 的输入上下文。

**工作原理：**

对完整 diff + PR 标题/描述 + 架构文档，用一次 LLM 调用生成包含 9 个章节的设计文档：

| 章节 | 内容 |
|------|------|
| 1. Background & Problem Definition | PR 解决的问题；在现有架构中的位置；新功能/bugfix/重构 |
| 2. Design Goals | 期望达到的效果；设计约束（性能/可扩展性/一致性） |
| 3. Design Approach | 核心思路；为何这样设计；是否有备选方案；是否符合架构分层 |
| 4. Module Impact Analysis | 修改/新增的模块；职责变化；新引入的依赖 |
| 5. API Design | 新增/修改的接口；输入输出；与现有 API 风格一致性 |
| 6. Usage Example | 典型调用示例；对用户使用方式的影响 |
| 7. Compatibility & Impact Scope | 是否影响已有功能；是否为 breaking change |
| 8. Risks & Edge Cases | 潜在问题；未覆盖场景；隐含假设 |
| 9. Extensibility Analysis | 后续类似需求的扩展难度；设计演进空间 |

设计文档保存在 checkpoint 中（`pr_design_doc`），也作为最终 review 结果的一部分返回。

---

### 4.3 Round 2：架构师设计评审

**目标**：从全局架构视角评审整个 PR，发现 R1 无法发现的设计层面问题。

**工作原理：**

以完整 diff + PR 设计文档（R2a 产物）+ 架构文档 + PR 摘要为输入，用一次 LLM 调用进行全局评审。

**11 个评估维度（`_ROUND2_ARCHITECT_PROMPT_TMPL`）：**

1. **Module Responsibility**：新代码是否放在了正确的模块？是否有逻辑被分散到不该放的地方？是否应该放在另一个模块？

2. **Layering & Dependencies**：是否遵守了现有的层次边界？是否引入了循环依赖？是否跨层直接访问（如 UI 层直接操作数据库）？

3. **API Design**：新增接口是否简洁、稳定、易用？参数是否过多？是否暴露了不必要的内部细节？

4. **Consistency（一致性）**：
   - 同模块的类/函数是否遵循相同的接口模式？
   - 若新增了与现有类同类型的类（如新的存储后端、新的模型供应商），是否共享公共基类或 Protocol？
   - 相似类是否遵循相同的构造模式（`__init__` 参数顺序、factory/client 入口）？
   - 相似类是否实现了相同的关键方法集合（参数名、顺序、返回类型一致）？
   - 若项目使用 `__call__` + `forward` 等分发模式，新类是否遵循同样的模式？

5. **Abstraction & Reuse（抽象与复用）**：
   - 是否有逻辑在其他地方已经存在？
   - **关键检查**：若系统此前只有一个概念的实现（如一个存储后端、一个模型供应商），而此 PR 新增了第二个，是否存在公共基类/Protocol/ABC？若不存在，这是设计问题——两个实现应统一在共同抽象下。
   - 新抽象的层次是否合适（不过度泛化，也不过度具体）？

6. **Complexity**：是否引入了不必要的复杂度？是否有可以用单行表达式替代的多步流程？

7. **Extensibility**：未来类似需求是否容易扩展？是否有硬编码的假设会阻碍扩展？

8. **Coupling**：模块间耦合是否增加？是否可以通过接口/事件/依赖注入降低耦合？

9. **Testability**：新代码是否容易测试？是否有全局状态或硬依赖使测试困难？

10. **Overall Design Verdict**：是否有更简单/更一致的替代设计？最重要的一个架构改进点是什么？

11. **Naming & Semantic Clarity（命名与语义清晰度）**：
    - 方法/函数名是否自解释且简洁？
    - 方法名是否避免了冗余地包含类名（如应用 `ClassA.get_instance()` 而非 `ClassA.get_classA_instance()`）？
    - 相似方法在兄弟类中的参数名是否一致？
    - 若项目提供语法糖（`__or__`、`__ror__`、`__getitem__` 等运算符重载），其语义是否与主流惯例一致（bash 管道 `|`、Python 切片 `[]` 等）？若可能误导用户则标记。

---

### 4.4 Round 3：ReactAgent 深度验证

**目标**：R1 和 R2 只能看到 diff 本身，R3 通过 ReactAgent 主动探索仓库，验证已有 issue 的准确性，并发现需要跨文件上下文才能发现的新问题。

**工作原理（两阶段）：**

#### 阶段一：Context Collect（上下文收集）

对每个需要验证的 diff 文件，ReactAgent 使用工具集主动探索仓库：

1. **读取文件骨架**：先用 `read_file_skeleton_scoped` 了解文件结构，再按需读取具体行。
2. **符号追踪**：用 `grep_symbol` 找到被修改符号的所有调用方，评估影响范围。
3. **基类/接口检查**：读取基类定义，确认子类是否满足接口契约。
4. **框架机制识别**：检查是否使用了 metaclass、`__init_subclass__`、registry 等自动注册机制（避免将自动注册的子类误报为"死代码"）。
5. **兄弟类探索**（新增）：当 diff 新增一个类时，搜索同基类或同目录下的兄弟类，对比其构造模式、关键方法签名、`__call__`/`forward` 分发模式。
6. **运算符语义验证**（新增）：若 diff 修改了 `__or__`、`__getitem__` 等 dunder 方法，grep 其使用模式，验证语义是否与主流惯例一致。

Context Collect 的输出是结构化 JSON：

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

#### 阶段二：Issue Extract（问题提取）

以 Context Collect 收集的上下文 + R1/R2 已有 issue 为输入，执行两类任务：

**任务 1：验证已有 issue**
- 确认 issue 是否真实存在（排除误报）
- 补充跨文件证据（如"调用方 X 会受到影响"）
- 若 issue 在仓库其他地方已有相同模式，标记为"项目一贯做法，非问题"

**任务 2：发现新问题**（需要跨文件上下文才能发现）：
- **调用方破坏**：修改了函数签名，但调用方未同步更新
- **孤儿符号**：重构后遗留的 helper/常量，原来只被删除代码使用
- **基类契约违反**：子类未实现基类的抽象方法，或覆盖了不应覆盖的方法
- **基类抽象缺口**（新增）：若 diff 新增了某概念的第二个实现，但不存在公共基类，报告设计问题
- **兄弟类一致性**（新增）：若兄弟类（同基类或同角色）的构造签名、关键方法签名、分发模式不一致，报告不一致
- **命名清晰度**（新增）：若新方法/函数名冗余地包含父类/模块名，或不够自解释，报告命名问题
- **语法糖语义**（新增）：若 diff 新增或修改了运算符重载（`__or__`、`__getitem__`、`__lshift__` 等），验证语义是否与主流语言/shell 惯例一致，若可能误导用户则报告

**R3 的规模控制：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `R3_MAX_FILES` | 20 | 最多处理 20 个文件 |
| `R3_MAX_CHUNKS_PER_FILE` | 3 | 每文件最多 3 个 chunk |
| `R3_MAX_CHUNKS_HARD` | 8 | 每文件硬上限 8 个 chunk |
| Agent 步数上限 | 15 步 | 每次 Context Collect 最多 15 次工具调用 |

大文件（diff 超过 `large_file_threshold`）会被跳过 R3 处理，避免消耗过多预算。

---

### 4.5 RMod：修改必要性分析（`_run_rmod_agent_round`）

**目标**：对每个被修改的文件，使用 ReactAgent 判断改动是否有架构层面的合理性——标记不必要的重构、过度设计或违反框架约定的改动。

**工作原理：**

1. `_rmod_collect_file_diffs` 提取逐文件 diff；`_rmod_new_file_paths` 识别新建文件（正确处理 unified diff 中 `--- /dev/null` → `+++ b/` 的顺序）。
2. 对每个文件启动一个 `ReactAgent`，工具集与 R3 相同（`read_file_scoped`、`search_scoped` 等）。Agent 以 PR 设计文档、架构文档和框架约定为上下文。
3. 每个文件有独立超时（`_RMOD_AGENT_TIMEOUT_SECS`），文件间并行处理（最多 `max_workers` 线程）。
4. issue 标记为 `source='rmod'`、`bug_category='design'`。
5. 结果保存到 checkpoint（`ReviewStage.RMOD`），支持断点续传。

**实现要点：**
- 在创建 Agent 前直接在 `tools` 上设置 `execute_in_sandbox = False`，不访问私有属性 `_tools_manager`。
- 使用 `concurrent.futures.TimeoutError`（别名 `FuturesTimeoutError`），而非内置 `TimeoutError`。

---

### 4.6 RScene：使用场景推断（`infer_usage_scenarios`）

**目标**：理解 PR 修改的公开 API，推断 2-4 个典型的端到端使用场景，为 RChain 提供输入。RScene 与主链（R1→R2→R3→RMod）**并行运行**，不阻塞主链。

**工作原理（两步）：**

**Step 1 — 理解功能模块**：ReactAgent 使用工具集主动探索仓库：
- 用 `read_file_scoped` 读取每个被修改文件的完整骨架（类定义、方法签名、关键常量）
- 用 `analyze_symbol` 理解核心类的状态机和数据流
- 用 `search_in_files` 搜索现有测试和示例，了解已有使用模式

**Step 2 — 推断典型使用场景**：基于理解，输出 2-4 个场景，每个场景包含：
- `title`：场景名称
- `description`：场景描述
- `api_sequence`：API 调用顺序（如 `init → configure → run → cleanup`）
- `call_chain`：预期调用链（函数/方法名列表）
- `edge_cases`：需要检查的边界情况

**规模控制**：
- 最多处理 `max_files_for_r3` 个文件（与 R3 共用策略参数）
- 只处理被修改的文件（非新建文件），新建文件无历史行为可推断
- 最多 3 个并行 worker（`_RSCENE_MAX_PARALLEL`）
- 每个 Agent 超时 `_RSCENE_AGENT_TIMEOUT_SECS`（180s），最多重试 `_RSCENE_AGENT_RETRIES`（10）次

**Checkpoint**：场景结果保存到 `ReviewStage.RSCENE`（key: `rscene_all`），断点续传时直接加载。

---

### 4.7 RChain：调用链 Bug 分析（`_rscenario_call_chain`）

**目标**：以 RScene 推断的使用场景为驱动，对每个场景的调用链进行深度 bug 分析和 API 可用性评审。RChain 在 RScene 完成后立即启动，与主链并行。

**工作原理（两个任务）：**

**Task A — 调用链 Bug 分析**：
1. 用 `read_file_scoped` / `analyze_symbol` 追踪调用链中每一步
2. 验证调用方与被调用方之间的输入/输出契约是否一致
3. 检查异常传播路径：异常是否被正确捕获或向上传递
4. 检查资源生命周期：文件/连接/锁是否在所有路径上正确释放
5. 检查并发安全：调用链中是否存在竞态条件或共享状态问题

**Task B — API 可用性评审**：
1. 验证 API 序列是否符合直觉（参数顺序、命名、默认值）
2. 检查错误信息是否足够清晰，便于用户调试
3. 检查是否存在容易误用的 API（如参数顺序容易混淆）

**输出**：issue 标记为 `source='rchain'`，Task A 使用 `bug_category` 为 `bug`/`security`/`performance`/`concurrency`/`safety`/`type`，Task B 使用 `design`/`exception`。

**规模控制**：
- 每个场景独立运行一个 ReactAgent，最多 `_RCHAIN_MAX_PARALLEL_SCENARIOS`（3）个并行
- 每个 Agent 超时 `_RCHAIN_AGENT_TIMEOUT_SECS`（240s），最多重试 `_RCHAIN_AGENT_RETRIES`（6）次
- 每个场景的 diff 预算受 `_RCHAIN_FIXED_OVERHEAD` 约束

**Checkpoint**：每个场景的结果独立缓存（key: `rchain_scene_{idx}_{title}`），整体结果缓存到 `rchain_all`（`ReviewStage.RCHAIN`）。

**与主链的关系**：RChain 的 issue **绕过 R4 去重**，在 `final_comments` 之后直接追加到 `all_comments`（与 RCov 类似）。这是因为 RChain 的场景驱动视角与 R1/R2/R3 的 diff 视角互补，强制去重会丢失有价值的场景级 bug。

---

### 4.8 Round 4：合并去重

**目标**：将 R1、R2、R3、RMod 的 issue 列表 + lint_issues 合并为最终的、无重复的评论列表。RCov 和 RChain 的 issue 绕过本阶段，在 R4 之后直接追加。

**工作原理（三步）：**

1. **确定性去重**：
   - 相同 `(path, line, bug_category)` 的 issue 只保留 severity 最高的一条
   - 已在 `existing_comments` 中存在的 issue 直接丢弃（避免重复评论）

2. **LLM 语义合并**：
   - 对同一文件内语义相近但行号不同的 issue，用 LLM 判断是否为同一问题的不同表述
   - 若是，合并为一条，保留更详细的描述和建议

3. **Lint 融合**：
   - `lint_issues`（来自 `lint_runner.py`，不经过 LLM）直接追加到最终列表
   - Lint issue 的 `source` 字段为 `'lint'`，在平台评论中有特殊标记

**输出**：`final_comments` 列表，每条包含完整的 `path`、`line`、`severity`、`bug_category`、`title`、`description`、`suggestion`、`source` 字段。

---

---

## 5. RCov：测试覆盖检查（`coverage_checker.py`）

RCov 在 R4 **之后**运行（独立，不经过去重），检查新增或修改的公开符号是否有充分的测试覆盖。

### 5.1 三步流程

**Step 1 — 识别可测符号**（`_rcov_identify_symbols`）：
- 将 diff 裁剪至 `SINGLE_CALL_CONTEXT_BUDGET - 20000` 字符。
- 调用 LLM 提取需要测试覆盖的公开函数/类列表。
- 过滤掉内部辅助函数（`is_internal=True`）。

**Step 1.5 — 按依赖分组**（`_rcov_group_symbols`）：
- 若符号数量 > 1，调用 LLM 将相关符号（如一个类及其关键方法）归为一组，一起评估。
- 分组失败时退化为每个符号单独一组。

**Step 2 — Grep + 评估**（`_rcov_evaluate_groups`）：
- `_find_test_files` 扫描 `clone_dir` 查找测试文件（`test_*.py`、`*_test.py` 等）。
- 对每组，`_build_grep_results` 并行 grep 各符号名。符号名通过 `re.fullmatch` 校验后以 `grep -F` 固定字符串模式搜索，防止正则注入。
- LLM 评估现有测试是否充分；issue 标记为 `source='rcov'`。
- 最多 4 个 worker 并行处理各组；每组有独立超时 `_RCOV_GROUP_TIMEOUT_SECS`（90s）。

### 5.2 超时策略

`runner.py` 中的外层超时根据 diff 大小动态推导：

```
timeout = clamp(3 × 90 + len(diff_text) // 10000 × 30, min=300, max=900)  # 秒
# 注：基础值 270 < min=300，diff 极小时实际取 min=300。
```

diff 越大，超时越长，避免大 PR 误超时，也避免小 PR 无谓等待。实际超时值会以 `INFO` 级别打印，便于排查。

### 5.3 Checkpoint

RCov 结果保存在独立的 `ReviewStage.RCOV` checkpoint key（`rcov_issues`）下。断点续传时直接加载缓存，跳过所有 LLM 调用。

输出指标：
- `rcov_issues_count`：发现的 issue 数量；若 RCov 被跳过（超时/无 clone）则为 `None`。
- `rcov_ran`：布尔值，标识 RCov 是否实际执行。

---

## 6. 静态 Lint 分析（`lint_runner.py`）

Lint 分析在所有 LLM 轮次（R1 开始）之前独立运行，不消耗 LLM 调用预算。

**工作原理：**

1. 从 diff 中提取变更的文件和行号范围。
2. 对 Python 文件运行 `ruff` / `flake8`，对 JS/TS 文件运行 `eslint`（若已安装）。
3. 过滤出仅在 diff 变更行上出现的 lint 错误（不报告未改动行的已有问题）。
4. 将结果转换为统一的 issue 格式（`source='lint'`），注入 R4。

Lint issue 不经过 LLM 验证，直接进入最终结果，因此准确率高、无幻觉风险。

---

## 7. Checkpoint 系统（`checkpoint.py`）

### 7.1 设计目标

- **断点续传**：review 中途失败（网络超时、LLM 限流）后，从上次完成的阶段继续，不重复已完成的 LLM 调用。
- **软失效**：`resume_from` 参数可指定从某个阶段重新开始，而不删除其他阶段的缓存。
- **版本控制**：`_REVIEW_ROUND_VERSION` 递增时，自动失效 R4（合并去重）阶段的缓存，确保 prompt 变更后重新计算最终结果。

### 7.2 阶段枚举（`ReviewStage`）

```
CLONE → ARCH → SPEC → PR_SUMMARY → R1 → R2A → R2 → R3 → RMOD →
RSCENE → RCHAIN → FINAL → RCOV → UPLOAD
```

每个阶段完成后调用 `ckpt.mark_stage_done(stage)`，重启时通过 `ckpt.should_use_cache(stage)` 判断是否跳过。

`_KEY_TO_STAGE` 在类定义时直接初始化（非懒初始化），避免并发场景下的 TOCTOU 竞态。

### 7.3 Head SHA 轮转

若 PR 在 review 过程中被 force-push（head SHA 变化），checkpoint 自动备份并清除所有 review 轮次数据（保留 clone、arch、spec），从 R1 重新开始。

---

## 8. 评论发布（`poster.py`）

### 8.1 发布策略

1. **批量提交**：优先使用平台的 `submit_review` API 一次性提交所有评论（GitHub 的 Pull Request Review）。
2. **逐条 fallback**：若批量提交失败（如评论行号不在 diff 范围内），逐条使用 `create_review_comment` 重试。
3. **速率限制重试**：遇到 429 / 403 限流响应时，指数退避重试（最多 3 次）。
4. **合并顺序**：`all_comments = final_comments + rchain_issues + rcov_issues`，三类 issue 均经过 `_filter_commentable` 过滤后发布。

### 8.2 可评论行过滤（`_filter_commentable`）

平台只允许对 diff 中实际变更的行添加行级评论。`_build_commentable_lines` 解析 diff，构建 `{path: set(lines)}` 映射。`_filter_commentable` 返回**三元组** `(inline, general, dropped)`：

- `inline`：有效行号且在 diff 范围内的 issue → 作为行级 review 评论发布。
- `general`：`line=None` 的 issue（如 RCov 覆盖问题）→ 作为 PR 级评论，附加在 review body 的"General Review Comments"节中。
- `dropped`：行号超出变更范围且无法映射的 issue → 丢弃（避免 GitHub 422 错误）。

### 8.3 评论正文策略

每条评论正文包含以下强制要求声明：

> *新引入的架构问题必须在合并前修复；已有问题必须通过 issue（新建或关联）追踪。风格问题也必须修复；测例缺失必须补全。*

---

## 9. 预算与限流（`constants.py`）

| 常量 | 默认值 | 说明 |
|------|--------|------|
| `SINGLE_CALL_CONTEXT_BUDGET` | 120000 字符 | 单次 LLM 调用的上下文上限 |
| `R1_DIFF_BUDGET` | 95000 字符 | R1 中 diff 内容的上限（预留 25000 给系统 prompt + 架构文档） |
| `TOTAL_CALL_BUDGET` | 60 次 | 整个 review 会话的 LLM 调用总上限 |
| `ISSUE_DENSITY_LINE_BLOCK` | 100 行 | issue 密度控制的块大小 |
| `ISSUE_DENSITY_MAX_PER_BLOCK` | 5 个 | 每 100 行最多输出 5 个 issue |
| `R3_MAX_FILES` | 20 | R3 最多处理的文件数 |
| `R3_MAX_CHUNKS_PER_FILE` | 3 | R3 每文件最多 chunk 数 |
| `R3_MAX_CHUNKS_HARD` | 8 | R3 每文件硬上限 |

`BudgetManager` 追踪已使用的 LLM 调用次数，各轮次在调用前检查剩余预算，超出时跳过非关键步骤。

---

## 10. Issue 字段规范

所有轮次输出的 issue 遵循统一格式：

| 字段 | 类型 | 说明 |
|------|------|------|
| `path` | str | 相对于仓库根目录的文件路径 |
| `line` | int | 问题所在行号（diff 中的新行号） |
| `severity` | str | `critical` / `high` / `medium` / `normal` |
| `bug_category` | str | `bug` / `security` / `performance` / `maintainability` / `design` / `style` |
| `title` | str | 简短标题（≤80 字符） |
| `description` | str | 问题详细描述 |
| `suggestion` | str | 修复建议（含代码示例，使用 markdown 代码块） |
| `source` | str | `r1` / `r2` / `r3` / `rmod` / `rcov` / `lint` / `meta` |

---

## 11. 已知限制

| 限制 | 说明 |
|------|------|
| Diff 截断 | 超大 PR（>120K 字符）只审查前 N 个文件，被截断的文件通过 meta warning 告知 |
| R3 文件上限 | 大 PR 下 R3 最多处理 10 个文件，其余文件只经过 R1/R2 |
| RCov clone 依赖 | RCov 需要 `clone_dir` 来 grep 测试文件；本地模式下无 clone 时，测试文件发现退化为使用本地 repo 路径 |
| 动态引用 | `grep_symbol` 无法追踪运行时动态生成的符号名（如 `getattr(obj, name)`） |
| 跨仓库依赖 | 只分析当前仓库，外部依赖的接口变更无法检测 |
| 语言支持 | 当前 Lint 分析主要支持 Python（ruff/flake8），JS/TS 需要本地安装 eslint |

---

*若后续调整 budget 常量、`ReviewStage.ordered()`、各轮 prompt、检查项或增加新轮次，请同步更新本文档。*
