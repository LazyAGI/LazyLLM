# Fix Progress

## Source: PR #1098 review comments (round >= ID 3136024626)
## Started: 2026-04-24

> 本文件是唯一真相来源。每次修复后立即更新 Status / Notes。
> 去重说明：166 条评论中存在大量重复（同一问题被多轮 review 各提一次），已合并为独立问题，保留最早 ID 作为主 ID，重复 ID 列在 Notes 中。

---

## CRITICAL 类（必须修复后才能合并）

| # | ID | File | Line | 问题摘要 | Verdict | Status | Notes |
|---|-----|------|------|---------|---------|--------|-------|
| C1 | 3136024626 | `git/review/runner.py` | 170 | `pr_dir` 缩进错误移到条件块外，运行时 NameError | Reliable | pending | |
| C2 | 3136024635 | `.github/workflows/main.yml` | 890 | `$GITHUB_TOKEN` 嵌入 clone URL，暴露在进程列表和 .git/config | Reliable | pending | |
| C3 | 3136024639 | `doc_manager.py` | 687 | `all([])` 返回 True，空记录被误判为"全部删除" | Reliable | pending | |
| C4 | 3136024649 | `doc_manager.py` | 1348 | `delete()` cancel/reject 混合循环：先取消部分任务再发现 WORKING 冲突，状态不一致 | Reliable | pending | 重复：3136032692、3136038892 |
| C5 | 3136024658 | `doc_manager.py` | 1682 | `on_task_callback` unbind 路径三个独立事务缺乏原子性 | Reliable | pending | 重复：3136039861 |
| C6 | 3136024667 | `parsing_service/server.py` | 387 | `_upsert_node_groups` 每个 item 独立 session，TOCTOU 竞态 + 部分提交 | Reliable | pending | 与 C7/C8 合并修复 |
| C7 | 3136024678 | `parsing_service/server.py` | 460 | `register_new_node_group` 在 `update_algorithm` session 内嵌套新 session，部分提交风险 | Reliable | pending | 与 C6/C8 合并修复 |
| C8 | 3136024685 | `parsing_service/server.py` | 344 | `_upsert_node_groups` 与 algorithm insert 分属不同事务，失败时 node group 孤立 | Reliable | pending | 与 C6/C7 合并修复 |
| C9 | 3136024695 | `parsing_service/impl.py` | 118 | `skip_ng_ids` 未透传给 `_create_nodes_recursive`，参数完全无效 | Reliable | pending | |
| C10 | 3136024702 | `parsing_service/impl.py` | 303 | skip 路径中 `doc_ids` 为空时 `get_nodes` 可能返回全量数据 | Reliable | pending | |
| C11 | 3136024707 | `doc_service/base.py` | 468 | `DOC_NODE_GROUP_STATUS_TABLE_INFO` 缺少 `(doc_id, kb_id, node_group_id)` 唯一约束 | Reliable | pending | 与 C12 合并修复 |
| C12 | 3136031430 | `doc_manager.py` | 601 | `_upsert_ng_status_pending` 并发重复写入漏洞（无唯一约束保护） | Reliable | pending | 与 C11 合并修复 |
| C13 | 3136024714 | `doc_service/doc_server.py` | 254 | 删除 `algo_id` 字段导致幂等性 key 变化，历史记录全部失效 | Reliable | pending | |
| C14 | 3136024716 | `doc_service/doc_server.py` | 811 | `list_docs` 删除 `algo_id` 参数是 breaking API change，现有客户端收到 422 | Reliable | pending | |
| C15 | 3136024722 | `migrate_collections.py` | 204 | Milvus 部分复制失败后未清理新集合，破坏幂等性 | Reliable | pending | |
| C16 | 3136025832 | `doc_impl.py` | 276 | `existing_sig` 为空时（旧数据）签名检查逻辑矛盾，既不报错也不更新 | Reliable | pending | 重复：3136032530、3136039915 |
| C17 | 3136032544 | `doc_impl.py` | 283 | `LOG.info` + `return` 在条件块外，无条件执行，跳过新注册流程 | Reliable | pending | |
| C18 | 3136031440 | `doc_manager.py` | 992 | `_enqueue_task` 中 `exclusive_ng_ids` 仅从 `extra_message` 取值，upload 路径未传递，DOC_ADD 路径下 ng_ids 丢失 | Reliable | pending | 重复：3136039883 |
| C19 | 3136031442 | `doc_manager.py` | 1376 | `delete()` 中 `all_canceled` 语义错误：任意一个 algo 取消就触发 purge，其他 algo 数据成孤儿 | Reliable | pending | 与 C4 同根因，合并修复 |
| C20 | 3136025759 | `doc_manager.py` | 1681 | `on_task_callback` unbind 路径：先物理删除 parse_state 行再 count，count 永远为 0 | Reliable | pending | 重复：3136038812 |
| C21 | 3136025761 | `doc_manager.py` | 2039 | `unbind_algo` 中 TOCTOU：检查状态到提交任务之间状态可能变化 | Reliable | pending | 重复：3136038732 |
| C22 | 3136032686 | `doc_manager.py` | 1664 | `on_task_callback` DOC_DELETE 路径：`_all_algo_snapshots_deleted` 检查时机与 unbind_algo 路径逻辑矛盾 | Reliable | pending | |
| C23 | 3136038887 | `doc_manager.py` | 1248 | `upload()` 异常处理中用 `algo_ids[0]` 查快照，丢失部分成功入队的 algo 状态 | Reliable | pending | 重复：3136031452 |
| C24 | 3136039853 | `doc_manager.py` | 1661 | `on_task_callback` `_all_algo_snapshots_deleted` 与 purge 之间 TOCTOU，多个并发回调都通过检查导致 purge 多次执行 | Reliable | pending | 重复：3136031553 |
| C25 | 3136038724 | `doc_manager.py` | 177 | `_ensure_kb_algorithm` IntegrityError 后 rollback 但不重查，静默失败 | Reliable | pending | 重复：3136038807 |

---

## MEDIUM 类（应修复后才能合并）

| # | ID | File | Line | 问题摘要 | Verdict | Status | Notes |
|---|-----|------|------|---------|---------|--------|-------|
| M1 | 3136024725 | `doc_manager.py` | 90 | `lazyllm_doc_node_group_status` 缺少索引，全表扫描 | Reliable | pending | |
| M2 | 3136024728 | `migrate_collections.py` | 373 | `--groups []` 被误判为未指定，触发意外的自动发现 | Reliable | pending | |
| M3 | 3136024734 | `migrate_collections.py` | 119 | SQLite 异常后 rollback 但返回非零 count，误导调用方 | Reliable | pending | |
| M4 | 3136024738 | `parsing_service/server.py` | 621 | `get_algo_groups` 返回原始 JSON 字符串而非解析后的列表 | Reliable | pending | |
| M5 | 3136024745 | `doc_impl.py` | 336 | 远程 `register_new_node_group` 后本地 `node_groups` 未同步更新 | Reliable | pending | |
| M6 | 3136024749 | `doc_service/base.py` | 95 | `extra='allow'` 静默接受任意字段，应改为 `extra='ignore'` 并加废弃警告 | Reliable | pending | 与 M7 合并修复 |
| M7 | 3136024753 | `doc_service/base.py` | 108 | validator 内 `from lazyllm import LOG` 违反顶部 import 规则 | Reliable | pending | 与 M6 合并修复 |
| M8 | 3136024757 | `doc_service/doc_server.py` | 928 | `algo_id` 默认值从 `'__default__'` 改为 `None`，静默破坏现有调用方 | Reliable | pending | |
| M9 | 3136024763 | `parsing_service/base.py` | 257 | 重复 group 名称且 signature 不同时直接抛 ValueError，用户无合法更新路径 | Reliable | pending | |
| M10 | 3136024769 | `parsing_service/base.py` | 254 | （待补充完整内容） | pending | pending | |
| M11 | 3136024774 | `parsing_service/base.py` | 91 | （待补充完整内容） | pending | pending | |
| M12 | 3136024776 | `data_loaders.py` | 38 | （待补充完整内容） | pending | pending | |
| M13 | 3136024781 | `data_loaders.py` | 35 | （待补充完整内容） | pending | pending | |
| M14 | 3136024785 | `doc_service/parser_client.py` | 94 | （待补充完整内容） | pending | pending | |
| M15 | 3136025765 | `doc_manager.py` | 183 | `_ensure_kb_algorithm` IntegrityError 后 `updated_at` 永远不更新，心跳语义失效 | Reliable | pending | 与 C25 合并修复 |
| M16 | 3136025768 | `doc_manager.py` | 1959 | `list_kbs`/`batch_get_kbs` N+1 查询问题 | Reliable | pending | 重复：3136031468 |
| M17 | 3136025777 | `doc_manager.py` | 755 | `_prepare_kb_delete_items` 用 `binding[0]` 作 default_algo_id，与快照 algo_id 不一致，primary 标记错误 | Reliable | pending | |
| M18 | 3136025786 | `doc_manager.py` | 2044 | `unbind_algo` 未处理 DELETING 状态，对同一文档重复发起 DOC_DELETE | Reliable | pending | 重复：3136031481、3136039901 |
| M19 | 3136025793 | `doc_manager.py` | 1672 | `on_task_callback` unbind 路径两步操作不在同一事务，崩溃后重试重复删除 | Reliable | pending | 重复：3136039908 |
| M20 | 3136025799 | `doc_manager.py` | 1248 | `upload()` except 分支只用 `algo_ids[0]` 查快照，错误信息不准确 | Reliable | pending | 与 C23 合并修复 |
| M21 | 3136032552 | `doc_impl.py` | 327 | `version` 参数改写 name 但 `parent` 参数不自动追加版本，父节点组查找失败 | Reliable | pending | |
| M22 | 3136032556 | `doc_impl.py` | 328 | `create_global_node_group` 和 `_create_builtin_node_group` 未同步添加 `version` 参数，API 不一致 | Reliable | pending | 重复：3136039911 |
| M23 | 3136039883 | `doc_manager.py` | 982 | `_enqueue_task` 中 `exclusive_ng_ids` 仅从 `extra_message` 取值，upload 路径未传递 | Reliable | pending | 与 C18 合并修复 |
| M24 | 3136039895 | `data_loaders.py` | 50 | `inspect.getsource()` 在某些环境（frozen/Jupyter）抛 OSError，fallback 到 repr 对 lambda 不稳定 | Reliable | pending | |
| M25 | 3136039899 | `doc_service/base.py` | 131 | （待补充完整内容） | pending | pending | |
| M26 | 3136039901 | `doc_manager.py` | 2044 | `unbind_algo` WAITING 状态应先取消再解绑，与 `delete()` 行为不一致 | Reliable | pending | 与 M18 合并修复 |
| M27 | 3136039908 | `doc_manager.py` | 1673 | `on_task_callback` unbind 路径两步操作不在同一事务 | Reliable | pending | 与 M19 合并修复 |
| M28 | 3136031472 | `doc_manager.py` | 1311 | `reparse()` 当 `pending_ng_ids` 为空时静默返回 `[]`，调用方无法区分原因 | Reliable | pending | 重复：3136025805 |
| M29 | 3136031476 | `doc_manager.py` | 625 | `_get_algo_node_group_ids` 错误消息暴露内部 resp 对象 repr | Reliable | pending | |
| M30 | 3136031487 | `doc_manager.py` | 1248 | `upload()` 异常处理丢失已成功入队的 algo 状态 | Reliable | pending | 与 C23/M20 合并修复 |

---

## NORMAL 类（风格/设计，合并前应修复）

| # | ID | File | Line | 问题摘要 | Verdict | Status | Notes |
|---|-----|------|------|---------|---------|--------|-------|
| N1 | 3136024787 | `document.py` | 22 | `from .doc_service import DocServer` import 位置违反 PEP 8 顺序 | Reliable | pending | |
| N2 | 3136024791 | `parsing_service/impl.py` | 297 | 空 `try/except Exception: raise` 块无意义，应移除 | Reliable | pending | |
| N3 | 3136025714 | `parsing_service/impl.py` | 267 | `target_doc_ids` 无条件计算但只在 skip 分支使用，应移入分支内 | Reliable | pending | |
| N4 | 3136025721 | `doc_service/base.py` | 107 | `DocItemsRequest` 用 `extra='allow'` 捕获废弃字段，typo 被静默接受 | Reliable | pending | 与 M6 合并修复 |
| N5 | 3136025724 | `doc_manager.py` | 1402 | `delete()` 多 algo 入队时第二个及之后 algo 传 `idempotency_key=None`，幂等性不一致 | Reliable | pending | |
| N6 | 3136025728 | `doc_manager.py` | 90 | `_ensure_indexes` 每次启动都执行 DROP+CREATE，并发时约束缺失窗口 | Reliable | pending | |
| N7 | 3136025734 | `doc_manager.py` | 1959 | N+1 查询（同 M16） | Reliable | pending | 与 M16 合并修复 |
| N8 | 3136025739 | `doc_manager.py` | 755 | primary 标记逻辑（同 M17） | Reliable | pending | 与 M17 合并修复 |
| N9 | 3136025743 | `doc_manager.py` | 2023 | `unbind_algo` 破坏性操作缺少 `dry_run` 参数和影响范围返回 | Reliable | pending | 重复：3136039928 |
| N10 | 3136025747 | `doc_manager.py` | 1311 | `reparse()` 静默返回 `[]`（同 M28） | Reliable | pending | 与 M28 合并修复 |
| N11 | 3136025753 | `doc_service/base.py` | 107 | 废弃字段应通过显式字段声明 + `@deprecated` 处理，而非 validator | Reliable | pending | 与 M6/N4 合并修复 |
| N12 | 3136025803 | `doc_manager.py` | 2023 | `unbind_algo` 缺少 `dry_run` 参数（同 N9） | Reliable | pending | 与 N9 合并修复 |
| N13 | 3136025805 | `doc_manager.py` | 1311 | `reparse()` 静默返回 `[]`（同 M28） | Reliable | pending | 与 M28 合并修复 |
| N14 | 3136025809 | `doc_service/base.py` | 107 | `extra='allow'` 副作用（同 N4） | Reliable | pending | 与 M6/N4 合并修复 |
| N15 | 3136025812 | `doc_manager.py` | 2031 | `unbind_algo` 错误消息未说明 `delete_kb` 的破坏性后果 | Reliable | pending | |
| N16 | 3136025818 | `doc_manager.py` | 1664 | `on_task_callback` DOC_DELETE 路径 `_delete_ng_status` 可能被重复调用 | Reliable | pending | |
| N17 | 3136025838 | `doc_service/base.py` | 107 | 废弃字段 validator 只在服务端触发，单元测试直接构造对象不触发警告 | Reliable | pending | 与 M6/N4/N11 合并修复 |
| N18 | 3136039919 | `doc_manager.py` | 1696 | （待补充完整内容） | pending | pending | |
| N19 | 3136039923 | `data_loaders.py` | 50 | `inspect.getsource()` OSError fallback（同 M24） | Reliable | pending | 与 M24 合并修复 |
| N20 | 3136039928 | `doc_manager.py` | 2059 | `unbind_algo` 返回值缺少影响范围（同 N9） | Reliable | pending | 与 N9 合并修复 |

---

## 待补充内容（需进一步查看代码确认）

| # | ID | File | Line | 备注 |
|---|-----|------|------|------|
| ? | 3136024769 | `parsing_service/base.py` | 254 | 需查看完整 body |
| ? | 3136024774 | `parsing_service/base.py` | 91 | 需查看完整 body |
| ? | 3136024776 | `data_loaders.py` | 38 | 需查看完整 body |
| ? | 3136024781 | `data_loaders.py` | 35 | 需查看完整 body |
| ? | 3136024785 | `doc_service/parser_client.py` | 94 | 需查看完整 body |
| ? | 3136039899 | `doc_service/base.py` | 131 | 需查看完整 body |
| ? | 3136039918 | `doc_manager.py` | 1696 | 需查看完整 body |

---

## 独立问题汇总（去重后）

- Critical 独立问题：**25 个**（含合并组）
- Medium 独立问题：**30 个**（含合并组）
- Normal 独立问题：**20 个**（含合并组）
- 待确认：**7 个**

