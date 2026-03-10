## 📌 PR 内容 / PR Description

为 `lazyllm/tools/tools/search` 增加统一搜索基类与多引擎支持：

- **基类与统一接口**：新增 `SearchBase`，子类实现 `search(query, ...)`，返回统一格式 `List[Dict]`（每项含 `title`、`url`、`snippet`、`source`，可选 `extra`）。`forward(query, **kwargs)` 透传参数给 `search`。
- **原有引擎改造**：`GoogleSearch`、`TencentSearch` 改为继承 `SearchBase`，结果规范为上述统一格式。
- **新增通用搜索引擎**：`BingSearch`（Azure Bing Web Search API v7）、`BochaSearch`（博查 AI）。
- **新增垂直/学术搜索**：`StackOverflowSearch`、`SemanticScholarSearch`、`GoogleBooksSearch`、`ArxivSearch`、`WikipediaSearch`。
- **文档与规范**：在 `lazyllm/docs/tools/search.py` 中为基类及各子类提供中英双语文档与 Example；每个子类文档中说明如何在对应官网申请 API Key/Token。代码内无文档级注释，统一使用单引号；`make_result` 改为私有 `_make_result`，`TITLE_KEY` 等常量不再从 `search/__init__.py` 导出。

## 🔍 相关 Issue / Related Issue

N/A

## ✅ 变更类型 / Type of Change

- [x] 新功能 / New feature (non-breaking change that adds functionality)
- [x] 重构 / Refactor (no functionality change, code structure optimized)
- [x] 文档更新 / Documentation update (changes to docs only)

## 🧪 如何测试 / How Has This Been Tested?

- 从 `lazyllm.tools.tools` 导入各搜索类并实例化（如 `ArxivSearch()`、`WikipediaSearch()`）。
- 调用 `engine.search(query, ...)` 或 `engine.forward(query, ...)`，确认返回 `List[Dict]` 且每项包含 `title`、`url`、`snippet`、`source`。
- 无 key 的引擎（如 Arxiv、Wikipedia）可直接跑通；需 key 的引擎需自备 key 后测试。

## 📷 截图 / Demo (Optional)

N/A

## ⚡ 更新后的用法示例 / Usage After Update

```python
from lazyllm.tools.tools import (
    SearchBase,
    GoogleSearch,
    TencentSearch,
    BingSearch,
    BochaSearch,
    StackOverflowSearch,
    SemanticScholarSearch,
    GoogleBooksSearch,
    ArxivSearch,
    WikipediaSearch,
)

# 无需 key 的引擎
arxiv = ArxivSearch()
results = arxiv.search('attention is all you need', max_results=5)

wiki = WikipediaSearch(base_url='https://zh.wikipedia.org')
results = wiki.search('机器学习', limit=5)

# 需 key 的引擎（以 Bing 为例）
bing = BingSearch(subscription_key='<your_key>')
results = bing.search('python tutorial', count=10)

# 统一入口：forward 会透传参数给 search
results = arxiv.forward('transformer', max_results=3)
```

## ⚠️ 注意事项 / Additional Notes

- **Breaking**：`TencentSearch.forward` 原返回 `package(...)`，现改为返回 `List[Dict]`，与其它引擎一致。
- 各引擎 API Key 申请方式见 `lazyllm/docs/tools/search.py` 中对应类的中英文档（如 Google、腾讯云、Azure、博查、Stack Exchange、Semantic Scholar、Google Books；Arxiv / Wikipedia 无需 key）。
