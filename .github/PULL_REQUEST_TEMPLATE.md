<!-- 
在提交PR之前，请认真阅读以下规则。
1. 我们每行的字符限制是120，提交代码前请在保持语义清晰的基础上，尽你所能的压缩代码行数
2. 我们强制要求单引号优先；但由于一些历史提交的文件并没有做相关要求，因此代码仓库中还存在一些不规范的代码，如果你修改到某个文件，请顺便处理整个文件，变成单引号优先。
3. 我们禁止在代码中使用print，请在提交之前删除所有的print语句
4. 我们禁止无意义的注释，包括重复一遍函数名，翻译代码，显而易见。注释应该讲“为什么这么写”，即对一大段代码讲解执行逻辑、设计思路、参考来源或解决了什么问题。
5. 在提交PR之前，请在你的分支上执行如下命令
```bash
pip install flake8-quotes
pip install flake8-bugbear
make lint-only-diff
```

Please read the following rules carefully before submitting a PR.
1. We have a character limit of 120 per line. Please compress the code lines as much as possible while maintaining semantic clarity before submitting code.
2. We enforce single quotes priority; however, due to some historical commits that didn't follow this requirement, there are still some non-standard codes in the codebase. If you modify a file, please also process the entire file to use single quotes priority.
3. We prohibit the use of print in code. Please delete all print statements before submitting.
4. We prohibit meaningless comments, including repeating function names, translating code, or stating the obvious. Comments should explain "why it's written this way", i.e., explain the execution logic, design ideas, reference sources, or problems solved for a large block of code.
5. Before submitting a PR, please execute the following commands on your branch
```bash
pip install flake8-quotes
pip install flake8-bugbear
make lint-only-diff
```
-->

## 📌 PR 内容 / PR Description
<!-- 简要描述本次 PR 的改动点 / Briefly describe the changes in this PR -->
- 

## 🔍 相关 Issue / Related Issue
<!-- 例如：Fix #123 / Close #456 -->
- 

## ✅ 变更类型 / Type of Change
<!-- 勾选对应选项 / Check the relevant options -->
- [ ] 修复 Bug / Bug fix (non-breaking change that fixes an issue)
- [ ] 新功能 / New feature (non-breaking change that adds functionality)
- [ ] 重构 / Refactor (no functionality change, code structure optimized)
- [ ] 重大变更 / Breaking change (fix or feature that would cause existing functionality to change)
- [ ] 文档更新 / Documentation update (changes to docs only)
- [ ] 性能优化 / Performance optimization

## 🧪 如何测试 / How Has This Been Tested?
<!-- 描述测试步骤 / Describe the tests that you ran to verify your changes -->
1. 
2. 
3. 

## 📷 截图 / Demo (Optional)
<!-- 如果是文档改动或者性能优化 / If document changes or performance optimization, please attach screenshots -->
- 

## ⚡ 更新后的用法示例 / Usage After Update
<!-- 请提供更新后的调用示例 / Provide example(s) of usage after your changes -->
```python
# 示例 / Example
```

## 🔄 重构前 / 重构后对比 (仅当 Type 为 Refactor) / Refactor Before & After (only for Refactor)
<!-- 请提供重构前后的调用对比 / Provide before & after usage for refactor -->

### 重构前 / Before:


### 重构后 / After:


## ⚠️ 注意事项 / Additional Notes
<!-- 是否有依赖更新、迁移步骤或其他注意点 / Mention dependencies, migration steps, or any other concerns -->
