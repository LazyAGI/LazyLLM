# Copyright (c) 2026 LazyAGI. All rights reserved.
# flake8: noqa E501
"""Git module docs: LazyLLMGitBase, PrInfo, ReviewCommentInfo, GitHub, GitLab, Gitee, GitCode."""
import importlib
import functools

from .. import utils

_add_git_chinese = functools.partial(
    utils.add_chinese_doc, module=importlib.import_module('lazyllm.tools.git'))
_add_git_english = functools.partial(
    utils.add_english_doc, module=importlib.import_module('lazyllm.tools.git'))
_add_git_example = functools.partial(
    utils.add_example, module=importlib.import_module('lazyllm.tools.git'))

# LazyLLMGitBase
_add_git_chinese('LazyLLMGitBase', '''\
Git 平台统一基类，借助 registry 注册各平台实现（如 GitHub、GitLab、Gitee、GitCode）。
子类需实现：认证方式、API 根地址、以及抽象方法。
Agent 可通过 lazyllm.git.github / lazyllm.git.gitlab 等获取实例并调用接口。

Args:
    token (str): 平台 Access Token / Private Token。
    repo (str): 仓库标识，格式一般为 "owner/repo" 或 "namespace/project"。
    api_base (str, optional): 自定义 API 根地址（如自建 GitLab）。
    return_trace (bool): 是否返回调用追踪信息。
''')

_add_git_english('LazyLLMGitBase', '''\
Unified Git platform base; implementations (GitHub, GitLab, Gitee, GitCode) are registered via registry.
Subclasses implement auth, API base URL, and abstract methods.
Agents get instances via lazyllm.git.github / lazyllm.git.gitlab etc.

Args:
    token (str): Platform access token or private token.
    repo (str): Repository identifier, e.g. "owner/repo" or "namespace/project".
    api_base (str, optional): Custom API base URL (e.g. self-hosted GitLab).
    return_trace (bool): Whether to return call trace.
''')

# PrInfo
_add_git_chinese('PrInfo', '''\
Pull Request / Merge Request 摘要。属性：number, title, state, body, source_branch, target_branch, html_url, raw。
''')
_add_git_english('PrInfo', '''\
Pull Request / Merge Request summary. Attributes: number, title, state, body, source_branch, target_branch, html_url, raw.
''')

# ReviewCommentInfo
_add_git_chinese('ReviewCommentInfo', '''\
单条评审评论（可含行级）。属性：id, body, path, line, side, user, raw。
''')
_add_git_english('ReviewCommentInfo', '''\
Single review comment (optionally line-level). Attributes: id, body, path, line, side, user, raw.
''')

# GitHub
_add_git_chinese('GitHub', '''\
GitHub 后端：使用 REST API (api.github.com)，push 使用本地 git 命令。
''')
_add_git_english('GitHub', '''\
GitHub backend: REST API (api.github.com), push via local git.
''')

# GitLab
_add_git_chinese('GitLab', '''\
GitLab 后端：使用 REST API (gitlab.com/api/v4)，push 使用本地 git。
''')
_add_git_english('GitLab', '''\
GitLab backend: REST API (gitlab.com/api/v4), push via local git.
''')

# Gitee
_add_git_chinese('Gitee', '''\
Gitee 后端：使用 OpenAPI v5 (gitee.com/api/v5)，push 使用本地 git。
''')
_add_git_english('Gitee', '''\
Gitee backend: OpenAPI v5 (gitee.com/api/v5), push via local git.
''')

# GitCode
_add_git_chinese('GitCode', '''\
GitCode 后端：华为云 CodeArts 代码托管，OpenAPI 与 Gitee v5 类似。
''')
_add_git_english('GitCode', '''\
GitCode backend: Huawei CodeArts, OpenAPI similar to Gitee v5.
''')

# LazyLLMGitBase abstract methods
_add_git_chinese('LazyLLMGitBase.push_branch', '''\
将本地分支推送到远程仓库。

Args:
    local_branch (str): 本地分支名。
    remote_branch (str, optional): 远程分支名，默认与 local_branch 相同。
    remote_name (str): 远程名称，默认为 "origin"。
    repo_path (str, optional): 本地仓库路径，不传则使用当前工作目录。

Returns:
    dict: 包含 success、message 等字段。
''')
_add_git_english('LazyLLMGitBase.push_branch', '''\
Push local branch to remote repository.

Args:
    local_branch (str): Local branch name.
    remote_branch (str, optional): Remote branch name; defaults to local_branch.
    remote_name (str): Remote name, default "origin".
    repo_path (str, optional): Local repo path; if omitted, uses current working directory.

Returns:
    dict: With keys such as success, message.
''')

_add_git_chinese('LazyLLMGitBase.create_pull_request', '''\
创建 Pull Request / Merge Request。

Args:
    source_branch (str): 源分支名。
    target_branch (str): 目标分支名。
    title (str): PR/MR 标题。
    body (str): PR/MR 正文描述，可选。
    **kwargs: 各平台扩展参数。

Returns:
    dict: 包含 success、number、html_url、message 等。
''')
_add_git_english('LazyLLMGitBase.create_pull_request', '''\
Create a Pull Request / Merge Request.

Args:
    source_branch (str): Source branch name.
    target_branch (str): Target branch name.
    title (str): PR/MR title.
    body (str): PR/MR body, optional.
    **kwargs: Platform-specific extra arguments.

Returns:
    dict: success, number, html_url, message, etc.
''')

_add_git_chinese('LazyLLMGitBase.update_pull_request', '''\
更新 PR/MR 的标题、正文或状态。

Args:
    number (int): PR/MR 编号。
    title (str, optional): 新标题。
    body (str, optional): 新正文。
    state (str, optional): 状态（如 open/closed）。
    **kwargs: 各平台扩展参数。

Returns:
    dict: 包含 success、message。
''')
_add_git_english('LazyLLMGitBase.update_pull_request', '''\
Update PR/MR title, body or state.

Args:
    number (int): PR/MR number.
    title (str, optional): New title.
    body (str, optional): New body.
    state (str, optional): State (e.g. open/closed).
    **kwargs: Platform-specific extra arguments.

Returns:
    dict: success, message.
''')

_add_git_chinese('LazyLLMGitBase.add_pr_labels', '''\
为 PR/MR 添加标签。

Args:
    number (int): PR/MR 编号。
    labels (list[str]): 要添加的标签名列表。

Returns:
    dict: 包含 success、message。
''')
_add_git_english('LazyLLMGitBase.add_pr_labels', '''\
Add labels to a PR/MR.

Args:
    number (int): PR/MR number.
    labels (list[str]): List of label names to add.

Returns:
    dict: success, message.
''')

_add_git_chinese('LazyLLMGitBase.get_pull_request', '''\
获取单条 PR/MR 详情。

Args:
    number (int): PR/MR 编号。

Returns:
    dict: 包含 success、pr（PrInfo 或 dict）、message。
''')
_add_git_english('LazyLLMGitBase.get_pull_request', '''\
Get a single PR/MR by number.

Args:
    number (int): PR/MR number.

Returns:
    dict: success, pr (PrInfo or dict), message.
''')

_add_git_chinese('LazyLLMGitBase.list_pull_requests', '''\
列出 PR/MR 列表。

Args:
    state (str): 状态筛选，如 "open"、"closed"，默认 "open"。
    head (str, optional): 按源分支筛选。
    base (str, optional): 按目标分支筛选。
    **kwargs: 各平台扩展参数。

Returns:
    dict: 包含 success、list（PrInfo 或 dict 列表）、message。
''')
_add_git_english('LazyLLMGitBase.list_pull_requests', '''\
List PRs/MRs with optional filters.

Args:
    state (str): State filter, e.g. "open", "closed"; default "open".
    head (str, optional): Filter by source branch.
    base (str, optional): Filter by target branch.
    **kwargs: Platform-specific extra arguments.

Returns:
    dict: success, list (of PrInfo or dict), message.
''')

_add_git_chinese('LazyLLMGitBase.get_pr_diff', '''\
获取 PR/MR 的 diff 文本。

Args:
    number (int): PR/MR 编号。

Returns:
    dict: 包含 success、diff、message。
''')
_add_git_english('LazyLLMGitBase.get_pr_diff', '''\
Get the diff text for a PR/MR.

Args:
    number (int): PR/MR number.

Returns:
    dict: success, diff, message.
''')

_add_git_chinese('LazyLLMGitBase.list_review_comments', '''\
列出 PR/MR 上的全部评审评论。

Args:
    number (int): PR/MR 编号。

Returns:
    dict: 包含 success、comments（ReviewCommentInfo 或 dict 列表）、message。
''')
_add_git_english('LazyLLMGitBase.list_review_comments', '''\
List all review comments on a PR/MR.

Args:
    number (int): PR/MR number.

Returns:
    dict: success, comments (list of ReviewCommentInfo or dict), message.
''')

_add_git_chinese('LazyLLMGitBase.create_review_comment', '''\
在 PR/MR 上创建一条评审评论（可指定文件与行）。

Args:
    number (int): PR/MR 编号。
    body (str): 评论内容。
    path (str): 文件路径。
    line (int, optional): 行号，用于行级评论。
    side (str): 左右侧，默认 "RIGHT"。
    commit_id (str, optional): 提交 ID，部分平台需要。
    **kwargs: 各平台扩展参数。

Returns:
    dict: 包含 success、comment_id、message。
''')
_add_git_english('LazyLLMGitBase.create_review_comment', '''\
Create a single review comment on a PR/MR (optionally line-level).

Args:
    number (int): PR/MR number.
    body (str): Comment body.
    path (str): File path.
    line (int, optional): Line number for line-level comment.
    side (str): Side (e.g. "RIGHT"), default "RIGHT".
    commit_id (str, optional): Commit ID, required on some platforms.
    **kwargs: Platform-specific extra arguments.

Returns:
    dict: success, comment_id, message.
''')

_add_git_chinese('LazyLLMGitBase.submit_review', '''\
提交评审结论（通过 / 请求修改 / 仅评论）。

Args:
    number (int): PR/MR 编号。
    event (str): 事件类型，如 APPROVE、REQUEST_CHANGES、COMMENT。
    body (str): 评审总结正文，可选。
    comment_ids (list, optional): 要一并提交的评论 ID 列表。
    **kwargs: 各平台扩展参数。

Returns:
    dict: 包含 success、message。
''')
_add_git_english('LazyLLMGitBase.submit_review', '''\
Submit a review (approve / request changes / comment).

Args:
    number (int): PR/MR number.
    event (str): Event type, e.g. APPROVE, REQUEST_CHANGES, COMMENT.
    body (str): Review summary body, optional.
    comment_ids (list, optional): Comment IDs to submit with the review.
    **kwargs: Platform-specific extra arguments.

Returns:
    dict: success, message.
''')

_add_git_chinese('LazyLLMGitBase.approve_pull_request', '''\
批准 PR/MR。

Args:
    number (int): PR/MR 编号。
    **kwargs: 各平台扩展参数。

Returns:
    dict: 包含 success、message。
''')
_add_git_english('LazyLLMGitBase.approve_pull_request', '''\
Approve a PR/MR.

Args:
    number (int): PR/MR number.
    **kwargs: Platform-specific extra arguments.

Returns:
    dict: success, message.
''')

_add_git_chinese('LazyLLMGitBase.merge_pull_request', '''\
合并 PR/MR。

Args:
    number (int): PR/MR 编号。
    merge_method (str, optional): 合并方式（如 merge、squash、rebase），依平台而定。
    commit_title (str, optional): 合并提交标题。
    commit_message (str, optional): 合并提交说明。
    **kwargs: 各平台扩展参数。

Returns:
    dict: 包含 success、sha、message。
''')
_add_git_english('LazyLLMGitBase.merge_pull_request', '''\
Merge a PR/MR.

Args:
    number (int): PR/MR number.
    merge_method (str, optional): Merge method (e.g. merge, squash, rebase), platform-dependent.
    commit_title (str, optional): Merge commit title.
    commit_message (str, optional): Merge commit message.
    **kwargs: Platform-specific extra arguments.

Returns:
    dict: success, sha, message.
''')

_add_git_example('LazyLLMGitBase', '''\
>>> from lazyllm.tools import git
>>> import lazyllm
>>> backend = lazyllm.git.GitHub(token='xxx', repo='owner/repo')
>>> backend.create_pull_request('feat', 'main', 'Title', 'Body')
>>> backend.merge_pull_request(1)
''')
