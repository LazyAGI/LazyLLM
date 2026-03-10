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
    repo (str, optional): 仓库标识，格式一般为 "owner/repo" 或 "namespace/project"。
    api_base (str, optional): 自定义 API 根地址（如自建 GitLab）。
    user (str, optional): 用户标识，部分接口默认使用该用户或 token 所属用户。
    return_trace (bool): 是否返回调用追踪信息。
''')

_add_git_english('LazyLLMGitBase', '''\
Unified Git platform base; implementations (GitHub, GitLab, Gitee, GitCode) are registered via registry.
Subclasses implement auth, API base URL, and abstract methods.
Agents get instances via lazyllm.git.github / lazyllm.git.gitlab etc.

Args:
    token (str): Platform access token or private token.
    repo (str, optional): Repository identifier, e.g. "owner/repo" or "namespace/project".
    api_base (str, optional): Custom API base URL (e.g. self-hosted GitLab).
    user (str, optional): User identifier; some APIs default to this user or token owner.
    return_trace (bool): Whether to return call trace.
''')

# PrInfo
_add_git_chinese('PrInfo', '''\
Pull Request / Merge Request 摘要。属性：number, title, state, body, source_branch, target_branch, html_url, raw。
''')
_add_git_english('PrInfo', '''\
Pull Request / Merge Request summary. Attributes: number, title, state, body, source_branch, target_branch, html_url, raw.
''')
_add_git_example('PrInfo', '''\
>>> from lazyllm.tools.git import PrInfo
>>> pr = PrInfo(1, 'Fix bug', 'open', 'Description', 'feat', 'main', 'https://github.com/owner/repo/pull/1')
>>> pr.number
... 1
>>> pr.title
... 'Fix bug'
''')

_add_git_chinese('PrInfo.to_dict', '''\
将 PR/MR 摘要转为字典，便于序列化或 JSON 输出。

Returns:
    dict: 包含 number, title, state, body, source_branch, target_branch, html_url, raw。
''')
_add_git_english('PrInfo.to_dict', '''\
Convert PR/MR summary to a dict for serialization or JSON output.

Returns:
    dict: Keys: number, title, state, body, source_branch, target_branch, html_url, raw.
''')
_add_git_example('PrInfo.to_dict', '''\
>>> pr = PrInfo(1, 'Fix bug', 'open', 'Description', 'feat', 'main', 'https://example.com/pull/1')
>>> pr.to_dict()
''')

# ReviewCommentInfo
_add_git_chinese('ReviewCommentInfo', '''\
单条评审评论（可含行级）。属性：id, body, path, line, side, user, raw。
''')
_add_git_english('ReviewCommentInfo', '''\
Single review comment (optionally line-level). Attributes: id, body, path, line, side, user, raw.
''')
_add_git_example('ReviewCommentInfo', '''\
>>> from lazyllm.tools.git import ReviewCommentInfo
>>> c = ReviewCommentInfo(101, 'Consider using constant', 'src/foo.py', 42, 'RIGHT', 'alice')
>>> c.body
... 'Consider using constant'
>>> c.path
... 'src/foo.py'
''')

_add_git_chinese('ReviewCommentInfo.to_dict', '''\
将评审评论转为字典，便于序列化或 JSON 输出。

Returns:
    dict: 包含 id, body, path, line, side, user, raw。
''')
_add_git_english('ReviewCommentInfo.to_dict', '''\
Convert review comment to a dict for serialization or JSON output.

Returns:
    dict: Keys: id, body, path, line, side, user, raw.
''')
_add_git_example('ReviewCommentInfo.to_dict', '''\
>>> c = ReviewCommentInfo(101, 'Consider using constant', 'src/foo.py', 42)
>>> c.to_dict()
''')

# GitHub
_add_git_chinese('GitHub', '''\
GitHub 后端：使用 REST API (api.github.com)，push 使用本地 git 命令。
''')
_add_git_english('GitHub', '''\
GitHub backend: REST API (api.github.com), push via local git.
''')
_add_git_example('GitHub', '''\
>>> from lazyllm.tools.git import GitHub
>>> backend = GitHub(token='ghp_xxx', repo='owner/repo')
>>> backend.get_pull_request(1)
''')

_add_git_chinese('GitHub.add_issue_comment', '''\
在 PR 的对话区添加一条评论（GitHub 中 PR 即 issue，评论显示在 Conversation）。

Args:
    number (int): PR 编号。
    body (str): 评论内容。

Returns:
    dict: 包含 success、message、url。
''')
_add_git_english('GitHub.add_issue_comment', '''\
Add a comment to the PR conversation (on GitHub, PR is an issue; comment appears in Conversation).

Args:
    number (int): PR number.
    body (str): Comment body.

Returns:
    dict: success, message, url.
''')
_add_git_example('GitHub.add_issue_comment', '''\
>>> backend.add_issue_comment(1, 'Looks good to me')
''')

# GitLab
_add_git_chinese('GitLab', '''\
GitLab 后端：使用 REST API (gitlab.com/api/v4)，push 使用本地 git。
''')
_add_git_english('GitLab', '''\
GitLab backend: REST API (gitlab.com/api/v4), push via local git.
''')
_add_git_example('GitLab', '''\
>>> from lazyllm.tools.git import GitLab
>>> backend = GitLab(token='glpat-xxx', repo='namespace/project')
>>> backend.list_pull_requests(state='open')
''')

# Gitee
_add_git_chinese('Gitee', '''\
Gitee 后端：使用 OpenAPI v5 (gitee.com/api/v5)，push 使用本地 git。
''')
_add_git_english('Gitee', '''\
Gitee backend: OpenAPI v5 (gitee.com/api/v5), push via local git.
''')
_add_git_example('Gitee', '''\
>>> from lazyllm.tools.git import Gitee
>>> backend = Gitee(token='xxx', repo='owner/repo')
>>> backend.get_pr_diff(1)
''')

# GitCode
_add_git_chinese('GitCode', '''\
GitCode 后端：华为云 CodeArts 代码托管，OpenAPI 与 Gitee v5 类似。
''')
_add_git_english('GitCode', '''\
GitCode backend: Huawei CodeArts, OpenAPI similar to Gitee v5.
''')
_add_git_example('GitCode', '''\
>>> from lazyllm.tools.git import GitCode
>>> backend = GitCode(token='xxx', repo='owner/repo')
>>> backend.create_pull_request('feat', 'main', 'Title', 'Body')
''')

# Git
_add_git_chinese('Git', '''\
统一 Git 客户端：根据 backend 或配置、环境变量、gh CLI 自动选择后端（GitHub/GitLab/Gitee/GitCode）。
传入 backend 时使用该后端；否则先读 config["git_backend"]，再按 GITHUB_TOKEN 等环境变量，再 gh 登录，最后默认 github。

Args:
    backend (str, optional): 后端名（github/gitlab/gitee/gitcode）；不传则自动检测。
    token (str, optional): Access Token；缺省时从环境变量或 gh 解析。
    repo (str): 仓库标识，如 owner/repo。
    api_base (str, optional): 后端 API 根地址。
    return_trace (bool): 是否返回调用追踪。
''')
_add_git_english('Git', '''\
Unified Git client: selects backend by argument, config, or auto-detect (env, gh CLI, default github).
If backend is passed, use it; else config["git_backend"], then env (GITHUB_TOKEN, etc.), then gh, then github.

Args:
    backend (str, optional): Backend name (github, gitlab, gitee, gitcode); if None, auto-detected.
    token (str, optional): Access token; resolved from env or gh when None.
    repo (str): Repository identifier, e.g. owner/repo.
    api_base (str, optional): API base URL for the backend.
    return_trace (bool): Whether to return call trace.
''')
_add_git_example('Git', '''\
>>> from lazyllm.tools.git import Git
>>> client = Git(backend='github', token='xxx', repo='owner/repo')
>>> client.get_pull_request(1)
>>> client.list_pull_requests(state='open')
''')

# review
_add_git_chinese('review', '''\
对 PR/MR 做代码评审：解析 diff、按 hunk 调用模型，可选地提交行级评论。后端随 Git 配置/backend；repo 支持完整 URL（如 https://.../owner/repo 或 .../repo.git）。

Args:
    pr_number (int): PR/MR 编号。
    repo (str): 仓库：owner/repo 或完整 URL；.git 会被去掉；未传 backend 时从 URL 推断。
    token (str, optional): Access Token；缺省按后端从环境变量或 gh 解析。
    backend (str, optional): 指定后端（github/gitlab/gitee/gitcode）；不传则用配置/环境/gh。
    llm: 推理用 LLM；None 时使用 lazyllm.OnlineChatModule()。
    api_base (str, optional): 后端 API 根地址。
    post_to_github (bool): 为 True 时把每条问题作为行级评论提交到平台。
    max_diff_chars (int, optional): diff 最大字符数；None 表示不限制。
    max_hunks (int, optional): 最多处理的 hunk 数；None 表示不限制。

Returns:
    dict: summary、comments_posted、comments。
''')
_add_git_english('review', '''\
Review a PR/MR: parse diff, call model per hunk, optionally post line-level comments. Backend follows Git config/backend; repo can be owner/repo or full URL (e.g. https://.../owner/repo or .../repo.git).

Args:
    pr_number (int): PR/MR number.
    repo (str): Repository: owner/repo or full URL; .git is stripped; backend inferred from URL when not passed.
    token (str, optional): Access token; resolved from env or gh per backend.
    backend (str, optional): If set, use this backend (github, gitlab, gitee, gitcode); else config/env/gh.
    llm: LLM for inference; None uses lazyllm.OnlineChatModule().
    api_base (str, optional): API base URL for the backend.
    post_to_github (bool): If True, post each issue as a line-level comment on the platform.
    max_diff_chars (int, optional): Max diff length; None for no limit.
    max_hunks (int, optional): Max hunks to process; None for no limit.

Returns:
    dict: summary, comments_posted, comments.
''')
_add_git_example('review', '''\
>>> from lazyllm.tools.git import review
>>> result = review(pr_number=1, repo='owner/repo', post_to_github=False)
>>> result['summary']
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
_add_git_example('LazyLLMGitBase.push_branch', '''\
>>> backend = lazyllm.tools.git.Git(backend='github', token='xxx', repo='owner/repo')
>>> backend.push_branch('feat', remote_branch='feat', remote_name='origin')
''')

_add_git_chinese('LazyLLMGitBase.create_pull_request', '''\
创建 Pull Request / Merge Request。

Args:
    source_branch (str): 源分支名。
    target_branch (str): 目标分支名。
    title (str): PR/MR 标题。
    body (str): PR/MR 正文描述，可选。

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

Returns:
    dict: success, number, html_url, message, etc.
''')
_add_git_example('LazyLLMGitBase.create_pull_request', '''\
>>> backend.create_pull_request('feat', 'main', 'Add feature', 'Description')
''')

_add_git_chinese('LazyLLMGitBase.update_pull_request', '''\
更新 PR/MR 的标题、正文或状态。

Args:
    number (int): PR/MR 编号。
    title (str, optional): 新标题。
    body (str, optional): 新正文。
    state (str, optional): 状态（如 open/closed）。

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

Returns:
    dict: success, message.
''')
_add_git_example('LazyLLMGitBase.update_pull_request', '''\
>>> backend.update_pull_request(1, title='New title', body='Updated body')
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
_add_git_example('LazyLLMGitBase.add_pr_labels', '''\
>>> backend.add_pr_labels(1, ['bug', 'priority-high'])
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
_add_git_example('LazyLLMGitBase.get_pull_request', '''\
>>> backend.get_pull_request(1)
''')

_add_git_chinese('LazyLLMGitBase.list_pull_requests', '''\
列出 PR/MR 列表。

Args:
    state (str): 状态筛选，如 "open"、"closed"，默认 "open"。
    head (str, optional): 按源分支筛选。
    base (str, optional): 按目标分支筛选。

Returns:
    dict: 包含 success、list（PrInfo 或 dict 列表）、message。
''')
_add_git_english('LazyLLMGitBase.list_pull_requests', '''\
List PRs/MRs with optional filters.

Args:
    state (str): State filter, e.g. "open", "closed"; default "open".
    head (str, optional): Filter by source branch.
    base (str, optional): Filter by target branch.

Returns:
    dict: success, list (of PrInfo or dict), message.
''')
_add_git_example('LazyLLMGitBase.list_pull_requests', '''\
>>> backend.list_pull_requests(state='open', base='main')
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
_add_git_example('LazyLLMGitBase.get_pr_diff', '''\
>>> backend.get_pr_diff(1)
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
_add_git_example('LazyLLMGitBase.list_review_comments', '''\
>>> backend.list_review_comments(1)
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

Returns:
    dict: success, comment_id, message.
''')
_add_git_example('LazyLLMGitBase.create_review_comment', '''\
>>> backend.create_review_comment(1, 'Consider refactoring', 'src/foo.py', line=10)
''')

_add_git_chinese('LazyLLMGitBase.submit_review', '''\
提交评审结论（通过 / 请求修改 / 仅评论）。

Args:
    number (int): PR/MR 编号。
    event (str): 事件类型，如 APPROVE、REQUEST_CHANGES、COMMENT。
    body (str): 评审总结正文，可选。
    comment_ids (list, optional): 要一并提交的评论 ID 列表。

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

Returns:
    dict: success, message.
''')
_add_git_example('LazyLLMGitBase.submit_review', '''\
>>> backend.submit_review(1, 'APPROVE', body='LGTM')
''')

_add_git_chinese('LazyLLMGitBase.approve_pull_request', '''\
批准 PR/MR。

Args:
    number (int): PR/MR 编号。

Returns:
    dict: 包含 success、message。
''')
_add_git_english('LazyLLMGitBase.approve_pull_request', '''\
Approve a PR/MR.

Args:
    number (int): PR/MR number.

Returns:
    dict: success, message.
''')
_add_git_example('LazyLLMGitBase.approve_pull_request', '''\
>>> backend.approve_pull_request(1)
''')

_add_git_chinese('LazyLLMGitBase.merge_pull_request', '''\
合并 PR/MR。

Args:
    number (int): PR/MR 编号。
    merge_method (str, optional): 合并方式（如 merge、squash、rebase），依平台而定。
    commit_title (str, optional): 合并提交标题。
    commit_message (str, optional): 合并提交说明。

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

Returns:
    dict: success, sha, message.
''')
_add_git_example('LazyLLMGitBase.merge_pull_request', '''\
>>> backend.merge_pull_request(1, merge_method='squash')
''')

_add_git_chinese('LazyLLMGitBase.list_repo_stargazers', '''\
列出给仓库加星的用户列表。

Args:
    page (int): 页码，默认 1。
    per_page (int): 每页数量，默认 20。

Returns:
    dict: 包含 success、list、message。部分平台可能返回不支持。
''')
_add_git_english('LazyLLMGitBase.list_repo_stargazers', '''\
List users who starred the repository.

Args:
    page (int): Page number, default 1.
    per_page (int): Items per page, default 20.

Returns:
    dict: success, list, message. Some platforms may return not supported.
''')
_add_git_example('LazyLLMGitBase.list_repo_stargazers', '''\
>>> backend.list_repo_stargazers(page=1, per_page=20)
''')

_add_git_chinese('LazyLLMGitBase.reply_to_review_comment', '''\
回复某条评审评论。

Args:
    number (int): PR/MR 编号。
    comment_id: 被回复的评论 ID。
    body (str): 回复内容。
    path (str): 文件路径。
    line (int, optional): 行号。
    commit_id (str, optional): 提交 ID。

Returns:
    dict: 包含 success、comment_id、message。
''')
_add_git_english('LazyLLMGitBase.reply_to_review_comment', '''\
Reply to a review comment.

Args:
    number (int): PR/MR number.
    comment_id: ID of the comment to reply to.
    body (str): Reply body.
    path (str): File path.
    line (int, optional): Line number.
    commit_id (str, optional): Commit ID.

Returns:
    dict: success, comment_id, message.
''')
_add_git_example('LazyLLMGitBase.reply_to_review_comment', '''\
>>> backend.reply_to_review_comment(1, 101, 'Agreed', 'src/foo.py')
''')

_add_git_chinese('LazyLLMGitBase.resolve_review_comment', '''\
将某条评审评论标记为已解决（若平台支持）。

Args:
    number (int): PR/MR 编号。
    comment_id: 评论 ID。

Returns:
    dict: 包含 success、message。
''')
_add_git_english('LazyLLMGitBase.resolve_review_comment', '''\
Mark a review comment as resolved (if supported by the platform).

Args:
    number (int): PR/MR number.
    comment_id: Comment ID.

Returns:
    dict: success, message.
''')
_add_git_example('LazyLLMGitBase.resolve_review_comment', '''\
>>> backend.resolve_review_comment(1, 101)
''')

_add_git_chinese('LazyLLMGitBase.get_user_info', '''\
获取用户信息。

Args:
    username (str, optional): 用户名；不传则返回构造时 user 或 token 对应用户。

Returns:
    dict: 包含 success、user、message。
''')
_add_git_english('LazyLLMGitBase.get_user_info', '''\
Get user profile.

Args:
    username (str, optional): Username; if None, returns instance user or token owner.

Returns:
    dict: success, user, message.
''')
_add_git_example('LazyLLMGitBase.get_user_info', '''\
>>> backend.get_user_info('octocat')
''')

_add_git_chinese('LazyLLMGitBase.list_user_starred_repos', '''\
列出用户加星过的仓库。

Args:
    username (str, optional): 用户名；不传则使用构造时 user 或 token 对应用户。
    page (int): 页码，默认 1。
    per_page (int): 每页数量，默认 20。

Returns:
    dict: 包含 success、list、message。
''')
_add_git_english('LazyLLMGitBase.list_user_starred_repos', '''\
List repositories starred by a user.

Args:
    username (str, optional): Username; if None, uses instance user or token owner.
    page (int): Page number, default 1.
    per_page (int): Items per page, default 20.

Returns:
    dict: success, list, message.
''')
_add_git_example('LazyLLMGitBase.list_user_starred_repos', '''\
>>> backend.list_user_starred_repos(username='octocat')
''')

_add_git_chinese('LazyLLMGitBase.stash_review_comment', '''\
将一条评审评论暂存，之后可用 batch_commit_review_comments 批量提交。

Args:
    number (int): PR/MR 编号。
    body (str): 评论内容。
    path (str): 文件路径。
    line (int, optional): 行号。

Returns:
    dict: 包含 success、message、stash_size。
''')
_add_git_english('LazyLLMGitBase.stash_review_comment', '''\
Stash a review comment for later batch submit via batch_commit_review_comments.

Args:
    number (int): PR/MR number.
    body (str): Comment body.
    path (str): File path.
    line (int, optional): Line number.

Returns:
    dict: success, message, stash_size.
''')
_add_git_example('LazyLLMGitBase.stash_review_comment', '''\
>>> backend.stash_review_comment(1, 'Fix this', 'src/foo.py', line=10)
''')

_add_git_chinese('LazyLLMGitBase.batch_commit_review_comments', '''\
将暂存的评审评论批量提交到 PR/MR。

Args:
    clear_stash (bool): 提交后是否清空暂存，默认 True。

Returns:
    dict: 包含 success、message、created。
''')
_add_git_english('LazyLLMGitBase.batch_commit_review_comments', '''\
Submit all stashed review comments to the PR/MR.

Args:
    clear_stash (bool): Whether to clear stash after submit, default True.

Returns:
    dict: success, message, created.
''')
_add_git_example('LazyLLMGitBase.batch_commit_review_comments', '''\
>>> backend.batch_commit_review_comments(clear_stash=True)
''')

_add_git_chinese('LazyLLMGitBase.check_review_resolution', '''\
检查评审评论是否已解决。默认实现列出评论；子类可覆盖为平台逻辑。

Args:
    number (int): PR/MR 编号。
    comment_ids (list, optional): 要检查的评论 ID 列表。

Returns:
    dict: 包含 success、resolved、comments、message。
''')
_add_git_english('LazyLLMGitBase.check_review_resolution', '''\
Check if review comments are resolved. Default: list comments; override for platform-specific logic.

Args:
    number (int): PR/MR number.
    comment_ids (list, optional): Comment IDs to check.

Returns:
    dict: success, resolved, comments, message.
''')
_add_git_example('LazyLLMGitBase.check_review_resolution', '''\
>>> backend.check_review_resolution(1)
''')

_add_git_example('LazyLLMGitBase', '''\
>>> from lazyllm.tools import git
>>> import lazyllm
>>> backend = lazyllm.tools.git.Git(backend='github', token='xxx', repo='owner/repo')
>>> backend.create_pull_request('feat', 'main', 'Title', 'Body')
>>> backend.merge_pull_request(1)
''')
