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

_add_git_example('LazyLLMGitBase', '''\
>>> from lazyllm.tools import git
>>> import lazyllm
>>> backend = lazyllm.git.GitHub(token='xxx', repo='owner/repo')
>>> backend.create_pull_request('feat', 'main', 'Title', 'Body')
>>> backend.merge_pull_request(1)
''')
