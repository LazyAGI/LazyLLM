# Copyright (c) 2026 LazyAGI. All rights reserved.
import os
import re
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
import json

from lazyllm import config, LOG

_AGENTSKILLHUB_API = 'https://agentskillhub.dev/api/v1'
_GITHUB_RAW = 'https://raw.githubusercontent.com'
_GITHUB_API = 'https://api.github.com'

# source format examples:
#   agentskillhub:username/skill-slug
#   github:owner/repo/path/to/skill          (SKILL.md lives at path/to/skill/SKILL.md)
#   github:owner/repo                         (SKILL.md lives at repo root)
#   https://...                               (direct URL to SKILL.md or a zip)
_SOURCE_RE = re.compile(
    r'^(?P<scheme>[a-zA-Z][a-zA-Z0-9+\-.]*):'
    r'(?P<rest>.+)$'
)


def _http_get(url: str, token: Optional[str] = None) -> bytes:
    headers = {'User-Agent': 'lazyllm-skill-installer/1.0'}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    req = Request(url, headers=headers)
    try:
        with urlopen(req, timeout=30) as resp:
            return resp.read()
    except HTTPError as e:
        raise RuntimeError(f'HTTP {e.code} fetching {url}: {e.reason}') from e
    except URLError as e:
        raise RuntimeError(f'Network error fetching {url}: {e.reason}') from e


def _write_file(dest_dir: str, rel_path: str, content: bytes) -> None:
    # Guard against path traversal (e.g. rel_path = "../../evil")
    dest_dir = os.path.realpath(dest_dir)
    full = os.path.realpath(os.path.join(dest_dir, rel_path))
    if not full.startswith(dest_dir + os.sep) and full != dest_dir:
        raise ValueError(f'Path traversal detected: {rel_path!r} escapes dest_dir {dest_dir!r}')
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, 'wb') as f:
        f.write(content)


# --- agentskillhub.dev ---

def _install_from_agentskillhub(slug: str, dest_dir: str, token: Optional[str] = None) -> str:
    parts = slug.strip('/').split('/', 1)
    if len(parts) != 2:
        raise ValueError(f'agentskillhub source must be "username/skill-slug", got: {slug!r}')
    username, skill_slug = parts
    url = f'{_AGENTSKILLHUB_API}/u/{username}/skills/{skill_slug}'
    LOG.info(f'Fetching skill metadata from {url}')
    data = json.loads(_http_get(url, token=token))

    skill_name = data['skill'].get('slug', skill_slug)
    skill_dir = os.path.join(dest_dir, skill_name)
    os.makedirs(skill_dir, exist_ok=True)

    latest = data.get('latestVersion', {})
    skill_md_raw = latest.get('skillMdRaw')
    file_manifest = latest.get('fileManifest', [])

    # Write SKILL.md from inline content (saves one request)
    if skill_md_raw:
        _write_file(skill_dir, 'SKILL.md', skill_md_raw.encode('utf-8'))

    # Fetch remaining files via GitHub blob API using gitBlobSha
    source_identifier = data['skill'].get('sourceIdentifier', '')  # e.g. "owner/repo"
    skill_path = data['skill'].get('skillPath', '')                 # e.g. "skills/webapp-testing"
    default_branch = data['skill'].get('defaultBranch', 'main')

    for entry in file_manifest:
        rel = entry['path']
        if rel == 'SKILL.md' and skill_md_raw:
            continue  # already written
        blob_sha = entry.get('gitBlobSha')
        if blob_sha and source_identifier:
            owner_repo = source_identifier
            blob_url = f'{_GITHUB_API}/repos/{owner_repo}/git/blobs/{blob_sha}'
            LOG.info(f'Fetching {rel} via blob {blob_sha[:8]}')
            blob_data = json.loads(_http_get(blob_url, token=token))
            import base64
            content = base64.b64decode(blob_data['content'])
        else:
            # Fallback: raw URL from GitHub
            if not source_identifier:
                LOG.warning(f'Cannot fetch {rel}: no sourceIdentifier in skill metadata')
                continue
            raw_url = f'{_GITHUB_RAW}/{source_identifier}/{default_branch}/{skill_path}/{rel}'
            LOG.info(f'Fetching {rel} from {raw_url}')
            content = _http_get(raw_url, token=token)
        _write_file(skill_dir, rel, content)

    LOG.info(f'Skill {skill_name!r} installed to {skill_dir}')
    return skill_dir


# --- GitHub ---

def _install_from_github(spec: str, dest_dir: str, token: Optional[str] = None,
                         branch: Optional[str] = None) -> str:
    # spec: owner/repo  or  owner/repo/path/to/skill
    parts = spec.strip('/').split('/', 2)
    if len(parts) < 2:
        raise ValueError(f'github source must be "owner/repo[/path]", got: {spec!r}')
    owner, repo = parts[0], parts[1]
    skill_path = parts[2].strip('/') if len(parts) == 3 else ''

    # Resolve default branch if not given
    if not branch:
        repo_url = f'{_GITHUB_API}/repos/{owner}/{repo}'
        repo_info = json.loads(_http_get(repo_url, token=token))
        branch = repo_info.get('default_branch', 'main')

    # Use GitHub Trees API to list all files under skill_path
    tree_url = f'{_GITHUB_API}/repos/{owner}/{repo}/git/trees/{branch}?recursive=1'
    LOG.info(f'Fetching file tree from {tree_url}')
    tree_data = json.loads(_http_get(tree_url, token=token))

    prefix = skill_path + '/' if skill_path else ''
    blobs = [
        item for item in tree_data.get('tree', [])
        if item['type'] == 'blob' and item['path'].startswith(prefix)
    ]

    if not blobs:
        raise RuntimeError(
            f'No files found under {spec!r} (branch={branch}). '
            f'Check that the path contains a SKILL.md.'
        )

    # Derive skill name from the last component of skill_path, or repo name
    skill_name = skill_path.rsplit('/', 1)[-1] if skill_path else repo
    skill_dir = os.path.join(dest_dir, skill_name)
    os.makedirs(skill_dir, exist_ok=True)

    import base64
    for item in blobs:
        rel = item['path'][len(prefix):]  # strip the skill_path prefix
        blob_url = f'{_GITHUB_API}/repos/{owner}/{repo}/git/blobs/{item["sha"]}'
        LOG.info(f'Fetching {rel}')
        blob_data = json.loads(_http_get(blob_url, token=token))
        content = base64.b64decode(blob_data['content'])
        _write_file(skill_dir, rel, content)

    if not os.path.exists(os.path.join(skill_dir, 'SKILL.md')):
        raise RuntimeError(
            f'Installed files from {spec!r} but no SKILL.md found. '
            f'Make sure the path points to a skill directory.'
        )

    LOG.info(f'Skill {skill_name!r} installed to {skill_dir}')
    return skill_dir


# --- public API ---

def install_skill(source: str, dest_dir: Optional[str] = None,
                  token: Optional[str] = None, branch: Optional[str] = None) -> str:
    dest_dir = dest_dir or config['skills_dir'].split(',')[0].strip()
    os.makedirs(dest_dir, exist_ok=True)

    m = _SOURCE_RE.match(source)
    if not m:
        raise ValueError(
            f'Cannot parse skill source: {source!r}. '
            f'Expected formats: "agentskillhub:user/slug", "github:owner/repo[/path]", '
            f'or a full URL.'
        )
    scheme = m.group('scheme').lower()
    rest = m.group('rest')

    if scheme == 'agentskillhub':
        return _install_from_agentskillhub(rest, dest_dir, token=token)
    elif scheme == 'github':
        return _install_from_github(rest, dest_dir, token=token, branch=branch)
    elif scheme in ('http', 'https'):
        # Direct URL: treat as raw SKILL.md or a zip (zip not yet supported)
        raise NotImplementedError(
            'Direct URL install is not yet supported. '
            'Use "github:owner/repo/path" or "agentskillhub:user/slug" instead.'
        )
    else:
        raise ValueError(
            f'Unknown skill source scheme: {scheme!r}. '
            f'Supported: agentskillhub, github.'
        )
