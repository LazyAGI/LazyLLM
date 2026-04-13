from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional

from lazyllm import LOG
from ..utils import gen_docid


def to_json(data: Optional[Dict[str, Any]]) -> str:
    return json.dumps(data or {}, ensure_ascii=False)


def from_json(raw: Optional[str]) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        LOG.warning('[DocService] Failed to decode json payload')
        return {}


def gen_doc_id(file_path: str, doc_id: Optional[str] = None) -> str:
    return doc_id or gen_docid(file_path)


def stable_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, default=str)


def hash_payload(data: Any) -> str:
    return hashlib.sha256(stable_json(data).encode()).hexdigest()


def sha256_file(file_path: str) -> str:
    digest = hashlib.sha256()
    with open(file_path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def merge_transfer_metadata(
    source_metadata: Dict[str, Any], target_metadata: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    metadata = dict(source_metadata or {})
    if target_metadata:
        metadata.update(target_metadata)
    return metadata


def resolve_transfer_target_path(
    source_path: str, target_filename: Optional[str], target_file_path: Optional[str]
) -> str:
    if target_file_path:
        return target_file_path
    if target_filename:
        base_dir = os.path.dirname(source_path) if source_path else ''
        return os.path.join(base_dir, target_filename) if base_dir else target_filename
    return source_path


def list_dataset_files(dataset_path: str) -> List[str]:
    """Recursively scan a directory, skip hidden files/dirs, return sorted absolute paths."""
    if not dataset_path or not os.path.exists(dataset_path):
        return []
    if not os.path.isdir(dataset_path):
        filename = os.path.basename(dataset_path)
        if filename.startswith('.'):
            return []
        return [dataset_path] if os.path.isfile(dataset_path) else []

    files: List[str] = []
    for root, dirs, names in os.walk(os.path.abspath(dataset_path)):
        path_parts = root.split(os.sep)
        if any(part.startswith('.') for part in path_parts if part):
            continue
        dirs[:] = [name for name in dirs if not name.startswith('.')]
        files.extend(os.path.join(root, name) for name in names if not name.startswith('.'))
    return sorted(files)


def normalize_api_base_url(url: str) -> str:
    url = url.rstrip('/')
    if url.endswith('/_call') or url.endswith('/generate'):
        return url.rsplit('/', 1)[0]
    return url
