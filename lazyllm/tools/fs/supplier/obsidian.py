# Copyright (c) 2026 LazyAGI. All rights reserved.
'''Local file access for Obsidian Vaults.

``OBSIDIAN_VAULT_PATH`` remains compatible with a single Vault.  It may also
point at a scan root: every directory below it that contains an ``.obsidian``
directory is then available as a Vault.
'''
import hashlib
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, unquote, urlsplit
from uuid import uuid4

from lazyllm import config
from ..base import LazyLLMFSBase, CloudFSBufferedFile

config.add('obsidian_vault_path', str, None, 'OBSIDIAN_VAULT_PATH', description='Obsidian Vault or scan root path.')
config.add(
    'obsidian_host_root', str, None, 'OBSIDIAN_HOST_ROOT',
    description='Host path corresponding to the Obsidian scan root.',
)


OBSIDIAN_IMAGE_SUFFIXES = frozenset({
    '.bmp', '.gif', '.jpeg', '.jpg', '.png', '.svg', '.tif', '.tiff', '.webp',
})


@dataclass(frozen=True)
class ObsidianVault:
    vault_id: str
    root: Path
    display_name: str


@dataclass(frozen=True)
class ObsidianNote:
    vault: ObsidianVault
    relative_path: str
    path: Path


_VAULT_CACHE_TTL_SECONDS = 30.0
_VAULT_DISCOVERY_CACHE: Dict[str, tuple[float, tuple[ObsidianVault, ...]]] = {}


class ObsidianFS(LazyLLMFSBase):

    '''File-system backed Obsidian access plus small Vault discovery helpers.'''

    def __init__(
        self,
        token: str = '',
        base_url: Optional[str] = None,
        asynchronous: bool = False,
        use_listings_cache: bool = False,
        skip_instance_cache: bool = False,
        loop: Optional[Any] = None,
        dynamic_auth: bool = False,
    ):
        token = str(token or config['obsidian_vault_path'] or '').strip()
        vault = (token or '').strip() or '.'
        self._vault_root = os.path.abspath(os.path.expanduser(vault))
        super().__init__(
            token=vault,
            base_url=base_url,
            asynchronous=asynchronous,
            use_listings_cache=use_listings_cache,
            skip_instance_cache=skip_instance_cache,
            loop=loop,
            dynamic_auth=dynamic_auth,
        )

    @classmethod
    def discover_vaults_for_root(cls, root: str) -> List[ObsidianVault]:
        root_path = Path(str(root or '').strip()).expanduser().resolve()
        if not root_path.is_dir():
            return []
        cache_key = str(root_path)
        now = time.monotonic()
        cached = _VAULT_DISCOVERY_CACHE.get(cache_key)
        if cached and now - cached[0] < _VAULT_CACHE_TTL_SECONDS:
            return list(cached[1])

        vault_paths = []
        for current, directories, _ in os.walk(root_path, topdown=True, followlinks=False):
            path = Path(current)
            if (path / '.obsidian').is_dir():
                vault_paths.append(path)
                directories[:] = []

        vaults = tuple(sorted((
            ObsidianVault(
                vault_id='vlt_' + hashlib.sha256(str(path).encode('utf-8')).hexdigest()[:16],
                root=path,
                display_name=path.name or str(path),
            )
            for path in vault_paths
        ), key=lambda item: (item.display_name.lower(), str(item.root))))
        _VAULT_DISCOVERY_CACHE[cache_key] = (now, vaults)
        return list(vaults)

    def discover_vaults(self) -> List[ObsidianVault]:
        return self.discover_vaults_for_root(self._vault_root)

    def resolve_locator(self, locator: str) -> ObsidianNote:
        value = str(locator or '').strip()
        parsed = urlsplit(value)
        if parsed.scheme.lower() != 'obsidian':
            raise ValueError('Invalid Obsidian locator.')
        vaults = self.discover_vaults()
        vault: Optional[ObsidianVault] = None
        relative_path = ''
        if parsed.netloc == 'open':
            query = parse_qs(parsed.query)
            vault_name = str((query.get('vault') or [''])[0]).strip()
            relative_path = unquote(str((query.get('file') or [''])[0]).strip())
            matches = [item for item in vaults if item.display_name == vault_name]
            if len(matches) != 1:
                raise ValueError('Obsidian Vault name is not uniquely configured; use canonical URI.')
            vault = matches[0]
        else:
            vault = next((item for item in vaults if item.vault_id == parsed.netloc), None)
            relative_path = unquote(parsed.path.lstrip('/'))
        if vault is None or not relative_path:
            raise ValueError('Obsidian note locator does not identify a configured note.')
        return self._note_from_relative(vault, relative_path)

    def _note_from_relative(self, vault: ObsidianVault, relative_path: str) -> ObsidianNote:
        candidate = Path(relative_path)
        if candidate.suffix.lower() != '.md':
            candidate = candidate.with_suffix('.md')
        full_path = (vault.root / candidate).resolve()
        try:
            full_path.relative_to(vault.root)
        except ValueError as exc:
            raise ValueError('Obsidian note path is outside the Vault.') from exc
        if not full_path.is_file():
            raise FileNotFoundError('Obsidian note was not found.')
        relative = full_path.relative_to(vault.root).as_posix()
        return ObsidianNote(vault=vault, relative_path=relative, path=full_path)

    def read_note(self, locator: str) -> tuple[ObsidianNote, str]:
        note = self.resolve_locator(locator)
        return note, note.path.read_text(encoding='utf-8')

    def resolve_image_reference(
        self,
        note: ObsidianNote,
        reference: str,
        *,
        markdown_relative: bool = False,
    ) -> Path:
        '''Resolve one local image reference without leaving the source Vault.

        Explicit paths use the syntax's normal base directory.  A bare image
        filename is accepted only when it has exactly one match in the Vault.
        '''
        target = self._image_reference_target(reference)
        if Path(target).suffix.lower() not in OBSIDIAN_IMAGE_SUFFIXES:
            raise ValueError('Obsidian image format is not supported by Writer.')

        if '/' not in target and not target.startswith('.'):
            matches = self._find_unique_image_name(note.vault, target)
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise ValueError('Obsidian image reference is ambiguous in this Vault.')
            raise FileNotFoundError('Obsidian image reference was not found.')

        relative = target.lstrip('/')
        if target.startswith('/'):
            candidate = note.vault.root / relative
        elif target.startswith(('./', '../')) or markdown_relative:
            candidate = note.path.parent / relative
        else:
            candidate = note.vault.root / relative
        resolved = self._vault_image_path(note.vault, candidate)
        if resolved is None:
            raise FileNotFoundError('Obsidian image reference was not found.')
        return resolved

    def display_note_path(self, note: ObsidianNote) -> str:
        '''Return the host-local path that corresponds to a resolved note.'''
        host_root = str(config['obsidian_host_root'] or '').strip()
        if not host_root:
            return str(note.path)
        relative = note.path.resolve().relative_to(Path(self._vault_root).resolve())
        if re.match(r'^[A-Za-z]:[\\/]', host_root):
            return host_root.rstrip('\\/') + '\\' + relative.as_posix().replace('/', '\\')
        return str(Path(host_root).expanduser() / relative)

    def write_note(self, note: ObsidianNote, content: str) -> None:
        temporary = note.path.with_suffix(f'{note.path.suffix}.{uuid4().hex}.lazymind-tmp')
        temporary.write_text(content, encoding='utf-8')
        os.replace(temporary, note.path)

    def create_note(self, title: str) -> ObsidianNote:
        '''Create a Markdown note in the first discovered Vault without overwriting one.'''
        vaults = self.discover_vaults()
        if not vaults:
            raise FileNotFoundError('No Obsidian Vault was found under OBSIDIAN_VAULT_PATH.')
        vault = vaults[0]
        stem = self._note_stem(title)
        index = 1
        while True:
            suffix = '' if index == 1 else f' {index}'
            path = vault.root / f'{stem}{suffix}.md'
            try:
                with path.open('x', encoding='utf-8'):
                    pass
            except FileExistsError:
                index += 1
                continue
            return ObsidianNote(vault=vault, relative_path=path.name, path=path)

    @staticmethod
    def _note_stem(title: str) -> str:
        value = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', '-', str(title or '').strip())
        value = ' '.join(value.split()).strip(' .')
        return value or '未命名文档'

    def copy_attachment(self, note: ObsidianNote, source: str | Path) -> str:
        source_path = Path(source).resolve()
        if not source_path.is_file():
            raise FileNotFoundError('Writer image asset was not found.')
        destination_dir = note.vault.root / 'assets' / 'lazymind'
        destination_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()[:12]
        suffix = source_path.suffix or '.png'
        destination = destination_dir / f'{digest}{suffix.lower()}'
        if not destination.exists():
            shutil.copy2(source_path, destination)
        return destination.relative_to(note.vault.root).as_posix()

    @staticmethod
    def _image_reference_target(reference: str) -> str:
        target = unquote(str(reference or '').split('|', 1)[0]).split('#', 1)[0].strip()
        parsed = urlsplit(target)
        if not target or parsed.scheme or target.startswith('//'):
            raise ValueError('Obsidian image reference must be a local Vault path.')
        return target

    @staticmethod
    def _vault_image_path(vault: ObsidianVault, candidate: Path) -> Path | None:
        resolved = candidate.resolve()
        try:
            resolved.relative_to(vault.root)
        except ValueError:
            return None
        if '.obsidian' in resolved.relative_to(vault.root).parts:
            return None
        return resolved if resolved.is_file() else None

    @classmethod
    def _find_unique_image_name(cls, vault: ObsidianVault, filename: str) -> List[Path]:
        matches: List[Path] = []
        for candidate in vault.root.rglob(filename):
            if '.obsidian' in candidate.relative_to(vault.root).parts:
                continue
            resolved = cls._vault_image_path(vault, candidate)
            if resolved is not None:
                matches.append(resolved)
        return matches

    def _setup_auth(self) -> None:
        if not os.path.isdir(self._vault_root):
            raise FileNotFoundError(
                'Obsidian vault path is not a directory: %r' % (self._vault_root,)
            )

    def _abspath(self, path: str) -> str:
        parts = self._parse_path(path)
        vault_p = Path(self._vault_root).resolve()
        if not (vault_p / '.obsidian').is_dir():
            raise PermissionError(
                'Generic ObsidianFS operations require a single Vault root; '
                'scan-root mode is discovery/provider-only.'
            )
        if not parts:
            return str(vault_p)
        full = vault_p.joinpath(*parts).resolve()
        try:
            full.relative_to(vault_p)
        except ValueError:
            raise PermissionError(
                'Path %r escapes Obsidian vault %r' % (path, self._vault_root)
            )
        return str(full)

    def _relpath(self, full_path: str) -> str:
        return os.path.relpath(full_path, self._vault_root).replace(os.sep, '/')

    def ls(self, path: str, detail: bool = True, **kwargs) -> List:
        full_dir = self._abspath(path)
        if not os.path.isdir(full_dir):
            raise FileNotFoundError(path)
        prefix = path.strip('/')
        results = []
        for name in sorted(os.listdir(full_dir)):
            if name.startswith('.'):
                continue
            child_full = os.path.join(full_dir, name)
            rel = prefix + '/' + name if prefix else name
            if detail:
                st = os.stat(child_full)
                ftype = 'directory' if os.path.isdir(child_full) else 'file'
                sz = 0 if os.path.isdir(child_full) else st.st_size
                results.append(
                    self._entry(name=rel, size=sz, ftype=ftype, mtime=st.st_mtime)
                )
            else:
                results.append(rel)
        return results

    def info(self, path: str, **kwargs) -> Dict[str, Any]:
        parts = self._parse_path(path)
        if not parts:
            return self._entry(name='/', ftype='directory')
        full = self._abspath(path)
        if not os.path.exists(full):
            raise FileNotFoundError(path)
        rel = self._relpath(full)
        if os.path.isdir(full):
            return self._entry(name=rel, ftype='directory')
        st = os.stat(full)
        return self._entry(name=rel, size=st.st_size, ftype='file', mtime=st.st_mtime)

    def _open(self, path: str, mode: str = 'rb',
              block_size: Optional[int] = None,
              autocommit: bool = True,
              cache_options: Optional[Dict] = None,
              **kwargs) -> CloudFSBufferedFile:
        return CloudFSBufferedFile(
            self, path, mode=mode,
            block_size=block_size or self.blocksize,
            autocommit=autocommit, cache_options=cache_options,
        )

    def mkdir(self, path: str, create_parents: bool = True, **kwargs) -> None:
        os.makedirs(self._abspath(path), exist_ok=create_parents)

    def rmdir(self, path: str) -> None:
        full = self._abspath(path)
        if not os.path.exists(full):
            raise FileNotFoundError(path)
        if not os.path.isdir(full):
            raise NotADirectoryError(path)
        os.rmdir(full)

    def rm_file(self, path: str) -> None:
        full = self._abspath(path)
        if os.path.isdir(full):
            raise IsADirectoryError(path)
        os.remove(full)

    def copy(self, path1: str, path2: str, recursive: bool = False, **kwargs) -> None:
        src, dst = self._abspath(path1), self._abspath(path2)
        if os.path.isdir(src):
            if not recursive:
                raise ValueError(f'Cannot copy directory {path1} without recursive=True')
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    def move(self, path1: str, path2: str, recursive: bool = False, **kwargs) -> None:
        if os.path.isdir(self._abspath(path1)) and not recursive:
            raise ValueError(f'Cannot move directory {path1} without recursive=True')
        shutil.move(self._abspath(path1), self._abspath(path2))

    def _download_range(self, path: str, start: int, end: int) -> bytes:
        full = self._abspath(path)
        with open(full, 'rb') as fh:
            fh.seek(start)
            return fh.read(end - start)

    def _upload_data(self, path: str, data: bytes) -> None:
        full = self._abspath(path)
        parent = os.path.dirname(full)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        with open(full, 'wb') as fh:
            fh.write(data)
