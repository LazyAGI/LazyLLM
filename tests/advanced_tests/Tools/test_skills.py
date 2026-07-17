import io
import os
import shutil
import tempfile

import lazyllm
from lazyllm.tools import ReactAgent
from lazyllm.cli.skills import skills as skills_cli
from lazyllm.tools.agent.skill_manager import SkillManager
from lazyllm.tools.fs.base import LazyLLMFSBase


def _make_skill(base_dir: str, folder_name: str, meta_name: str) -> str:
    skill_dir = os.path.join(base_dir, folder_name)
    os.makedirs(skill_dir, exist_ok=True)
    skill_md = os.path.join(skill_dir, 'SKILL.md')
    with open(skill_md, 'w', encoding='utf-8') as f:
        f.write(
            '---\n'
            f'name: {meta_name}\n'
            f'description: {meta_name} skill for tests\n'
            '---\n'
            f'# {meta_name}\n'
            'Test skill\n'
        )
    return skill_dir


class _MemoryCloudFS(LazyLLMFSBase):
    protocol = 'memory'
    _fs_protocol_key = 'memory'

    def __init__(self, entries, files, info_error=False, include_size=True):
        self._entries = entries
        self._files = files
        self._info_error = info_error
        self._include_size = include_size

    def _setup_auth(self):
        pass

    def ls(self, path: str, detail: bool = True, **kwargs):
        return self._entries.get(path, [])

    def info(self, path: str, **kwargs):
        if self._info_error:
            raise RuntimeError('info unavailable')
        info = {'name': path}
        if self._include_size and path in self._files:
            info['size'] = len(self._files[path])
        return info

    def _open(self, path: str, mode: str = 'rb', block_size=None, autocommit: bool = True,
              cache_options=None, **kwargs):
        return self.open(path, mode=mode, **kwargs)

    def open(self, path: str, mode: str = 'rb', **kwargs):
        body = io.BytesIO(self._files[path])
        if 'b' in mode:
            return body
        return io.TextIOWrapper(body, encoding=kwargs.get('encoding') or 'utf-8',
                                errors=kwargs.get('errors') or 'strict')

    def exists(self, path: str, **kwargs):
        return path in self._files or path in self._entries


class TestSkills(object):
    @classmethod
    def setup_class(cls):
        cls._home = lazyllm.config['home']
        cls._skills_dir = lazyllm.config['skills_dir']
        cls._src_root = os.path.join(cls._home, '_test_skills_src')
        os.makedirs(cls._src_root, exist_ok=True)
        cls._alpha_folder = 'test-alpha'
        cls._beta_folder = 'test-beta'
        cls._alpha_name = 'test-alpha'
        cls._beta_name = 'test-beta'
        _make_skill(cls._src_root, cls._alpha_folder, cls._alpha_name)
        _make_skill(cls._src_root, cls._beta_folder, cls._beta_name)

    @classmethod
    def teardown_class(cls):
        for folder in (cls._alpha_folder, cls._beta_folder):
            path = os.path.join(cls._skills_dir, folder)
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
        if os.path.isdir(cls._src_root):
            shutil.rmtree(cls._src_root, ignore_errors=True)

    def test_skills_cli(self):
        skills_cli(['init'])
        assert os.path.isdir(self._skills_dir)

        skills_cli(['import', self._src_root])
        assert os.path.isdir(os.path.join(self._skills_dir, self._alpha_folder))
        assert os.path.isdir(os.path.join(self._skills_dir, self._beta_folder))

    def test_skill_manager(self):
        manager = SkillManager(dir=self._skills_dir)
        listing = manager.list_skill()
        assert self._alpha_name in listing
        assert self._beta_name in listing

    def test_parse_dirs_local_expands_paths(self):
        parsed = SkillManager._parse_dirs('~/skills')
        assert parsed == [os.path.abspath(os.path.expanduser('~/skills'))]

    def test_parse_dirs_cloud_preserves_paths(self):
        parsed = SkillManager._parse_dirs('s3:/remote/skills')
        assert parsed == ['s3:/remote/skills']

    def test_parse_dirs_non_local_fs_preserves_bare_paths(self):
        fs = _MemoryCloudFS({}, {})

        parsed = SkillManager._parse_dirs('skills', fs=fs)
        manager = SkillManager(dir='skills', fs=fs)

        assert parsed == ['skills']
        assert manager._skills_dir == ['skills']

    def test_skill_manager_uses_content_when_info_fails(self):
        fs = _MemoryCloudFS(
            {
                'skills': [{'name': 'skills/demo', 'type': 'directory'}],
                'skills/demo': [{'name': 'skills/demo/SKILL.md', 'type': 'file'}],
            },
            {
                'skills/demo/SKILL.md': (
                    b'---\n'
                    b'name: demo\n'
                    b'description: demo skill for tests\n'
                    b'---\n'
                    b'# Demo\n'
                ),
            },
            info_error=True,
        )
        manager = SkillManager(dir='skills', fs=fs)

        listing = manager.list_skill()
        skill = manager.get_skill('demo')

        assert 'demo skill for tests' in listing
        assert skill['status'] == 'ok'
        assert '# Demo' in skill['content']

    def test_invalid_required_metadata_type_does_not_block_valid_skills(self):
        fs = _MemoryCloudFS(
            {
                'skills': [
                    {'name': 'skills/valid-skill', 'type': 'directory'},
                    {'name': 'skills/bad-name', 'type': 'directory'},
                    {'name': 'skills/bad-description', 'type': 'directory'},
                ],
                'skills/valid-skill': [
                    {'name': 'skills/valid-skill/SKILL.md', 'type': 'file'},
                ],
                'skills/bad-name': [
                    {'name': 'skills/bad-name/SKILL.md', 'type': 'file'},
                ],
                'skills/bad-description': [
                    {'name': 'skills/bad-description/SKILL.md', 'type': 'file'},
                ],
            },
            {
                'skills/valid-skill/SKILL.md': (
                    b'---\n'
                    b'name: valid-skill\n'
                    b'description: valid skill remains available\n'
                    b'---\n'
                    b'# Valid Skill\n'
                ),
                'skills/bad-name/SKILL.md': (
                    b'---\n'
                    b'name: 123\n'
                    b'description: invalid name type\n'
                    b'---\n'
                    b'# Bad Name\n'
                ),
                'skills/bad-description/SKILL.md': (
                    b'---\n'
                    b'name: bad-description\n'
                    b'description: 123\n'
                    b'---\n'
                    b'# Bad Description\n'
                ),
            },
        )
        manager = SkillManager(dir='skills', fs=fs)

        prompt = manager.build_prompt()

        assert 'valid skill remains available' in prompt
        assert 'skills/bad-name' not in prompt
        assert 'skills/bad-description' not in prompt

    def test_skill_manager_enforces_size_limit_when_info_has_no_size(self):
        fs = _MemoryCloudFS(
            {
                'skills': [{'name': 'skills/large', 'type': 'directory'}],
                'skills/large': [{'name': 'skills/large/SKILL.md', 'type': 'file'}],
            },
            {
                'skills/large/SKILL.md': (
                    b'---\n'
                    b'name: large\n'
                    b'description: large skill for tests\n'
                    b'---\n'
                    b'# Large\n'
                    b'x' * 128
                ),
            },
            include_size=False,
        )
        manager = SkillManager(dir='skills', fs=fs, max_skill_md_bytes=64)

        listing = manager.list_skill()

        assert 'large skill for tests' not in listing

    def test_run_script_materializes_non_local_fs_with_bare_dir(self):
        fs = _MemoryCloudFS(
            {
                'skills': [{'name': 'skills/script-skill', 'type': 'directory'}],
                'skills/script-skill': [
                    {'name': 'skills/script-skill/SKILL.md', 'type': 'file'},
                    {'name': 'skills/script-skill/scripts', 'type': 'directory'},
                ],
                'skills/script-skill/scripts': [
                    {'name': 'skills/script-skill/scripts/ok.py', 'type': 'file'},
                ],
            },
            {
                'skills/script-skill/SKILL.md': (
                    b'---\n'
                    b'name: script-skill\n'
                    b'description: script skill for tests\n'
                    b'---\n'
                    b'# Script Skill\n'
                ),
                'skills/script-skill/scripts/ok.py': b'print("ok")\n',
            },
        )
        manager = SkillManager(dir='skills', fs=fs)

        result = manager.run_script('script-skill', 'scripts/ok.py', allow_unsafe=True)

        assert result['status'] == 'ok'
        assert result['exit_code'] == 0
        assert result['stdout'] == 'ok\n'

    def test_run_script_marks_nonzero_exit_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = _make_skill(tmp, 'script-skill', 'script-skill')
            scripts_dir = os.path.join(skill_dir, 'scripts')
            os.makedirs(scripts_dir, exist_ok=True)
            ok_script = os.path.join(scripts_dir, 'ok.py')
            fail_script = os.path.join(scripts_dir, 'fail.py')
            with open(ok_script, 'w', encoding='utf-8') as f:
                f.write('print("ok")\n')
            with open(fail_script, 'w', encoding='utf-8') as f:
                f.write('import sys\nprint("bad")\nsys.exit(7)\n')

            manager = SkillManager(dir=tmp)

            ok_result = manager.run_script('script-skill', 'scripts/ok.py', allow_unsafe=True)
            fail_result = manager.run_script('script-skill', 'scripts/fail.py', allow_unsafe=True)

            assert ok_result['status'] == 'ok'
            assert ok_result['exit_code'] == 0
            assert fail_result['status'] == 'failed'
            assert fail_result['exit_code'] == 7
            assert 'bad' in fail_result['stdout']

    def test_run_script_reports_missing_cwd_as_tool_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = _make_skill(tmp, 'cwd-skill', 'cwd-skill')
            scripts_dir = os.path.join(skill_dir, 'scripts')
            os.makedirs(scripts_dir, exist_ok=True)
            script = os.path.join(scripts_dir, 'ok.py')
            with open(script, 'w', encoding='utf-8') as f:
                f.write('print("ok")\n')

            manager = SkillManager(dir=tmp)
            result = manager.run_script('cwd-skill', 'scripts/ok.py', allow_unsafe=True, cwd='missing')

            assert result['status'] == 'error'
            assert result['error_type'] == 'FileNotFoundError'
            assert result['rel_path'] == 'scripts/ok.py'
            assert result['cwd'].endswith(os.path.join('cwd-skill', 'missing'))
            assert 'cwd not found' in result['error']

    def test_materialize_dir_preserves_paths_when_root_is_empty(self):
        fs = _MemoryCloudFS(
            {
                '': [{'name': 'pkg', 'type': 'directory'}],
                'pkg': [
                    {'name': 'pkg/SKILL.md', 'type': 'file'},
                    {'name': 'pkg/scripts', 'type': 'directory'},
                ],
                'pkg/scripts': [{'name': 'pkg/scripts/run.py', 'type': 'file'}],
            },
            {
                'pkg/SKILL.md': b'# skill\n',
                'pkg/scripts/run.py': b'print("ok")\n',
            },
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = fs.materialize_dir('', tmp)

            assert result['files'] == ['pkg/SKILL.md', 'pkg/scripts/run.py']
            assert os.path.exists(os.path.join(tmp, 'pkg', 'SKILL.md'))
            assert os.path.exists(os.path.join(tmp, 'pkg', 'scripts', 'run.py'))
            assert not os.path.exists(os.path.join(tmp, 'SKILL.md'))
            assert not os.path.exists(os.path.join(tmp, 'run.py'))

    def test_materialize_dir_rejects_paths_that_escape_local_dir(self):
        fs = _MemoryCloudFS(
            {'root': [{'name': 'root/..', 'type': 'file'}]},
            {'root/..': b'bad\n'},
        )
        with tempfile.TemporaryDirectory() as tmp:
            try:
                fs.materialize_dir('root', tmp)
            except RuntimeError as exc:
                assert 'invalid relative path' in str(exc)
            else:
                raise AssertionError('expected materialize_dir to reject parent path segments')

    def test_react_agent_with_skills(self):
        llm = lazyllm.TrainableModule('Qwen2.5-32B-Instruct')
        agent = ReactAgent(llm=llm, skills=[self._alpha_name, self._beta_name],
                           skills_dir=self._src_root)
        res = agent('what skills do you have?')
        assert self._alpha_name in res
        assert self._beta_name in res
