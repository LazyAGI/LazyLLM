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

    def test_extract_protocol_does_not_treat_windows_drive_as_protocol(self):
        assert SkillManager._extract_protocol('C:/Users/test/skills') is None
        assert SkillManager._extract_protocol(r'C:\Users\test\skills') is None
        assert SkillManager._extract_protocol('s3:/remote/skills') == 's3'

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

    def test_run_script_extracts_missing_env_guidance_from_stderr(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = _make_skill(tmp, 'stderr-skill', 'stderr-skill')
            scripts_dir = os.path.join(skill_dir, 'scripts')
            os.makedirs(scripts_dir, exist_ok=True)
            script = os.path.join(scripts_dir, 'needs_key.py')
            with open(script, 'w', encoding='utf-8') as f:
                f.write(
                    'import sys\n'
                    'sys.stderr.write("缺少 API_KEY，请设置环境变量 REDFOX_API_KEY。"\n'
                    '                 "获取方式：https://redfox.hk/settings/api-keys?source=workbuddy")\n'
                    'sys.exit(1)\n'
                )

            manager = SkillManager(dir=tmp)
            result = manager.run_script('stderr-skill', 'scripts/needs_key.py', allow_unsafe=True)

            assert result['status'] == 'failed'
            assert result['error_type'] == 'MissingCredential'
            assert result['missing_env'] == ['REDFOX_API_KEY']
            assert result['api_key_url'] == 'https://redfox.hk/settings/api-keys?source=workbuddy'
            assert result['setup_commands'] == ['export REDFOX_API_KEY="<your REDFOX_API_KEY>"']
            assert 'REDFOX_API_KEY' in result['hint']
            assert 'generic KEY' not in result['hint']
            assert 'Missing REDFOX_API_KEY required by skill stderr-skill.' in result['error']
            assert 'REDFOX_API_KEY (' not in result['error']

    def test_run_script_extracts_missing_env_guidance_from_skill_md_body(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = _make_skill(tmp, 'body-skill', 'body-skill')
            with open(os.path.join(skill_dir, 'SKILL.md'), 'a', encoding='utf-8') as f:
                f.write(
                    '\n## 鉴权\n'
                    '请前往 https://example.test/body-key 获取 API Key。\n'
                    '然后执行 export BODY_API_KEY="ak_xxx" 后再运行技能。\n'
                )
            scripts_dir = os.path.join(skill_dir, 'scripts')
            os.makedirs(scripts_dir, exist_ok=True)
            script = os.path.join(scripts_dir, 'needs_key.py')
            with open(script, 'w', encoding='utf-8') as f:
                f.write('print("should not run")\n')

            previous = os.environ.pop('BODY_API_KEY', None)
            try:
                manager = SkillManager(dir=tmp)
                result = manager.run_script('body-skill', 'scripts/needs_key.py', allow_unsafe=True)
            finally:
                if previous is None:
                    os.environ.pop('BODY_API_KEY', None)
                else:
                    os.environ['BODY_API_KEY'] = previous

            assert result['status'] == 'failed'
            assert result['error_type'] == 'MissingCredential'
            assert result['missing_env'] == ['BODY_API_KEY']
            assert result['api_key_url'] == 'https://example.test/body-key'
            assert result['setup_commands'] == ['export BODY_API_KEY="<your BODY_API_KEY>"']
            assert 'generic KEY' not in result['hint']

    def test_run_script_extracts_missing_env_from_fenced_skill_md(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = _make_skill(tmp, 'fence-skill', 'fence-skill')
            with open(os.path.join(skill_dir, 'SKILL.md'), 'a', encoding='utf-8') as f:
                f.write(
                    '\n## 鉴权\n'
                    '```bash\n'
                    'export FENCE_API_KEY="ak_xxx"\n'
                    '```\n'
                    '获取方式：https://example.test/fence-key\n'
                )
            scripts_dir = os.path.join(skill_dir, 'scripts')
            os.makedirs(scripts_dir, exist_ok=True)
            script = os.path.join(scripts_dir, 'needs_key.py')
            with open(script, 'w', encoding='utf-8') as f:
                f.write('print("should not run")\n')

            previous = os.environ.pop('FENCE_API_KEY', None)
            try:
                manager = SkillManager(dir=tmp)
                result = manager.run_script('fence-skill', 'scripts/needs_key.py', allow_unsafe=True)
            finally:
                if previous is None:
                    os.environ.pop('FENCE_API_KEY', None)
                else:
                    os.environ['FENCE_API_KEY'] = previous

            assert result['status'] == 'failed'
            assert result['error_type'] == 'MissingCredential'
            assert result['missing_env'] == ['FENCE_API_KEY']
            assert result['api_key_url'] == 'https://example.test/fence-key'
            assert result['setup_commands'] == ['export FENCE_API_KEY="<your FENCE_API_KEY>"']

    def test_run_script_runs_when_required_env_is_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = _make_skill(tmp, 'ready-skill', 'ready-skill')
            with open(os.path.join(skill_dir, 'SKILL.md'), 'a', encoding='utf-8') as f:
                f.write(
                    '\n## 鉴权\n'
                    'export READY_API_KEY="ak_xxx"\n'
                )
            scripts_dir = os.path.join(skill_dir, 'scripts')
            os.makedirs(scripts_dir, exist_ok=True)
            script = os.path.join(scripts_dir, 'ok.py')
            with open(script, 'w', encoding='utf-8') as f:
                f.write('print("ok")\n')

            previous = os.environ.get('READY_API_KEY')
            os.environ['READY_API_KEY'] = 'present'
            try:
                manager = SkillManager(dir=tmp)
                result = manager.run_script('ready-skill', 'scripts/ok.py', allow_unsafe=True)
            finally:
                if previous is None:
                    os.environ.pop('READY_API_KEY', None)
                else:
                    os.environ['READY_API_KEY'] = previous

            assert result['status'] == 'ok'
            assert result['stdout'].strip() == 'ok'

    def test_run_script_does_not_treat_unrelated_api_key_mention_as_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = _make_skill(tmp, 'note-skill', 'note-skill')
            with open(os.path.join(skill_dir, 'SKILL.md'), 'a', encoding='utf-8') as f:
                f.write(
                    '\n## Examples\n'
                    'This document may mention SAMPLE_API_KEY in an example, but the skill does not require it.\n'
                )
            scripts_dir = os.path.join(skill_dir, 'scripts')
            os.makedirs(scripts_dir, exist_ok=True)
            script = os.path.join(scripts_dir, 'ok.py')
            with open(script, 'w', encoding='utf-8') as f:
                f.write('print("ok")\n')

            previous = os.environ.pop('SAMPLE_API_KEY', None)
            try:
                manager = SkillManager(dir=tmp)
                result = manager.run_script('note-skill', 'scripts/ok.py', allow_unsafe=True)
            finally:
                if previous is None:
                    os.environ.pop('SAMPLE_API_KEY', None)
                else:
                    os.environ['SAMPLE_API_KEY'] = previous

            assert result['status'] == 'ok'
            assert result['stdout'].strip() == 'ok'

    def test_credential_sections_ignore_code_fences_and_unrelated_headings(self):
        selected = SkillManager._credential_sections_from_skill_doc(
            '# Skill\n'
            '```\n'
            '# Include authoritative source hints\n'
            'export SAMPLE_API_KEY=x\n'
            '```\n'
            '## Development\n'
            'export DEV_API_KEY=x\n'
            '## 配置\n'
            'export PAGE_SIZE=20\n'
            '### 配置 API Key\n'
            'export BODY_API_KEY=x\n'
            'https://example.test/body-key\n'
        )
        assert 'BODY_API_KEY' in selected
        assert 'https://example.test/body-key' in selected
        assert 'SAMPLE_API_KEY' not in selected
        assert 'DEV_API_KEY' not in selected
        assert 'PAGE_SIZE' not in selected

    def test_credential_sections_keep_fenced_export_in_auth_section(self):
        selected = SkillManager._credential_sections_from_skill_doc(
            '# Skill\n'
            '## 鉴权\n'
            '```bash\n'
            'export REDFOX_API_KEY=x\n'
            '```\n'
            'https://redfox.hk/settings/api-keys\n'
        )
        assert 'REDFOX_API_KEY' in selected
        assert 'https://redfox.hk/settings/api-keys' in selected

    def test_credential_entries_prefer_specific_over_generic_names(self):
        entries = SkillManager._credential_entries_from_message(
            'Set API_KEY or REDFOX_API_KEY. Get it from https://redfox.hk/keys\n'
            'export PAGE_SIZE=20\n'
        )
        names = [entry['name'] for entry in entries]
        assert names == ['REDFOX_API_KEY']
        assert entries[0]['url'] == 'https://redfox.hk/keys'

    def test_credential_entries_keep_generic_name_when_it_is_the_only_hint(self):
        entries = SkillManager._credential_entries_from_message('Missing API_KEY')
        assert [entry['name'] for entry in entries] == ['API_KEY']

    def test_run_script_docstring_mentions_missing_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_skill(tmp, 'doc-skill', 'doc-skill')
            manager = SkillManager(dir=tmp)
            run_script = next(tool for tool in manager.get_skill_tools() if tool.__name__ == 'run_script')
            assert 'missing_env' in run_script.__doc__
            assert 'Never replace a concrete variable' in run_script.__doc__

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
            assert result.get('cwd') == 'missing'
            assert 'cwd not found' in result['error']

    def test_run_script_failure_omits_sandbox_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = _make_skill(tmp, 'cwd-skill', 'cwd-skill')
            scripts_dir = os.path.join(skill_dir, 'scripts')
            os.makedirs(scripts_dir, exist_ok=True)
            script = os.path.join(scripts_dir, 'fail.py')
            with open(script, 'w', encoding='utf-8') as f:
                f.write('import sys\nsys.exit(1)\n')

            manager = SkillManager(dir=tmp)
            result = manager.run_script('cwd-skill', 'scripts/fail.py', allow_unsafe=True)

            assert result['status'] == 'failed'
            assert 'cwd' not in result or not os.path.isabs(str(result['cwd']))

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
