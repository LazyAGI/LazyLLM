import io
import os
import shutil
import tempfile

import lazyllm
import pytest
from lazyllm.tools import ReactAgent
from lazyllm.tools.agent import ToolExecutionError
from lazyllm.cli.skills import skills as skills_cli
from lazyllm.tools.agent.skill_manager import SkillManager
from lazyllm.tools.fs.base import LazyLLMFSBase


def _make_skill(base_dir: str, folder_name: str, meta_name: str, description: str | None = None,
                extra_meta: str = '') -> str:
    skill_dir = os.path.join(base_dir, folder_name)
    os.makedirs(skill_dir, exist_ok=True)
    skill_md = os.path.join(skill_dir, 'SKILL.md')
    desc = description or f'{meta_name} skill for tests'
    extra = f'{extra_meta.rstrip()}\n' if extra_meta else ''
    with open(skill_md, 'w', encoding='utf-8') as f:
        f.write(
            '---\n'
            f'name: {meta_name}\n'
            f'description: {desc}\n'
            f'{extra}'
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

    def test_prompt_catalog_does_not_limit_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_skill(tmp, 'resident', 'resident')
            _make_skill(tmp, 'on-demand', 'on-demand')
            manager = SkillManager(dir=tmp, skills=['resident', 'on-demand'], prompt_skills=['resident'])
            prompt = manager.build_prompt()
            assert 'resident skill for tests' in prompt
            assert 'on-demand skill for tests' not in prompt
            assert 'source:' not in prompt
            assert 'Test skill' in manager.get_skill('on-demand')['content']
            assert 'on-demand skill for tests' not in ''.join(
                part['content'] for part in manager.describe_prompt()
            )
            manager.set_prompt_skills([])
            assert 'resident skill for tests' not in manager.build_prompt()
            assert 'Test skill' in manager.get_skill('on-demand')['content']
            manager.set_prompt_skills(None)
            restored = manager.build_prompt()
            assert 'resident skill for tests' in restored
            assert 'on-demand skill for tests' in restored
            assert 'source:' in restored

    def test_inherit_skill_scope_keeps_loadable_and_filters_catalog(self):
        from lazyllm.tools.agent.skill_manager import inherit_skill_scope
        loadable, catalog = inherit_skill_scope(
            ['design/image', 'research/deep'],
            ['research/deep', 'missing/gone'],
            extra_catalog=['design/image'],
        )
        assert loadable == ['design/image', 'research/deep']
        assert catalog == ['research/deep', 'design/image']

    def test_search_tools_use_backend_without_shrinking_loadable_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_skill(tmp, 'resident', 'resident')
            _make_skill(tmp, 'on-demand', 'on-demand')
            captured = {}

            def backend(request):
                captured['request'] = request
                return {'skills': [
                    {'skill_key': 'on-demand', 'name': 'on-demand', 'description': 'hidden until search'},
                    {'skill_key': 'denied', 'name': 'denied', 'description': 'out of scope'},
                ]}

            manager = SkillManager(
                dir=tmp, skills=['resident', 'on-demand'], prompt_skills=['resident'],
                excluded_skills=['denied'], skill_search=backend,
            )
            assert manager.list_prompt_skills()['skills'] == ['resident']
            result = manager.search_skill('anything')
            assert captured['request']['allowed_skill_keys'] == ['on-demand', 'resident']
            assert result['hits'] == [{
                'skill_key': 'on-demand',
                'name': 'on-demand',
                'description': 'hidden until search',
                'match_reason': 'hidden until search',
            }]
            assert result['has_more'] is False
            names = [tool.__name__ for tool in manager.get_skill_tools()]
            assert names == ['search_skill', 'get_skill', 'read_skill_resource', 'run_skill_script']
            assert 'filters' not in captured['request']

    def test_search_skill_uses_local_catalog_without_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_skill(tmp, 'writing/xiaohongshu-copywriter', 'xiaohongshu-copywriter')
            _make_skill(tmp, 'learning/anki', 'anki')
            manager = SkillManager(dir=tmp)
            names = [tool.__name__ for tool in manager.get_skill_tools()]
            assert names[0] == 'search_skill'
            hits = manager.search_skill('xiaohongshu copy')['hits']
            assert [item['skill_key'] for item in hits] == ['writing/xiaohongshu-copywriter']

    def test_local_search_ranks_chinese_and_rejects_noise(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_skill(
                tmp, 'writing/article', 'article',
                description='Long-form article and blog writing',
            )
            _make_skill(
                tmp, 'design/diagram-design', 'diagram-design',
                description='Visual diagrams posters and image layout in SVG',
            )
            _make_skill(
                tmp, 'learning/vocabulary-learning', 'vocabulary-learning',
                description='Vocabulary flashcards and spaced repetition review',
            )
            _make_skill(tmp, 'career/resume', 'resume', description='Resume generator')
            manager = SkillManager(dir=tmp)
            writing_hits = [item['skill_key'] for item in manager.search_skill(
                '帮我写文章', enforce_budget=False,
            )['hits']]
            assert writing_hits[0] == 'writing/article'
            image_hits = [item['skill_key'] for item in manager.search_skill(
                '生成图片 图像生成 视觉设计 海报 配图', enforce_budget=False,
            )['hits']]
            assert image_hits[0] == 'design/diagram-design'
            anki_hits = [item['skill_key'] for item in manager.search_skill(
                'anki', enforce_budget=False,
            )['hits']]
            assert anki_hits[0] == 'learning/vocabulary-learning'
            assert manager.search_skill('a', enforce_budget=False)['hits'] == []
            _make_skill(tmp, 'tiny/a', 'a', description='Named exactly a')
            manager = SkillManager(dir=tmp)
            assert [item['skill_key'] for item in manager.search_skill(
                'a', enforce_budget=False,
            )['hits']] == ['tiny/a']
            mixed = [item['skill_key'] for item in manager.search_skill(
                'writing article', enforce_budget=False,
            )['hits']]
            assert mixed[0] == 'writing/article'
            assert [tool.__name__ for tool in manager.get_skill_tools()][:2] == ['search_skill', 'get_skill']
            discovery = SkillManager(dir=tmp, skill_tool_mode='discovery')
            assert [tool.__name__ for tool in discovery.get_skill_tools()] == ['search_skill', 'get_skill']

    def test_search_skill_has_more_when_catalog_exceeds_max_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            for index in range(25):
                _make_skill(
                    tmp, f'noise/skill-{index:02d}', f'skill-{index:02d}',
                    description=f'Generic helper for miscellaneous tasks number {index}',
                )
            _make_skill(
                tmp, 'writing/xiaohongshu-copywriter', 'xiaohongshu-copywriter',
                description='Xiaohongshu copywriting and travel product posts',
                extra_meta='field: writing',
            )
            manager = SkillManager(dir=tmp)
            assert manager.list_prompt_skills()['count'] == 26
            wide = manager.search_skill(
                'generic helper miscellaneous tasks', limit=20, enforce_budget=False,
            )
            assert len(wide['hits']) == 20
            assert wide['has_more'] is True
            hits = [item['skill_key'] for item in manager.search_skill(
                'xiaohongshu copywriting', enforce_budget=False,
            )['hits']]
            assert hits[0] == 'writing/xiaohongshu-copywriter'

    def test_get_skill_resolves_unique_name_and_basename(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_skill(tmp, 'writing/xiaohongshu-copywriter', 'xiaohongshu-copywriter')
            _make_skill(tmp, 'external/okr-writer', 'okr-writer')
            manager = SkillManager(dir=tmp)
            assert manager.get_skill('xiaohongshu-copywriter')['skill_key'] == (
                'writing/xiaohongshu-copywriter'
            )
            assert manager.get_skill('writing/xiaohongshu-copywriter')['skill_key'] == (
                'writing/xiaohongshu-copywriter'
            )
            assert manager.get_skill('`xiaohongshu-copywriter`')['skill_key'] == (
                'writing/xiaohongshu-copywriter'
            )
            unresolved = manager.get_skill('external/xiaohongshu-copywriter')
            assert unresolved['code'] == 'skill_not_installed'

    def test_get_skill_reports_resolution_and_visibility_states_without_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_skill(tmp, 'writing/enabled', 'enabled')
            _make_skill(tmp, 'external/enabled', 'enabled')
            _make_skill(tmp, 'writing/disabled', 'disabled')
            _make_skill(tmp, 'writing/hidden', 'hidden')
            _make_skill(tmp, 'external/shared', 'shared')
            _make_skill(tmp, 'writing/shared', 'shared')
            manager = SkillManager(
                dir=tmp,
                skills=['writing/enabled', 'writing/hidden'],
                excluded_skills=['writing/hidden'],
            )

            before = manager._search_count
            assert manager.get_skill('enabled')['skill_key'] == 'writing/enabled'
            assert manager._search_count == before
            assert manager.get_skill('external/enabled')['code'] == 'skill_disabled'
            assert manager.get_skill('disabled')['code'] == 'skill_disabled'
            assert manager.get_skill('hidden')['code'] == 'skill_exists_but_not_visible'
            assert manager.get_skill('missing')['code'] == 'skill_not_installed'
            assert manager.get_skill('shared')['code'] == 'identifier_not_resolved'
            assert manager.get_skill('')['code'] == 'identifier_not_resolved'

    def test_search_skill_enforces_budget_and_resource_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = _make_skill(tmp, 'demo', 'demo')
            os.makedirs(os.path.join(skill_dir, 'references'), exist_ok=True)
            with open(os.path.join(skill_dir, 'references', 'guide.md'), 'w', encoding='utf-8') as handle:
                handle.write('guide')

            def backend(request):
                return {'skills': [
                    {'skill_key': 'demo', 'name': 'demo', 'description': 'Use for routing tests'},
                ]}

            manager = SkillManager(dir=tmp, skill_search=backend)
            first = manager.search_skill('routing tests')
            assert first['status'] == 'ok'
            assert manager.search_skill('routing tests')['error'] == 'duplicate_search'
            manager.search_skill('another query')
            exhausted = manager.search_skill('third query')
            assert exhausted['error'] == 'retrieval_budget_exhausted'
            blocked = manager._read_loaded_skill_resource('demo', 'references/guide.md')
            assert blocked['error'] == 'skill_not_loaded'
            loaded = manager.get_skill('demo')
            assert loaded['resources'] == ['references/guide.md']
            assert manager.get_skill('demo').get('already_loaded') is True
            missing = manager._read_loaded_skill_resource('demo', 'references/missing.md')
            assert missing['error'] == 'resource_not_declared'
            assert manager._read_loaded_skill_resource('demo', 'references/guide.md')['content'] == 'guide'

    def test_routing_description_strips_execution_steps(self):
        info = {
            'name': 'xiaohongshu-copywriter',
            'description': (
                '用于生成小红书种草内容，按照标题吸引力、emoji、正文结构、'
                '标签、互动引导等步骤生成小红书文案'
            ),
            'raw_meta': {},
        }
        assert SkillManager._routing_description_text(info['description']) == '用于生成小红书种草内容'
        info['raw_meta'] = {'when_to_use': 'Use for Xiaohongshu travel or product posts'}
        with tempfile.TemporaryDirectory() as tmp:
            manager = SkillManager(dir=tmp)
            assert manager._routing_description(info) == 'Use for Xiaohongshu travel or product posts'

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

        result = manager.run_script('script-skill', 'scripts/ok.py')

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

            ok_result = manager.run_script('script-skill', 'scripts/ok.py')
            with pytest.raises(ToolExecutionError) as exc_info:
                manager.run_script('script-skill', 'scripts/fail.py')

            assert ok_result['status'] == 'ok'
            assert ok_result['exit_code'] == 0
            assert 'exit code 7' in str(exc_info.value)
            assert 'bad' in str(exc_info.value)

    def test_run_script_uses_dynamic_env_vars(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = _make_skill(tmp, 'env-skill', 'env-skill')
            scripts_dir = os.path.join(skill_dir, 'scripts')
            os.makedirs(scripts_dir, exist_ok=True)
            script = os.path.join(scripts_dir, 'print_env.py')
            with open(script, 'w', encoding='utf-8') as f:
                f.write('import os\nprint(os.getenv("DYNAMIC_TEST_API_KEY", ""))\n')

            old_dynamic_env = lazyllm.globals.get('dynamic_env_vars')
            lazyllm.globals['dynamic_env_vars'] = {'DYNAMIC_TEST_API_KEY': 'secret-from-session'}
            try:
                manager = SkillManager(dir=tmp)
                result = manager.run_script('env-skill', 'scripts/print_env.py')
            finally:
                if old_dynamic_env is None:
                    lazyllm.globals.pop('dynamic_env_vars', None)
                else:
                    lazyllm.globals['dynamic_env_vars'] = old_dynamic_env

            assert result['status'] == 'ok'
            assert result['stdout'].strip() == 'secret-from-session'

    def test_run_script_retries_after_dynamic_env_injection(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = _make_skill(tmp, 'retry-env-skill', 'retry-env-skill')
            scripts_dir = os.path.join(skill_dir, 'scripts')
            os.makedirs(scripts_dir, exist_ok=True)
            script = os.path.join(scripts_dir, 'needs_key.py')
            with open(script, 'w', encoding='utf-8') as f:
                f.write(
                    'import os\nimport sys\n'
                    'value = os.getenv("DYNAMIC_TEST_API_KEY", "")\n'
                    'if not value:\n'
                    '    sys.stderr.write("missing DYNAMIC_TEST_API_KEY")\n'
                    '    sys.exit(1)\n'
                    'print(value)\n'
                )

            from lazyllm.tools.tool_config_inject import inject_env_vars
            old_dynamic_env = lazyllm.globals.get('dynamic_env_vars')
            lazyllm.globals['dynamic_env_vars'] = {}
            try:
                manager = SkillManager(dir=tmp)
                with pytest.raises(ToolExecutionError) as exc_info:
                    manager.run_script(
                        'retry-env-skill', 'scripts/needs_key.py',
                    )
                assert exc_info.value.missing_env == ['DYNAMIC_TEST_API_KEY']
                assert 'missing_env: ["DYNAMIC_TEST_API_KEY"]' in str(exc_info.value)
                inject_env_vars({'DYNAMIC_TEST_API_KEY': 'secret-after-set'})
                result = manager.run_script(
                    'retry-env-skill', 'scripts/needs_key.py',
                )
            finally:
                if old_dynamic_env is None:
                    lazyllm.globals.pop('dynamic_env_vars', None)
                else:
                    lazyllm.globals['dynamic_env_vars'] = old_dynamic_env

            assert result['status'] == 'ok'
            assert result['stdout'].strip() == 'secret-after-set'

    def test_run_script_does_not_preflight_block_unset_required_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = _make_skill(tmp, 'optional-env-skill', 'optional-env-skill')
            with open(os.path.join(skill_dir, 'SKILL.md'), 'w', encoding='utf-8') as f:
                f.write(
                    '---\n'
                    'name: optional-env-skill\n'
                    'description: optional env skill\n'
                    'required_env:\n'
                    '  - OPTIONAL_API_KEY\n'
                    '---\n'
                    '# optional\n'
                )
            scripts_dir = os.path.join(skill_dir, 'scripts')
            os.makedirs(scripts_dir, exist_ok=True)
            with open(os.path.join(scripts_dir, 'ok.py'), 'w', encoding='utf-8') as f:
                f.write('print("ran-without-key")\n')

            manager = SkillManager(dir=tmp)
            result = manager.run_script(
                'optional-env-skill', 'scripts/ok.py',
            )

            assert result['status'] == 'ok'
            assert result['stdout'].strip() == 'ran-without-key'

    def test_run_script_hints_declared_required_env_only_after_failure(self, monkeypatch):
        monkeypatch.delenv('DECLARED_API_KEY', raising=False)
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = _make_skill(tmp, 'declared-env-skill', 'declared-env-skill')
            with open(os.path.join(skill_dir, 'SKILL.md'), 'w', encoding='utf-8') as f:
                f.write(
                    '---\n'
                    'name: declared-env-skill\n'
                    'description: declared env skill\n'
                    'required_env:\n'
                    '  - DECLARED_API_KEY\n'
                    '---\n'
                    '# declared\n'
                )
            scripts_dir = os.path.join(skill_dir, 'scripts')
            os.makedirs(scripts_dir, exist_ok=True)
            with open(os.path.join(scripts_dir, 'fail.py'), 'w', encoding='utf-8') as f:
                f.write('import sys\nprint("boom")\nsys.exit(1)\n')

            old_dynamic_env = lazyllm.globals.get('dynamic_env_vars')
            lazyllm.globals['dynamic_env_vars'] = {}
            try:
                manager = SkillManager(dir=tmp)
                with pytest.raises(ToolExecutionError) as exc_info:
                    manager.run_script(
                        'declared-env-skill', 'scripts/fail.py',
                    )
            finally:
                if old_dynamic_env is None:
                    lazyllm.globals.pop('dynamic_env_vars', None)
                else:
                    lazyllm.globals['dynamic_env_vars'] = old_dynamic_env

            assert 'boom' in str(exc_info.value)
            assert exc_info.value.missing_env == ['DECLARED_API_KEY']

    def test_run_script_hints_convention_missing_env_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = _make_skill(tmp, 'convention-env-skill', 'convention-env-skill')
            scripts_dir = os.path.join(skill_dir, 'scripts')
            os.makedirs(scripts_dir, exist_ok=True)
            with open(os.path.join(scripts_dir, 'fail.py'), 'w', encoding='utf-8') as f:
                f.write(
                    'import sys\n'
                    'sys.stderr.write("MISSING_ENV=CONVENTION_API_KEY\\n")\n'
                    'sys.exit(1)\n'
                )

            manager = SkillManager(dir=tmp)
            with pytest.raises(ToolExecutionError) as exc_info:
                manager.run_script(
                    'convention-env-skill', 'scripts/fail.py',
                )

            assert exc_info.value.missing_env == ['CONVENTION_API_KEY']

    def test_run_script_reports_missing_cwd_as_tool_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = _make_skill(tmp, 'cwd-skill', 'cwd-skill')
            scripts_dir = os.path.join(skill_dir, 'scripts')
            os.makedirs(scripts_dir, exist_ok=True)
            script = os.path.join(scripts_dir, 'ok.py')
            with open(script, 'w', encoding='utf-8') as f:
                f.write('print("ok")\n')

            manager = SkillManager(dir=tmp)
            with pytest.raises(ToolExecutionError) as exc_info:
                manager.run_script('cwd-skill', 'scripts/ok.py', cwd='missing')

            assert 'scripts/ok.py' in str(exc_info.value)
            assert 'missing' in str(exc_info.value)
            assert 'cwd not found' in str(exc_info.value)
            assert not exc_info.value.missing_env

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
