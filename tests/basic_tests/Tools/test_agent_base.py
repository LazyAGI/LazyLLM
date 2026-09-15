import json

import lazyllm.tools.agent.base as agent_base_module
from lazyllm.tools.agent.base import LazyLLMAgentBase
from lazyllm.tools.agent.skill_manager import SkillManager


class _DummyAgent(LazyLLMAgentBase):
    def build_agent(self):
        self._agent = lambda x: x


class TestLazyLLMAgentBase(object):
    def test_filesystem_group_is_eager_without_changing_other_groups(self):
        from lazyllm.tools.agent import FileSystemToolkit
        from lazyllm.tools.agent.file_tool import read
        from lazyllm.tools.agent.toolsManager import ToolGroup
        agent = _DummyAgent(tools=[FileSystemToolkit(), ToolGroup([read], name='Other')], skills=False)
        names = [item['function']['name'] for item in agent._tools_manager.tools_description]
        assert len(names) == len(set(names))
        assert {'read', 'write', 'edit', 'ls', 'glob', 'grep', 'mkdir', 'move', 'remove', 'stat'} <= set(names)
        assert 'get_FileSystemToolkit_methods' not in names
        assert 'get_Other_methods' in names
        assert 'Other_read' not in names

    def test_enable_builtin_tools_warns_when_skills_disabled(self):
        agent = _DummyAgent(skills=False, enable_builtin_tools=False)
        assert agent._skill_manager is None
        assert agent._enable_builtin_tools is False
        assert 'read' not in agent._builtin_tool_names
        assert all(not (isinstance(tool, str) and tool.startswith('builtin_tools.')) for tool in agent._tools)

    def test_enable_builtin_tools_default_does_not_warn_when_skills_disabled(self):
        agent = _DummyAgent(skills=False)
        assert agent._skill_manager is None
        assert agent._enable_builtin_tools is True
        assert 'read' in agent._builtin_tool_names
        assert any(isinstance(tool, str) and tool.startswith('builtin_tools.read') for tool in agent._tools)
        assert 'read' in {tool.name for tool in agent._tools_manager.all_tools}

    def test_builtin_tools_are_added_without_skills(self):
        agent = _DummyAgent(skills=False, enable_builtin_tools=True)
        assert agent._skill_manager is None
        assert any(isinstance(tool, str) and tool.startswith('builtin_tools.read') for tool in agent._tools)
        assert {'read', 'shell_tool'}.issubset({tool.name for tool in agent._tools_manager.all_tools})

    def test_skills_only_add_skill_tools_when_builtin_tools_disabled(self, monkeypatch):
        monkeypatch.setattr(SkillManager, 'get_skill_tools', lambda self: [
            self._build_get_skill_tool(),
            self._build_read_reference_tool(),
            self._build_run_script_tool(),
        ])
        agent = _DummyAgent(skills=['demo-skill'], enable_builtin_tools=False)
        assert agent._skill_manager is not None
        assert agent._builtin_tool_names == set()
        assert agent._skill_tool_names == {'get_skill', 'read_reference', 'run_script'}
        assert all(not (isinstance(tool, str) and tool.startswith('builtin_tools.')) for tool in agent._tools)
        assert all(
            tool.execute_in_sandbox is False
            for tool in agent._tools_manager.all_tools
            if tool.name in agent._skill_tool_names
        )

    def test_agent_sandbox_auto_creates_sandbox(self, monkeypatch):
        sentinel = object()
        monkeypatch.setattr(agent_base_module, 'create_sandbox', lambda: sentinel)
        agent = _DummyAgent(skills=False, enable_builtin_tools=False)
        assert agent.sandbox is sentinel
        assert agent._tools_manager.sandbox is sentinel

    def test_run_script_dispatches_through_tool_manager_and_executes_in_sandbox(self, tmp_path):
        skill_dir = tmp_path / 'demo-skill'
        scripts_dir = skill_dir / 'scripts'
        scripts_dir.mkdir(parents=True)
        (skill_dir / 'SKILL.md').write_text(
            '---\nname: demo-skill\ndescription: Demo skill.\n---\n'
            'Run `scripts/check.py`.\n',
            encoding='utf-8',
        )
        (scripts_dir / 'check.py').write_text('print("sandboxed skill")\n', encoding='utf-8')
        agent = _DummyAgent(
            skills=['demo-skill'],
            skills_dir=str(tmp_path),
            enable_builtin_tools=False,
        )

        call = {
            'function': {
                'name': 'run_script',
                'arguments': json.dumps({
                    'name': 'demo-skill',
                    'rel_path': 'scripts/check.py',
                }),
            },
        }
        prepared = agent._tools_manager.prepare_tool_calls(call)
        result = agent._tools_manager.execute_prepared(prepared, approved_indices=(0,)).results[0]

        assert result['ok'] is True
        assert result['value']['status'] == 'ok'
        assert result['value']['stdout'] == 'sandboxed skill\n'

    def test_agent_sandbox_none_skips_creation(self, monkeypatch):
        def _unexpected_create():
            raise AssertionError('create_sandbox should not be called when sandbox=None')

        monkeypatch.setattr(agent_base_module, 'create_sandbox', _unexpected_create)
        agent = _DummyAgent(skills=False, enable_builtin_tools=False, sandbox=None)
        assert agent.sandbox is None
        assert agent._tools_manager.sandbox is None
