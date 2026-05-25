import lazyllm.tools.agent.base as agent_base_module
from lazyllm.tools.agent.base import LazyLLMAgentBase
from lazyllm.tools.agent.skill_manager import SkillManager


class _DummyAgent(LazyLLMAgentBase):
    def build_agent(self):
        self._agent = lambda x: x


class TestLazyLLMAgentBase(object):
    def test_enable_builtin_tools_warns_when_skills_disabled(self):
        agent = _DummyAgent(skills=False, enable_builtin_tools=False)
        assert agent._skill_manager is None
        assert agent._enable_builtin_tools is False
        assert 'read_file' not in agent._builtin_tool_names
        assert all(not (isinstance(tool, str) and tool.startswith('builtin_tools.')) for tool in agent._tools)

    def test_enable_builtin_tools_default_does_not_warn_when_skills_disabled(self):
        agent = _DummyAgent(skills=False)
        assert agent._skill_manager is None
        assert agent._enable_builtin_tools is True
        assert 'read_file' in agent._builtin_tool_names
        assert any(isinstance(tool, str) and tool.startswith('builtin_tools.read_file') for tool in agent._tools)
        assert 'read_file' in {tool.name for tool in agent._tools_manager.all_tools}

    def test_builtin_tools_are_added_without_skills(self):
        agent = _DummyAgent(skills=False, enable_builtin_tools=True)
        assert agent._skill_manager is None
        assert any(isinstance(tool, str) and tool.startswith('builtin_tools.read_file') for tool in agent._tools)
        assert {'read_file', 'shell_tool'}.issubset({tool.name for tool in agent._tools_manager.all_tools})

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

    def test_agent_sandbox_auto_creates_sandbox(self, monkeypatch):
        sentinel = object()
        monkeypatch.setattr(agent_base_module, 'create_sandbox', lambda: sentinel)
        agent = _DummyAgent(skills=False, enable_builtin_tools=False)
        assert agent.sandbox is sentinel
        assert agent._tools_manager.sandbox is sentinel

    def test_agent_sandbox_none_skips_creation(self, monkeypatch):
        def _unexpected_create():
            raise AssertionError('create_sandbox should not be called when sandbox=None')

        monkeypatch.setattr(agent_base_module, 'create_sandbox', _unexpected_create)
        agent = _DummyAgent(skills=False, enable_builtin_tools=False, sandbox=None)
        assert agent.sandbox is None
        assert agent._tools_manager.sandbox is None
