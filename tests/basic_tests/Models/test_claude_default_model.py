from lazyllm.module.llms.onlinemodule.supplier.claude import ClaudeChat

# Anthropic retired these (https://platform.claude.com/docs/en/about-claude/model-deprecations):
# claude-3-5-sonnet-20241022 on 2025-10-28, claude-3-haiku-20240307 on 2026-04-20.
RETIRED = {'claude-3-5-sonnet-20241022', 'claude-3-haiku-20240307'}


def test_claude_chat_default_model_is_the_documented_served_one():
    chat = ClaudeChat(api_key='test-key')
    assert chat._model_name not in RETIRED
    assert chat._model_name == 'claude-opus-4-5'
