import pytest
from lazyllm import ActorPrompt
from lazyllm import DataPrompt, ChatPrompter

class TestActorPrompt:
    def test_initialization(self):
        lib = ActorPrompt()
        assert 'zh' in lib.supported_langs
        assert 'en' in lib.supported_langs

    def test_get_all_acts(self):
        lib = ActorPrompt(lang='en')
        acts = lib.get_all_acts()
        assert isinstance(acts, list)
        assert len(acts) > 1000

        acts_zh = lib.get_all_acts('zh')
        assert isinstance(acts_zh, list)
        assert len(acts_zh) > 100

    def test_get_prompt(self):
        lib = ActorPrompt()
        acts_en = lib.get_all_acts('en')
        if acts_en:
            act = acts_en[0]
            # Test __call__ return raw string
            prompt_raw = lib(act, lang='en', return_raw=True)
            assert isinstance(prompt_raw, str)
            # Test get_prompt returns same raw string
            prompt_from_get = lib.get_prompt(act, lang='en')
            assert prompt_raw == prompt_from_get
            assert len(prompt_raw) > 0
            # Test __call__ returns ChatPrompter when return_raw=False
            prompter = lib(act, lang='en', return_raw=False)
            assert isinstance(prompter, ChatPrompter)

    def test_invalid_inputs(self):
        lib = ActorPrompt()
        # Test non-existent act
        with pytest.raises(ValueError):
            lib.get_prompt('non_existent_act_12345', lang='en')
        # Test non-existent language
        with pytest.raises(ValueError):
            lib.get_prompt('Translator', lang='fr')
        # Test get_all_acts for non-existent language
        assert lib.get_all_acts('fr') == []

class TestDataPrompt:
    def test_add_and_get_prompt(self):
        DataPrompt.add_prompt(
            act='simple_summarize',
            system_prompt='You are a concise summarizer.',
            user_prompt='Please summarize the following text: {text}',
            lang='en'
        )
        lib = DataPrompt(lang='en')
        prompter = lib('simple_summarize')
        assert isinstance(prompter, ChatPrompter)

    def test_return_raw_dict(self):
        lib = DataPrompt(lang='en')
        raw = lib('simple_summarize', return_raw=True)
        assert isinstance(raw, dict)
        assert 'system' in raw and 'user' in raw
        assert raw['system'].startswith('You are a concise')

    def test_missing_key_raises(self):
        lib = DataPrompt(lang='en')
        with pytest.raises(ValueError):
            lib.get_prompt('non_existent_key_123', lang='en')

    def test_add_prompt_requires_content(self):
        with pytest.raises(AssertionError):
            DataPrompt.add_prompt(act='bad_prompt', system_prompt=None, user_prompt=None, lang='en')
