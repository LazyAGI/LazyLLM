from lazyllm import PromptLibrary

class TestPromptLibrary:
    def test_initialization(self):
        lib = PromptLibrary()
        assert 'zh' in lib.supported_langs
        assert 'en' in lib.supported_langs

    def test_get_all_acts(self):
        lib = PromptLibrary(lang='en')
        acts = lib.get_all_acts()
        assert isinstance(acts, list)
        assert len(acts) > 1000

        acts_zh = lib.get_all_acts('zh')
        assert isinstance(acts_zh, list)
        assert len(acts_zh) > 100

    def test_get_prompt(self):
        lib = PromptLibrary()
        acts_en = lib.get_all_acts('en')
        if acts_en:
            act = acts_en[0]
            # Test __call__
            prompt1 = lib(act, lang='en')
            # Test get_prompt
            prompt2 = lib.get_prompt(act, lang='en')
            assert prompt1 == prompt2
            assert len(prompt1) > 0

    def test_invalid_inputs(self):
        lib = PromptLibrary()
        # Test non-existent act
        assert lib.get_prompt('non_existent_act_12345', lang='en') == ''
        # Test non-existent language
        assert lib.get_prompt('Translator', lang='fr') == ''
        # Test get_all_acts for non-existent language
        assert lib.get_all_acts('fr') == []
