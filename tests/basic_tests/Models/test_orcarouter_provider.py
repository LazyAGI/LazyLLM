import lazyllm
import pytest
from lazyllm.module.llms.onlinemodule.supplier.orcarouter import OrcaRouterChat


class TestOrcaRouterChat:

    def test_registered_in_online_chat(self):
        assert 'orcarouter' in lazyllm.online.chat
        assert lazyllm.online.chat.orcarouter is OrcaRouterChat

    def test_default_base_url_and_model(self):
        chat = OrcaRouterChat(api_key='sk-orca-test')
        assert chat._base_url == 'https://api.orcarouter.ai/v1/'
        assert chat._model_name == 'orcarouter/auto'

    def test_custom_base_url_and_model(self):
        chat = OrcaRouterChat(api_key='sk-orca-test', base_url='https://custom.example/v1',
                              model='orcarouter/fusion')
        assert chat._base_url == 'https://custom.example/v1'
        assert chat._model_name == 'orcarouter/fusion'

    def test_api_key_required(self):
        with pytest.raises(ValueError, match='api_key is required'):
            OrcaRouterChat(api_key=None)
