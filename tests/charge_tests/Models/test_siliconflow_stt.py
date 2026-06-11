import os
from unittest.mock import MagicMock, patch

import pytest

import lazyllm
from lazyllm import AutoModel
from lazyllm.module.llms.onlinemodule.supplier.siliconflow import SiliconFlowSTT

from ...utils import get_api_key

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'automodel_stt_config.yaml')
AUDIO_PATH = os.path.join(lazyllm.config['data_path'], 'ci_data/shuidiaogetou.mp3')


def _live_stt_ready() -> bool:
    return os.path.exists(AUDIO_PATH) and bool(get_api_key('siliconflow'))


class TestSiliconFlowSTT:
    def test_forward_posts_multipart_and_returns_text(self, tmp_path):
        audio_file = tmp_path / 'audio.mp3'
        audio_file.write_bytes(b'fake-audio')

        mock_response = MagicMock()
        mock_response.json.return_value = {'text': 'hello world'}
        mock_response.raise_for_status = MagicMock()

        with patch('lazyllm.module.llms.onlinemodule.supplier.siliconflow.requests.post',
                   return_value=mock_response) as mock_post:
            module = SiliconFlowSTT(api_key='test-key')
            result = module(files=[str(audio_file)])

        assert result == 'hello world'
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs['headers'] == {'Authorization': 'Bearer test-key'}
        assert 'file' in call_kwargs['files']
        assert call_kwargs['files']['model'] == (None, 'FunAudioLLM/SenseVoiceSmall')

    def test_forward_accepts_audio_path_as_input(self, tmp_path):
        audio_file = tmp_path / 'speech.mp3'
        audio_file.write_bytes(b'fake-audio')

        mock_response = MagicMock()
        mock_response.json.return_value = {'text': 'path input'}
        mock_response.raise_for_status = MagicMock()

        with patch('lazyllm.module.llms.onlinemodule.supplier.siliconflow.requests.post',
                   return_value=mock_response):
            module = SiliconFlowSTT(api_key='test-key')
            result = module(str(audio_file))

        assert result == 'path input'

    def test_forward_rejects_multiple_files(self):
        module = SiliconFlowSTT(api_key='test-key')
        files = [AUDIO_PATH, AUDIO_PATH]
        with pytest.raises(ValueError, match='only supports one audio file'):
            module(files=files)


@pytest.mark.skipif(not _live_stt_ready(), reason='ci_data/shuidiaogetou.mp3 or siliconflow api key unavailable')
class TestSiliconFlowSTTLive:
    def test_siliconflow_stt_live(self):
        module = SiliconFlowSTT(api_key=get_api_key('siliconflow'))
        result = module(AUDIO_PATH)
        assert isinstance(result, str)
        assert result.strip()

    def test_automodel_speech_to_text_live(self, monkeypatch):
        monkeypatch.setitem(lazyllm.config._impl, 'auto_model_config_map_path', CONFIG_PATH)
        module = AutoModel(model='speech_to_text')
        result = module(AUDIO_PATH)
        assert isinstance(result, str)
        assert result.strip()
