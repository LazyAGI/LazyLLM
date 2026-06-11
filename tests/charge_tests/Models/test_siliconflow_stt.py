import os
from unittest.mock import MagicMock, patch

import pytest

import lazyllm
from lazyllm import AutoModel
from lazyllm.module.llms.onlinemodule.supplier.siliconflow import SiliconFlowSTT

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'automodel_stt_config.yaml')
AUDIO_PATH = '/mnt/c/Users/cuishaoting/Downloads/test_content/speech.mp3'


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


@pytest.mark.skipif(
    not os.environ.get('LAZYLLM_SILICONFLOW_API_KEY'),
    reason='LAZYLLM_SILICONFLOW_API_KEY is required for live STT test',
)
@pytest.mark.skipif(not os.path.exists(AUDIO_PATH), reason=f'audio file not found: {AUDIO_PATH}')
class TestSiliconFlowSTTLive:
    def test_siliconflow_stt_live(self):
        module = SiliconFlowSTT(api_key=os.environ['LAZYLLM_SILICONFLOW_API_KEY'])
        result = module(AUDIO_PATH)
        assert isinstance(result, str)
        assert result.strip()

    def test_automodel_speech_to_text_live(self, monkeypatch):
        monkeypatch.setitem(lazyllm.config._impl, 'auto_model_config_map_path', CONFIG_PATH)
        module = AutoModel(model='speech_to_text')
        result = module(AUDIO_PATH)
        assert isinstance(result, str)
        assert result.strip()
