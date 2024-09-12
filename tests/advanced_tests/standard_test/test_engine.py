from lazyllm.engine import LightEngine

class TestEngine(object):
    # This test requires 4 GPUs and takes about 4 minutes to execute, skip this test to save time.
    def _test_vqa(self):
        resource = [dict(id='0', kind='web', name='web', args=dict(port=None, title='多模态聊天机器人', history=[], audio=True))]
        node = [dict(id='1', kind='VQA', name='vqa', args=dict(base_model='Mini-InternVL-Chat-2B-V1-5'))]
        edge = [dict(iid="__start__", oid="1"), dict(iid="1", oid="__end__")]
        engine = LightEngine()
        engine.start(node, edge, resource)

    def _test_multimedia(self):
        painter_p = 'Now you are a master of drawing prompts, capable of converting any Chinese content entered by the user into English drawing prompts. In this task, you need to convert any input content into English drawing prompts, and you can enrich and expand the prompt content.'  # noqa E501
        musician_p = 'Now you are a master of music composition prompts, capable of converting any Chinese content entered by the user into English music composition prompts. In this task, you need to convert any input content into English music composition prompts, and you can enrich and expand the prompt content.'    # noqa E501

        resources = [dict(id='0', kind='LocalLLM', name='base', args=dict(base_model='internlm2-chat-7b')),
                     dict(id='1', kind='web', name='web', args=dict(port=None, title='多模态聊天机器人', audio=True))]
        nodes = [dict(id='2', kind='Intention', name='intent', args=dict(base_model='0', nodes={
            'Chat': dict(id='3', kind='SharedLLM', name='chat', args=dict(llm='0')),
            'Speech Recognition': dict(id='4', kind='STT', name='stt', args=dict(base_model='SenseVoiceSmall')),
            'Image QA': dict(id='5', kind='VQA', name='vqa', args=dict(base_model='Mini-InternVL-Chat-2B-V1-5')),
            'Drawing': [dict(id='6', kind='SharedLLM', name='drow_prompt', args=dict(llm='0', prompt=painter_p)),
                        dict(id='7', kind='SD', name='sd', args=dict(base_model='stable-diffusion-3-medium'))],
            'Generate Music': [dict(id='8', kind='SharedLLM', name='translate', args=dict(llm='0', prompt=musician_p)),
                               dict(id='9', kind='TTS', name='music', args=dict(base_model='musicgen-small'))],
            'Text to Speech': dict(id='10', kind='TTS', name='speech', args=dict(base_model='ChatTTS')),
        }))]

        edges = [dict(iid="__start__", oid="2"), dict(iid="2", oid="__end__")]
        engine = LightEngine()
        engine.start(nodes, edges, resources)
