import os
import lazyllm
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

    def test_http(self):
        nodes = [
            dict(
                id="1",
                kind="HTTP",
                name="visit_sensetime",
                args=dict(
                    method="GET",
                    url="https://www.sensetime.com/cn",
                    api_key=None,
                    headers=None,
                    params=None,
                    body=None,
                )
            )
        ]
        edges = [dict(iid="__start__", oid="1"), dict(iid="1", oid="__end__")]
        engine = LightEngine()
        gid = engine.start(nodes, edges)
        ret = engine.run(gid)
        assert '商汤科技' in ret['content']

    def test_multimedia2(self):
        painter_prompt = 'Now you are a master of drawing prompts, capable of converting any Chinese content entered by the user into English drawing prompts. In this task, you need to convert any input content into English drawing prompts, and you can enrich and expand the prompt content.'  # noqa E501
        musician_prompt = 'Now you are a master of music composition prompts, capable of converting any Chinese content entered by the user into English music composition prompts. In this task, you need to convert any input content into English music composition prompts, and you can enrich and expand the prompt content.'  # noqa E501
        translator_prompt = 'Now you are a master of translation prompts, capable of converting any Chinese content entered by the user into English translation prompts. In this task, you need to convert any input content into English translation prompts, and you can enrich and expand the prompt content.'  # noqa E501

        resources = [dict(id='0', kind='LocalLLM', name='base', args=dict(base_model='internlm2-chat-7b')),
                     dict(id='1', kind='web', name='web', args=dict(port=None, title='多模态聊天机器人', audio=True))]

        nodes1 = [
            dict(id='2', kind='SharedLLM', name='draw_prompt', args=dict(llm='0', prompt=painter_prompt)),
            dict(id='3', kind='SD', name='sd', args=dict(base_model='stable-diffusion-3-medium')),
            dict(id='4', kind='Code', name='vqa_query', args='def static_str(x):\n    return "描述图片"\n'),
            dict(id='5', kind='Formatter', name='merge_sd_vqa1', args=dict(ftype='file', rule='merge')),
            dict(id='6', kind='VQA', name='vqa', args=dict(base_model='Mini-InternVL-Chat-2B-V1-5')),
            dict(id='7', kind='Formatter', name='merge_sd_vqa2', args=dict(ftype='file', rule='merge')),
        ]
        edges1 = [
            dict(iid='__start__', oid='2'), dict(iid='7', oid='__end__'),
            dict(iid="2", oid="3"), dict(iid="3", oid="4"), dict(iid="3", oid="5"),
            dict(iid="4", oid="5"), dict(iid="5", oid="6"), dict(iid="3", oid="7"), dict(iid="6", oid="7"),
        ]

        speech_recog = dict(id='8', kind='STT', name='stt', args=dict(base_model='SenseVoiceSmall'))
        ident = dict(id='9', kind='Code', name='ident', args='def ident(x):\n    return x\n')
        nodes = [dict(id='10', kind='Formatter', name='encode_input', args=dict(ftype='file', rule='encode')),
                 dict(id='11', kind='Ifs', name='voice_or_txt', args=dict(
                      cond='def cond(x): return "<lazyllm-query>" in x', true=[speech_recog], false=[ident])),
                 dict(id='12', kind='Intention', name='intent', args=dict(base_model='0', nodes={
                     'Drawing': dict(id='14', kind='SubGraph', name='draw_vqa', args=dict(nodes=nodes1, edges=edges1)),
                     'Translate': dict(id='15', kind='SharedLLM', name='translate_prompt',
                                       args=dict(llm='0', prompt=translator_prompt)),
                     'Generate Music': [dict(id='16', kind='SharedLLM', name='translate',
                                             args=dict(llm='0', prompt=musician_prompt)),
                                        dict(id='17', kind='TTS', name='music',
                                             args=dict(base_model='musicgen-small'))],
                     'Chat': dict(id='18', kind='SharedLLM', name='chat', args=dict(llm='0'))}))]
        edges = [dict(iid="__start__", oid="10"), dict(iid="10", oid="11"),
                 dict(iid="11", oid="12"), dict(iid="12", oid="__end__")]

        engine = LightEngine()
        gid = engine.start(nodes, edges, resources)

        r = engine.run(gid, '画个猪')
        assert '.png' in r

        r = engine.run(gid, '翻译：我喜欢敲代码。')
        assert 'code' in r

        audio_path = os.path.join(lazyllm.config['data_path'], 'ci_data/draw_pig.mp3')
        r = engine.run(gid, {"query": "", "files": [f"{audio_path}"]})
        assert '.png' in r

        r = engine.run(gid, '你好，很高兴认识你')
        assert '你好' in r

        r = engine.run(gid, '生成音乐，长笛独奏，大自然之声。')
        assert '.wav' in r
