import lazyllm
import os
from lazyllm.engine import LightEngine

class TestEngine(object):
    # This test requires 4 GPUs and takes about 4 minutes to execute, skip this test to save time.
    def _test_vqa(self):
        resource = [dict(id='0', kind='web', name='web', args=dict(port=None, title='多模态聊天机器人', history=[], audio=True))]
        node = [dict(id='1', kind='VQA', name='vqa', args=dict(base_model='Mini-InternVL-Chat-2B-V1-5'))]
        edge = [dict(iid="__start__", oid="1"), dict(iid="1", oid="__end__")]
        engine = LightEngine()
        engine.start(node, edge, resource)

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

    def test_multimedia(self):
        painter_prompt = 'Now you are a master of drawing prompts, capable of converting any Chinese content entered by the user into English drawing prompts. In this task, you need to convert any input content into English drawing prompts, and you can enrich and expand the prompt content.'  # noqa E501
        musician_prompt = 'Now you are a master of music composition prompts, capable of converting any Chinese content entered by the user into English music composition prompts. In this task, you need to convert any input content into English music composition prompts, and you can enrich and expand the prompt content.'  # noqa E501
        translator_prompt = 'Now you are a master of translation prompts, capable of converting any Chinese content entered by the user into English translation prompts. In this task, you need to convert any input content into English translation prompts, and you can enrich and expand the prompt content.'  # noqa E501

        resources = [dict(id='llm', kind='LocalLLM', name='base', args=dict(base_model='internlm2-chat-7b')),
                     dict(id='file-resource', kind='File', name='file', args=dict(id='file-resource')),
                     dict(id='vqa', kind='VQA', name='vqa', args=dict(base_model='Mini-InternVL-Chat-2B-V1-5')),
                     dict(id='web', kind='web', name='web', args=dict(port=None, title='多模态聊天机器人', audio=True))]

        nodes1 = [
            dict(id='2', kind='SharedLLM', name='draw_prompt', args=dict(llm='llm', prompt=painter_prompt)),
            dict(id='3', kind='SD', name='sd', args=dict(base_model='stable-diffusion-3-medium')),
            dict(id='5', kind='SharedLLM', name='vqa1', args=dict(llm='vqa')),
            dict(id='6', kind='JoinFormatter', name='merge_sd_vqa2', args=dict(type='file')),
        ]
        edges1 = [
            dict(iid='__start__', oid='2'), dict(iid='6', oid='__end__'), dict(iid="2", oid="3"),
            dict(constant='描述图片', oid="5"), dict(iid="3", oid="5"), dict(iid="3", oid="6"), dict(iid="5", oid="6"),
        ]

        nodes = [dict(id='7', kind='STT', name='stt', args=dict(base_model='SenseVoiceSmall')),
                 dict(id='8', kind='Intention', name='intent', args=dict(base_model='llm', nodes={
                     'Drawing': dict(id='9', kind='SubGraph', name='draw_vqa', args=dict(nodes=nodes1, edges=edges1)),
                     'Translate': dict(id='10', kind='SharedLLM', name='translate_prompt',
                                       args=dict(llm='llm', prompt=translator_prompt)),
                     'Generate Music': [dict(id='11', kind='SharedLLM', name='translate',
                                             args=dict(llm='llm', prompt=musician_prompt)),
                                        dict(id='12', kind='TTS', name='music',
                                             args=dict(base_model='musicgen-small'))],
                     'Image Question Answering': dict(id='13', kind='SharedLLM', name='vqa2',
                                                      args=dict(llm='vqa', file_resource_id='file-resource')),
                     'Chat': dict(id='14', kind='SharedLLM', name='chat', args=dict(llm='llm'))}))]
        edges = [dict(iid="__start__", oid="7"), dict(iid="7", oid="8"), dict(iid="8", oid="__end__")]

        engine = LightEngine()
        gid = engine.start(nodes, edges, resources)

        r = engine.run(gid, '画个猪')
        assert '.png' in r

        r = engine.run(gid, '翻译：我喜欢敲代码。')
        assert 'code' in r

        r = engine.run(gid, "", _lazyllm_files=os.path.join(lazyllm.config['data_path'], 'ci_data/draw_pig.mp3'))
        assert '.png' in r

        r = engine.run(gid, "这张图片描述的是什么？", _lazyllm_files=os.path.join(lazyllm.config['data_path'], 'ci_data/ji.jpg'))
        assert '鸡' in r or 'chicken' in r

        r = engine.run(gid, "这张图片描述的是什么？",
                       _file_resources={'file-resource': os.path.join(lazyllm.config['data_path'], 'ci_data/ji.jpg')})
        assert '鸡' in r or 'chicken' in r

        r = engine.run(gid, '你好，很高兴认识你')
        assert '你好' in r

        r = engine.run(gid, '生成音乐，长笛独奏，大自然之声。')
        assert '.wav' in r
