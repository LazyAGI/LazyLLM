import lazyllm
from lazyllm import OnlineEmbeddingModule, launchers
from lazyllm.engine import LightEngine
import time
from gradio_client import Client


class TestExample(object):
    def setup_method(self):
        self.launcher = launchers.k8s()

    def teardown_method(self):
        lazyllm.launcher.cleanup()

    def test_fc(self):
        class Graph(lazyllm.ModuleBase):
            def __init__(self):
                super().__init__()
                self._engine_conf = {
                    'nodes': [
                        {
                            'id': (
                                'a0550cf15ea0f29f4d172d867d0df559c3c75a5b3fefb8ff419c61dd87e6d8f9'
                                '_1729740030006'
                            ),
                            'kind': 'FunctionCall',
                            'name': '1729740030006',
                            'args': {
                                'llm': '4ec882c0_d7a6_4796_b498_ca1ee6b2bd27',
                                'tools': ['782b588f_7e87_415f_9e3b_3bfc9de2de7f'],
                                'algorithm': 'ReWOO'
                            }
                        }
                    ],
                    'edges': [
                        {
                            'iid': '__start__',
                            'oid': (
                                'a0550cf15ea0f29f4d172d867d0df559c3c75a5b3fefb8ff419c61dd87e6d8f9'
                                '_1729740030006'
                            )
                        },
                        {
                            'iid': (
                                'a0550cf15ea0f29f4d172d867d0df559c3c75a5b3fefb8ff419c61dd87e6d8f9'
                                '_1729740030006'
                            ),
                            'oid': '__end__'
                        }
                    ],
                    'resources': [
                        {
                            'id': '782b588f_7e87_415f_9e3b_3bfc9de2de7f',
                            'kind': 'HttpTool',
                            'name': 'Func20',
                            'extras': {'provider_name': 'Func20'},
                            'args': {
                                'timeout': 1,
                                'doc': (
                                    '\n奇数偶数判断\n\nArgs:\n  number (int): 输入数值\n\nReturns:\n  output (str): 输出'
                                ),
                                'code_str': (
                                    'def is_even_or_odd(number):\r\n'
                                    '    """\r\n'
                                    '    定义一个函数，用于判断一个数字是奇数还是偶数\r\n'
                                    '\r\n'
                                    '    args:\r\n'
                                    '        int: number\r\n'
                                    '\r\n'
                                    '    returns:\r\n'
                                    '        str: result\r\n'
                                    '    """\r\n'
                                    '    if number % 2 == 0:\r\n'
                                    '        return f"这是工具代码的输出：{number}偶数"\r\n'
                                    '    else:\r\n'
                                    '        return f"这是工具代码的输出：{number}是奇数"'
                                )
                            }
                        },
                        {
                            'id': '4ec882c0_d7a6_4796_b498_ca1ee6b2bd27',
                            'kind': 'OnlineLLM',
                            'name': '4ec882c0_d7a6_4796_b498_ca1ee6b2bd27',
                            'args': {
                                'source': 'glm',
                                'prompt': None,
                                'stream': False
                            }
                        }
                    ]
                }
                self.start_engine()

            def start_engine(self):
                self._engine = LightEngine()
                self._eid = self._engine.start(self._engine_conf.get("nodes", []),
                                               self._engine_conf.get("edges", []),
                                               self._engine_conf.get("resources", []))

            def forward(self, query, **kw):
                res = self._engine.run(self._eid, query)
                return res
        g = Graph()
        web_module = lazyllm.ServerModule(g, launcher=self.launcher)
        web_module.start()
        r = web_module("3是奇数还是偶数")
        assert '3是奇数' in r

    def test_rag(self):
        def demo(query):
            prompt = ("作为国学大师，你将扮演一个人工智能国学问答助手的角色，完成一项对话任务。在这个任务中，你需要根据给定的已知国学篇章以及"
                      "问题，给出你的结论。请注意，你的回答应基于给定的国学篇章，而非你的先验知识，且注意你回答的前后逻辑不要出现"
                      "重复，且不需要提到具体文件名称。\n任务示例如下：\n示例国学篇章：《礼记 大学》大学之道，在明明德，在亲民，在止于至善"
                      "。\n问题：什么是大学？\n回答：“大学”在《礼记》中代表的是一种理想的教育和社会实践过程，旨在通过个人的"
                      "道德修养和社会实践达到最高的善治状态。\n注意以上仅为示例，禁止在下面任务中提取或使用上述示例已知国学篇章。"
                      "\n现在，请对比以下给定的国学篇章和给出的问题。如果已知国学篇章中有该问题相关的原文，请提取相关原文出来。\n"
                      "已知国学篇章：{context_str}\n")
            resources = [
                dict(id='00', kind='OnlineEmbedding', name='e1', args=dict(source='glm')),
                dict(id='0', kind='Document', name='d1', args=dict(dataset_path='rag_master', embed='00', node_group=[
                    dict(name='sentence', transform='SentenceSplitter', chunk_size=100, chunk_overlap=10)]))]
            nodes = [dict(id='1', kind='Retriever', name='ret1',
                          args=dict(doc='0', group_name='CoarseChunk', similarity='bm25_chinese', topk=3)),
                     dict(id='2', kind='Retriever', name='ret2',
                          args=dict(doc='0', group_name='sentence', similarity='cosine', topk=3)),
                     dict(id='3', kind='JoinFormatter', name='c', args=dict(type='sum')),
                     dict(id='4', kind='Reranker', name='rek1',
                          args=dict(type='ModuleReranker', output_format='content', join=True,
                                    arguments=dict(model=OnlineEmbeddingModule(type="rerank"), topk=1))),
                     dict(id='5', kind='Code', name='c1',
                          args='def test(nodes, query): return dict(context_str=nodes, query=query)'),
                     dict(id='6', kind='OnlineLLM', name='m1',
                          args=dict(source='glm', prompt=dict(system=prompt, user='问题: {query}')))]
            edges = [dict(iid='__start__', oid='1'), dict(iid='__start__', oid='2'), dict(iid='1', oid='3'),
                     dict(iid='2', oid='3'), dict(iid='3', oid='4'), dict(iid='__start__', oid='4'),
                     dict(iid='4', oid='5'), dict(iid='__start__', oid='5'), dict(iid='5', oid='6'),
                     dict(iid='6', oid='__end__')]
            engine = LightEngine()
            gid = engine.start(nodes, edges, resources)
            r = engine.run(gid, query)
            return r
        web_module = lazyllm.ServerModule(demo, launcher=self.launcher)
        web_module.start()
        r = web_module("何为天道?")
        assert '观天之道，执天之行' in r or '天命之谓性，率性之谓道' in r

    def test_engine_server(self):
        nodes = [dict(id='1', kind='Code', name='m1', args='def test(x: int):\n    return 2 * x\n')]
        edges = [dict(iid='__start__', oid='1'), dict(iid='1', oid='__end__')]
        resources = [dict(id='2', kind='server', name='s1', args=dict(port=None, launcher=self.launcher)),
                     dict(id='3', kind='web', name='w1', args=dict(port=None, title='网页', history=[], audio=False))
                    ]
        engine = LightEngine()
        gid = engine.start(nodes, edges, resources, gid='graph-1')
        assert engine.status(gid) == {'1': 'running', '2': lazyllm.launcher.Status.Running, '3': 'running'}
        assert engine.run(gid, 1) == 2
        time.sleep(3)
        web = engine.build_node('graph-1').func._web
        assert engine.build_node('graph-1').func.api_url is not None
        assert engine.build_node('graph-1').func.web_url == web.url
        client = Client(web.url, download_files=web.cach_path)
        chat_history = [['123', None]]
        ans = client.predict(False, chat_history, False, False, api_name="/_respond_stream")
        assert ans[0][-1][-1] == '123123'
        client.close()
        web.stop()
