from lazyllm import LightEngine
import pytest
from . import tools as _  # noqa F401


class TestEngine(object):

    @pytest.fixture(autouse=True)
    def run_around_tests(self):
        yield
        LightEngine().reset()

    def test_intent_classifier(self):
        resources = [dict(id="0", kind="OnlineLLM", name="llm", args=dict(source=None))]
        music = dict(id='1', kind='Code', name='m1', args='def music(x): return f"Music get {x}"')
        draw = dict(id='2', kind='Code', name='m2', args='def draw(x): return f"Draw get {x}"')
        chat = dict(id='3', kind='Code', name='m3', args='def chat(x): return f"Chat get {x}"')
        nodes = [dict(id="4", kind="Intention", name="int1",
                      args=dict(base_model="0", nodes={'music': music, 'draw': draw, 'chat': chat}))]
        edges = [dict(iid="__start__", oid="4"), dict(iid="4", oid="__end__")]
        engine = LightEngine()
        engine.start(nodes, edges, resources)
        assert engine.run("sing a song") == 'Music get sing a song'
        assert engine.run("draw a hourse") == 'Draw get draw a hourse'

    def test_toolsforllm(self):
        nodes = [dict(id="1", kind="ToolsForLLM", name="fc",
                      args=dict(tools=['get_current_weather', 'get_n_day_weather_forecast',
                                       'multiply_tool', 'add_tool']))]
        edges = [dict(iid="__start__", oid="1"), dict(iid="1", oid="__end__")]
        engine = LightEngine()
        engine.start(nodes, edges)
        assert '22' in engine.run([dict(name='get_current_weather', arguments=dict(location='Paris'))])[0]

    def test_fc(self):
        resources = [dict(id="0", kind="OnlineLLM", name="llm", args=dict(source='glm'))]
        nodes = [dict(id="1", kind="FunctionCall", name="fc",
                      args=dict(llm='0', tools=['get_current_weather', 'get_n_day_weather_forecast',
                                                'multiply_tool', 'add_tool']))]
        edges = [dict(iid="__start__", oid="1"), dict(iid="1", oid="__end__")]
        engine = LightEngine()
        engine.start(nodes, edges, resources)
        assert '10' in engine.run("What's the weather like today in celsius in Tokyo.")

        nodes = [dict(id="2", kind="FunctionCall", name="re",
                      args=dict(llm='0', tools=['multiply_tool', 'add_tool'], algorithm='React'))]
        edges = [dict(iid="__start__", oid="2"), dict(iid="2", oid="__end__")]
        engine = LightEngine()
        engine.start(nodes, edges, resources)
        assert '5440' in engine.run("Calculate 20*(45+23)*4, step by step.")

        nodes = [dict(id="3", kind="FunctionCall", name="re",
                      args=dict(llm='0', tools=['multiply_tool', 'add_tool'], algorithm='PlanAndSolve'))]
        edges = [dict(iid="__start__", oid="3"), dict(iid="3", oid="__end__")]
        engine = LightEngine()
        engine.start(nodes, edges, resources)
        assert '5440' in engine.run("Calculate 20*(45+23)*(1+3), step by step.")

    def test_rag(self):
        prompt = ("作为国学大师，你将扮演一个人工智能国学问答助手的角色，完成一项对话任务。在这个任务中，你需要根据给定的已知国学篇章以及"
                  "问题，给出你的结论。请注意，你的回答应基于给定的国学篇章，而非你的先验知识，且注意你回答的前后逻辑不要出现"
                  "重复，且不需要提到具体文件名称。\n任务示例如下：\n示例国学篇章：《礼记 大学》大学之道，在明明德，在亲民，在止于至善"
                  "。\n问题：什么是大学？\n回答：“大学”在《礼记》中代表的是一种理想的教育和社会实践过程，旨在通过个人的"
                  "道德修养和社会实践达到最高的善治状态。\n注意以上仅为示例，禁止在下面任务中提取或使用上述示例已知国学篇章。"
                  "\n现在，请对比以下给定的国学篇章和给出的问题。如果已知国学篇章中有该问题相关的原文，请提取相关原文出来。\n"
                  "已知国学篇章：{context_str}\n")
        resources = [dict(id='0', kind='Document', name='d1', args=dict(
            dataset_path='rag_master', node_group=[dict(name='sentence', transform='SentenceSplitter',
                                                        chunk_size=100, chunk_overlap=10)]))]
        nodes = [dict(id='1', kind='Retriever', name='ret1',
                      args=dict(doc='0', group_name='CoarseChunk', similarity='bm25_chinese', topk=3)),
                 dict(id='2', kind='Retriever', name='ret2',
                      args=dict(doc='0', group_name='sentence', similarity='bm25', topk=3)),
                 dict(id='3', kind='JoinFormatter', name='c', args=dict(method='sum')),
                 dict(id='4', kind='Reranker', name='rek1',
                      args=dict(type='ModuleReranker', output_format='content', join=True,
                                arguments=dict(model="bge-reranker-large", topk=1))),
                 dict(id='5', kind='Code', name='c1',
                      args='def test(nodes, query): return dict(context_str=nodes, query=query)'),
                 dict(id='6', kind='OnlineLLM', name='m1',
                      args=dict(source='glm', prompt=dict(system=prompt, user='问题: {query}')))]
        edges = [dict(iid='__start__', oid='1'), dict(iid='__start__', oid='2'), dict(iid='1', oid='3'),
                 dict(iid='2', oid='3'), dict(iid='3', oid='4'), dict(iid='__start__', oid='4'),
                 dict(iid='4', oid='5'), dict(iid='__start__', oid='5'), dict(iid='5', oid='6'),
                 dict(iid='6', oid='__end__')]
        engine = LightEngine()
        engine.start(nodes, edges, resources)
        assert '观天之道，执天之行' in engine.run('何为天道?')
