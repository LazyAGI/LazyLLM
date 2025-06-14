import lazyllm
from lazyllm.engine import LightEngine, NodeMetaHook
import pytest
from .utils import SqlEgsData, get_db_init_keywords
from lazyllm.tools import SqlManager, DBStatus
from .tools import (get_current_weather_code, get_current_weather_vars, get_current_weather_doc,
                    get_n_day_weather_forecast_code, multiply_tool_code, add_tool_code, dummy_code)
import unittest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
import json

app = FastAPI()


@app.post("/mock_post")
async def receive_json(data: dict):
    return JSONResponse(content=data)


@pytest.mark.skip_on_win
@pytest.mark.skip_on_mac
class TestEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        client = TestClient(app)

        def mock_report(self):
            headers = {"Content-Type": "application/json; charset=utf-8"}
            json_data = json.dumps(self._meta_info, ensure_ascii=False)
            try:
                lazyllm.LOG.info(f"meta_info: {self._meta_info}")
                response = client.post(self.URL, data=json_data, headers=headers)
                assert response.json() == self._meta_info, "mock response should be same as input"
            except Exception as e:
                lazyllm.LOG.warning(f"Error sending collected data: {e}")

        NodeMetaHook.report = mock_report

    @pytest.fixture(autouse=True)
    def run_around_tests(self):
        yield
        LightEngine().reset()
        lazyllm.FileSystemQueue().dequeue()
        lazyllm.FileSystemQueue(klass="lazy_trace").dequeue()

    def test_intent_classifier(self):
        resources = [dict(id='0', kind='OnlineLLM', name='llm', args=dict(source=None))]
        music = dict(id='1', kind='Code', name='m1',
                     args=dict(code='def music(x): return f"Music get {x}"'))
        draw = dict(id='2', kind='Code', name='m2',
                    args=dict(code='def draw(x): return f"Draw get {x}"'))
        chat = dict(id='3', kind='Code', name='m3',
                    args=dict(code='def chat(x): return f"Chat get {x}"'))
        nodes = [dict(id='4', kind='Intention', name='int1',
                      args=dict(base_model='0', prompt='', constrain='', attention='',
                                nodes={'music': music, 'draw': draw, 'chat': chat}))]
        edges = [dict(iid="__start__", oid="4"), dict(iid="4", oid="__end__")]
        engine = LightEngine()
        gid = engine.start(nodes, edges, resources)
        assert engine.run(gid, "sing a song") == 'Music get sing a song'
        assert engine.run(gid, "draw a hourse") == 'Draw get draw a hourse'

    def test_toolsforllm(self):
        resources = [
            dict(id="1001", kind="Code", name="get_current_weather",
                 args=(dict(code=get_current_weather_code,
                            vars_for_code=get_current_weather_vars))),
            dict(id="1002", kind="Code", name="get_n_day_weather_forecast",
                 args=dict(code=get_n_day_weather_forecast_code,
                           vars_for_code=get_current_weather_vars)),
            dict(id="1003", kind="Code", name="multiply_tool",
                 args=dict(code=multiply_tool_code)),
            dict(id="1004", kind="Code", name="add_tool",
                 args=dict(code=add_tool_code)),
        ]
        nodes = [dict(id="1", kind="ToolsForLLM", name="fc",
                      args=dict(tools=['1001', '1002', '1003', '1004']))]
        edges = [dict(iid="__start__", oid="1"), dict(iid="1", oid="__end__")]
        engine = LightEngine()
        engine.set_report_url("mock_post")
        gid = engine.start(nodes, edges, resources)
        assert '22' in engine.run(gid, [dict(name='get_current_weather', arguments=dict(location='Paris'))])[0]

    def test_fc(self):
        resources = [
            dict(id="0", kind="OnlineLLM", name="llm", args=dict(source='glm')),
            dict(id="1001", kind="Code", name="get_current_weather",
                 args=(dict(code=get_current_weather_code,
                            vars_for_code=get_current_weather_vars))),
            dict(id="1002", kind="Code", name="get_n_day_weather_forecast",
                 args=dict(code=get_n_day_weather_forecast_code,
                           vars_for_code=get_current_weather_vars)),
            dict(id="1003", kind="Code", name="multiply_tool",
                 args=dict(code=multiply_tool_code)),
            dict(id="1004", kind="Code", name="add_tool",
                 args=dict(code=add_tool_code)),
        ]
        nodes = [dict(id="1", kind="FunctionCall", name="fc",
                      args=dict(llm='0', tools=['1001', '1002', '1003', '1004']))]
        edges = [dict(iid="__start__", oid="1"), dict(iid="1", oid="__end__")]
        engine = LightEngine()
        gid = engine.start(nodes, edges, resources)
        assert '10' in engine.run(gid, "What's the weather like today in celsius in Tokyo.")
        assert '22' in engine.run(gid, "What will the temperature be in degrees Celsius in Paris tomorrow?")

        nodes = [dict(id="2", kind="FunctionCall", name="re",
                      args=dict(llm='0', tools=['1003', '1004'], algorithm='React'))]
        edges = [dict(iid="__start__", oid="2"), dict(iid="2", oid="__end__")]
        engine = LightEngine()
        gid = engine.start(nodes, edges, resources)
        assert '5440' in engine.run(gid, "Calculate 20*(45+23)*4, step by step.")

        nodes = [dict(id="3", kind="FunctionCall", name="re",
                      args=dict(llm='0', tools=['1003', '1004'], algorithm='PlanAndSolve'))]
        edges = [dict(iid="__start__", oid="3"), dict(iid="3", oid="__end__")]
        engine = LightEngine()
        gid = engine.start(nodes, edges, resources)
        assert '5440' in engine.run(gid, "Calculate 20*(45+23)*(1+3), step by step.")

    def test_rag(self):
        prompt = ("作为国学大师，你将扮演一个人工智能国学问答助手的角色，完成一项对话任务。在这个任务中，你需要根据给定的已知国学篇章以及"
                  "问题，给出你的结论。请注意，你的回答应基于给定的国学篇章，而非你的先验知识，且注意你回答的前后逻辑不要出现"
                  "重复，且不需要提到具体文件名称。\n任务示例如下：\n示例国学篇章：《礼记 大学》大学之道，在明明德，在亲民，在止于至善"
                  "。\n问题：什么是大学？\n回答：“大学”在《礼记》中代表的是一种理想的教育和社会实践过程，旨在通过个人的"
                  "道德修养和社会实践达到最高的善治状态。\n注意以上仅为示例，禁止在下面任务中提取或使用上述示例已知国学篇章。"
                  "\n现在，请对比以下给定的国学篇章和给出的问题。如果已知国学篇章中有该问题相关的原文，请提取相关原文出来。\n"
                  "已知国学篇章：{context_str}\n")
        resources = [
            dict(id='00', kind='OnlineEmbedding', name='e0', args=dict(source='glm')),
            dict(id='01', kind='OnlineEmbedding', name='e1', args=dict(type='rerank')),
            dict(id='0', kind='Document', name='d1', args=dict(dataset_path='rag_master', node_group=[
                dict(name='sentence', embed='00', transform='SentenceSplitter', chunk_size=100, chunk_overlap=10)]))]
        nodes = [dict(id='1', kind='Retriever', name='ret1',
                      args=dict(doc='0', group_name='CoarseChunk', similarity='bm25_chinese', topk=3)),
                 dict(id='2', kind='Retriever', name='ret2',
                      args=dict(doc='0', group_name='sentence', similarity='cosine', topk=3)),
                 dict(id='3', kind='JoinFormatter', name='c', args=dict(type='sum')),
                 dict(id='4', kind='Reranker', name='rek1',
                      args=dict(type='ModuleReranker', output_format='content', join=True,
                                arguments=dict(model="01", topk=3))),
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
        r = engine.run(gid, '何为天道?')
        assert '观天之道，执天之行' in r or '天命之谓性，率性之谓道' in r or '执古之道，以御今之有' in r

    def test_sql_call(self):
        db_type = "PostgreSQL"
        username, password, host, port, database = get_db_init_keywords(db_type)

        # 1.  Init: insert data to database
        tmp_sql_manager = SqlManager(db_type, username, password, host, port, database,
                                     tables_info_dict=SqlEgsData.TEST_TABLES_INFO)
        for table_name in SqlEgsData.TEST_TABLES:
            tmp_sql_manager.execute_commit(f"DELETE FROM {table_name}")
        for insert_script in SqlEgsData.TEST_INSERT_SCRIPTS:
            tmp_sql_manager.execute_commit(insert_script)

        # 2. Engine: build and chat
        resources = [
            dict(
                id="0",
                kind="SqlManager",
                name="sql_manager",
                args=dict(
                    db_type=db_type,
                    user=username,
                    password=password,
                    host=host,
                    port=port,
                    db_name=database,
                    options_str="",
                    tables_info_dict=SqlEgsData.TEST_TABLES_INFO,
                ),
            ),
            dict(id="1", kind="OnlineLLM", name="llm", args=dict(source="sensenova")),
        ]
        nodes = [
            dict(
                id="2",
                kind="SqlCall",
                name="sql_call",
                args=dict(sql_manager="0", llm="1", sql_examples="", _lazyllm_enable_report=True),
            )
        ]
        edges = [dict(iid="__start__", oid="2"), dict(iid="2", oid="__end__")]
        engine = LightEngine()
        # Note: Set real http://ip:port/uri ...
        engine.set_report_url("mock_post")
        gid = engine.start(nodes, edges, resources)
        str_answer = engine.run(gid, "员工编号是11的人来自哪个部门？")
        assert "销售三部" in str_answer

        # 3. Release: delete data and table from database
        for table_name in SqlEgsData.TEST_TABLES:
            db_result = tmp_sql_manager.drop_table(table_name)
            assert db_result.status == DBStatus.SUCCESS

    def test_register_tools(self):
        resources = [
            dict(id="0", kind="OnlineLLM", name="llm", args=dict(source='glm')),
            dict(id="3", kind="HttpTool", name="weather_12345",
                 args=dict(code_str=get_current_weather_code,
                           vars_for_code=get_current_weather_vars,
                           doc=get_current_weather_doc)),
            dict(id="2", kind="HttpTool", name="dummy_111",
                 args=dict(code_str=dummy_code, doc='dummy')),
        ]
        # `tools` in `args` is a list of ids in `resources`
        nodes = [dict(id="1", kind="FunctionCall", name="fc",
                      args=dict(llm='0', tools=['3', '2']))]
        edges = [dict(iid="__start__", oid="1"), dict(iid="1", oid="__end__")]
        engine = LightEngine()
        # TODO handle duplicated node id
        gid = engine.start(nodes, edges, resources)

        city_name = 'Tokyo'
        unit = 'Celsius'
        ret = engine.run(gid, f"What is the temperature in {city_name} today in {unit}?")
        assert city_name in ret and unit in ret and '10' in ret

    def test_stream_and_hostory(self):
        builtin_history = [['水的沸点是多少？', '您好，我的答案是：水的沸点在标准大气压下是100摄氏度。'],
                           ['世界上最大的动物是什么？', '您好，我的答案是：蓝鲸是世界上最大的动物。'],
                           ['人一天需要喝多少水？', '您好，我的答案是：一般建议每天喝8杯水，大约2升。']]
        nodes = [dict(id='1', kind='LLM', name='m1', args=dict(source='glm', stream=True, type='online', keys=['query'],
                                                               prompt=dict(
                      system='请将我的问题翻译成中文。请注意，请直接输出翻译后的问题，不要反问和发挥',
                      user='问题: {query} \n, 翻译:'))),
                 dict(id='2', kind='LLM', name='m2',
                      args=dict(source='glm', stream=True, type='online',
                                prompt=dict(system='请参考历史对话，回答问题，并保持格式不变。', user='{query}'))),
                 dict(id='3', kind='LLM', stream=False, name='m3',
                      args=dict(source='glm', type='online', keys=['query', 'answer'], history=builtin_history,
                                prompt=dict(
                          system='你是一个问答机器人，会根据用户的问题作出回答。',
                          user='请结合历史对话和本轮的问题，总结我们的全部对话。本轮情况如下:\n {query}, 回答: {answer}')))]
        engine = LightEngine()
        gid = engine.start(nodes, edges=[['__start__', '1'], ['1', '2'], ['1', '3'], ['2', '3'],
                                         ['3', '__end__']], _history_ids=['2', '3'])
        history = [['雨后为什么会有彩虹？', '您好，我的答案是：雨后阳光通过水滴发生折射和反射形成了彩虹。'],
                   ['月亮会发光吗？', '您好，我的答案是：月亮本身不会发光，它反射太阳光。'],
                   ['一年有多少天', '您好，我的答案是：一年有365天，闰年有366天。']]

        stream_result = ''
        with lazyllm.ThreadPoolExecutor(1) as executor:
            future = executor.submit(engine.run, gid, 'How many hours are there in a day?', _lazyllm_history=history)
            while True:
                if value := lazyllm.FileSystemQueue().dequeue():
                    stream_result += f"{''.join(value)}"
                elif future.done():
                    break
            result = future.result()
            assert '一天' in stream_result and '小时' in stream_result
            assert '您好，我的答案是' in stream_result and '24' in stream_result
            assert ('蓝鲸' in result or '动物' in result) and '水' in result

    def test_egine_online_serve_train(self):
        envs = ['glm_api_key', 'qwen_api_key']
        sources = ['glm', 'qwen']
        engine = LightEngine()

        for env, source in list(zip(envs, sources)):
            token = lazyllm.config[env]
            res = engine.online_model_get_all_trained_models(token, source=source)
            assert isinstance(res, list)

            res = engine.online_model_validate_api_key(token, source=source)
            assert res is True

            res = engine.online_model_validate_api_key(token + 'ss', source=source)
            assert res is False

    def test_tools_with_llm(self):
        resources = [dict(id='0', kind='OnlineLLM', name='base', args=dict(source="qwen"))]
        nodes = [dict(id="1", kind="QustionRewrite", name="m1", args=dict(base_model='0', formatter="str")),
                 dict(id="2", kind="QustionRewrite", name="m2", args=dict(base_model='0', formatter="list")),
                 dict(id="3", kind="ParameterExtractor", name="m3", args=dict(
                      base_model='0', param=["year"], type=["int"], description=["年份"], require=[True])),
                 dict(id="4", kind="ParameterExtractor", name="m4", args=dict(
                      base_model='0', param=["year", "month"], type=["int", "int"], description=["年份", "月份"],
                      require=[True, True])),
                 dict(id="5", kind="CodeGenerator", name="m5", args=dict(base_model='0'))]

        engine = LightEngine()
        gid = engine.start(nodes, [['__start__', '1'], ['1', '__end__']], resources)
        res = engine.run(gid, "Model Context Protocol是啥")
        assert isinstance(res, str)

        gid = engine.start(nodes, [['__start__', '2'], ['2', '__end__']], resources)
        res = engine.run(gid, "RAG是什么？")
        assert isinstance(res, list) and len(res) > 0

        gid = engine.start(nodes, [['__start__', '3'], ['3', '__end__']], resources)
        res = engine.run(gid, "This year is 2023")
        assert res == 2023

        gid = engine.start(nodes, [['__start__', '4'], ['4', '__end__']], resources)
        res = engine.run(gid, "Today is 2022/06/06")
        assert res[0] == 2022 and res[1] == 6

        gid = engine.start(nodes, [['__start__', '5'], ['5', '__end__']], resources)
        res = engine.run(gid, "帮我写一个函数，计算两数之和")
        compiled = lazyllm.common.utils.compile_func(res)
        assert compiled(2, 3) == 5
