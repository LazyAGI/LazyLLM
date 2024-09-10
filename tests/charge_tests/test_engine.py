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
