from lazyllm import LightEngine
import pytest

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
