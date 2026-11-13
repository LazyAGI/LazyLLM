import uuid

import pytest

from lazyllm.engine import LightEngine


@pytest.fixture
def owned_engine():
    engine = LightEngine()
    identifiers = []
    yield engine, identifiers
    for identifier in identifiers:
        engine.release_node(identifier)


def code_node(identifier):
    return {'id': identifier, 'name': 'increment_' + identifier, 'kind': 'Code',
            'args': {'code': 'def invoke(value): return value + 1'}}


@pytest.mark.parametrize('count', [0, 1, 3])
def test_stop_all_releases_actual_nodes_and_preserves_sentinels(owned_engine, count):
    engine, identifiers = owned_engine
    start, end = engine.build_node('__start__'), engine.build_node('__end__')
    for _ in range(count):
        identifier = uuid.uuid4().hex
        identifiers.append(identifier)
        engine.build_node(code_node(identifier))
        assert engine.run(identifier, 1) == 2
    engine.stop()
    assert all(engine.status(identifier) == 'unknown' for identifier in identifiers)
    assert engine.build_node('__start__') is start
    assert engine.build_node('__end__') is end
    engine.stop()


def test_stop_all_releases_started_graph_and_descendants(owned_engine):
    engine, identifiers = owned_engine
    children = [uuid.uuid4().hex for _ in range(2)]
    graph = uuid.uuid4().hex
    identifiers.extend([graph, *children])
    assert engine.start([code_node(identifier) for identifier in children], gid=graph) == graph
    assert engine.run(graph, 1) == 3
    engine.stop()
    assert all(engine.status(identifier) == 'unknown' for identifier in identifiers)


def test_single_node_stop_keeps_unrelated_node_and_existing_contract(owned_engine):
    engine, identifiers = owned_engine
    identifiers.extend(uuid.uuid4().hex for _ in range(2))
    for identifier in identifiers:
        engine.build_node(code_node(identifier))
    engine.stop(identifiers[0])
    assert engine.run(identifiers[1], 1) == 2
    engine.release_node(identifiers[0])
    assert engine.status(identifiers[0]) == 'unknown'
    assert engine.run(identifiers[1], 2) == 3
