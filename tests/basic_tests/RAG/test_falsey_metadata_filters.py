import json

import pytest

from lazyllm.tools.rag import Document, Retriever
from lazyllm.tools.rag.doc_node import DocNode
from lazyllm.tools.rag.store import LAZY_ROOT_NAME
from lazyllm.tools.rag.utils import generic_process_filters


@pytest.mark.parametrize('value', [0, False, ''])
@pytest.mark.parametrize('container', [list, set])
def test_explicit_falsey_candidates(value, container):
    match = DocNode(text='apple', global_metadata={'field': value})
    other = DocNode(text='orange', global_metadata={'field': 'different'})
    missing = DocNode(text='pear')
    assert generic_process_filters([match, other, missing], {'field': container([value])}) == [match]


@pytest.mark.parametrize('candidates', [[], set(), [None], {None}])
def test_absent_metadata_stays_excluded(candidates):
    missing = DocNode(text='pear')
    assert generic_process_filters([missing], {'field': candidates}) == []


def test_scalar_and_multiple_filter_controls():
    match = DocNode(text='apple', global_metadata={'field': 0, 'kind': 'fruit'})
    assert generic_process_filters([match], {'field': 0}) == [match]
    assert generic_process_filters([match], {'field': [0], 'kind': ['vegetable']}) == []
    assert generic_process_filters([match], {'field': [1]}) == []


def read_filter_records(path, extra_info=None):
    records = json.loads(path.read_text())
    return [DocNode(text=item['text'], global_metadata=item['metadata']) for item in records]


def test_document_retriever_bm25_falsey_filter(tmp_path):
    path = tmp_path / 'fruit.filterrecords'
    path.write_text(json.dumps([
        {'text': 'apple fruit harvest', 'metadata': {'year': 0}},
        {'text': 'orange fruit harvest', 'metadata': {'year': 1}},
    ]))
    document = Document(doc_files=[str(path)], manager=False)
    document.add_reader('*.filterrecords', read_filter_records)
    retriever = Retriever(document, LAZY_ROOT_NAME, similarity='bm25', topk=1)
    results = retriever('apple', filters={'year': [0]})
    assert len(results) == 1
    assert results[0].text == 'apple fruit harvest'
    assert results[0].global_metadata['year'] == 0
