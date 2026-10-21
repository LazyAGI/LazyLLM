import pytest

from lazyllm.tools.rag import Document, Retriever, TransformArgs


def split_pipe(text):
    return text.split('|')


def split_comma(text):
    return text.split(',')


@pytest.mark.parametrize('container', [tuple, list])
@pytest.mark.parametrize('as_dict', [False, True])
def test_pattern_transforms_register_and_retrieve(tmp_path, container, as_dict):
    text = tmp_path / 'fruit.txt'
    text.write_text('apple|banana')
    xml = tmp_path / 'vegetables.xml'
    xml.write_text('carrot,pear')
    document = Document(doc_files=[str(text), str(xml)], manager=False)
    specs = [{'f': split_pipe, 'pattern': '*.txt'}, {'f': split_comma, 'pattern': '*.xml'}]
    transforms = container(specs if as_dict else [TransformArgs(**spec) for spec in specs])
    document.create_node_group('Pieces', transform=transforms)
    retriever = Retriever(document, 'Pieces', similarity='bm25', topk=10)
    results = retriever('apple')
    assert sorted(node.text for node in results) == ['apple', 'banana', 'carrot', 'pear']


def test_single_callable_transform_still_retrieves(tmp_path):
    path = tmp_path / 'fruit.txt'
    path.write_text('apple|banana')
    document = Document(doc_files=[str(path)], manager=False)
    document.create_node_group('Pieces', transform=split_pipe)
    results = Retriever(document, 'Pieces', similarity='bm25', topk=10)('apple')
    assert sorted(node.text for node in results) == ['apple', 'banana']
