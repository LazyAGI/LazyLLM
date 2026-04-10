import pytest

lazyllm_cpp = pytest.importorskip('lazyllm.lazyllm_cpp')


@pytest.mark.parametrize(
    ('impl_name', 'expected_methods', 'expected_attrs', 'expected_init_types', 'expected_method_signatures'),
    [
        (
            '_TextSplitterBaseCPPImpl',
            ('split_text', '_split', '_merge'),
            ('_chunk_size', '_overlap'),
            {'chunk_size': int, 'overlap': int, 'encoding_name': str},
            {
                'split_text': ('text', 'metadata_size'),
                '_split': ('text', 'chunk_size'),
                '_merge': ('splits', 'chunk_size'),
            },
        ),
        (
            'SentenceSplitterCPPImpl',
            ('split_text', '_split', '_merge'),
            ('_chunk_size', '_overlap'),
            {'chunk_size': int, 'chunk_overlap': int, 'encoding_name': str},
            {
                'split_text': ('text', 'metadata_size'),
                '_split': ('text', 'chunk_size'),
                '_merge': ('splits', 'chunk_size'),
            },
        ),
    ],
)
def test_cpp_proxy_contract_snapshot(
    impl_name, expected_methods, expected_attrs, expected_init_types, expected_method_signatures
):
    impl_cls = getattr(lazyllm_cpp, impl_name)

    assert tuple(impl_cls.__proxy_methods__) == expected_methods
    assert tuple(impl_cls.__proxy_attrs__) == expected_attrs
    assert dict(impl_cls.__proxy_method_signatures__) == expected_method_signatures

    init_param_types = dict(impl_cls.__init_param_types__)
    assert set(init_param_types.keys()) == set(expected_init_types.keys())
    for name, expected_type in expected_init_types.items():
        assert init_param_types[name] is expected_type


def test_text_splitter_base_binding_keyword_names():
    cls = lazyllm_cpp._TextSplitterBaseCPPImpl
    splitter = cls(chunk_size=128, overlap=8, encoding_name='gpt2')
    assert isinstance(splitter.split_text(text='hello world', metadata_size=0), list)

    with pytest.raises(TypeError):
        cls(chunk=128, overlap=8, encoding_name='gpt2')

    with pytest.raises(TypeError):
        splitter.split_text(content='hello world', metadata_size=0)


def test_sentence_splitter_binding_keyword_names():
    cls = lazyllm_cpp.SentenceSplitterCPPImpl
    splitter = cls(chunk_size=128, chunk_overlap=8, encoding_name='gpt2')
    assert isinstance(splitter.split_text(text='hello world', metadata_size=0), list)

    with pytest.raises(TypeError):
        cls(chunk_size=128, overlap=8, encoding_name='gpt2')

    with pytest.raises(TypeError):
        splitter._split(content='hello world', chunk_size=128)
