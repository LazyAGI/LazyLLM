import pytest

from lazyllm.module.llms.onlinemodule.base.onlineEmbeddingModuleBase import (
    OnlineEmbeddingModuleBase,
    _format_embed_request_error,
)


def _make_base(model_name='test-embed'):
    # Bypass __init__ (which would resolve providers / model type); _parse_response
    # only relies on self._embed_model_name, so a bare instance is enough to unit-test
    # the response-parsing and error-formatting behavior without any network access.
    inst = OnlineEmbeddingModuleBase.__new__(OnlineEmbeddingModuleBase)
    inst._embed_model_name = model_name
    return inst


class TestFormatRequestError:
    def test_includes_model_url_and_status(self):
        msg = _format_embed_request_error('m-1', 'http://x/embeddings', 503, 'upstream down')
        assert "'m-1'" in msg
        assert 'http://x/embeddings' in msg
        assert '503' in msg
        assert 'upstream down' in msg

    def test_empty_body_is_marked(self):
        msg = _format_embed_request_error('m-1', 'http://x', 500, '')
        assert '<empty>' in msg

    def test_long_body_truncated(self):
        msg = _format_embed_request_error('m-1', 'http://x', 500, 'a' * 5000)
        assert 'truncated' in msg
        assert len(msg) < 5000


class TestParseResponse:
    def test_valid_str_input(self):
        base = _make_base()
        resp = {'data': [{'embedding': [0.1, 0.2, 0.3]}]}
        assert base._parse_response(resp, input='hello') == [0.1, 0.2, 0.3]

    def test_valid_list_input(self):
        base = _make_base()
        resp = {'data': [{'embedding': [0.1]}, {'embedding': [0.2]}]}
        assert base._parse_response(resp, input=['a', 'b']) == [[0.1], [0.2]]

    def test_no_data_raises_with_context(self):
        base = _make_base('m-x')
        with pytest.raises(ValueError) as exc:
            base._parse_response({'error': 'bad key'}, input='hello')
        assert "'m-x'" in str(exc.value)

    def test_parse_response_missing_embedding_key_raises(self):
        # Bug repro: previously this returned an empty list silently, producing
        # zero-length embeddings downstream with no signal. It must now raise.
        base = _make_base()
        resp = {'data': [{'index': 0}]}  # no 'embedding' field
        with pytest.raises(ValueError):
            base._parse_response(resp, input='hello')

    def test_parse_response_missing_embedding_key_in_batch_raises(self):
        base = _make_base()
        resp = {'data': [{'embedding': [0.1]}, {'index': 1}]}  # second item malformed
        with pytest.raises(ValueError):
            base._parse_response(resp, input=['a', 'b'])
