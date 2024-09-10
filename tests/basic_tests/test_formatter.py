from lazyllm import formatter

class TestFormatter(object):

    def test_jsonlike_formatter(self):
        origin = [dict(a=[1, 2], b=[2, 3]), dict(a=[3, 4], b=[4, 5]), dict(a=[5, 6], b=[6, 7])]
        assert formatter.JsonLike('[:]')(origin) == origin
        assert formatter.JsonLike('[:][a]')(origin) == [[1, 2], [3, 4], [5, 6]]
        assert formatter.JsonLike('[:][a, b][0, 1]') == [[[1, 2], [2, 3]], [[3, 4], [4, 5]], [[5, 6], [6, 7]]]
