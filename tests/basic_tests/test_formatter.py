from lazyllm import formatter
import lazyllm

class TestFormatter(object):

    def test_jsonlike_formatter_base(self):
        jsf = formatter.JsonLike

        for tp in [list, tuple, lazyllm.package]:
            origin = tp([1, 2, 3, 4, 5, 6, 7, 8])
            assert jsf('[0]')(origin) == 1
            assert jsf('[0, 3, 6]')(origin) == tp([1, 4, 7])
            assert isinstance(jsf('[0, 3, 6]')(origin), tp)
            assert jsf('[0:7:3]')(origin) == tp([1, 4, 7])
            assert isinstance(jsf('[0:7:3]')(origin), tp)
            assert jsf('[:]')(origin) == origin

        origin = dict(a=1, b=2, c=3, d=4, e=5, f=6)
        assert jsf('[a]')(origin) == 1
        assert jsf('[a, c, e]')(origin) == [1, 3, 5]
        assert jsf('[:]')(origin) == [1, 2, 3, 4, 5, 6]
        assert jsf('{a}')(origin) == dict(a=1)
        assert jsf('{a, b, c}')(origin) == dict(a=1, b=2, c=3)
        assert jsf('{:}')(origin) == origin

    def test_jsonlike_formatter_complex(self):
        jsf = formatter.JsonLike
        origin = [dict(a=[1, 2], b=[2, 3], c=[3, 4], d=[4, 5], e=[5, 6], f=[6, 7]),
                  dict(a=[10, 20], b=[20, 30], c=[30, 40], d=[40, 50], e=[50, 60], f=[60, 70])]
        assert jsf('[:]')(origin) == origin
        assert jsf('[:]{:}')(origin) == origin
        assert jsf('[:]{a, b, c, d, e, f}')(origin) == origin
        assert jsf('[0, 1]{:}')(origin) == origin
        assert jsf('[0, 1]{:}[:]')(origin) == origin
        assert jsf('[0:]{:}[0, 1]')(origin) == origin
        assert jsf('[:1]{:}[0, 1]')(origin) == [origin[0]]
        assert jsf('[1]{a, b, c, d, e, f}[:]')(origin) == origin[1]
        assert jsf('[0]{:}[0, 1]')(origin) == origin[0]

        assert jsf('[:]{a, c, e}[0:2]')(origin) == [dict(a=[1, 2], c=[3, 4], e=[5, 6]),
                                                    dict(a=[10, 20], c=[30, 40], e=[50, 60])]
        assert jsf('[:]{a, c, e}[:1]')(origin) == [dict(a=[1], c=[3], e=[5]), dict(a=[10], c=[30], e=[50])]
        assert jsf('[:]{a, c, e}[1]')(origin) == [dict(a=2, c=4, e=6), dict(a=20, c=40, e=60)]
        assert jsf('[:][a, c, e]')(origin) == [[[1, 2], [3, 4], [5, 6]], [[10, 20], [30, 40], [50, 60]]]
        assert jsf('[:][a, c, e][1:]')(origin) == [[[2], [4], [6]], [[20], [40], [60]]]
        assert jsf('[:][e, c, a][1]')(origin) == [[6, 4, 2], [60, 40, 20]]
