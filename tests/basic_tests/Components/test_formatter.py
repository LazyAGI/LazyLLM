from lazyllm import formatter
import lazyllm
from lazyllm.flow import pipeline
import pytest

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

    def test_file_formatter(self):
        # Decode
        filef = formatter.FileFormatter()
        normal_output = 'hi'
        encode_output = '<lazyllm-query>{"query": "aha", "files": ["path/to/file"]}'
        other_output = ['a', 'b']
        assert filef(normal_output) == normal_output
        assert filef(other_output) == other_output
        decode_output = filef(encode_output)
        assert decode_output == {'query': 'aha', 'files': ['path/to/file']}

        # Encode
        filef = formatter.FileFormatter(formatter='encode')
        assert filef(normal_output) == normal_output
        assert filef(encode_output) == encode_output
        assert filef(other_output) == other_output
        assert filef(decode_output) == encode_output

        # Merge
        filef = formatter.FileFormatter(formatter='merge')
        assert filef(normal_output) == normal_output
        assert filef(normal_output, normal_output, normal_output) == normal_output * 3
        assert filef(normal_output, encode_output) == '<lazyllm-query>{"query": "hiaha", "files": ["path/to/file"]}'
        assert filef(encode_output, encode_output) == ('<lazyllm-query>{"query": "ahaaha", "files": '
                                                       '["path/to/file", "path/to/file"]}')
        assert filef(encode_output, normal_output, normal_output) == ('<lazyllm-query>{"query": "ahahihi", '
                                                                      '"files": ["path/to/file"]}')

    def test_json_formatter_from_string(self):
        jsf = formatter.JsonFormatter
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

        assert jsf('[:]')('[1,2,3]') == [1, 2, 3]
        assert (jsf('[:]')('[{"age":23,"name":"张三"},{"age":24,"name":"李四"},{"age":25,"name":"王五"}]'
                           ) == [{'age': 23, 'name': '张三'}, {'age': 24, 'name': '李四'}, {'age': 25, 'name': '王五'}])
        assert (jsf('[:]')('[["张三",22],["李四",24],["王五",30],["陈六",33]]'
                           ) == [['张三', 22], ['李四', 24], ['王五', 30], ['陈六', 33]])
        assert jsf('[birthday][year]')('{"birthday":{"year":2024,"month":10},"name":"张三","age": 23}') == 2024
        assert (jsf('[0:2][age,name]')('[{"age":23,"name":"张三"},{"age":24,"name":"李四"},'
                                       '{"age":25,"name":"王五"}]') == [[23, '张三'], [24, '李四']])

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

    def test_formatter_ror(self):
        jsf = formatter.JsonFormatter('[0,2]{:}[1:3]')
        f = (lambda x, y, z: [x, y, z]) | jsf
        assert f(dict(a=[1, 2, 3, 4, 5], b=[2, 3, 4, 5, 6]), dict(a=[3, 4, 5, 6, 7], b=[4, 5, 6, 7, 8]),
                 dict(a=[5, 6, 7, 8, 9], b=[6, 7, 8, 9, 10])) == [dict(a=[2, 3], b=[3, 4]), dict(a=[6, 7], b=[7, 8])]

        f = lazyllm.pipeline(lambda x, y, z: [x, y, z])
        f2 = f | jsf
        assert f is f2
        assert f(dict(a=[1, 2, 3, 4, 5], b=[2, 3, 4, 5, 6]), dict(a=[3, 4, 5, 6, 7], b=[4, 5, 6, 7, 8]),
                 dict(a=[5, 6, 7, 8, 9], b=[6, 7, 8, 9, 10])) == [dict(a=[2, 3], b=[3, 4]), dict(a=[6, 7], b=[7, 8])]
        with pytest.raises(AssertionError):
            f._add('k', lambda x: x)
        f3 = f | formatter.JsonFormatter('[:]{a}')
        assert f3 is f
        assert f3(dict(a=[1, 2, 3, 4, 5], b=[2, 3, 4, 5, 6]), dict(a=[3, 4, 5, 6, 7], b=[4, 5, 6, 7, 8]),
                  dict(a=[5, 6, 7, 8, 9], b=[6, 7, 8, 9, 10])) == [dict(a=[2, 3]), dict(a=[6, 7])]

        f = lazyllm.pipeline(a=(lambda x, y, z: [x, y, z])) | jsf
        assert f(dict(a=[1, 2, 3, 4, 5], b=[2, 3, 4, 5, 6]), dict(a=[3, 4, 5, 6, 7], b=[4, 5, 6, 7, 8]),
                 dict(a=[5, 6, 7, 8, 9], b=[6, 7, 8, 9, 10])) == [dict(a=[2, 3], b=[3, 4]), dict(a=[6, 7], b=[7, 8])]

        with pipeline() as p:
            p.a = lambda x, y, z: [x, y, z]
            p | jsf
            p.b = formatter.JsonFormatter('[:]{a}')
        assert p(dict(a=[1, 2, 3, 4, 5], b=[2, 3, 4, 5, 6]), dict(a=[3, 4, 5, 6, 7], b=[4, 5, 6, 7, 8]),
                 dict(a=[5, 6, 7, 8, 9], b=[6, 7, 8, 9, 10])) == [dict(a=[2, 3]), dict(a=[6, 7])]

        f = lazyllm.pipeline(lambda x, y, z: [x, y, z])
        f | (jsf | formatter.JsonFormatter('[:]{a}'))
        assert f(dict(a=[1, 2, 3, 4, 5], b=[2, 3, 4, 5, 6]), dict(a=[3, 4, 5, 6, 7], b=[4, 5, 6, 7, 8]),
                 dict(a=[5, 6, 7, 8, 9], b=[6, 7, 8, 9, 10])) == [dict(a=[2, 3]), dict(a=[6, 7])]

    def test_json_repair(self):
        jsf = formatter.JsonFormatter()
        origin = '{"name": "Bob"}后面还有{name: "Charlie"}'
        result = jsf(origin)
        assert result == [{'name': 'Bob'}, {'name': 'Charlie'}]

    def test_json_repair_nested(self):
        jsf = formatter.JsonFormatter()
        origin = '{"outer": {"inner": "value"}, "arr": [1, 2, 3]}其他文本{"simple": "obj"}'
        result = jsf(origin)
        assert result == [{'outer': {'inner': 'value'}, 'arr': [1, 2, 3]}, {'simple': 'obj'}]

        origin2 = '{"level1": {"level2": {"level3": [{"nested": "array"}]}}}后面{"flat": "obj"}'
        result2 = jsf(origin2)
        assert result2 == [{'level1': {'level2': {'level3': [{'nested': 'array'}]}}}, {'flat': 'obj'}]

    def test_json_repair_string_with_braces(self):
        jsf = formatter.JsonFormatter()
        origin = '{"message": "This is {not} a brace"}后面{"code": "function() { return 1; }"}'
        result = jsf(origin)
        assert result == [{'message': 'This is {not} a brace'}, {'code': 'function() { return 1; }'}]

        origin2 = '{"pattern": "match {x} and [y]"}其他{"text": "nested {inner {braces}}"}'
        result2 = jsf(origin2)
        assert result2 == [{'pattern': 'match {x} and [y]'}, {'text': 'nested {inner {braces}}'}]

        origin3 = '{"escaped": "quote \\"with {braces}\\""}'
        result3 = jsf(origin3)
        assert result3 == {'escaped': 'quote "with {braces}"'}


class TestModuleFormatter(object):
    def test_module_formatter(self):
        jsf = formatter.JsonFormatter('[0,2]{:}[1:3]')
        m = lazyllm.ServerModule(lambda x, y, z: [x, y, z]) | jsf | formatter.JsonFormatter('[:]{a}')
        m | formatter.JsonFormatter('[:]{a}[1]')
        m.start()
        assert isinstance(m, lazyllm.ServerModule)
        assert isinstance(m._formatter, lazyllm.formatter.pipeline)
        assert m(dict(a=[1, 2, 3, 4, 5], b=[2, 3, 4, 5, 6]), dict(a=[3, 4, 5, 6, 7], b=[4, 5, 6, 7, 8]),
                 dict(a=[5, 6, 7, 8, 9], b=[6, 7, 8, 9, 10])) == [dict(a=3), dict(a=7)]
